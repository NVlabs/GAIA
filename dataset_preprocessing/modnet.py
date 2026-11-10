# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Stage 2: MODNet portrait matting (RGB -> RGBA with alpha matte).

The original modnet_util.py loads the checkpoint at MODULE IMPORT TIME onto
CUDA:0. That breaks multi-GPU sharding because every child process would
race for device 0. Here we expose a clean `load_modnet(ckpt_path) -> (modnet,
im_transform)` function and call it inside each worker after CUDA_VISIBLE_DEVICES
has been pinned.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

# Resolve paths relative to this file so the function works from any cwd.
_THIS = Path(__file__).resolve()
_REPO_ROOT = _THIS.parents[1]
_MODNET_DIR = _REPO_ROOT / 'gaia' / 'third_party' / 'MODNet'
_DEFAULT_MODNET_CKPT = _MODNET_DIR / 'pretrained' / 'modnet_webcam_portrait_matting.ckpt'


def load_modnet(ckpt_path: Path = _DEFAULT_MODNET_CKPT, device: str = 'cuda',
                modnet_dir: Path = _MODNET_DIR):
    """Load the MODNet model + image transform. Call inside each worker, after
    CUDA_VISIBLE_DEVICES has been set, so the model lands on the worker's GPU.

    `modnet_dir` is the MODNet clone providing `src/models/modnet.py`; override it
    if you cloned MODNet somewhere other than gaia/third_party/MODNet.
    """
    import torch
    import torch.nn as nn
    import torchvision.transforms as transforms

    # `from src.models.modnet import MODNet` requires the MODNet clone to be on
    # sys.path. Always insert the absolute path so this works from any cwd.
    modnet_dir = Path(modnet_dir)
    if not (modnet_dir / 'src' / 'models' / 'modnet.py').exists():
        raise FileNotFoundError(
            f'MODNet clone not found at {modnet_dir}. Clone it with:\n'
            f'  git clone https://github.com/ZHKKKe/MODNet {modnet_dir}'
        )
    if str(modnet_dir) not in sys.path:
        sys.path.insert(0, str(modnet_dir))
    from src.models.modnet import MODNet

    modnet = MODNet(backbone_pretrained=False)
    modnet = nn.DataParallel(modnet)
    if torch.cuda.is_available():
        modnet = modnet.to('cuda')
        weights = torch.load(str(ckpt_path))
    else:
        weights = torch.load(str(ckpt_path), map_location=torch.device('cpu'))
    modnet.load_state_dict(weights)
    modnet.eval()

    im_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    return modnet, im_transform


def get_mask(modnet, im_transform, input_image):
    """Run MODNet on a single PIL image and return an HxW uint8 alpha matte."""
    import numpy as np
    import torch

    im = im_transform(input_image)
    im = im[None, :, :, :]
    if torch.cuda.is_available():
        im = im.to('cuda')
    _, _, matte = modnet(im, True)
    matte = matte[0].detach().cpu().numpy()
    matte = (matte * 255).astype(np.uint8)
    return matte


def _worker_main(rank: int, world: int, in_folder: Path, out_folder: Path,
                 modnet_ckpt: Path, skip_mirror: bool,
                 modnet_dir: Path = _MODNET_DIR):
    """Worker entry: run MODNet on this rank's slice of the sorted file list."""
    import glob
    from PIL import Image
    from tqdm import tqdm

    in_folder = Path(in_folder)
    out_folder = Path(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)

    files = sorted(
        list(glob.glob(str(in_folder / '*.png')))
        + list(glob.glob(str(in_folder / '*.jpg')))
    )
    if skip_mirror:
        files = [f for f in files if 'mirror' not in Path(f).name]
    # Round-robin shard so the workload distributes evenly even if file
    # processing time varies.
    my_files = files[rank::world]

    if rank == 0:
        print(f'[modnet rank=0] {len(my_files)}/{len(files)} files this shard, world={world}')
    modnet, im_transform = load_modnet(modnet_ckpt, modnet_dir=modnet_dir)

    iterator = tqdm(my_files, desc=f'modnet rank={rank}', position=rank, disable=(rank != 0))
    for file in iterator:
        img = Image.open(file)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        alpha = get_mask(modnet, im_transform, img)            # HxW uint8
        alpha_pil = Image.fromarray(alpha.squeeze(0))
        img.putalpha(alpha_pil)                                 # RGB -> RGBA
        out_path = out_folder / Path(file).name
        img.save(out_path, format='png')


def run_modnet(
    in_folder: Path,
    out_folder: Path,
    modnet_ckpt: Path = _DEFAULT_MODNET_CKPT,
    modnet_dir: Path = _MODNET_DIR,
    num_workers: int = 1,
    gpu_ids: Optional[list[int]] = None,
    skip_mirror: bool = True,
) -> dict:
    """Run MODNet matting on every PNG/JPG in `in_folder`, writing RGBA PNGs
    to `out_folder`. If num_workers > 1, launches one subprocess per GPU.

    Returns {'out_folder', 'n_processed'}.
    """
    in_folder = Path(in_folder).resolve()
    out_folder = Path(out_folder).resolve()
    modnet_ckpt = Path(modnet_ckpt).resolve()
    modnet_dir = Path(modnet_dir).resolve()
    out_folder.mkdir(parents=True, exist_ok=True)

    if num_workers <= 1:
        # Run inline in this process.
        _worker_main(0, 1, in_folder, out_folder, modnet_ckpt, skip_mirror,
                     modnet_dir=modnet_dir)
    else:
        from ._sharding import spawn_workers
        # Build a one-liner that re-invokes this module's worker entry.
        cmd = [
            sys.executable, '-c',
            'import sys; sys.path.insert(0, "{pp}"); '
            'from dataset_preprocessing.modnet import _worker_main; '
            'from pathlib import Path; '
            '_worker_main({rank}, {world}, '
            f'Path("{in_folder}"), Path("{out_folder}"), '
            f'Path("{modnet_ckpt}"), {skip_mirror!r}, '
            f'modnet_dir=Path("{modnet_dir}"))'
            .replace('{pp}', str(_REPO_ROOT)),
        ]
        spawn_workers(cmd, num_workers=num_workers, gpu_ids=gpu_ids)

    n = sum(1 for _ in out_folder.glob('*.png'))
    return {'out_folder': out_folder, 'n_processed': n}
