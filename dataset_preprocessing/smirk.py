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

"""Stage 3: SMIRK FLAME parameter extraction.

Wraps preprocess_smirk.py, run against the gaia/third_party/smirk submodule. Each input image
gets two output JSONs: `{stem}.json` (the original) and `{stem}_mirror.json`
(the horizontally-flipped FLAME params; image is NOT actually mirrored here —
that's mirror_dataset's job in Stage 4).
"""
from __future__ import annotations

import os
import sys
import subprocess
from pathlib import Path
from typing import Optional

# Resolve paths relative to this file so it works from any cwd.
_THIS = Path(__file__).resolve()
_REPO_ROOT = _THIS.parents[1]
_SMIRK_DIR = _REPO_ROOT / 'gaia' / 'third_party' / 'smirk'   # existing submodule
_DEFAULT_SMIRK_CKPT = _SMIRK_DIR / 'pretrained_models' / 'SMIRK_em1.pt'
_DEFAULT_LANDMARKER = _SMIRK_DIR / 'assets' / 'face_landmarker.task'
_PREPROCESS_SMIRK_PY = _THIS.parent / 'preprocess_smirk.py'


def _shard_run(rank: int, world: int, in_folder: Path, out_folder: Path,
               crop: bool, max_images: Optional[int],
               skip_mirror: bool = True) -> None:
    """Run preprocess_smirk.py on this rank's slice of the sorted files.

    We pick the slice ourselves (round-robin) and hand the child a symlink dir
    holding exactly it, with length=len(slice) and offset=0. The child re-globs
    its own --in_folder, so this is the only way its slice is guaranteed to be
    ours.
    """
    import glob

    in_folder = Path(in_folder)
    out_folder = Path(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)

    files = sorted(
        list(glob.glob(str(in_folder / '*.png')))
        + list(glob.glob(str(in_folder / '*.jpg')))
    )
    if skip_mirror:
        # EG3D's runme.py mirrors final_crops/ before we ever see it.
        # preprocess_smirk.py skips `*_mirror.png` itself, but it still counts
        # them in the glob it slices with --length/--offset, so they must not
        # reach it. run_modnet drops them for the same reason.
        files = [f for f in files if 'mirror' not in Path(f).name]
    if max_images is not None:
        files = files[:max_images]
    my_files = files[rank::world] if world > 1 else files

    # Always hand the child a directory holding exactly our file list, even
    # at world=1. preprocess_smirk.py re-globs its own --in_folder and takes
    # files[length*offset : length*(offset+1)], so --length is only meaningful
    # if the child's glob matches ours. Pointing it at an in_folder that holds
    # anything we filtered out (EG3D's mirrors) silently truncates the run --
    # 400 files in the folder against --length 200 processed half the ids.
    slice_in = out_folder / f'.shard_{rank}_input'
    slice_in.mkdir(parents=True, exist_ok=True)
    for f in slice_in.glob('*.png'):
        f.unlink()
    for f in my_files:
        (slice_in / Path(f).name).symlink_to(f)

    cmd = [
        sys.executable, str(_PREPROCESS_SMIRK_PY),
        '--in_folder', str(slice_in),
        '--out_folder', str(out_folder),
        '--length', str(len(my_files)),
        '--offset', '0',
    ]
    if crop:
        cmd.append('--crop')

    # Run from the smirk/ directory so the script's relative asset paths
    # (pretrained_models/SMIRK_em1.pt, assets/face_landmarker.task) resolve.
    # This script lives outside the submodule, so sys.path[0] is our own
    # directory, not SMIRK's -- put SMIRK on PYTHONPATH for its `src.`,
    # `utils.` and `datasets.` imports.
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join(
        [str(_SMIRK_DIR)] + ([env['PYTHONPATH']] if env.get('PYTHONPATH') else [])
    )
    rc = subprocess.run(cmd, cwd=_SMIRK_DIR, env=env, check=False).returncode
    if rc != 0:
        raise RuntimeError(f'preprocess_smirk.py rank={rank} exited {rc}')


def run_smirk(
    in_folder: Path,
    out_folder: Path,
    smirk_ckpt: Path = _DEFAULT_SMIRK_CKPT,
    landmarker_task: Path = _DEFAULT_LANDMARKER,
    crop: bool = True,
    num_workers: int = 1,
    gpu_ids: Optional[list[int]] = None,
    max_images: Optional[int] = None,
    skip_mirror: bool = True,
) -> dict:
    """Run SMIRK FLAME extraction on every PNG/JPG in `in_folder`. For each
    image `{stem}.png`, writes `{stem}.json` (FLAME params) and
    `{stem}_mirror.json` (mirrored FLAME params for the data-augmentation
    flipped copy).

    smirk_ckpt and landmarker_task must already exist on disk; preprocess_smirk.py
    reads them from paths relative to its cwd, which is the SMIRK submodule, so as
    long as those files are there they'll be found.

    Returns {'out_folder', 'n_processed'}.
    """
    in_folder = Path(in_folder).resolve()
    out_folder = Path(out_folder).resolve()
    out_folder.mkdir(parents=True, exist_ok=True)
    if not Path(smirk_ckpt).exists():
        raise FileNotFoundError(f'SMIRK checkpoint not found: {smirk_ckpt}')
    if not Path(landmarker_task).exists():
        raise FileNotFoundError(f'face_landmarker.task not found: {landmarker_task}')
    if not _PREPROCESS_SMIRK_PY.exists():
        raise FileNotFoundError(f'preprocess_smirk.py not found: {_PREPROCESS_SMIRK_PY}')
    # SMIRK's src/FLAME/FLAME.py hardcodes 'assets/FLAME2020/generic_model.pkl'
    # relative to its own repo root, so a FLAME download living anywhere else
    # (e.g. gaia/flame/assets/) is invisible to it. Check here: otherwise this
    # surfaces as a bare FileNotFoundError from inside FLAME.__init__ after the
    # models have already loaded.
    flame_pkl = _SMIRK_DIR / 'assets' / 'FLAME2020' / 'generic_model.pkl'
    if not flame_pkl.exists():
        raise FileNotFoundError(
            f'FLAME 2020 not found at {flame_pkl}.\nSMIRK resolves it relative to '
            f'its own repo root. If you already downloaded FLAME 2020 for GAIA, '
            f'symlink it:\n  ln -s "$PWD/gaia/flame/assets/FLAME2020" '
            f'{_SMIRK_DIR}/assets/FLAME2020\nSee dataset_preprocessing/README.md '
            f'and gaia/flame/NOTE.md.')

    if num_workers <= 1:
        _shard_run(0, 1, in_folder, out_folder, crop, max_images, skip_mirror)
    else:
        from ._sharding import spawn_workers
        cmd = [
            sys.executable, '-c',
            'import sys; sys.path.insert(0, "{pp}"); '
            'from dataset_preprocessing.smirk import _shard_run; '
            'from pathlib import Path; '
            '_shard_run({rank}, {world}, '
            f'Path("{in_folder}"), Path("{out_folder}"), '
            f'crop={crop!r}, max_images={max_images!r}, '
            f'skip_mirror={skip_mirror!r})'
            .replace('{pp}', str(_REPO_ROOT)),
        ]
        spawn_workers(cmd, num_workers=num_workers, gpu_ids=gpu_ids)

    n = sum(1 for p in out_folder.glob('*.json') if not p.name.endswith('_mirror.json'))
    return {'out_folder': out_folder, 'n_processed': n}
