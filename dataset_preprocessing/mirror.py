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
#
# Based on EG3D (https://github.com/NVlabs/eg3d) and modified for GAIA.

"""Stage 4: mirror dataset (horizontal flip) and append flipped entries to
dataset.json with flip_yaw'd camera poses.

Wraps the logic from dataset_preprocessing/mirror_dataset.py.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm

COMPRESS_LEVEL = 0


def flip_yaw(pose_matrix: np.ndarray) -> np.ndarray:
    """Flip a 4x4 cam2world pose matrix horizontally (yaw axis)."""
    flipped = pose_matrix.copy()
    flipped[0, 1] *= -1
    flipped[0, 2] *= -1
    flipped[1, 0] *= -1
    flipped[2, 0] *= -1
    flipped[0, 3] *= -1
    return flipped


def mirror_dataset(
    source_dir: Path,
    dest_dir: Optional[Path] = None,
    max_images: Optional[int] = None,
) -> dict:
    """Mirror images and labels in `source_dir/dataset.json`.

    For each non-mirror entry, writes a `{stem}_mirror.png` PIL.ImageOps.mirror'd
    copy to `dest_dir` and appends a flipped entry to dataset.json.

    Parameters
    ----------
    source_dir : Path
        Contains dataset.json and the original PNG files (RGBA after MODNet).
    dest_dir : Path, optional
        If None (default), mirror in-place: writes mirrored PNGs into source_dir
        and overwrites source_dir/dataset.json. If given, copies originals AND
        writes mirrored PNGs there, plus a fresh dataset.json.
    max_images : int, optional
        Process at most this many original entries (helpful for testing).

    Returns
    -------
    dict with 'dest' (Path), 'n_total' (entries in resulting dataset.json),
    'n_added' (mirrored entries added).
    """
    source_dir = Path(source_dir)
    dest_dir = Path(dest_dir) if dest_dir is not None else source_dir

    dataset_file = source_dir / 'dataset.json'
    if not dataset_file.exists():
        raise FileNotFoundError(f'no dataset.json in {source_dir}')

    with open(dataset_file) as f:
        dataset = json.load(f)

    in_place = (dest_dir == source_dir)
    if not in_place:
        dest_dir.mkdir(parents=True, exist_ok=True)

    n_originals = len(dataset['labels'])
    if max_images is not None:
        n_originals = min(n_originals, max_images)

    n_added = 0
    new_entries = []
    for i, example in enumerate(tqdm(dataset['labels'][:n_originals],
                                     total=n_originals, desc='mirror')):
        filename, label = example
        if '_mirror' in filename:
            continue

        image_path = source_dir / filename
        img = Image.open(image_path)

        if not in_place:
            dst_orig = dest_dir / filename
            dst_orig.parent.mkdir(parents=True, exist_ok=True)
            img.save(dst_orig, compress_level=COMPRESS_LEVEL)

        flipped_img = ImageOps.mirror(img)
        if len(label) < 25:
            raise ValueError(
                f'expected at least 25-dim label (16 pose + 9 intrinsics), '
                f'got {len(label)} for {filename}'
            )
        pose = np.array(label[:16]).reshape(4, 4)
        intrinsics = np.array(label[16:25]).reshape(3, 3)
        flipped_pose = flip_yaw(pose)
        # Preserve any extra label dims (shape, FLAME) verbatim — only the
        # cam pose is flipped. (Note: this matches the SMIRK *_mirror.json
        # convention which also keeps shape/exp identical to the un-mirrored
        # version. flip_yaw only operates on the camera.)
        new_label = (
            flipped_pose.reshape(-1).tolist()
            + intrinsics.reshape(-1).tolist()
            + list(label[25:])
        )
        base, ext = os.path.splitext(filename)
        flipped_filename = base + '_mirror' + ext
        new_entries.append([flipped_filename, new_label])
        n_added += 1

        flipped_path = dest_dir / flipped_filename
        flipped_path.parent.mkdir(parents=True, exist_ok=True)
        flipped_img.save(flipped_path, compress_level=COMPRESS_LEVEL)

    dataset['labels'].extend(new_entries)
    out_json = dest_dir / 'dataset.json'
    with open(out_json, 'w') as f:
        json.dump(dataset, f)

    return {
        'dest': dest_dir,
        'n_total': len(dataset['labels']),
        'n_added': n_added,
    }
