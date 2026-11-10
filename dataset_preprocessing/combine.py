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

"""Stage 5: combine source dataset.json (cam labels) + per-image SMIRK json
into a single dataset.json with 380-dim labels:

    label = cam(16+9=25) + shape(300) + expression(50) + jaw(3) + eyelid(2)

This corrects a bug in the original combine_flame_json.py: that script only
appended expression+jaw+eyelid (55), missing the 300-dim shape entirely.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


def combine_to_dataset_json(
    source_dataset_json: Path,
    smirk_dir: Path,
    dest_dataset_json: Path,
    image_dir: Optional[Path] = None,
    require_all: bool = True,
) -> dict:
    """For each (filename, cam_label[25]) in source_dataset_json:
        flame = read smirk_dir / f'{stem}.json'
        out_label = cam_label + shape(300) + expression(50) + jaw(3) + eyelid(2)
                  = 380-dim
    Writes {'labels': [...]} to dest_dataset_json.

    Parameters
    ----------
    source_dataset_json : Path
        EG3D-style dataset.json with 25-dim labels (cam pose 16 + intrinsics 9).
        Filenames are basenames like '00000.png' or '00000_mirror.png'.
    smirk_dir : Path
        Directory with per-image SMIRK json files. Each SMIRK json has fields
        `shape_params`, `expression_params`, `jaw_params`, `eyelid_params`,
        each shaped [1, *]. For mirrored entries we look for `{stem}.json`
        (e.g. for '00000_mirror.png' we read '00000_mirror.json').
    dest_dataset_json : Path
        Where to write the merged dataset.json.
    image_dir : Path, optional
        If given, asserts each entry has a matching PNG file in this directory.
    require_all : bool
        If True (default), raise on any missing SMIRK json. If False, skip
        and report n_skipped in the return dict.
    """
    src = json.load(open(source_dataset_json))
    assert 'labels' in src, f'{source_dataset_json} missing "labels" key'

    new_labels = []
    n_skipped = 0
    missing = []
    for fname, cam_label in src['labels']:
        if len(cam_label) != 25:
            raise ValueError(
                f'expected 25-dim source label, got {len(cam_label)} for {fname}'
            )
        stem = Path(fname).stem
        flame_path = Path(smirk_dir) / f'{stem}.json'
        if not flame_path.exists():
            if require_all:
                raise FileNotFoundError(f'missing SMIRK json: {flame_path}')
            missing.append(fname)
            n_skipped += 1
            continue
        flame = json.load(open(flame_path))
        if image_dir is not None:
            img_path = Path(image_dir) / fname
            if not img_path.exists():
                if require_all:
                    raise FileNotFoundError(f'missing image: {img_path}')
                missing.append(fname)
                n_skipped += 1
                continue
        full = (
            list(cam_label)                            # 25
            + list(flame['shape_params'][0])           # +300
            + list(flame['expression_params'][0])      # +50
            + list(flame['jaw_params'][0])             # +3
            + list(flame['eyelid_params'][0])          # +2
        )
        if len(full) != 380:
            raise ValueError(
                f'concatenated label is {len(full)}-dim (expected 380) for {fname}'
            )
        new_labels.append([fname, full])

    Path(dest_dataset_json).parent.mkdir(parents=True, exist_ok=True)
    with open(dest_dataset_json, 'w') as f:
        json.dump({'labels': new_labels}, f)

    return {
        'dest': Path(dest_dataset_json),
        'n_labels': len(new_labels),
        'n_skipped': n_skipped,
        'missing': missing[:20],  # first 20 for debugging
    }
