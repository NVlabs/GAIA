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

"""Stage 6: pack the post-combine directory into a zip via gaia/dataset_tool.py."""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

# Default location of EG3D-style dataset_tool.py inside this repo.
_REPO_ROOT = Path(__file__).resolve().parents[1]   # repo root (parent of dataset_preprocessing/)
_DEFAULT_DATASET_TOOL = _REPO_ROOT / 'gaia' / 'dataset_tool.py'


def pack_zip(
    source_dir: Path,
    dest_zip: Path,
    resolution: tuple[int, int] = (512, 512),
    max_images: Optional[int] = None,
    transform: Optional[str] = None,
    dataset_tool_path: Path = _DEFAULT_DATASET_TOOL,
    python_exe: str | None = None,
) -> dict:
    """Wrap gaia/dataset_tool.py to pack `source_dir` (RGBA PNGs + dataset.json)
    into `dest_zip`.

    `transform` is one of None, 'center-crop', 'center-crop-wide' (passed
    through to dataset_tool.py).
    """
    source_dir = Path(source_dir)
    dest_zip = Path(dest_zip)
    if not source_dir.exists():
        raise FileNotFoundError(f'source_dir does not exist: {source_dir}')
    dataset_tool_path = Path(dataset_tool_path)
    if not dataset_tool_path.exists():
        raise FileNotFoundError(f'dataset_tool.py not found at {dataset_tool_path}')

    cmd = [
        python_exe or sys.executable,
        str(dataset_tool_path),
        '--source', str(source_dir),
        '--dest', str(dest_zip),
        '--resolution', f'{resolution[0]}x{resolution[1]}',
    ]
    if max_images is not None:
        cmd += ['--max-images', str(max_images)]
    if transform is not None:
        cmd += ['--transform', transform]

    dest_zip.parent.mkdir(parents=True, exist_ok=True)
    rc = subprocess.run(cmd, check=False).returncode
    if rc != 0:
        raise RuntimeError(f'dataset_tool.py exited with code {rc}: {" ".join(cmd)}')

    return {
        'dest_zip': dest_zip,
        'size_bytes': dest_zip.stat().st_size if dest_zip.exists() else 0,
    }
