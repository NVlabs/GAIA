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

"""GAIA dataset preprocessing, Stages 2-6.

Stage 1 (align + crop) is EG3D's, not shipped here -- see README.md.
Starting from EG3D's `final_crops/` (512x512 RGB + 25-dim dataset.json):

    2. run_modnet              final_crops/ -> preprocessed/ RGBA with alpha matte
    3. run_smirk               final_crops/ -> per-image FLAME json (+ _mirror json)
    4. mirror_dataset          preprocessed/ -> doubled with horizontally-flipped entries
    5. combine_to_dataset_json 25-dim + SMIRK FLAME -> 380-dim labels
    6. pack_zip                preprocessed/ -> FFHQ_512.zip via gaia/dataset_tool.py

Stages 2 and 3 are independent -- both read final_crops/ and can run in parallel.
"""
from .modnet import run_modnet
from .smirk import run_smirk
from .mirror import mirror_dataset
from .combine import combine_to_dataset_json
from .pack import pack_zip

__all__ = [
    'run_modnet',
    'run_smirk',
    'mirror_dataset',
    'combine_to_dataset_json',
    'pack_zip',
]
