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


import torch
import numpy as np
from PIL import Image


# This function is from guided-diffusion
# https://github.com/openai/guided-diffusion/blob/22e0df8183507e13a7813f8d38d51b072ca1e67c/guided_diffusion/nn.py#L68
# License: MIT (https://github.com/openai/guided-diffusion/blob/22e0df8183507e13a7813f8d38d51b072ca1e67c/LICENSE)
def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def sample_uv_map(inputs, flame_faces, faceim, barim, uv_grid, mode='bilinear'):
    '''
    inputs: B, N, K
    faceim: B, H, W, 1
    barim: B, H, W, 3
    uv_grid: B, G, 1, 2
    
    return: B, G, K
    '''
    map0 = inputs[:, flame_faces[faceim[..., 0]][..., 0], :]  #(B, H, W, 5)
    map1 = inputs[:, flame_faces[faceim[..., 0]][..., 1], :]
    map2 = inputs[:, flame_faces[faceim[..., 0]][..., 2], :]
    bary_map = barim[..., [0]] * map0 + barim[..., [1]] * map1 + barim[..., [2]] * map2
    torch_bary_map = bary_map.permute(0, 3, 1, 2)  # [5, H_map, W_map]
    sampled_outputs = torch.nn.functional.grid_sample(
        torch_bary_map, uv_grid,
        align_corners=False,
        mode=mode
    )[:, :, :, 0]  # [B, 3, G]
    sampled_outputs = sampled_outputs.permute(0, 2, 1)  # [B, G, 3]
    return sampled_outputs


def get_uv_sample_coords(sample_res):
    xs = torch.linspace(-1, 1, steps=sample_res)
    ys = torch.linspace(-1, 1, steps=sample_res)
    xs, ys = torch.meshgrid(xs, ys, indexing='ij')
    sampled_uv_coords = torch.stack([ys, xs], dim=-1)
    return sampled_uv_coords


def find_indices(valid_uv_coords, mouth_valid_coords):
    indices = []
    for coord in mouth_valid_coords:
        mask = (valid_uv_coords == coord).all(dim=-1)
        idx = mask.nonzero().squeeze()
        indices.append(idx.item())
    return indices


def load_mask(path):
    mask_image = Image.open(path)
    mask_gray = mask_image.convert('L')
    mask = np.array(mask_gray)
    mask = mask / 255.0
    return mask
