# -*- coding: utf-8 -*-
#
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
# Attribution: this file subclasses `FLAME` from SMIRK
# (https://github.com/georgeretsi/smirk), MIT License, Copyright (c) 2024
# George Retsinas. SMIRK is consumed as a git submodule at
# `gaia/third_party/smirk` (pinned c7de404) and is not redistributed here.
#
# No upstream code is contained in this file. Its base class, and the
# Max Planck (MPG) notice that upstream carries, live in
# `gaia/third_party/smirk/src/FLAME/FLAME.py`, unmodified, where they apply.

"""
GAIA's FLAME model
We define a thin subclass of the upstream SMIRK `FLAME` (as a submodule) 
The upstream SMIRK FLAME (https://github.com/georgeretsi/smirk) is linked from "gaia/third_party/smirk".
version: c7de404. Make sure you clone the submodules.
"""

import os
import torch

from flame._smirk_flame import FLAME as _SmirkFLAME
from flame._smirk_util import face_vertices  # noqa: F401

# We use a close-mouth template. See gaia/flame/assets/setup_assets.sh for details.
STOCK_FLAME_NUM_FACES = 9976
TEMPLATE_NUM_UVS = 5150
TEMPLATE_NUM_FACES = 10006


class FLAME(_SmirkFLAME):
    """
    GAIA FLAME: subclass of the upstream SMIRK `FLAME`, consumed from the submodule.

    The upstream constructor is called as-is; this class adds only two things:

    1. `_make_buffers_contiguous()`: four of upstream's buffers (`posedirs`,
       `J_regressor`, `lmk_bary_coords`, `mp_lmk_bary_coords`) come out
       non-contiguous, and GAIA's rasterization path requires contiguous memory.
    2. `register_updated_template()`: GAIA renders with a close-mouth UV
       template (10006 faces) rather than the stock FLAME topology. Call it
       explicitly after construction; `load_flame()` does so for the runtime path.

    Upstream hardcodes three asset paths relative to the working directory
    (`assets/l_eyelid.npy`, `assets/r_eyelid.npy` and the mediapipe embedding).
    Since GAIA's assets already live in a directory named `assets`, we simply run
    the upstream constructor with the working directory set to this file's own
    directory, which makes those paths resolve to ours.
    """

    def __init__(self, asset_dir=None, n_shape=300, n_exp=50):
        assets = FLAMEAssets(asset_dir) if asset_dir is not None else FLAMEAssets()
        asset_root = os.path.abspath(assets.asset_root)
        if os.path.basename(asset_root) != "assets":
            raise ValueError(
                f"asset_dir must be a directory named 'assets' (got {asset_root}); "
                "upstream SMIRK resolves several of its own assets as 'assets/...' "
                "relative to the working directory."
            )

        # Resolve before chdir: these are relative to the *caller's* working directory.
        model_path = os.path.abspath(assets.flame_model_path)
        lmk_path = os.path.abspath(assets.flame_lmk_embedding_path)

        cwd = os.getcwd()
        try:
            os.chdir(os.path.dirname(asset_root))
            super().__init__(
                flame_model_path=model_path,
                flame_lmk_embedding_path=lmk_path,
                n_shape=n_shape,
                n_exp=n_exp,
            )
        finally:
            os.chdir(cwd)

        self._make_buffers_contiguous()

    def _make_buffers_contiguous(self):
        """`.contiguous()` returns self when already contiguous, so this copies only what it must."""
        for name, buf in list(self._buffers.items()):
            if buf is not None:
                self._buffers[name] = buf.contiguous()

    def register_updated_template(self, flame_template_mesh_path):
        # We use a different FLAME template topology with mouth closed
        # add faces and uvs
        from pytorch3d.io import load_obj
        _, faces, aux = load_obj(flame_template_mesh_path, load_textures=False)
        vertex_uvs = aux.verts_uvs
        face_uvs_idx = faces.textures_idx  # index into verts_uvs

        # TEMPLATE_NUM_FACES is 10006, not the 9976 of the original FLAME template.
        n_faces, n_uvs = faces.verts_idx.shape[0], aux.verts_uvs.shape[0]
        if (n_faces, n_uvs) != (TEMPLATE_NUM_FACES, TEMPLATE_NUM_UVS):
            raise ValueError(
                f"{flame_template_mesh_path} has {n_faces} faces / {n_uvs} uvs, expected "
                f"{TEMPLATE_NUM_FACES}/{TEMPLATE_NUM_UVS}. "
                + "Re-run gaia/flame/assets/setup_assets.sh to fix the assets."
            )

        # create uvcoords per face --> this is what you can use for uv map rendering
        # range from -1 to 1 (-1, -1) = left top; (+1, +1) = right bottom
        # pad 1 to the end
        pad = torch.ones(vertex_uvs.shape[0], 1)
        vertex_uvs = torch.cat([vertex_uvs, pad], dim=-1)
        vertex_uvs = vertex_uvs * 2 - 1
        vertex_uvs[..., 1] = -vertex_uvs[..., 1]
        face_uv_coords = face_vertices(vertex_uvs[None], face_uvs_idx[None])[0]
        self.register_buffer("face_uvcoords", face_uv_coords, persistent=False)
        self.register_buffer("faces", faces.verts_idx, persistent=False)
        self.register_buffer("verts_uvs", aux.verts_uvs, persistent=False)
        self.register_buffer("textures_idx", faces.textures_idx, persistent=False)


class FLAMEAssets:
    """
    A manifest of every FLAME-related asset path.
    All assets are expected at ``flame/assets``. See ``flame/assets/setup_assets.sh`` for details.
    """
    def __init__(self, asset_root="flame/assets"):
        self.asset_root = asset_root

        # FLAME model and mappings
        self.flame_model_path         = os.path.join(asset_root, "FLAME2020/generic_model.pkl")
        self.flame_lmk_embedding_path = os.path.join(asset_root, "landmark_embedding.npy")
        self.l_eyelid_path            = os.path.join(asset_root, "l_eyelid.npy")
        self.r_eyelid_path            = os.path.join(asset_root, "r_eyelid.npy")
        self.lmk_embeddings_mp_path   = os.path.join(asset_root, "mediapipe_landmark_embedding/mediapipe_landmark_embedding.npz")

        # uv template mesh
        self.flame_template_mesh_path    = os.path.join(asset_root, "flame_uv_close_mouth.obj")

        # flame to canonical space conversion
        self.flame_to_canonical_params_path = os.path.join(asset_root, "flame_to_canonical_params.json")

        # UV region masks
        self.uv_masks_dir = os.path.join(asset_root, "flame_uv_masks")

    def uv_mask(self, name, res512=False):
        """
        Path to a UV region mask
        Valid names: "face_eye_teeth", "eye", "lips", "eye_region", "forehead".
        Default resolution is 256. If res512 is True, resolution is 512.

        Example:
        uv_mask('eye', res512=True) returns "<self.asset_root>/flame_uv_masks/uv_eye_mask_512.png"
        """
        suffix = "_512" if res512 else ""
        return os.path.join(self.uv_masks_dir, f"uv_{name}_mask{suffix}.png")


# a util function to load with default settings
def load_flame(asset_dir="flame/assets", n_shape=300, n_exp=50, device="cuda:0"):
    # Set current assumption
    assert n_shape == 300 and n_exp == 50, "We currently assume using these two numbers."
    import time
    tic = time.time()
    assets = FLAMEAssets(asset_dir)
    flame = FLAME(asset_dir=asset_dir, n_shape=n_shape, n_exp=n_exp)
    flame.register_updated_template(assets.flame_template_mesh_path)
    flame = flame.to(device)
    print(f"loaded FLAME model to device ({device}) in {float(time.time() - tic):.2f} sec")
    return flame
