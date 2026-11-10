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

"""
Generate ``flame_uv_close_mouth.obj`` from the FLAME model + ``head_template.obj``.

This file cannot be downloaded from anywhere: it is the FLAME neutral shape posed into the
canonical frame, carrying the *closed-mouth* topology and UVs of ``head_template.obj``.
Concretely::

    vertices = scale * FLAME.forward(all params zero except pose_params) + trans
    faces / uvs / uv-faces = taken from head_template.obj (via FLAME.__init__'s load_obj)

with ``scale`` / ``trans`` / ``pose_params`` read from the repo-tracked
``flame_to_canonical_params.json``. This is the same computation the training code performs at
``gaia/training/uvplane.py:233-236``.

``save_obj_with_uv`` below is restored verbatim from the pre-refactor
``gaia/flame/FLAME.py`` (``git show 0d8768b^:gaia/flame/FLAME.py``, line 1127), which is the
function that produced the original asset -- hence the byte-for-byte identical output format
(``v``/``vt``/``f`` only; no normals, no ``mtllib``).

Usage (from anywhere)::

    python gaia/flame/assets/make_close_mouth_mesh.py
    python gaia/flame/assets/make_close_mouth_mesh.py --device cpu   # ~1 ULP drift, see below

NOTE on reproducibility: the original was generated on ``cuda:0``. A CPU run agrees to within
one float32 ULP (max|dv| ~ 9e-08) but is not byte-identical, so prefer a GPU when regenerating.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

# Allow running this script directly: `gaia/` is the import root (see gaia/flame/FLAME.py).
_GAIA_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _GAIA_ROOT not in sys.path:
    sys.path.insert(0, _GAIA_ROOT)

from flame.FLAME import FLAME, FLAMEAssets  # noqa: E402

# Build-time input, deliberately not part of FLAMEAssets: head_template.obj is never loaded at
# training/inference time, only here, to source the close-mouth topology and UVs.
HEAD_TEMPLATE_FILENAME = "head_template.obj"


def save_obj_with_uv(file_path, verts, faces, verts_uvs, faces_uvs):
    with open(file_path, 'w') as f:
        for v in verts:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for vt in verts_uvs:
            f.write(f"vt {vt[0]} {vt[1]}\n")
        for face, face_uv in zip(faces, faces_uvs):
            f.write(f"f {face[0]+1}/{face_uv[0]+1} {face[1]+1}/{face_uv[1]+1} {face[2]+1}/{face_uv[2]+1}\n")


def build_close_mouth_vertices(flame, params, device):
    """Apply the pre-estimated transform to the vertices"""

    def as_tensor(key):
        return torch.from_numpy(np.asarray(params[key]).astype(np.float32))[None, :].to(device)

    # Everything except the global pose is zero in flame_to_canonical_params.json; we zero them
    # explicitly anyway so a future edit to the json cannot silently change the template.
    flame_parms = {
        "shape_params":      0. * as_tensor("shape_params"),
        "expression_params": 0. * as_tensor("expression_params"),
        "pose_params":            as_tensor("pose_params"),
        "neck_pose_params":  0. * as_tensor("neck_pose_params"),
        "jaw_params":        0. * as_tensor("jaw_params"),
        "eye_pose_params":   0. * as_tensor("eye_pose_params"),
        "eyelid_params":     0. * as_tensor("eyelid_params"),
    }
    trans = as_tensor("trans")
    scale = as_tensor("scale_params")

    vertices = flame.forward(flame_parms)["vertices"]  # (1, 5023, 3)
    # NOTE: scale first, then translate -- same order as uvplane.py.
    vertices = scale * vertices + trans[:, None, :]
    return vertices


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--asset-dir", default=os.path.dirname(os.path.abspath(__file__)),
                        help="directory holding FLAME2020/, head_template.obj, flame_to_canonical_params.json")
    parser.add_argument("--output", default=None,
                        help="output path (default: <asset-dir>/flame_uv_close_mouth.obj)")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu",
                        help="torch device; the original asset was produced on cuda:0")
    args = parser.parse_args()

    # FLAME assets
    assets = FLAMEAssets(args.asset_dir)
    output = args.output or assets.flame_template_mesh_path

    head_template_path = os.path.join(args.asset_dir, HEAD_TEMPLATE_FILENAME)
    if not os.path.isfile(head_template_path):
        raise SystemExit(f"missing {head_template_path} -- run setup_assets.sh first")

    # Generate in true fp32, never TF32.
    #
    # On Ampere/Hopper, torch defaults to TF32 for matmul: a 10-bit mantissa, i.e. ~1e-3 relative
    # error. FLAME's forward is all matmuls (shapedirs/posedirs blends, LBS), so with TF32 on, this
    # GPU produced vertices differing from the reference asset by 1.9e-04 (4e-4 relative) -- about
    # 2000x worse than a CPU run, which matches the reference to 1 ULP. The reference was clearly
    # built in real fp32, so pin that here: it makes the output reproducible across CPU/GPU and
    # across GPU generations, instead of silently depending on which card you happen to be on.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # Load model and parameters.
    # NOTE: build FLAME directly rather than via load_flame(), so we can take the topology/UVs from
    # head_template.obj. load_flame() is the runtime path and always uses the runtime template --
    # which is flame_uv_close_mouth.obj, the very file we are generating (absent on a fresh clone,
    # and stale on a rerun).
    flame = FLAME(asset_dir=assets.asset_root)
    # Register head_template.obj rather than the runtime template: load_flame() would use
    # flame_uv_close_mouth.obj, the very file this script generates.
    flame.register_updated_template(head_template_path)
    flame = flame.to(args.device)
    with open(assets.flame_to_canonical_params_path, "r") as fp:
        params = json.load(fp)

    # Apply the transformation
    vertices = build_close_mouth_vertices(flame, params, args.device)

    # Topology and UVs come from head_template.obj, loaded by FLAME.__init__.
    mesh_verts = vertices[0].detach().cpu().numpy()
    mesh_faces = flame.faces.detach().cpu().numpy()
    uv = flame.verts_uvs.detach().cpu().numpy()
    faces_uvs = flame.textures_idx.detach().cpu().numpy()

    # Save
    save_obj_with_uv(output, verts=mesh_verts, faces=mesh_faces, verts_uvs=uv, faces_uvs=faces_uvs)
    print(f"wrote {output}  ({len(mesh_verts)} verts, {len(uv)} uvs, {len(mesh_faces)} faces)")


if __name__ == "__main__":
    main()
