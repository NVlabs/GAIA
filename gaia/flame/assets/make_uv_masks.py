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
Generate the ten ``flame_uv_masks/*.png`` UV region masks.

Each mask marks the texels of the FLAME UV layout belonging to one region of the head.
``gaia/training/uvplane.py`` loads them to restrict losses and regularizers to a region,
so training cannot run without them.

Recipe: rasterize the UV layout of ``flame_uv_close_mouth.obj`` with GAIA's own
``gen_tritex`` (``gaia/gaia_utils/uv.py``), which returns the source triangle and its three
vertex indices for every texel. A texel is set iff it is covered **and any one** of its
triangle's vertices belongs to the region. Region vertex sets come from SMIRK's
``FLAME_masks.pkl``.

    region            vertex set
    ----------------- ------------------------------------------------------
    eye               left_eyeball + right_eyeball
    lips              lips
    forehead          forehead
    eye_region        eye_region
    face_eye_teeth    face + both eyeballs + the mouth-sealing triangles

"teeth" is the 30 triangles that ``head_template.obj`` adds on top of stock FLAME's 9976 to
seal the mouth. They are selected **by triangle index** (>= N_STOCK_FLAME_FACES), not by
vertex membership: their vertices are shared with neighbouring lip triangles, so selecting
by vertex over-reaches by ~24 texels at 256.

This reproduces all ten masks **bit-exactly** -- verified against the original assets, whose
hashes are pinned in ``verify_assets.py``. Run ``--check`` to re-verify without writing.

Both inputs are already required by a normal install: ``flame_uv_close_mouth.obj`` is built
one step earlier by ``make_close_mouth_mesh.py``, and ``FLAME_masks.pkl`` is tracked in the
SMIRK submodule. Nothing new is downloaded, and no FLAME-derived binary needs redistributing.

Usage (from anywhere)::

    python gaia/flame/assets/make_uv_masks.py
    python gaia/flame/assets/make_uv_masks.py --check
"""

import argparse
import os
import pickle
import sys

import numpy as np
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
_GAIA_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))   # the import root is gaia/
if _GAIA_ROOT not in sys.path:
    sys.path.insert(0, _GAIA_ROOT)

from gaia_utils.uv import gen_tritex   # noqa: E402

# Stock FLAME has 9976 faces; head_template.obj adds 30 to seal the mouth.
N_STOCK_FLAME_FACES = 9976
RESOLUTIONS = (256, 512)

# name -> (FLAME_masks keys to union, include the mouth-sealing triangles)
REGIONS = {
    "face_eye_teeth": (("face", "left_eyeball", "right_eyeball"), True),
    "eye":            (("left_eyeball", "right_eyeball"), False),
    "lips":           (("lips",), False),
    "eye_region":     (("eye_region",), False),
    "forehead":       (("forehead",), False),
}


def load_obj(path):
    """Minimal OBJ reader -> (uvs, face_vertex_idx, face_uv_idx). Triangles only."""
    uvs, fv, ft = [], [], []
    with open(path) as fp:
        for line in fp:
            w = line.split()
            if not w:
                continue
            if w[0] == "vt":
                uvs.append([float(x) for x in w[1:3]])
            elif w[0] == "f":
                corners = [t.split("/") for t in w[1:4]]
                fv.append([int(c[0]) - 1 for c in corners])
                ft.append([int(c[1]) - 1 for c in corners])
    return (np.array(uvs, dtype=np.float32), np.array(fv, dtype=np.int32),
            np.array(ft, dtype=np.int32))


def build_masks(obj_path, flame_masks_path, resolutions=RESOLUTIONS):
    """Return {relative filename: (H, W) uint8 array of 0/255}."""
    uvs, fv, ft = load_obj(obj_path)
    with open(flame_masks_path, "rb") as fp:
        flame_masks = pickle.load(fp, encoding="latin1")

    n_verts = int(fv.max()) + 1
    out = {}
    for res in resolutions:
        viim, _, _, faceim = gen_tritex(uvs, fv, ft, res)
        covered = faceim[..., 0] >= 0
        sealing = faceim[..., 0] >= N_STOCK_FLAME_FACES
        for name, (keys, with_teeth) in REGIONS.items():
            in_region = np.zeros(n_verts, dtype=bool)
            for k in keys:
                in_region[np.asarray(flame_masks[k], dtype=np.int64)] = True
            sel = in_region[viim].any(axis=-1)
            if with_teeth:
                sel |= sealing
            suffix = "_512" if res == 512 else ""
            out[f"uv_{name}_mask{suffix}.png"] = ((sel & covered) * 255).astype(np.uint8)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--asset-dir", default=_HERE, help="where flame_uv_close_mouth.obj lives")
    p.add_argument("--flame-masks", default=os.path.join(
        _GAIA_ROOT, "third_party", "smirk", "assets", "FLAME_masks", "FLAME_masks.pkl"),
        help="SMIRK's FLAME_masks.pkl")
    p.add_argument("--outdir", default=None, help="default: <asset-dir>/flame_uv_masks")
    p.add_argument("--check", action="store_true",
                   help="compare against the masks already on disk instead of writing")
    args = p.parse_args()

    obj = os.path.join(args.asset_dir, "flame_uv_close_mouth.obj")
    for path, hint in ((obj, "run make_close_mouth_mesh.py first"),
                       (args.flame_masks, "run: git submodule update --init gaia/third_party/smirk")):
        if not os.path.isfile(path):
            sys.exit(f"missing input: {path}\n  {hint}")

    outdir = args.outdir or os.path.join(args.asset_dir, "flame_uv_masks")
    masks = build_masks(obj, args.flame_masks)

    if args.check:
        bad = 0
        for fname, arr in sorted(masks.items()):
            path = os.path.join(outdir, fname)
            if not os.path.isfile(path):
                print(f"  MISSING  {fname}")
                bad += 1
                continue
            ref = np.array(Image.open(path))
            n = int((ref != arr).sum())
            print(f"  {'OK  ' if n == 0 else 'DIFF'}     {fname}"
                  + ("" if n == 0 else f"  ({n} texels)"))
            bad += n != 0
        print(f"\n{len(masks) - bad}/{len(masks)} masks match")
        return 1 if bad else 0

    os.makedirs(outdir, exist_ok=True)
    for fname, arr in sorted(masks.items()):
        # PIL defaults (compress_level=6, optimize off) reproduce the original assets
        # byte-for-byte; do not "optimize" here or the pinned hashes stop matching.
        Image.fromarray(arr, mode="L").save(os.path.join(outdir, fname))
        print(f"  wrote {fname}  ({(arr > 0).mean():.4f} coverage)")
    print(f"\n{len(masks)} masks -> {outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
