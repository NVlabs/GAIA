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
# Portions derived from MVP (https://github.com/facebookresearch/mvp).

import numpy as np
import torch
import torch.nn as nn


# ----- defaults for upper-half-lips constraint --------------------------
# upper-half lips = first UPPER_N row-major pixels of G._lips_mask
UPPER_N_DEFAULT   = 4500
EXP_CAP_DEFAULT   = 0.01      # ch 0:3 (pos offset) L2 magcap
SCALE_CAP_DEFAULT = 0.1       # ch 3:6 (scale offset) L2 magcap
ID_CAP_DEFAULT    = 0.60      # pos_conv (id branch) L2 magcap


def build_upper_lips_mask(G, n=UPPER_N_DEFAULT):
    """Bool [H, W] mask = first n row-major True pixels of G._lips_mask, on G's device."""
    lm = G._lips_mask.detach().cpu().numpy()
    H, W = lm.shape
    rows, cols = np.nonzero(lm)
    n = min(n, len(rows))
    upper = np.zeros((H, W), dtype=bool)
    upper[rows[:n], cols[:n]] = True
    return torch.from_numpy(upper).to(next(G.parameters()).device)


def cap_l2_on_mask(value, mask, cap):
    mag = torch.linalg.norm(value, dim=1, keepdim=True)
    factor = torch.clamp(float(cap) / (mag + 1e-12), max=1.0)
    capped = value * factor
    mask = mask.unsqueeze(0).unsqueeze(0)
    return torch.where(mask.expand_as(value), capped, value)


class ExpConvPosCapOnly(nn.Module):
    """Wrap exp_conv. ONLY applies an L2 magcap on ch 0:3 (pos offset)
    """
    def __init__(self, orig_conv, damp_mask, exp_cap=EXP_CAP_DEFAULT):
        super().__init__()
        self.orig = orig_conv
        self.register_buffer('dm', damp_mask.bool())
        self.cap = float(exp_cap)

    def forward(self, x):
        out = self.orig(x).clone()
        v = out[:, 0:3]
        mag = torch.linalg.norm(v, dim=1, keepdim=True)
        factor = torch.clamp(self.cap / (mag + 1e-12), max=1.0)
        capped = v * factor
        dm = self.dm.unsqueeze(0).unsqueeze(0)
        out[:, 0:3] = torch.where(dm.expand_as(v), capped, v)
        return out


class ExpConvFreezeAndCap(nn.Module):
    """Wrap exp_conv. For temporal animation.

    Constraints (default, `enable_forehead_dyn=False`):
      Upper-lips (`damp_mask`):
        - L2 magcap on ch 0:3   (pos offset, cap=exp_cap)
        - L2 magcap on ch 3:6   (scale offset, cap=scale_cap)
        - freeze     ch 3:6     to frame-0 cache (then capped)
        - freeze     ch 10:21   (color SH) to frame-0 cache
      Whole face_mask:
        - freeze     ch 10:21   to frame-0 cache (so forehead skin color
                                stays put across frames; lips/mouth/etc. is
                                a subset and gets the same treatment).
      Untouched: ch 6:9 (rotation), ch 9 (opacity).

    `enable_forehead_dyn=True` disables the face_mask-wide ch 10:21 freeze,
    so forehead colour stays expression-dynamic. The upper-lips freeze still
    applies. Useful when you actually want forehead lighting to change.
    """
    def __init__(self, orig_conv, damp_mask, exp_cap=EXP_CAP_DEFAULT,
                 scale_cap=SCALE_CAP_DEFAULT,
                 face_mask=None, enable_forehead_dyn=False):
        super().__init__()
        self.orig = orig_conv
        self.register_buffer('dm', damp_mask.bool())
        if face_mask is not None:
            self.register_buffer('fm', face_mask.bool())
        else:
            self.fm = None
        self.cap = float(exp_cap)
        self.scale_cap = float(scale_cap)
        self.enable_forehead_dyn = bool(enable_forehead_dyn)
        self.cached = None

    def forward(self, x):
        out = self.orig(x)
        if self.cached is None:
            self.cached = out.detach().clone()
            return out
        out = out.clone()
        m = self.dm.unsqueeze(0)
        # upper-lips: freeze ch 3:6 (scale) + ch 10:21 (color SH)
        for ch in range(3, 6):
            out[:, ch] = torch.where(m, self.cached[:, ch], out[:, ch])
        for ch in range(10, 22):
            out[:, ch] = torch.where(m, self.cached[:, ch], out[:, ch])
        # whole face_mask: freeze ch 10:21 (when forehead is NOT dynamic)
        if (not self.enable_forehead_dyn) and self.fm is not None:
            fm = self.fm.unsqueeze(0)
            for ch in range(10, 22):
                out[:, ch] = torch.where(fm, self.cached[:, ch], out[:, ch])
        # ch 0:3 magcap on upper-lips
        v = out[:, 0:3]
        mag = torch.linalg.norm(v, dim=1, keepdim=True)
        factor = torch.clamp(self.cap / (mag + 1e-12), max=1.0)
        capped = v * factor
        dm = self.dm.unsqueeze(0).unsqueeze(0)
        out[:, 0:3] = torch.where(dm.expand_as(v), capped, v)
        # ch 3:6 magcap on upper-lips (applied after the ch 3:6 freeze, so
        # the post-freeze frame-0 cached values are also bounded)
        s = out[:, 3:6]
        s_mag = torch.linalg.norm(s, dim=1, keepdim=True)
        s_factor = torch.clamp(self.scale_cap / (s_mag + 1e-12), max=1.0)
        s_capped = s * s_factor
        out[:, 3:6] = torch.where(dm.expand_as(s), s_capped, s)
        return out


class PosConvMagCap(nn.Module):
    """Wrap pos_conv (3-channel output). L2 magcap."""
    def __init__(self, orig_conv, damp_mask, id_cap=ID_CAP_DEFAULT):
        super().__init__()
        self.orig = orig_conv
        self.register_buffer('dm', damp_mask.bool())
        self.cap = float(id_cap)

    def forward(self, x):
        out = self.orig(x)
        mag = torch.linalg.norm(out, dim=1, keepdim=True)
        factor = torch.clamp(self.cap / (mag + 1e-12), max=1.0)
        capped = out * factor
        dm = self.dm.unsqueeze(0).unsqueeze(0)
        return torch.where(dm.expand_as(out), capped, out)


def install_upper_lips_wrappers(G, mode='animation',
                                 upper_n=UPPER_N_DEFAULT,
                                 exp_cap=EXP_CAP_DEFAULT,
                                 scale_cap=SCALE_CAP_DEFAULT,
                                 id_cap=ID_CAP_DEFAULT,
                                 enable_forehead_dyn=False):
    """Install the upper-lips constraint wrappers on G.exp_conv and G.pos_conv.

    mode='animation' : full freeze+cap (gen_animation.py)
    mode='novel'     : pos magcap only, no freeze (suitable for single-pose,
                       multi-camera renders where temporal cache is not needed)

    enable_forehead_dyn (animation mode only):
      False (default) -> wrapper freezes ch 10:21 (color SH) across the WHOLE
                         face_mask, so forehead colour stays put.
      True            -> ch 10:21 face_mask freeze is disabled; forehead and
                         the rest of the skin colour is expression-dynamic.
                         Upper-lips freeze still applies.
    """
    upper_mask = build_upper_lips_mask(G, upper_n)
    device = next(G.parameters()).device
    if mode == 'animation':
        G.exp_conv = ExpConvFreezeAndCap(
            G.exp_conv, upper_mask, exp_cap, scale_cap=scale_cap,
            face_mask=G._face_mask, enable_forehead_dyn=enable_forehead_dyn,
        ).to(device)
    elif mode == 'novel':
        G.exp_conv = ExpConvPosCapOnly(G.exp_conv, upper_mask, exp_cap).to(device)
    else:
        raise ValueError(f'unknown mode {mode!r}')
    G.pos_conv = PosConvMagCap(G.pos_conv, upper_mask, id_cap).to(device)
    return upper_mask


def gen_tritex(vt: np.ndarray, vi: np.ndarray, vti: np.ndarray, texsize: int):
    """
    Rasterize a UV layout into per-texel triangle attributes. For a
    ``texsize x texsize`` UV image it returns, for every texel, the source
    triangle's vertex indices, UV (texel) indices, barycentric weights, and
    triangle index.

    Texels covered by more than one triangle are resolved in favor of the
    lowest triangle index. Uncovered texels keep the defaults: 0 for the index
    and barycentric maps, -1 for ``faceim``.

    Parameters
    ----------
        vt: (Nvt, 2) UV coordinates in [0, 1].
        vi: (Ntri, 3) triangle list indexing into vertex positions.
        vti: (Ntri, 3) triangle list indexing into UV coordinates.
        texsize: side length of the generated maps.

    Returns
    -------
        viim:   (texsize, texsize, 3) int32 vertex indices.
        vtiim:  (texsize, texsize, 3) int32 UV indices.
        baryim: (texsize, texsize, 3) float32 barycentric weights (u, v, w).
        faceim: (texsize, texsize, 1) int32 triangle index (-1 if uncovered).
    """
    vt = np.asarray(vt, dtype=np.float32)
    vi = np.asarray(vi, dtype=np.int32)
    vti = np.asarray(vti, dtype=np.int32)
    ntris = vi.shape[0]

    # Texel centers in UV space: pixel (r, c) samples ((c + 0.5), (r + 0.5)) / texsize.
    # Kept float64 (like the reference impl) so the barycentric math has the
    # same intermediate precision and the outputs stay bit-identical.
    axis = (np.arange(texsize) + 0.5) / texsize
    texu, texv = np.meshgrid(axis, axis)          # both (texsize, texsize)
    texuv = np.stack((texu, texv), axis=-1)       # (texsize, texsize, 2)

    # UV corners per triangle: tri_uv[i, k] is the UV of the k-th corner. (Ntri, 3, 2)
    tri_uv = vt[vti]

    viim = np.zeros((texsize, texsize, 3), dtype=np.int32)
    vtiim = np.zeros((texsize, texsize, 3), dtype=np.int32)
    baryim = np.zeros((texsize, texsize, 3), dtype=np.float32)
    faceim = np.full((texsize, texsize, 1), -1, dtype=np.int32)

    # Walk triangles high index -> low so that low-index triangles are written
    # last and therefore win any texel overlap (matches the reference).
    for i in range(ntris - 1, -1, -1):
        a, b, c = tri_uv[i]                        # three UV corners, each (2,)

        # Integer texel bounding box (1-texel margin), clamped to the image.
        lo = np.minimum(np.minimum(a, b), c) * texsize
        hi = np.maximum(np.maximum(a, b), c) * texsize
        x0 = max(0, int(lo[0]) - 1)
        x1 = min(texsize, int(hi[0]) + 2)
        y0 = max(0, int(lo[1]) - 1)
        y1 = min(texsize, int(hi[1]) + 2)
        if x1 <= x0 or y1 <= y0:
            continue

        # Barycentric coordinates via the standard edge-vector (Cramer) form.
        # Dot products use explicit element-wise multiply-and-sum (rather than
        # `@`/BLAS) so the float32 rounding matches the reference impl exactly.
        e0 = b - a
        e1 = c - a
        e2 = texuv[y0:y1, x0:x1] - a               # (h, w, 2)
        d00 = np.sum(e0 * e0)
        d01 = np.sum(e0 * e1)
        d11 = np.sum(e1 * e1)
        denom = d00 * d11 - d01 * d01
        if denom == 0.0:                           # degenerate triangle
            continue

        d20 = np.sum(e2 * e0, axis=-1)             # (h, w)
        d21 = np.sum(e2 * e1, axis=-1)             # (h, w)
        wv = (d11 * d20 - d01 * d21) / denom       # weight of corner b
        ww = (d00 * d21 - d01 * d20) / denom       # weight of corner c
        wu = 1.0 - wv - ww                         # weight of corner a
        inside = (wu >= 0.0) & (wv >= 0.0) & (ww >= 0.0)
        if not inside.any():
            continue

        baryim[y0:y1, x0:x1][inside] = np.stack((wu, wv, ww), axis=-1)[inside]
        viim[y0:y1, x0:x1][inside] = vi[i]
        vtiim[y0:y1, x0:x1][inside] = vti[i]
        faceim[y0:y1, x0:x1][inside] = i

    return viim, vtiim, baryim, faceim