# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
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
# Derived from SMIRK (https://github.com/georgeretsi/smirk), MIT License,
# Copyright (c) 2024 George Retsinas. Adapted from the upstream `demo.py`
# to extract per-frame FLAME parameters over a video's frames.

"""Make a --flame-params json for gaia/gen_animation.py from a folder of video frames.

Per frame, SMIRK's `demo.py --crop` path: MediaPipe face landmarks -> square box at
scale 1.4 -> similarity warp to 224x224 -> SmirkEncoder (SMIRK_em1.pt). The raw encoder
outputs are stored per frame -- no smoothing, no shape averaging.

    python dataset_preprocessing/make_flame_params.py --frames <dir of *.png|*.jpg> \
        --out driving.json [--start 0] [--count 500] [--device cuda]

Frames are taken in filename order, numerically when the names are numbers
(0.png .. 1000.png). Needs the same environment and SMIRK downloads as Stage 3 (see README.md).
"""
import argparse, glob, json, os, sys

import numpy as np

_SMIRK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'gaia', 'third_party', 'smirk')
KEYS = ['pose_params', 'cam', 'shape_params', 'expression_params', 'eyelid_params', 'jaw_params']

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--frames', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--start', type=int, default=0, help='index into the sorted frame list')
ap.add_argument('--count', type=int, default=0, help='0 = to the end')
ap.add_argument('--device', default='cuda')
args = ap.parse_args()


def frame_key(path):  # numeric when the stem is a number: unpadded 0.png, 1.png, ..., 10.png sort in time order
    stem = os.path.splitext(os.path.basename(path))[0]
    return (0, int(stem), '') if stem.isdigit() else (1, 0, stem)


files = sorted(glob.glob(os.path.join(args.frames, '*.png')) + glob.glob(os.path.join(args.frames, '*.jpg')), key=frame_key)
files = [os.path.abspath(f) for f in files[args.start:args.start + args.count if args.count else None]]
if not files:
    sys.exit(f'no frames selected in {args.frames}')
out = os.path.abspath(args.out)

sys.path.insert(0, os.path.abspath(_SMIRK_DIR))
os.chdir(_SMIRK_DIR)  # SMIRK resolves assets/ and pretrained_models/ relative to its root
import cv2, torch
from skimage.transform import estimate_transform, warp
from tqdm import tqdm
from src.smirk_encoder import SmirkEncoder
from utils.mediapipe_utils import run_mediapipe


def crop_face(landmarks, scale=1.4, image_size=224):  # demo.py
    left, right = np.min(landmarks[:, 0]), np.max(landmarks[:, 0])
    top, bottom = np.min(landmarks[:, 1]), np.max(landmarks[:, 1])
    old_size = (right - left + bottom - top) / 2
    center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
    size = int(old_size * scale)
    src = np.array([[center[0] - size / 2, center[1] - size / 2], [center[0] - size / 2, center[1] + size / 2],
                    [center[0] + size / 2, center[1] - size / 2]])
    dst = np.array([[0, 0], [0, image_size - 1], [image_size - 1, 0]])
    return estimate_transform('similarity', src, dst)


enc = SmirkEncoder().to(args.device)
ck = torch.load('pretrained_models/SMIRK_em1.pt', map_location=args.device)
enc.load_state_dict({k.replace('smirk_encoder.', ''): v for k, v in ck.items() if 'smirk_encoder' in k})
enc.eval()

params = {k: [] for k in KEYS}
missed = []
for f in tqdm(files):
    img = cv2.imread(f)
    kpt = run_mediapipe(img)
    if kpt is None:
        missed.append(os.path.basename(f))
        continue
    t = crop_face(kpt[..., :2])
    crop = warp(img, t.inverse, output_shape=(224, 224), preserve_range=True).astype(np.uint8)
    x = torch.tensor(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).unsqueeze(0).float().div(255.0)
    with torch.no_grad():
        o = enc(x.to(args.device))
    for k in KEYS:
        params[k].append(o[k][0].cpu().numpy().tolist())

# A dropped frame would silently shift every later frame, so refuse rather than write a gappy file.
if missed:
    sys.exit(f'MediaPipe found no face in {len(missed)} frame(s), e.g. {missed[:5]}; nothing written')
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, 'w') as fh:
    json.dump(params, fh)
print(f'wrote {out}: {len(files)} frames ({os.path.basename(files[0])} .. {os.path.basename(files[-1])})')
