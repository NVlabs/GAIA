#!/bin/bash
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

# Set up the FLAME assets GAIA needs. Run from this directory:
#
#     cd gaia/flame/assets && bash setup_assets.sh
#
# You supply FLAME 2020 yourself (licence-gated). SMIRK assets are symlinked from the
# pinned submodule; only head_template.obj is downloaded, from a pinned commit.
# verify_assets.py runs at the end and holds the expected hashes -- do not repeat them here.

set -euo pipefail

SMIRK_ASSETS=../../third_party/smirk/assets
FLASHAVATAR_COMMIT=5d6cf43   # the commit recorded in THIRD-PARTY.txt

# Everything below uses paths relative to this directory, so refuse to run from anywhere
# else -- otherwise the symlinks and the download land in the wrong place.
if [ ! -f make_close_mouth_mesh.py ]; then
    echo "run this from gaia/flame/assets/"; exit 1
fi

# ---------------------------------------------------------------------------
# FLAME 2020 -- you download it. Not scripted on purpose: the MPI licence is one a
# person accepts, so no credentials are handled here.
# ---------------------------------------------------------------------------
if [ ! -f FLAME2020/generic_model.pkl ]; then
    cat <<'MSG'

FLAME 2020 is missing. It is not redistributed with GAIA and is not downloaded
by this script. Get it yourself:

  1. Accept the licence at https://flame.is.tue.mpg.de/ and download FLAME2020.zip
  2. Unzip it here, so that FLAME2020/generic_model.pkl exists
  3. Re-run this script

Dataset preprocessing needs FLAME 2020 too, but that path already symlinks here.

MSG
    exit 1
fi
echo "FLAME 2020 present."

# ---------------------------------------------------------------------------
# SMIRK assets -- taken from the pinned submodule, never downloaded.
# ---------------------------------------------------------------------------
if [ ! -f "$SMIRK_ASSETS/landmark_embedding.npy" ]; then
    echo
    echo "SMIRK submodule assets not found at $SMIRK_ASSETS"
    echo "Initialise it first, from the repo root:"
    echo "    ./install.sh submodules"
    exit 1
fi

echo -e "\nLinking FLAME-related assets from the pinned SMIRK submodule..."
for f in landmark_embedding.npy l_eyelid.npy r_eyelid.npy mediapipe_landmark_embedding; do
    # Remove first. `ln -sfn` into an existing *real* directory creates the link INSIDE it
    # (mediapipe_landmark_embedding/mediapipe_landmark_embedding) instead of replacing it,
    # which is exactly what an earlier install of this script leaves behind.
    rm -rf "$f"
    ln -s "$SMIRK_ASSETS/$f" "$f"
    echo "  $f -> $SMIRK_ASSETS/$f"
done

# NOTE: do NOT link head_template.obj from SMIRK. The submodule ships the 9976-face
# open-mouth FLAME head; GAIA's UV pipeline needs FlashAvatar's 10006-face close-mouth
# topology below. Using SMIRK's is accepted silently by every loader and then crashes
# training with a CUDA device-side assert. verify_assets.py recognises that file by hash.

# ---------------------------------------------------------------------------
# head_template.obj -- FlashAvatar, pinned commit + hash check.
# ---------------------------------------------------------------------------
echo -e "\nDownloading head_template.obj from FlashAvatar (pinned ${FLASHAVATAR_COMMIT:0:7})..."
curl --fail --location --retry 3 \
     "https://raw.githubusercontent.com/USTC3DV/FlashAvatar-code/${FLASHAVATAR_COMMIT}/flame/FlameMesh.obj" \
     --output head_template.obj

# ---------------------------------------------------------------------------
# Generated locally from the above.
# ---------------------------------------------------------------------------
echo -e "\nGenerating flame_uv_close_mouth.obj..."
python make_close_mouth_mesh.py

echo -e "\nGenerating flame_uv_masks/..."
python make_uv_masks.py

# ---------------------------------------------------------------------------
# Gate: every downloaded artifact is sha256-pinned in verify_assets.py.
# ---------------------------------------------------------------------------
echo -e "\nVerifying assets..."
python verify_assets.py

echo -e "\nAsset setup is done."
