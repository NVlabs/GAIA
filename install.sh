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

# GAIA setup. Run from the repo root, in this order:
#   ./install.sh submodules
#   ./install.sh env
#   conda activate <name printed by the env step>
#   ./install.sh extensions
# Please see details in README.md

set -e
ENV_NAME=$(awk '/^name:/{print $2}' environment.yml)
DGR=gaia/third_party/diff-gaussian-rasterization
SKNN=gaia/third_party/simple-knn

case "${1:-}" in

submodules)
    # No blanket --recursive: gaussian-splatting's own .gitmodules points at
    # SIBR_viewers and at different pins of the two CUDA extensions.
    git submodule update --init
    git submodule update --init --recursive $DGR   # needs third_party/glm to compile
    git submodule status
    ;;

env)
    conda env create -f environment.yml
    echo "Now run: conda activate $ENV_NAME && ./install.sh extensions"
    ;;

extensions)
    python -c "import torch" || { echo "activate the env first"; exit 1; }
    # No GPU visible (e.g. a login node)? torch cannot detect the arch.
    python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
        || export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;9.0}"
    pip install $DGR
    pip install $SKNN
    ;;

verify)
    # Import root is gaia/, not the repo root: scene/utils/gaussian_renderer/
    # arguments are symlinks that live there, and train.py runs from there too.
    cd gaia
    python -c "
import scene, utils, gaussian_renderer, arguments
import diff_gaussian_rasterization, simple_knn._C
import flame.FLAME
import gaia_utils.general_utils, gaia_utils.graphics_utils, gaia_utils.uv
import gaia_utils.activation, gaia_utils.camera_utils
import gaia_utils.gaussian_model, gaia_utils.gs_render
print('ok')
"
    ;;

*)
    sed -n '2,6p' "$0"
    exit 1
    ;;
esac
