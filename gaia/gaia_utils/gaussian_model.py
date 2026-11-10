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
GAIA's GaussianModel: a thin subclass of the upstream gaussian-splatting
``GaussianModel``.

Persistence note: the activation helpers must be *module-level* functions. StyleGAN/EG3D
``torch_utils.persistence`` freezes the module source and pickles each instance's ``__dict__``,
which stores ``opacity_activation`` / ``covariance_activation`` as function references — those
only pickle by qualified name if they live at module scope (upstream's nested closure would
not be picklable).
"""

import warnings

import torch
from torch_utils import persistence
from utils.general_utils import strip_symmetric, build_scaling_rotation
from scene.gaussian_model import GaussianModel as _BaseGaussianModel
from gaia_utils.activation import mip_sigmoid


def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
    L = build_scaling_rotation(scaling_modifier * scaling, rotation)
    actual_covariance = L @ L.transpose(1, 2)
    symm = strip_symmetric(actual_covariance)
    return symm


def _apply_opacity_activation(value: torch.Tensor) -> torch.Tensor:
    return mip_sigmoid(value, overshoot=0.001, clamp=False)


@persistence.persistent_class
class GaussianModel(_BaseGaussianModel):
    def __init__(self, sh_degree: int, max_scaling=0.05):
        """
        Differences from upstream:
        * use mip-style opacity activation (``_apply_opacity_activation``) instead of ``torch.sigmoid``;
        * a module-level covariance activation so instances stay picklable;
        * ``active_sh_degree`` initialised to ``sh_degree``;
        * a ``max_scaling`` field;
        * attribute setters + a ``get_color`` property used by the generator;
        * ``save_ply`` overridden to a warning -- GAIA writes no PLY output.
        """
        super().__init__(sh_degree)
        self.active_sh_degree = sh_degree
        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = _apply_opacity_activation
        self.max_scaling = max_scaling

    def save_ply(self, path):
        """
        Overrides upstream's PLY writer, which GAIA does not use.
        """
        warnings.warn(
            f"GaussianModel.save_ply({path!r}) ignored: GAIA does not write PLY files.",
            RuntimeWarning,
            stacklevel=2,
        )
