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
Placeholder for the ``plyfile`` package (GPL-3.0-or-later), which GAIA does not use.

The unmodified gaussian-splatting submodule imports it at module scope
(``scene/gaussian_model.py:18``, ``scene/dataset_readers.py:22``), so the name has to resolve.
GAIA reads and writes no PLY files, so it never has to work. ``gaia/`` is ``sys.path[0]`` for
every entry point, so this module resolves ahead of a real ``plyfile`` if one is installed.
"""


class _Placeholder:
    """Binds an upstream import; raises on any use."""

    def __init__(self, name):
        self._name = name

    def _raise(self, attr):
        raise NotImplementedError(
            f"plyfile.{self._name}.{attr} was called, but GAIA ships a placeholder instead of "
            f"the GPL-licensed plyfile package: GAIA does not read or write PLY files."
        )

    def __getattr__(self, attr):
        self._raise(attr)

    def __call__(self, *args, **kwargs):
        self._raise("__call__")


PlyData = _Placeholder("PlyData")
PlyElement = _Placeholder("PlyElement")
