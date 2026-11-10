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

"""Sharding helper: launches N child processes, each pinned to a single GPU
via CUDA_VISIBLE_DEVICES, runs a Python callable inside each child.

Why subprocess (not torch.multiprocessing.spawn): MODNet's modnet_util.py
and SMIRK's preprocess_smirk.py both call torch.cuda at import time, so we
need each child to be a clean Python interpreter starting from a single
visible device. CUDA_VISIBLE_DEVICES set in the parent's env propagates to
each child cleanly.
"""
from __future__ import annotations

import os
import sys
import subprocess


def spawn_workers(
    cmd_template: list[str],
    num_workers: int,
    gpu_ids: list[int] | None = None,
    extra_env: dict | None = None,
) -> list[int]:
    """Run `num_workers` subprocesses in parallel, one per GPU.

    `cmd_template` is a list of command-line tokens. The substring "{rank}"
    inside any token is replaced with the worker's rank. The token
    "{world}" is replaced with `num_workers`.

    Each child gets `CUDA_VISIBLE_DEVICES=str(gpu_ids[rank])` in its env.

    Returns the list of exit codes (one per worker). Raises if any worker
    exited with non-zero.
    """
    if num_workers < 1:
        raise ValueError(f'num_workers must be >= 1, got {num_workers}')
    if gpu_ids is None:
        gpu_ids = list(range(num_workers))
    if len(gpu_ids) < num_workers:
        raise ValueError(f'need {num_workers} gpu_ids, got {len(gpu_ids)}: {gpu_ids}')

    procs = []
    for rank in range(num_workers):
        cmd = [tok.replace('{rank}', str(rank)).replace('{world}', str(num_workers))
               for tok in cmd_template]
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_ids[rank])
        if extra_env:
            env.update(extra_env)
        # rank 0 streams stdout/stderr; others go silent (their errors still
        # bubble up via exit code).
        if rank == 0:
            stdout = stderr = None
        else:
            stdout = stderr = subprocess.DEVNULL
        p = subprocess.Popen(cmd, env=env, stdout=stdout, stderr=stderr)
        procs.append(p)

    rcs = [p.wait() for p in procs]
    failed = [i for i, rc in enumerate(rcs) if rc != 0]
    if failed:
        raise RuntimeError(f'workers {failed} exited non-zero: {rcs}')
    return rcs
