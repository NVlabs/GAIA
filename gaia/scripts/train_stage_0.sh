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

# Before running:
#   --data    the dataset zip built by dataset_preprocessing/ (see dataset_preprocessing/README.md).
#   --gpus / --batch  as shipped: 8 GPUs at 4 images per GPU (--batch=32). The authors trained on 8x A100.
#             --batch must be a multiple of --gpus, and --batch/--gpus must be at least
#             --mbstd-group (default 4). Going below 4 per GPU means lowering --mbstd-group too,
#             which changes the discriminator's minibatch-std layer.

python train.py --outdir=./output/stage_0 \
--cfg=ffhq --data=/path/to/FFHQ_512.zip \
--gpus=8 --batch=32 --gamma=0.2 --gen_pose_cond=True \
--glr=0.0025 --dlr=0.002 --regularization=True \
--use_tv_reg=False --use_tv_uv_gen=False --use_opacity_reg=False --use_pos_reg=True \
--use_scale_reg=True --sample_res=256 --use_new_layer=False \
--lambda_raw_gaussian_position=0.1 \
--lambda_raw_gaussian_scale=0.05 --max_offset=0.25 \
--kimg=25000 \
--use_eye_uv_reg=True \
--shape_cond_disc=True \
--use_d2=False \
--use_lanczos=True