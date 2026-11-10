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
#             Stage 2 peaked at about 78 GB per GPU at 4 images per GPU.
#   --resume  a stage 1 snapshot. Training resumes its kimg counter from the number in the
#             filename (network-snapshot-KIMG.pkl), or from 0 if the name has no number.
#   INIT_K    (--init_k) set it to that KIMG. Left empty, train.py refuses to start. The
#             second discriminator's weight ramps from 0 to --d2_end_weight over the 1000 kimg
#             after this value: if it is larger than the snapshot's kimg, D2 stays off until
#             training gets there (wasted compute); if it is smaller, there is no warm-up.
#   With --gpus > 1, a --resume run may hang at exit after the final snapshot is written;
#   it is safe to kill it then.

INIT_K=

python train.py --outdir=./output/stage_2 \
--cfg=ffhq --data=/path/to/FFHQ_512.zip \
--gpus=8 --batch=32 --gamma=1 --gamma2=2 --gen_pose_cond=True \
--glr=0.0025 --dlr=0.002 --regularization=True \
--use_tv_reg=False --use_tv_uv_gen=False --use_opacity_reg=True --use_pos_reg=True \
--use_scale_reg=True --sample_res=512 --use_new_layer=True --pretrained_resolution=256 \
--lambda_raw_gaussian_position=0.1 \
--lambda_raw_gaussian_scale=0.05 --max_offset=0.25 \
--kimg=50000 \
--use_eye_uv_reg=True \
--shape_cond_disc=True \
--init_k=$INIT_K \
--warmup_weight_d2=True \
--d2_start_weight=0 \
--d2_end_weight=0.5 \
--use_d2=True \
--use_dynamic_featurev2=True \
--use_exp_reg=True \
--lambda_exp_reg=60 \
--direct_512=True \
--resume=/path/to/output/stage_1/RUN/network-snapshot-KIMG.pkl