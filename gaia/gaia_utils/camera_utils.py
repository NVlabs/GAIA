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

# Camera

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getProjectionMatrix, focal2fov


class Camera(nn.Module):
    """
    GAIA's camera: built from a precomputed world-to-camera matrix (w2c) + fov, instead
    of upstream's (R, T, image). Used only via create_new_camera in the generator render path.
    """
    def __init__(self, w2c, FoVx, FoVy, image_height, image_width, gt_alpha_mask=None,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device="cuda"):
        super(Camera, self).__init__()
        self.FoVx = FoVx
        self.FoVy = FoVy
        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device")
            self.data_device = torch.device("cuda")
        self.image_height = image_height
        self.image_width = image_width
        self.zfar = 100.0
        self.znear = 0.01
        self.trans = trans
        self.scale = scale
        self.world_view_transform = torch.tensor(w2c).transpose(0, 1).to(data_device)
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0, 1).to(data_device)
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]


def create_new_camera(c2w, K, device, image_size=256):
    c2w = c2w.detach().cpu().numpy()
    K = K.detach().cpu().numpy()
    if (c2w.sum() == 0):
        w2c = np.eye(4)
        w2c[:3, 1] = -w2c[:3, 1]
        w2c[:3, 2] = -w2c[:3, 2]
        w2c[2, 3] += 2.7
        fov = 0.233417
    else:
        w2c = np.linalg.inv(c2w)
        fov = focal2fov(K[0][0] * image_size, image_size)
    return Camera(w2c=w2c.astype(np.float32), FoVx=fov, FoVy=fov, image_height=image_size, image_width=image_size, data_device=device)
