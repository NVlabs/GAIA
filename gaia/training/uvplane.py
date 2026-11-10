# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES.
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
# Based on EG3D (https://github.com/NVlabs/eg3d) and modified for GAIA.

import torch
from torch_utils import persistence
from training.networks_stylegan2 import Generator as StyleGAN2Backbone

import torch.nn as nn
from gaia_utils.general_utils import zero_module, load_mask, sample_uv_map, get_uv_sample_coords
import numpy as np
from gaia_utils.gaussian_model import GaussianModel
from gaia_utils.gs_render import render
from gaia_utils.camera_utils import create_new_camera
from gaia_utils.activation import mip_tanh
from utils.sh_utils import C0

from gaia_utils.graphics_utils import axis_angle_to_quaternion, compute_face_orientation, Pytorch3dRasterizer
from gaia_utils.general_utils import find_indices

from torch.nn.functional import grid_sample
from gaia_utils.uv import build_upper_lips_mask, cap_l2_on_mask, gen_tritex
import trimesh
from training.deformer import BarycentricBlendDeform
from copy import copy
from roma import rotmat_to_unitquat, quat_xyzw_to_wxyz, quat_product, quat_wxyz_to_xyzw
import json
from flame.FLAME import load_flame, FLAMEAssets, STOCK_FLAME_NUM_FACES
import os


@persistence.persistent_class
class UVPlaneGenerator(torch.nn.Module):
    def __init__(
        self,
        z_dim,                      # Input latent (Z) dimensionality.
        c_dim,                      # Conditioning label (C) dimensionality.
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output resolution.
        img_channels,               # Number of output color channels.
        device,
        mapping_kwargs      = {},   # Arguments for MappingNetwork.
        rendering_kwargs    = {},
        triplane_resolution = 256,  # Triplane resolution
        sample_res = 256,           # sample resolution in uv plane
        max_offset = 0.25,
        freeze_backbone=True,
        use_dynamic_featurev2=False,
        black_background=False,
        direct_512=False,
        use_eye_uv_reg=False,
        face_nosh=False,
        use_upper_lips_constraint=False,
        upper_lips_n=4500,
        upper_lips_exp_cap=0.01,
        upper_lips_scale_cap=0.1,
        upper_lips_id_cap=0.60,
        **synthesis_kwargs,         # Arguments for SynthesisNetwork.
    ):
        super().__init__()
        self.z_dim=z_dim
        self.c_dim=c_dim
        self.w_dim=w_dim
        self.img_resolution=img_resolution
        self.img_channels=img_channels
        self.freeze_backbone = freeze_backbone
        self.use_dynamic_featurev2 = use_dynamic_featurev2
        self.black_background = black_background
        self.direct_512 = direct_512
        self.use_eye_uv_reg = use_eye_uv_reg
        self.face_nosh = face_nosh
        self.use_upper_lips_constraint = use_upper_lips_constraint
        self.upper_lips_n = upper_lips_n
        self.upper_lips_exp_cap = upper_lips_exp_cap
        self.upper_lips_scale_cap = upper_lips_scale_cap
        self.upper_lips_id_cap = upper_lips_id_cap
        c_dim = 325  # hardcoded

        if self.direct_512:
            self.backbone = StyleGAN2Backbone(z_dim, c_dim, w_dim, img_resolution=triplane_resolution*2, img_channels=22, mapping_kwargs=mapping_kwargs, **synthesis_kwargs)
        else:
            self.backbone = StyleGAN2Backbone(z_dim, c_dim, w_dim, img_resolution=triplane_resolution, img_channels=22, mapping_kwargs=mapping_kwargs, **synthesis_kwargs)

        if self.use_dynamic_featurev2:
            if self.direct_512:
                self.backbone_exp = StyleGAN2Backbone(z_dim, self.c_dim-c_dim, w_dim+55, img_resolution=triplane_resolution*2, img_channels=22, mapping_kwargs=mapping_kwargs, **synthesis_kwargs)#use one sh degree
            else:
                self.backbone_exp = StyleGAN2Backbone(z_dim, self.c_dim-c_dim, w_dim+55, img_resolution=triplane_resolution, img_channels=22, mapping_kwargs=mapping_kwargs, **synthesis_kwargs)#use one sh degree

        self.max_offset = max_offset
        self.sample_res = sample_res

        self.pos_conv = zero_module(nn.Conv2d(3, 3, 1))

        if self.use_dynamic_featurev2:
            self.exp_conv = zero_module(nn.Conv2d(22, 22, 1))

        self.rot_conv = zero_module(nn.Conv2d(3, 3, 1))
        self.scale_conv = zero_module(nn.Conv2d(3, 3, 1))

        # all FLAME asset paths come from this manifest (single source of truth)
        flame_assets = FLAMEAssets()

        if  self.use_dynamic_featurev2:
            if self.direct_512:
                face_mask = load_mask(flame_assets.uv_mask('face_eye_teeth', res512=True))
            else:
                face_mask = load_mask(flame_assets.uv_mask('face_eye_teeth'))
            face_mask = torch.from_numpy(face_mask).bool().to(device)
            self.register_buffer("_face_mask", face_mask.contiguous())

        if self.direct_512:
            face_mask_512 = load_mask(flame_assets.uv_mask('face_eye_teeth', res512=True))
            face_mask_512 = torch.from_numpy(face_mask_512).bool().to(device)
            self.register_buffer("_face_mask_512", face_mask_512.contiguous())

        if self.use_eye_uv_reg:
            if self.direct_512:
                eye_mask = load_mask(flame_assets.uv_mask('eye', res512=True))
            else:
                eye_mask = load_mask(flame_assets.uv_mask('eye'))
            eye_mask = torch.from_numpy(eye_mask).bool().to(device)
            self.register_buffer("_eye_mask", eye_mask.contiguous())

        self.neural_rendering_resolution = 256  # hardcoded
        self.rendering_kwargs = rendering_kwargs
    
        self._last_planes = None
    
        self.device = device

        # Gaussian Model
        self.gaussians = GaussianModel(sh_degree=1)
        self.gaussians.active_sh_degree = 1

        # canonical parameters, which defines the canonical space
        flame_to_canonical_json_path = flame_assets.flame_to_canonical_params_path
        with open(flame_to_canonical_json_path, "r") as fp:
            canonical_params = json.load(fp)
        shape_params      = torch.from_numpy(np.asarray(canonical_params["shape_params"]).astype(np.float32))[None, :].to(device)       # (1, 300)
        pose_params       = torch.from_numpy(np.asarray(canonical_params["pose_params"]).astype(np.float32))[None, :].to(device)        # (1, 3)
        neck_pose_params  = torch.from_numpy(np.asarray(canonical_params["neck_pose_params"]).astype(np.float32))[None, :].to(device)   # (1, 3)
        eye_pose_params   = torch.from_numpy(np.asarray(canonical_params["eye_pose_params"]).astype(np.float32))[None, :].to(device)    # (1, 6)
        trans             = torch.from_numpy(np.asarray(canonical_params["trans"]).astype(np.float32))[None, :].to(device)              # (1, 3)
        scale_params      = torch.from_numpy(np.asarray(canonical_params["scale_params"]).astype(np.float32))[None, :].to(device)       # (1, 1)
        self.register_buffer("_shape_params", shape_params.contiguous())
        self.register_buffer("_pose_params", pose_params.contiguous())
        self.register_buffer("_neck_pose_params", neck_pose_params.contiguous())
        self.register_buffer("_eye_pose_params", eye_pose_params.contiguous())
        self.register_buffer("_trans", trans.contiguous())
        self.register_buffer("_scale", scale_params.contiguous())

        # load flame model
        self.flame = load_flame(device=self.device)

        # template
        mesh_path = flame_assets.flame_template_mesh_path
        flame_template_mesh = trimesh.load(mesh_path, process=False)
        uvs_per_flame_vertex = flame_template_mesh.visual.uv
        uv_coords = uvs_per_flame_vertex
        uv_faces = flame_template_mesh.faces
        vertices = flame_template_mesh.vertices
        faces = flame_template_mesh.faces

        # `faceim` below indexes self.flame.faces, which comes from head_template.obj, so the two
        # template meshes must share a topology. Checked here because a mismatch otherwise only
        # surfaces as a CUDA device-side assert. See flame/assets/verify_assets.py.
        if faces.shape[0] != self.flame.faces.shape[0]:
            raise ValueError(
                f"{mesh_path} has {faces.shape[0]} faces but head_template.obj has "
                f"{self.flame.faces.shape[0]}; the two must match. "
                f"Re-run gaia/flame/assets/setup_assets.sh."
            )

        if self.direct_512:
            idxim, tidxim, barim, faceim = gen_tritex(uv_coords, faces, uv_faces, triplane_resolution * 2)
        else:
            idxim, tidxim, barim, faceim = gen_tritex(uv_coords, faces, uv_faces, triplane_resolution)

        # UV masks
        # Faces at/after STOCK_FLAME_NUM_FACES are the 30 mouth-seal triangles added on top of the
        # stock FLAME head; everything before them is stock FLAME.
        mouth_mask = faceim >= STOCK_FLAME_NUM_FACES
        mouth_mask = torch.from_numpy(mouth_mask).squeeze(-1)
        self.register_buffer("_mouth_mask", mouth_mask.contiguous())

        if self.use_dynamic_featurev2 or self.use_upper_lips_constraint:
            if self.direct_512:
                lips_mask = load_mask(flame_assets.uv_mask('lips', res512=True))
            else:
                lips_mask = load_mask(flame_assets.uv_mask('lips'))
            lips_mask = torch.from_numpy(lips_mask).bool().to(device)
            self.register_buffer("_lips_mask", lips_mask.contiguous())
            if self.use_upper_lips_constraint:
                upper_lips_mask = build_upper_lips_mask(self, self.upper_lips_n)
                self.register_buffer("_upper_lips_mask", upper_lips_mask.contiguous())

        if self.use_dynamic_featurev2:
            if self.direct_512:
                eye_region_mask = load_mask(flame_assets.uv_mask('eye_region', res512=True))
                forehead_mask = load_mask(flame_assets.uv_mask('forehead', res512=True))
            else:
                eye_region_mask = load_mask(flame_assets.uv_mask('eye_region'))
                forehead_mask = load_mask(flame_assets.uv_mask('forehead'))
            eye_region_mask = torch.from_numpy(eye_region_mask).bool().to(device)
            forehead_mask = torch.from_numpy(forehead_mask).bool().to(device)
            self.register_buffer("_eye_region_mask", eye_region_mask.contiguous())
            self.register_buffer("_forehead_mask", forehead_mask.contiguous())

        self.register_buffer("_faceim", torch.from_numpy(faceim).long().contiguous())

        flame_parms_src = {
            "shape_params": self._shape_params.repeat(1, 1),
            "expression_params": torch.from_numpy(np.asarray(canonical_params["expression_params"]).astype(np.float32))[None, :].to(device),
            "pose_params": self._pose_params.repeat(1, 1),
            "neck_pose_params": self._neck_pose_params.repeat(1, 1),
            "jaw_params": torch.from_numpy(np.asarray(canonical_params["jaw_params"]).astype(np.float32))[None, :].to(device),
            "eye_pose_params": self._eye_pose_params.repeat(1, 1),
            "eyelid_params": torch.from_numpy(np.asarray(canonical_params["eyelid_params"]).astype(np.float32))[None, :].to(device),
        }
        flame_output_src = self.flame.forward(flame_parms_src)
        vertices_src = flame_output_src["vertices"]
        vertices_src = self._scale * vertices_src + self._trans[:, None, :]

        flame_faces = self.flame.faces
        v0_map = vertices_src[:, flame_faces[faceim[..., 0]][..., 0], :]  #(B, H, W, 3)
        v1_map = vertices_src[:, flame_faces[faceim[..., 0]][..., 1], :]
        v2_map = vertices_src[:, flame_faces[faceim[..., 0]][..., 2], :]
        self.register_buffer("_vertices_src", vertices_src.contiguous())
        
        sampled_uv_coords = get_uv_sample_coords(sample_res)
        self.register_buffer("_sample_res", torch.tensor([sample_res]))

        if sample_res == 512:
            sampled_uv_coords_256 = get_uv_sample_coords(256)
        
        torch_face_index_map = torch.from_numpy(idxim).permute(2, 0, 1)

        valid_uv_map = (torch_face_index_map > 0).any(dim=0).float()[None]  # [1, H_map, W_map]
        valid_samples = torch.nn.functional.grid_sample(valid_uv_map.unsqueeze(0), sampled_uv_coords.unsqueeze(0),
                                                        align_corners=False,
                                                        mode='bilinear')[0].permute(1, 2, 0)
        valid_samples = valid_samples[:, :, 0] > 0.99

        if sample_res == 512:
            valid_samples_256 = torch.nn.functional.grid_sample(valid_uv_map.unsqueeze(0), sampled_uv_coords_256.unsqueeze(0),
                                                        align_corners=False,
                                                        mode='bilinear')[0].permute(1, 2, 0)
            valid_samples_256 = valid_samples_256[:, :, 0] > 0.99
            self.register_buffer("_valid_samples_256", valid_samples_256.contiguous())

        else:
            self.register_buffer("_valid_samples_256", valid_samples.contiguous())
            
        valid_uv_coords = sampled_uv_coords[valid_samples]  # [G, 2]

        mouth_valid_samples = valid_samples * (sampled_uv_coords[:, :, 1] >= 0.8787) * (sampled_uv_coords[:, :, 1] <= 1) * (sampled_uv_coords[:, :, 0] >= -0.1843) * (sampled_uv_coords[:, :, 0] <= 0.1922)
        mouth_valid_coords = sampled_uv_coords[mouth_valid_samples]

        # get the mouth index
        mouth_indices = find_indices(valid_uv_coords, mouth_valid_coords)
        assert torch.all(valid_uv_coords[mouth_indices] == mouth_valid_coords)
        mouth_indices = torch.tensor(mouth_indices).long().to(device)
        self.register_buffer("_mouth_indices", mouth_indices.contiguous())

        uv_grid = valid_uv_coords.unsqueeze(0).unsqueeze(2).to(device)  # [1, G, 1, 2]
        self.register_buffer("_uv_grid", uv_grid.contiguous())

        barim = torch.from_numpy(barim).float().to(device)
        self.register_buffer("_barim", barim.contiguous())
        self.register_buffer("_valid_samples", valid_samples.contiguous())
        flame_position_map = barim[..., [0]] * v0_map + barim[..., [1]] * v1_map + barim[..., [2]] * v2_map  # Maps texels to 3D positions
        torch_position_map = flame_position_map.permute(0, 3, 1, 2)  # [3, H_map, W_map]

        ##-------------------sample neutral position map--------------------------------
        sampled_positions_src = torch.nn.functional.grid_sample(torch_position_map, uv_grid,
                                                            align_corners=False,
                                                            mode='bilinear')[0, :, :, 0].T  # [G, 3]
        self.register_buffer("_sampled_positions_src", sampled_positions_src.unsqueeze(0).contiguous())

        upper_teeth_indices = sampled_positions_src[mouth_indices][:, 1] >= -0.087
        lower_teeth_indices = sampled_positions_src[mouth_indices][:, 1] < -0.087
        self.register_buffer("_upper_teeth_indices", upper_teeth_indices.contiguous())
        self.register_buffer("_lower_teeth_indices", lower_teeth_indices.contiguous())

        ##-------------------sample lbs weights--------------------------------
        if sample_res == 512:
            lbs_weights = self.flame.lbs_weights
            lbs_weights = lbs_weights.unsqueeze(0)
            sampled_lbs_weights = sample_uv_map(inputs=lbs_weights, flame_faces=flame_faces, faceim=faceim, barim=barim, uv_grid=uv_grid, mode='bilinear')
            self.register_buffer("_sampled_lbs_weights", sampled_lbs_weights.contiguous())
        ##-------------------sample lbs weights--------------------------------

        face_orien_mat, face_scaling = compute_face_orientation(torch.from_numpy(vertices).float(), torch.from_numpy(faces), return_scale=True) #[n_faces, 3, 3], [n_faces, 3]
        face_orien_quat_src = quat_xyzw_to_wxyz(rotmat_to_unitquat(face_orien_mat))
        self.register_buffer("_face_orien_quat_src", face_orien_quat_src.contiguous())

        if sample_res == 512:
            torch_rot_map = face_orien_mat.reshape(face_orien_mat.shape[0], -1)[None, ...]
            torch_rot_map = torch_rot_map[:, faceim[..., 0], :].permute(0, 3, 1, 2).to(device)
            sample_orien_mat = torch.nn.functional.grid_sample(torch_rot_map.float(), self._uv_grid,
                                                    align_corners=False,
                                                    mode='nearest').squeeze(3).permute(0, 2, 1)  # [B, G, 9]
            sample_orien_mat = sample_orien_mat.reshape(sample_orien_mat.shape[0], sample_orien_mat.shape[1], 3, 3)
        else:
            sample_orien_mat = face_orien_mat[faceim[..., 0][self._valid_samples], :]
            sample_orien_mat = sample_orien_mat.unsqueeze(0).float()

        flame_quat = quat_xyzw_to_wxyz(rotmat_to_unitquat(sample_orien_mat))  # roma [B, n_faces, 4]
        self.register_buffer("_flame_quat", flame_quat.contiguous())
        sample_orien_mat = sample_orien_mat.permute(0, 1, 3, 2)
        self.register_buffer("_flame_rot", sample_orien_mat.contiguous())


    def _setup_gaussian_model(self, gaussian_attributes, i: int) -> GaussianModel:
        gaussian_positions = gaussian_attributes['pos']
        gaussian_scales = gaussian_attributes['scale']
        gaussian_rotations = gaussian_attributes['rot']
        gaussian_opacities = gaussian_attributes['opacity']
        gaussian_colors = gaussian_attributes['color']  # [B, G, SH*3]
        B, G, _ = gaussian_colors.shape
        gaussian_colors = gaussian_colors.view(B, G, -1, 3)
        self.gaussians._xyz = gaussian_positions[i]
        self.gaussians._features_dc = gaussian_colors[i][:, [0]]  # [G, 1, 3]
        self.gaussians._features_rest = gaussian_colors[i][:, 1:]  # [G, SH-1, 3]
        self.gaussians._scaling = gaussian_scales[i]
        self.gaussians._rotation = gaussian_rotations[i].contiguous()
        self.gaussians._opacity = gaussian_opacities[i]
        return self.gaussians

    def offset_activation(self, pos_map_offset):
        return self.max_offset * torch.tanh(pos_map_offset)

    def scale_activation(self, scale_map, scale_offset=-5, max_scale=-3):
        scale_map = -(scale_map + scale_offset)
        scale_map = max_scale - torch.nn.functional.softplus(scale_map + max_scale)
        return scale_map
    
    def rot_activation(self, rot_map):
        rot_map = torch.tanh(rot_map) * (2 * torch.pi)
        rot_map = axis_angle_to_quaternion(rot_map)
        rot_map = torch.cat([rot_map[..., [3]], rot_map[..., :3]], dim=-1)  # xyzr -> rxyz
        return rot_map
    
    def color_activation(self, color_map, has_sh=True):
        color_value = color_map[..., :3]  # First 3 channels are always color values
        color_value = mip_tanh(color_value, overshoot=0.001)
        color_value = color_value * (0.5 / C0)  # Force colors between [-1.78, 1.78]

        if has_sh:
            sh_value = color_map[..., 3:]
            sh_value = mip_tanh(sh_value, overshoot=0.001)
            sh_value = sh_value * (0.5 / C0)  # Force colors between [-1.78, 1.78]
            color_map = torch.cat([color_value, sh_value], dim=-1)
        else:
            color_map = color_value

        return color_map

    def get_uv_rendering(self, c, output, include_transparent_gaussians: bool = False) -> torch.Tensor:
        B = len(c)
        device = c.device
        resolution = output['image'].shape[2]
        debug_gaussian_attributes = copy(output['gaussian_attributes'])
        uv_colors = self._uv_grid.squeeze(2)  # [1, G, 2]
        uv_colors = torch.cat([uv_colors, -torch.ones((1, uv_colors.shape[1], 1), device=device)], dim=-1)  # [1, G, 3]
        debug_gaussian_attributes['color'] = uv_colors.repeat(B, 1, 1)
        if include_transparent_gaussians:
            debug_gaussian_attributes['opacity'] = torch.ones_like(debug_gaussian_attributes['opacity'])

        all_uv_renders = []
        for i, single_c in enumerate(c.cpu()):
            gaussian_model = self._setup_gaussian_model(debug_gaussian_attributes, i)
            c2w = single_c[:16].view(4, 4)
            K = single_c[16:25].view(3, 3)
            viewpoint_cam = create_new_camera(c2w, K, self.device, image_size=resolution)
            with torch.cuda.device(self.device):
                uv_render = render(
                    viewpoint_cam,
                    self.gaussians,
                    torch.tensor([1., 1., 1.], device=device),
                    device=self.device,
                    override_color=uv_colors[0]
                )
            all_uv_renders.append(uv_render['render'])

        all_uv_renders = torch.stack(all_uv_renders)
        return all_uv_renders

    def set_gaussian(self, pos, scale, rot, features_dc, features_rest, opacity):
        self.gaussians._xyz = pos
        self.gaussians._scaling = scale
        self.gaussians._rotation = rot
        self.gaussians._features_dc = features_dc
        self.gaussians._features_rest = features_rest
        self.gaussians._opacity = opacity

    def mapping(self, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False):
        c = c[:, :325]
        if self.rendering_kwargs['c_gen_conditioning_zero']:
            c = torch.zeros_like(c)
        return self.backbone.mapping(
            z,
            c * self.rendering_kwargs.get('c_scale', 0),
            truncation_psi=truncation_psi,
            truncation_cutoff=truncation_cutoff,
            update_emas=update_emas
        )

    def render_uv(self, c, vertices=None):
        cam2world_matrix = c[:, :16].view(-1, 4, 4)
        intrinsics = c[:, 16:25].view(-1, 3, 3)
        B = len(c)
        uv_imgs = []
        if vertices is None:
            shape_params = self._shape_params.repeat(len(c), 1)
            expression_params = c[:, 325:375]
            jaw_params = c[:, 375:378]
            eyelid_params = c[:, 378:380]
            flame_parms = {
                "shape_params": shape_params,
                "expression_params": expression_params,
                "pose_params": self._pose_params.repeat(len(c), 1),
                "neck_pose_params": self._neck_pose_params.repeat(len(c), 1),
                "jaw_params": jaw_params,
                "eye_pose_params": self._eye_pose_params.repeat(len(c), 1),
                "eyelid_params": eyelid_params,
            }
            flame_output = self.flame.forward(flame_parms)
            vertices = flame_output["vertices"]  # (B, 5023, 3)
            vertices = self._scale * vertices + self._trans[:, None, :]
        else:
            pass
        faces = self.flame.faces

        for bs in range(len(c)):
            c2w = cam2world_matrix[bs]
            K = intrinsics[bs]
            vertices_uv_render = vertices[bs].clone().detach()
            vertices_uv_render = torch.cat((vertices_uv_render, torch.ones_like(vertices_uv_render[:, 0:1])), dim=-1)
            try:
                w2c = torch.inverse(c2w)
            except:
                w2c = torch.eye(4).to(self.device)
            vertices_uv_render = torch.matmul(vertices_uv_render, w2c.T)
            vertices_uv_render = vertices_uv_render[:, :3]
            scale_K = K.clone()
            scale_K[:2, :] = scale_K[:2, :] * self.sample_res
            vertices_uv_render = torch.matmul(vertices_uv_render[:, :3], scale_K.T)
            vertices_uv_render[:, :2] = vertices_uv_render[:, :2] / vertices_uv_render[:, 2:3]
            vertices_uv_render[:, :2] = (vertices_uv_render[:, :2] / (self.sample_res-1)) * 2.0 - 1.0
            vertices_uv_render = vertices_uv_render.unsqueeze(0)
            vertices_uv_render[:, :, 2] = vertices_uv_render[:, :, 2] + 10  # shift past pytorch3d's near plane so nothing is culled
            attributes = self._face_uvcoords.detach().clone().expand(B, -1, -1, -1)
            rasterizer = Pytorch3dRasterizer(image_size=self.sample_res)
            flame_rendering = rasterizer(vertices_uv_render, faces[None, ...], attributes, self.sample_res, self.sample_res)
            alpha_image = flame_rendering[:, -1, :, :][:, None, :, :].detach()
            uvcoords_image = flame_rendering[:, :-1, :, :]
            uvcoords_image = alpha_image * uvcoords_image
            uvcoords_image = uvcoords_image[0]
            uv_imgs.append(uvcoords_image)
        return torch.stack(uv_imgs)

    def synthesis(
        self,
        ws,
        c,
        neural_rendering_resolution=None,
        update_emas=False,
        cache_backbone=False,
        use_cached_backbone=False,
        ws_exp=None,
        exp_cond=None,
        pose_params=None,
        exp_plane_cache=None,
        neck_pose=None,
        eye_pose=None,
        **synthesis_kwargs
    ):
        cam2world_matrix = c[:, :16].view(-1, 4, 4)
        intrinsics = c[:, 16:25].view(-1, 3, 3)
        if self.use_dynamic_featurev2:
            face_mask = self._face_mask
        else:
            face_mask = None

        if self.use_eye_uv_reg:
            eye_mask = self._eye_mask
        else:
            eye_mask = None
        if self.use_dynamic_featurev2:
            mouth_lips_mask = self._mouth_mask | self._lips_mask
            forehead_mask = self._forehead_mask
        else:
            forehead_mask = None
            mouth_lips_mask = None

        B = len(c)
        ###-------------------for animation---------------------#############
        with torch.no_grad():
            shape_params = self._shape_params.repeat(len(c), 1)
            expression_params = c[:, 325:375]
            jaw_params = c[:, 375:378]
            eyelid_params = c[:, 378:380]

            flame_parms = {
                "shape_params": shape_params,
                "expression_params": expression_params,
                "pose_params": self._pose_params.repeat(len(c), 1),
                "neck_pose_params": self._neck_pose_params.repeat(len(c), 1),
                "jaw_params": jaw_params,
                "eye_pose_params": self._eye_pose_params.repeat(len(c), 1),
                "eyelid_params": eyelid_params,
            }
            if pose_params is not None:
                flame_parms['pose_params'] = pose_params
            if neck_pose is not None:
                flame_parms['neck_pose_params'] = neck_pose
            if eye_pose is not None:
                flame_parms['eye_pose_params'] = eye_pose

        if neural_rendering_resolution is None:
            neural_rendering_resolution = self.neural_rendering_resolution
        else:
            self.neural_rendering_resolution = neural_rendering_resolution

        # Create triplanes by running StyleGAN backbone
        if use_cached_backbone and self._last_planes is not None:
            planes = self._last_planes
        else:
            planes = self.backbone.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        if cache_backbone:
            self._last_planes = planes

        re_uv_map_exp = None
        re_exp_planes_cache = None
        if self.use_dynamic_featurev2:
            ws_exp = torch.cat([ws, c[:, 325:].unsqueeze(1).repeat(1, ws.shape[1], 1)], dim=-1)
            exp_planes = self.backbone_exp.synthesis(ws_exp, update_emas=update_emas, **synthesis_kwargs)
            re_uv_map_exp = exp_planes.clone()
            exp_planes = self.exp_conv(exp_planes)
            if self.use_upper_lips_constraint and hasattr(self, '_upper_lips_mask'):
                exp_0_3 = cap_l2_on_mask(exp_planes[:, 0:3], self._upper_lips_mask, self.upper_lips_exp_cap)
                exp_3_6 = cap_l2_on_mask(exp_planes[:, 3:6], self._upper_lips_mask, self.upper_lips_scale_cap)
                exp_planes = torch.cat([exp_0_3, exp_3_6, exp_planes[:, 6:]], dim=1)

            exp_planes[:, :6, :, :] = exp_planes[:, :6, :, :] * (self._mouth_mask | self._lips_mask).unsqueeze(0).unsqueeze(0)
            exp_planes = exp_planes * self._face_mask.unsqueeze(0).unsqueeze(0)
            re_exp_planes_cache = exp_planes.clone()

            if self.face_nosh:
                exp_planes[:, 16:19, :] = exp_planes[:, 16:19, :]  * 0.
            if exp_plane_cache is not None:
                exp_planes[:, 16:19, self._face_mask & (~(self._forehead_mask))& (~(self._mouth_mask)) & (~(self._lips_mask))] = exp_plane_cache[:, 16:19, self._face_mask & (~(self._forehead_mask))& (~(self._mouth_mask)) & (~(self._lips_mask))]

        B = len(planes)
        planes = planes.view(B, 22, planes.shape[-2], planes.shape[-1])
        if self.use_dynamic_featurev2:
            exp_planes = exp_planes.view(B, 22, exp_planes.shape[-2], exp_planes.shape[-1])

        gaussian_attributes = dict()
        raw_gaussian_attributes = dict()
        raw_implict_gaussian_attributes = dict()

        if self.use_dynamic_featurev2:
            uv_map_exp = exp_planes

        zeroed_positions = self.pos_conv(
            planes[:, 0:3])
        if self.use_upper_lips_constraint and hasattr(self, '_upper_lips_mask'):
            zeroed_positions = cap_l2_on_mask(zeroed_positions, self._upper_lips_mask, self.upper_lips_id_cap)
        uv_map = torch.cat([zeroed_positions, planes[:, 3:]], dim=1)
        
        if self.use_dynamic_featurev2:
            uv_map = uv_map + uv_map_exp

        uv_attributes = grid_sample(uv_map, self._uv_grid.repeat(B, 1, 1, 1),
                                    align_corners=False,
                                    mode='bilinear')  # [B, C_uv, G, 1]

        uv_attributes = uv_attributes.squeeze(3).permute(0, 2, 1)

        pos_map_offset = uv_attributes[:, :, :3]
        scale_map = uv_attributes[:, :, 3:6]
        rot_map = uv_attributes[:, :, 6:9]
        opacity_map = uv_attributes[:, :, 9:10]
        color_map = uv_attributes[:, :, 10:22]

        raw_gaussian_attributes['pos_offset'] = pos_map_offset
        raw_gaussian_attributes['scale'] = scale_map
        raw_gaussian_attributes['rot'] = rot_map
        raw_gaussian_attributes['color'] = color_map
        raw_gaussian_attributes['opacity'] = opacity_map

        pos_map_offset = self.offset_activation(pos_map_offset)#0.25

        # Rodriguez -> Quaternion
        rot_map = self.rot_activation(rot_map)
        rot_map = rot_map.contiguous()  # Important: Rotation needs to be contiguous!
        scale_map = self.scale_activation(scale_map)
        color_map = self.color_activation(color_map)

        ############-------------------barycentric blend weight---------------------#############
        pts = self._sampled_positions_src + pos_map_offset
        if self.sample_res == 512 and not self.direct_512:
            final_pos, transform = BarycentricBlendDeform(
                query_pts=pts,
                flame=self.flame,
                param_dictionary=flame_parms,
                scale=self._scale,
                translation=self._trans[:, None, :],
                pts_faces=None,
                pts_bary=None,
                sampled_lbs_weights=self._sampled_lbs_weights,
                faceim=self._faceim,
                barim=self._barim,
                uv_grid=self._uv_grid.repeat(B, 1, 1, 1)
            )
        else:
            final_pos, transform = BarycentricBlendDeform(
                query_pts=pts,
                flame=self.flame,
                param_dictionary=flame_parms,
                scale=self._scale,
                translation=self._trans[:, None, :],
                pts_faces=self._faceim[self._valid_samples],
                pts_bary=self._barim[self._valid_samples],
                mouth_indices=self._mouth_indices,
                upper_teeth_indices=self._upper_teeth_indices,
                lower_teeth_indices=self._lower_teeth_indices
            )

        quat_transform = rotmat_to_unitquat(transform)
        final_rot = quat_product(quat_transform, quat_wxyz_to_xyzw(rot_map))
        final_rot = quat_xyzw_to_wxyz(final_rot)
        final_rot = final_rot.contiguous()

        gaussian_attributes['pos'] = final_pos
        gaussian_attributes['scale'] = scale_map
        gaussian_attributes['rot'] = rot_map
        gaussian_attributes['opacity'] = opacity_map
        gaussian_attributes['color'] = color_map

        color_map = color_map.view(color_map.shape[0], color_map.shape[1], -1, 3)

        if self.black_background:
            bg_color = [0.0, 0.0, 0.0]
        else:
            bg_color = [1.0, 1.0, 1.0]
        background = torch.tensor(bg_color, dtype=torch.float32, device=self.device)

        # get attributes of gaussians
        images = []
        depth_images = []
        uv_images = []
        neutral_images = []
        images_raw = []
        for bs in range(len(planes)):

            pos = final_pos[bs]
            scale = scale_map[bs]
            rot = final_rot[bs]
            opacity = opacity_map[bs]

            self.set_gaussian(pos=pos, scale=scale, rot=rot.contiguous(), 
                            features_dc=color_map[bs][:, [0]], 
                            features_rest=color_map[bs][:, 1:], 
                            opacity=opacity)

            c2w = cam2world_matrix[bs]
            K = intrinsics[bs]

            viewpoint_cam = create_new_camera(c2w, K, self.device, image_size=self.img_resolution)

            with torch.cuda.device(self.device):
                render_pkg = render(viewpoint_cam, self.gaussians, background, device=self.device)# 6.6869 deg * 2
                rendering = render_pkg['render'] * 2 - 1
                images.append(rendering)
                depth_images.append(rendering)
                images_raw.append(rendering)

        images = torch.stack(images)
        images_raw = torch.stack(images_raw)
        depth_images = torch.stack(depth_images)

        return {
            'image': images,
            'image_raw': images_raw,
            'image_depth': depth_images,
            'gaussian_attributes': gaussian_attributes,
            'raw_gaussian_attributes': raw_gaussian_attributes,
            'uv_map': uv_map,
            'valid_samples':self._valid_samples,
            'uv_render': uv_images,
            'valid_samples_256':self._valid_samples_256,
            'raw_implict_gaussian_attributes': raw_implict_gaussian_attributes,
            'neutral_images': neutral_images,
            'exp_cond': exp_cond,
            'face_mask': face_mask,
            'eye_mask': eye_mask,
            'uv_exp': re_uv_map_exp,
            'forehead_mask': forehead_mask,
            'mouth_lips_mask': mouth_lips_mask,
            'exp_planes_cache': re_exp_planes_cache
        }

    def forward(
        self,
        z,
        c,
        truncation_psi=1,
        truncation_cutoff=None,
        neural_rendering_resolution=None,
        update_emas=False,
        cache_backbone=False,
        use_cached_backbone=False,
        **synthesis_kwargs
    ):
        # Render a batch of generated images.
        ws = self.mapping(
            z,
            c,
            truncation_psi=truncation_psi,
            truncation_cutoff=truncation_cutoff,
            update_emas=update_emas
        )
        return self.synthesis(
            ws,
            c,
            update_emas=update_emas,
            neural_rendering_resolution=neural_rendering_resolution,
            cache_backbone=cache_backbone,
            use_cached_backbone=use_cached_backbone,
            ws_exp=None,
            exp_cond=None,
            **synthesis_kwargs
        )