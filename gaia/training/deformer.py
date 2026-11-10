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
Deformation of Gaussian attributes
"""

import torch


from flame.lbs import blend_shapes, vertices2joints, batch_rodrigues, batch_rigid_transform
from gaia_utils.general_utils import sample_uv_map

def BarycentricBlendDeform(
    query_pts,
    flame,
    param_dictionary,
    scale,
    translation,
    pts_faces,
    pts_bary,
    sampled_lbs_weights=None,
    faceim=None,
    barim=None,
    uv_grid=None,
    mouth_indices=None,
    upper_teeth_indices=None,
    lower_teeth_indices=None
):
    '''
    query_pts: (B, N, 3) tensor of points to deform
    vertices_src: (B, V, 3) tensor of source vertices
    pts_faces: (N, 1) tensor of point's face id
    pts_bary: (N, 3) tensor of point's bary centric coefficients
    '''
    # Extract parameters
    shape_params = param_dictionary['shape_params']
    expression_params = param_dictionary['expression_params']
    pose_params = param_dictionary.get('pose_params', None)
    jaw_params = param_dictionary.get('jaw_params', None)
    eye_pose_params = param_dictionary.get('eye_pose_params', None)
    neck_pose_params = param_dictionary.get('neck_pose_params', None)
    eyelid_params = param_dictionary.get('eyelid_params', None)
    batch_size = shape_params.shape[0]

    n_exp = 50
    n_shape = 300
    if expression_params.shape[1] < n_exp:
        expression_params = torch.cat(
            [
                expression_params,
                torch.zeros(expression_params.shape[0], n_exp - expression_params.shape[1]).to(shape_params.device)
            ],
            dim=1
        )

    if shape_params.shape[1] < n_shape:
        shape_params = torch.cat(
            [
                shape_params,
                torch.zeros(shape_params.shape[0], n_shape - shape_params.shape[1]).to(shape_params.device)
            ],
            dim=1
        )

    default_eyball_pose = torch.zeros([1, 6], dtype=expression_params.dtype, requires_grad=False)

    if pose_params is None:
        raise ValueError("pose_params is required for BarycentricBlendDeform")
    
    if eye_pose_params is None:
        default_eyball_pose = torch.zeros([1, 6], dtype=expression_params.dtype, requires_grad=False)
        eye_pose_params = default_eyball_pose.expand(batch_size, -1)

    if neck_pose_params is None:
        default_neck_pose = torch.zeros([1, 3], dtype=expression_params.dtype, requires_grad=False)
        neck_pose_params = default_neck_pose.expand(batch_size, -1)

    # extract meshes
    betas = torch.cat([shape_params, expression_params], dim=1)
    full_pose = torch.cat([pose_params, neck_pose_params, jaw_params, eye_pose_params], dim=1)
    template_vertices = flame.v_template.unsqueeze(0).expand(batch_size, -1, -1)
    device = template_vertices.device

    batch_size = max(betas.shape[0], full_pose.shape[0])

    lbs_weights = flame.lbs_weights
    dtype = flame.dtype

    B = query_pts.shape[0]
    faces = flame.faces # F, 3

    # Skinning transforms and the two vertex offsets, composed from SMIRK's LBS
    # primitives. FLAME's own lbs() computes all of this and then returns only the
    # skinned vertices, but GAIA skins Gaussians rather than mesh vertices, so it
    # needs the pieces: A (per-joint transforms) and the shape/pose offsets, which
    # are resampled onto the Gaussians below.
    shape_offsets = blend_shapes(betas, flame.shapedirs)                    # (B, V, 3)
    v_shaped = template_vertices + shape_offsets
    joints = vertices2joints(flame.J_regressor, v_shaped)                   # (B, J, 3)

    rot_mats = batch_rodrigues(full_pose.view(-1, 3), dtype=flame.dtype).view([batch_size, -1, 3, 3])

    # Pose blendshapes: FLAME parameterises them by the joint rotations relative to
    # the rest pose, so subtract the identity from every joint but the root and
    # project through posedirs.
    ident = torch.eye(3, dtype=flame.dtype, device=device)
    pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1])   # (B, (J-1)*9)
    pose_offsets = torch.matmul(pose_feature, flame.posedirs).view(batch_size, -1, 3)

    _, A = batch_rigid_transform(rot_mats, joints, flame.parents, dtype=flame.dtype)

    if faceim is not None:
        bary_shape_offsets = sample_uv_map(inputs=shape_offsets, flame_faces=faces, faceim=faceim, barim=barim, uv_grid=uv_grid, mode='bilinear')
        bary_pose_offsets = sample_uv_map(inputs=pose_offsets, flame_faces=faces, faceim=faceim, barim=barim, uv_grid=uv_grid, mode='bilinear')
        
    else:
        idx = faces[pts_faces.squeeze(-1)] #N, 3 # vertex indices for each face
        idx = idx.unsqueeze(0).expand(B, -1, -1) #B, N, 3
        pts_bary = pts_bary.unsqueeze(0).expand(B, -1, -1) #B, N, 3
        #shape offset: B, V, 3
        shape_offsets_nearest = torch.gather(shape_offsets.unsqueeze(2).expand(-1, -1, 3, -1), 1, idx.unsqueeze(3).expand(-1, -1, -1, 3))
        pose_offsets_nearest = torch.gather(pose_offsets.unsqueeze(2).expand(-1, -1, 3, -1), 1, idx.unsqueeze(3).expand(-1, -1, -1, 3))

        #B, N, 3, 3 & B, N, 3 -> B, N, 3
        bary_shape_offsets = shape_offsets_nearest * pts_bary.unsqueeze(-1).expand(-1, -1, -1, 3)
        bary_shape_offsets = torch.sum(bary_shape_offsets, dim=2)
        
        bary_pose_offsets = pose_offsets_nearest * pts_bary.unsqueeze(-1).expand(-1, -1, -1, 3)
        bary_pose_offsets = torch.sum(bary_pose_offsets, dim=2)

    if mouth_indices is not None:
        ## move low the upper teeth a bit
        query_pts[:, mouth_indices[upper_teeth_indices], 1] -= 0.01
        ## move up the lower teeth a bit
        query_pts[:, mouth_indices[lower_teeth_indices], 1] += 0.02

    # scale and translate back to the original space
    query_pts = (query_pts - translation) / scale
    if mouth_indices is not None:
        #bary_shape_offsets[:, mouth_indices] = 0. * bary_shape_offsets[:, mouth_indices]
        bary_pose_offsets[:, mouth_indices] = 0. * bary_pose_offsets[:, mouth_indices]


    query_pts += bary_shape_offsets
    v_posed = query_pts + bary_pose_offsets

    # Do skinning:
    # W is N x V x (J + 1)
    if sampled_lbs_weights is not None:
        bary_lbs_weights = sampled_lbs_weights.expand(B, -1, -1)
    else:
        # get nearest weights
        nearest_lbs_weights = lbs_weights[idx.squeeze(-1), :]#N, V, J+1
        # B, N, 3, 5 & B, N, 3 -> B, N, 5
        bary_lbs_weights = nearest_lbs_weights * pts_bary.unsqueeze(-1).expand(-1, -1, -1, 5)
        bary_lbs_weights = torch.sum(bary_lbs_weights, dim=2)

    if mouth_indices is not None:
        bary_lbs_weights[:, mouth_indices[upper_teeth_indices], 0] = 0
        bary_lbs_weights[:, mouth_indices[upper_teeth_indices], 1] = 1
        bary_lbs_weights[:, mouth_indices[upper_teeth_indices], 2] = 0
        bary_lbs_weights[:, mouth_indices[upper_teeth_indices], 3] = 0
        bary_lbs_weights[:, mouth_indices[upper_teeth_indices], 4] = 0

        bary_lbs_weights[:, mouth_indices[lower_teeth_indices], 0] = 0
        bary_lbs_weights[:, mouth_indices[lower_teeth_indices], 1] = 0
        bary_lbs_weights[:, mouth_indices[lower_teeth_indices], 2] = 1
        bary_lbs_weights[:, mouth_indices[lower_teeth_indices], 3] = 0
        bary_lbs_weights[:, mouth_indices[lower_teeth_indices], 4] = 0
        #print(bary_lbs_weights[:, mouth_indices[upper_teeth_indices]])

    W = bary_lbs_weights
    # (N x V x (J + 1)) x (N x (J + 1) x 16)
    num_joints = flame.J_regressor.shape[0]
    T = torch.matmul(W, A.view(batch_size, num_joints, 16)) \
        .view(batch_size, -1, 4, 4)# N, V, 4, 4

    homogen_coord = torch.ones([batch_size, v_posed.shape[1], 1],
                            dtype=dtype, device=device)
    v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
    v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))

    verts = v_homo[:, :, :3, 0]

    if eyelid_params is not None:
        r_eyelid_offset = flame.r_eyelid.expand(batch_size, -1, -1) * eyelid_params[:, 1:2, None]
        l_eyelib_offset = flame.l_eyelid.expand(batch_size, -1, -1) * eyelid_params[:, 0:1, None]
        
        if faceim is not None:
            bary_r_eyelid_offsets = sample_uv_map(inputs=r_eyelid_offset, flame_faces=faces, faceim=faceim, barim=barim, uv_grid=uv_grid, mode='bilinear')
            bary_l_eyelib_offsets = sample_uv_map(inputs=l_eyelib_offset, flame_faces=faces, faceim=faceim, barim=barim, uv_grid=uv_grid, mode='bilinear')
        else:
            r_eyelid_offset_nearest = torch.gather(r_eyelid_offset.unsqueeze(2).expand(-1, -1, 3, -1), 1, idx.unsqueeze(3).expand(-1, -1, -1, 3))
            l_eyelib_offset_nearest = torch.gather(l_eyelib_offset.unsqueeze(2).expand(-1, -1, 3, -1), 1, idx.unsqueeze(3).expand(-1, -1, -1, 3))
            
            #B, N, 3, 3 & B, N, 3 -> B, N, 3
            bary_r_eyelid_offsets = r_eyelid_offset_nearest * pts_bary.unsqueeze(-1).expand(-1, -1, -1, 3)
            bary_r_eyelid_offsets = torch.sum(bary_r_eyelid_offsets, dim=2)

            bary_l_eyelib_offsets = l_eyelib_offset_nearest * pts_bary.unsqueeze(-1).expand(-1, -1, -1, 3)
            bary_l_eyelib_offsets = torch.sum(bary_l_eyelib_offsets, dim=2)
        
        verts = verts + bary_r_eyelid_offsets
        verts = verts + bary_l_eyelib_offsets        

    verts = verts * scale + translation
    transforms = T[:, :, :3, :3]

    return verts, transforms