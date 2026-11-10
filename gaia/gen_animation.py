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

import os
import re
import json
from typing import List, Optional, Tuple, Union

import click
import dnnlib
import imageio.v2 as imageio
import numpy as np
import scipy.interpolate
import torch
from tqdm import tqdm
from pytorch3d.transforms import euler_angles_to_matrix, matrix_to_axis_angle

import legacy
from camera_utils import LookAtPoseSampler
from torch_utils import misc
from training.uvplane import UVPlaneGenerator
from gaia_utils.uv import (
    EXP_CAP_DEFAULT as EXP_CAP,
    ID_CAP_DEFAULT  as ID_CAP,
    UPPER_N_DEFAULT as UPPER_N,
    install_upper_lips_wrappers,
)

SHAPE_IDX = 15


# ---------- helpers ----------------------------------------------------

def layout_grid(img, grid_w=None, grid_h=1, float_to_uint8=True, chw_to_hwc=True, to_numpy=True):
    batch_size, channels, img_h, img_w = img.shape
    if grid_w is None:
        grid_w = batch_size // grid_h
    assert batch_size == grid_w * grid_h
    if float_to_uint8:
        img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    img = img.reshape(grid_h, grid_w, channels, img_h, img_w)
    img = img.permute(2, 0, 3, 1, 4)
    img = img.reshape(channels, grid_h * img_h, grid_w * img_w)
    if chw_to_hwc:
        img = img.permute(1, 2, 0)
    if to_numpy:
        img = img.cpu().numpy()
    return img


def euler_to_axis_angle(euler_angles, convention='XYZ'):
    euler_angles = torch.tensor(euler_angles).unsqueeze(0)
    R = euler_angles_to_matrix(euler_angles, convention)
    return matrix_to_axis_angle(R).squeeze().numpy()


def parse_range(s: Union[str, List[int]]) -> List[int]:
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        if m := range_re.match(p):
            ranges.extend(range(int(m.group(1)), int(m.group(2)) + 1))
        else:
            ranges.append(int(p))
    return ranges


def parse_tuple(s: Union[str, Tuple[int, int]]) -> Tuple[int, int]:
    if isinstance(s, tuple): return s
    if m := re.match(r'^(\d+)[x,](\d+)$', s):
        return (int(m.group(1)), int(m.group(2)))
    raise ValueError(f'cannot parse tuple {s}')


# ---------- core render --------------------------------------------------

def gen_interp_video(G, mp4: str, seeds, shuffle_seed=None, w_frames=None,
                     kind='cubic', grid_dims=(1, 1), num_keyframes=None,
                     wraps=2, psi=1, truncation_cutoff=14, cfg='FFHQ',
                     device=torch.device('cuda'), flame_params=None,
                     save_images=False, eye_gaze_file=None,
                     init_frame=0, shape_idx=SHAPE_IDX, **video_kwargs):
    grid_w, grid_h = grid_dims

    w_frames = len(flame_params["pose_params"])

    # eye gaze (optional)
    eye_gaze_params = None
    if eye_gaze_file is not None:
        eye_gaze_params = np.loadtxt(eye_gaze_file)
        print(f'frames: {w_frames}, eye_gaze length: {eye_gaze_params.shape}')

    if num_keyframes is None:
        if len(seeds) % (grid_w * grid_h) != 0:
            raise ValueError('Number of input seeds must be divisible by grid W*H')
        num_keyframes = len(seeds) // (grid_w * grid_h)

    all_seeds = np.zeros(num_keyframes * grid_h * grid_w, dtype=np.int64)
    for idx in range(num_keyframes * grid_h * grid_w):
        all_seeds[idx] = seeds[idx % len(seeds)]
    if shuffle_seed is not None:
        rng = np.random.RandomState(seed=shuffle_seed)
        rng.shuffle(all_seeds)

    # camera setup
    camera_lookat_point = torch.tensor(G.rendering_kwargs['avg_camera_pivot'], device=device)
    zs = torch.from_numpy(np.stack([np.random.RandomState(s).randn(G.z_dim) for s in all_seeds])).to(device)
    cam2world_pose_map = LookAtPoseSampler.sample(3.14/2, 3.14/2, camera_lookat_point,
                                                  radius=G.rendering_kwargs['avg_camera_radius'] / 2,
                                                  device=device)
    focal = 4.2647 if cfg != 'Shapenet' else 1.7074
    intrinsics = torch.tensor([[focal, 0, 0.5], [0, focal, 0.5], [0, 0, 1]], device=device)

    # shape300 from flame_params (frame `shape_idx`; clamp to last if list shorter)
    sp_arr = np.asarray(flame_params['shape_params'], dtype=np.float32)
    idx = min(shape_idx, len(sp_arr) - 1)
    shape300 = torch.from_numpy(sp_arr[idx])[None, :].to(device)
    if shape300.shape[1] != 300:
        shape300 = torch.cat([shape300,
                              torch.zeros(1, 300 - shape300.shape[1], device=device)], dim=1)

    c_map = torch.cat([cam2world_pose_map.reshape(-1, 16), intrinsics.reshape(-1, 9), shape300], 1)
    c_map = c_map.repeat(len(zs), 1)

    ws = G.mapping(z=zs, c=c_map, truncation_psi=psi, truncation_cutoff=truncation_cutoff)
    ws = ws.reshape(grid_h, grid_w, num_keyframes, *ws.shape[1:])

    # interpolation across keyframes
    grid = []
    for yi in range(grid_h):
        row = []
        for xi in range(grid_w):
            x = np.arange(-num_keyframes * wraps, num_keyframes * (wraps + 1))
            y = np.tile(ws[yi][xi].cpu().numpy(), [wraps * 2 + 1, 1, 1])
            interp = scipy.interpolate.interp1d(x, y, kind=kind, axis=0)
            row.append(interp)
        grid.append(row)

    _eye_pose_params = torch.zeros((1, 6), dtype=torch.float32, device=device)

    name = mp4[:-4]
    mp4 = name + '.mp4'
    video_out = imageio.get_writer(mp4, mode='I', fps=30, codec='libx264', **video_kwargs)

    # warmup at init_frame to populate (a) wrapper.cached, (b) native exp_planes_cache
    G.exp_conv.cached = None
    init_fi = max(0, min(init_frame, w_frames - 1))
    init_pose = np.asarray(flame_params['pose_params'][init_fi]).astype(np.float32)
    init_exp = torch.cat([
        torch.from_numpy(np.asarray(flame_params['expression_params'][init_fi]).astype(np.float32))[None],
        torch.from_numpy(np.asarray(flame_params['jaw_params'][init_fi]).astype(np.float32))[None],
        torch.from_numpy(np.asarray(flame_params['eyelid_params'][init_fi]).astype(np.float32))[None],
    ], dim=1).to(device)
    cam_init = LookAtPoseSampler.sample(3.14/2, 3.14/2 - 0.3, camera_lookat_point,
                                        radius=G.rendering_kwargs['avg_camera_radius'], device=device)
    c_init = torch.cat([cam_init.reshape(-1, 16), intrinsics.reshape(-1, 9), shape300, init_exp], dim=1)
    interp0 = grid[0][0]
    w_init = torch.from_numpy(interp0(0.0)).to(device)
    pkg_init = G.synthesis(ws=w_init.unsqueeze(0), c=c_init, noise_mode='const',
                           pose_params=torch.from_numpy(init_pose)[None].to(device),
                           eye_pose=_eye_pose_params)
    exp_plane_cache = pkg_init['exp_planes_cache']

    for frame_idx in tqdm(range(num_keyframes * w_frames)):
        imgs = []
        for yi in range(grid_h):
            for xi in range(grid_w):
                # camera fixed (no animation cycling)
                cam2world_pose = LookAtPoseSampler.sample(
                    3.14/2, 3.14/2 - 0.3, camera_lookat_point,
                    radius=G.rendering_kwargs['avg_camera_radius'], device=device)

                # eye gaze
                eye_pose = _eye_pose_params.clone()
                if eye_gaze_params is not None and frame_idx < len(eye_gaze_params):
                    e = euler_to_axis_angle([eye_gaze_params[frame_idx][0],
                                             eye_gaze_params[frame_idx][1], 0])
                    eye_pose[:, 1] = float(-e[1] * 0.2)
                    eye_pose[:, 4] = float(-e[1] * 0.2)
                    eye_pose[:, 1] = torch.clamp(eye_pose[:, 1], -0.1, 0.1)
                    eye_pose[:, 4] = torch.clamp(eye_pose[:, 4], -0.1, 0.1)

                # smoothed pose params (3-tap mean)
                if 0 < frame_idx < w_frames - 1:
                    pose_params = (np.asarray(flame_params['pose_params'][frame_idx - 1]).astype(np.float32) +
                                   np.asarray(flame_params['pose_params'][frame_idx]).astype(np.float32) +
                                   np.asarray(flame_params['pose_params'][frame_idx + 1]).astype(np.float32)) / 3
                else:
                    pose_params = np.asarray(flame_params['pose_params'][frame_idx]).astype(np.float32)

                exp_p = torch.from_numpy(np.asarray(flame_params['expression_params'][frame_idx]).astype(np.float32))[None].to(device)
                jaw_p = torch.from_numpy(np.asarray(flame_params['jaw_params'][frame_idx]).astype(np.float32))[None].to(device)
                eyl_p = torch.from_numpy(np.asarray(flame_params['eyelid_params'][frame_idx]).astype(np.float32))[None].to(device)
                exp = torch.cat([exp_p, jaw_p, eyl_p], dim=1)
                c = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9), shape300, exp], dim=1)

                w = torch.from_numpy(grid[yi][xi](frame_idx / w_frames)).to(device)
                pkg = G.synthesis(ws=w.unsqueeze(0), c=c, noise_mode='const',
                                  pose_params=torch.from_numpy(pose_params)[None].to(device),
                                  eye_pose=eye_pose,
                                  exp_plane_cache=exp_plane_cache)
                img = pkg['image'][0]

                if save_images:
                    out_dir = mp4[:-len(mp4.split('/')[-1])]
                    video_name = mp4.split('/')[-1][:-4]
                    img_dir = os.path.join(out_dir, f'images_{video_name}')
                    os.makedirs(img_dir, exist_ok=True)
                    img_w8 = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                    imageio.imwrite(os.path.join(img_dir, f'{frame_idx:04d}.png'),
                                    img_w8.permute(1, 2, 0).cpu().numpy())
                imgs.append(img)
        video_out.append_data(layout_grid(torch.stack(imgs), grid_w=grid_w, grid_h=grid_h))
    video_out.close()


# ---------- CLI ----------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seeds', type=parse_range, help='List of random seeds', required=True)
@click.option('--shuffle-seed', type=int, default=None)
@click.option('--grid', type=parse_tuple, default=(1, 1))
@click.option('--num-keyframes', type=int, default=None)
@click.option('--w-frames', type=int, default=120)
@click.option('--trunc', 'truncation_psi', type=float, default=1, show_default=True)
@click.option('--trunc-cutoff', 'truncation_cutoff', type=int, default=14, show_default=True)
@click.option('--outdir', type=str, required=True, metavar='DIR')
@click.option('--reload_modules', type=bool, default=True, show_default=True)
@click.option('--flame-source', 'flame_source', type=click.Choice(['auto', 'checkpoint', 'assets']),
              default='auto', show_default=True,
              help='Where FLAME comes from. auto: rehydrate from gaia/flame/assets only if the checkpoint is a stripped release build; checkpoint: always use the embedded FLAME (errors on a stripped build); assets: always rehydrate.')
@click.option('--cfg', type=click.Choice(['FFHQ', 'AFHQ', 'Shapenet']), default='FFHQ', show_default=True)
@click.option('--sample_mult', 'sampling_multiplier', type=float, default=2, show_default=True)
@click.option('--nrr', type=int, default=None)
@click.option('--interpolate', type=bool, default=True, show_default=True)
@click.option('--flame-params', 'flame_params_path', type=str, required=True)
@click.option('--save_images', type=bool, default=False, show_default=True)
@click.option('--eye_gaze', 'eye_gaze', type=str, default=None)
@click.option('--init_frame', type=int, default=0, show_default=True,
              help='which frame to use for warmup / native exp_plane_cache')
@click.option('--shape_idx', type=int, default=SHAPE_IDX, show_default=True,
              help='index into flame_params[shape_params] (clamped to last if shorter)')
@click.option('--enable_forehead_dyn', type=bool, default=False, show_default=True,
              help='True keeps native narrow ch 16:19 freeze (forehead dynamic); '
                   'False additionally freezes ch 16:19 across the whole face_mask')
def generate_images(network_pkl, seeds, shuffle_seed, truncation_psi, truncation_cutoff,
                    grid, num_keyframes, w_frames, outdir, reload_modules, flame_source, cfg,
                    sampling_multiplier, nrr, interpolate, flame_params_path,
                    save_images, eye_gaze, init_frame, shape_idx, enable_forehead_dyn):
    os.makedirs(outdir, exist_ok=True)

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f, flame_source=flame_source)['G_ema'].to(device)

    if reload_modules:
        print('Reloading Modules!')
        G_new = UVPlaneGenerator(*G.init_args, **G.init_kwargs).eval().requires_grad_(False).to(device)
        misc.copy_params_and_buffers(G, G_new, require_all=False)
        G_new.neural_rendering_resolution = G.neural_rendering_resolution
        G_new.rendering_kwargs = G.rendering_kwargs
        G = G_new

    G.rendering_kwargs['depth_resolution'] = int(G.rendering_kwargs['depth_resolution'] * sampling_multiplier)
    G.rendering_kwargs['depth_resolution_importance'] = int(G.rendering_kwargs['depth_resolution_importance'] * sampling_multiplier)
    if nrr is not None: G.neural_rendering_resolution = nrr

    # ---- bake in the wrappers ----
    _ = install_upper_lips_wrappers(G, mode='animation',
                                             exp_cap=EXP_CAP, id_cap=ID_CAP,
                                             enable_forehead_dyn=enable_forehead_dyn)

    if truncation_cutoff == 0:
        truncation_psi = 1.0
    if truncation_psi == 1.0:
        truncation_cutoff = 14

    name = os.path.splitext(os.path.basename(flame_params_path))[0]
    flame_params = json.load(open(flame_params_path, 'r'))

    if interpolate:
        output = os.path.join(outdir, f'{name}_{seeds[0]}.mp4')
        gen_interp_video(G=G, mp4=output, bitrate='10M', grid_dims=grid,
                         num_keyframes=num_keyframes, w_frames=w_frames,
                         seeds=seeds, shuffle_seed=shuffle_seed,
                         psi=truncation_psi, truncation_cutoff=truncation_cutoff,
                         cfg=cfg, flame_params=flame_params,
                         save_images=save_images, eye_gaze_file=eye_gaze,
                         init_frame=init_frame, shape_idx=shape_idx, device=device)
    else:
        for seed in seeds:
            output = os.path.join(outdir, f'{name}_{seed}.mp4')
            G.exp_conv.cached = None  # reset wrapper cache per seed
            gen_interp_video(G=G, mp4=output, bitrate='10M', grid_dims=grid,
                             num_keyframes=num_keyframes, w_frames=w_frames,
                             seeds=[seed], shuffle_seed=shuffle_seed,
                             psi=truncation_psi, truncation_cutoff=truncation_cutoff,
                             cfg=cfg, flame_params=flame_params,
                             save_images=save_images, eye_gaze_file=eye_gaze,
                             init_frame=init_frame, shape_idx=shape_idx, device=device)


if __name__ == '__main__':
    generate_images()
