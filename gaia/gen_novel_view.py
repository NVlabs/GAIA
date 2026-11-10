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

import legacy
from camera_utils import LookAtPoseSampler
from torch_utils import misc
from training.uvplane import UVPlaneGenerator
from gaia_utils.uv import (
    EXP_CAP_DEFAULT as EXP_CAP,
    ID_CAP_DEFAULT  as ID_CAP,
    install_upper_lips_wrappers,
)


# ---------- helpers ------------------------------------------------------

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

def gen_novel_view_video(G, mp4: str, seeds, dataset_json,
                         shuffle_seed=None, w_frames=120, kind='cubic',
                         grid_dims=(1, 1), num_keyframes=None, wraps=2,
                         psi=1, truncation_cutoff=14, cfg='FFHQ',
                         device=torch.device('cuda'),
                         save_images=False, **video_kwargs):
    grid_w, grid_h = grid_dims

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

    camera_lookat_point = torch.tensor(G.rendering_kwargs['avg_camera_pivot'], device=device)
    zs = torch.from_numpy(np.stack([np.random.RandomState(s).randn(G.z_dim)
                                    for s in all_seeds])).to(device)
    cam_map = LookAtPoseSampler.sample(3.14/2, 3.14/2, camera_lookat_point,
                                       radius=G.rendering_kwargs['avg_camera_radius'] / 2,
                                       device=device)
    focal = 4.2647 if cfg != 'Shapenet' else 1.7074
    intrinsics = torch.tensor([[focal, 0, 0.5], [0, focal, 0.5], [0, 0, 1]], device=device)

    # ---- sample (shape, exp) per seed from dataset.json ----
    with open(dataset_json, 'rb') as f:
        label_list = json.load(f)['labels']
    label = torch.tensor([label_list[np.random.RandomState(s).randint(len(label_list))][1]
                          for s in all_seeds]).to(device)
    shape = label[:, 25:325]    # [N, 300]
    exp   = label[:, 325:380]   # [N, 55]

    # ---- mapping -> ws, interpolate across keyframes ----
    c_map = torch.cat([cam_map.reshape(-1, 16).repeat(len(zs), 1),
                       intrinsics.reshape(-1, 9).repeat(len(zs), 1),
                       shape], 1)
    ws = G.mapping(z=zs, c=c_map, truncation_psi=psi, truncation_cutoff=truncation_cutoff)
    ws = ws.reshape(grid_h, grid_w, num_keyframes, *ws.shape[1:])

    grid = []
    for yi in range(grid_h):
        row = []
        for xi in range(grid_w):
            x = np.arange(-num_keyframes * wraps, num_keyframes * (wraps + 1))
            y = np.tile(ws[yi][xi].cpu().numpy(), [wraps * 2 + 1, 1, 1])
            interp = scipy.interpolate.interp1d(x, y, kind=kind, axis=0)
            row.append(interp)
        grid.append(row)

    name = mp4[:-4]
    mp4 = name + '.mp4'
    video_out = imageio.get_writer(mp4, mode='I', fps=30, codec='libx264', **video_kwargs)

    pitch_range = 0.25
    yaw_range = 0.35
    total = num_keyframes * w_frames
    for frame_idx in tqdm(range(total)):
        imgs = []
        for yi in range(grid_h):
            for xi in range(grid_w):
                cam2world_pose = LookAtPoseSampler.sample(
                    3.14/2 + yaw_range * np.sin(2 * 3.14 * frame_idx / total),
                    3.14/2 + pitch_range * np.cos(2 * 3.14 * frame_idx / total),
                    camera_lookat_point,
                    radius=G.rendering_kwargs['avg_camera_radius'], device=device)
                idx_n = yi * grid_w + xi
                c = torch.cat([cam2world_pose.reshape(-1, 16),
                               intrinsics.reshape(-1, 9),
                               shape[idx_n][None],
                               exp[idx_n][None]], dim=1)
                w = torch.from_numpy(grid[yi][xi](frame_idx / w_frames)).to(device)
                pkg = G.synthesis(ws=w.unsqueeze(0), c=c, noise_mode='const')
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
@click.option('--seeds', type=parse_range, default='272', show_default=True,
              help='List of random seeds')
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
@click.option('--dataset_json', type=str, default='./dataset/dataset.json',
              show_default=True, help='dataset.json with labels for shape/exp sampling')
@click.option('--save_images', type=bool, default=False, show_default=True)
def generate_images(network_pkl, seeds, shuffle_seed, truncation_psi, truncation_cutoff,
                    grid, num_keyframes, w_frames, outdir, reload_modules, flame_source, cfg,
                    sampling_multiplier, nrr, interpolate, dataset_json, save_images):
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

    upper_mask = install_upper_lips_wrappers(G, mode='novel',
                                              exp_cap=EXP_CAP, id_cap=ID_CAP)
    print(f'[novel_view] upper-lips {int(upper_mask.sum())} pts; '
          f'exp_cap={EXP_CAP}, id_cap={ID_CAP}; dataset={dataset_json}')

    if truncation_cutoff == 0:
        truncation_psi = 1.0
    if truncation_psi == 1.0:
        truncation_cutoff = 14

    if interpolate:
        output = os.path.join(outdir, f'{seeds[0]}.mp4')
        gen_novel_view_video(G=G, mp4=output, bitrate='10M', grid_dims=grid,
                             num_keyframes=num_keyframes, w_frames=w_frames,
                             seeds=seeds, dataset_json=dataset_json,
                             shuffle_seed=shuffle_seed,
                             psi=truncation_psi, truncation_cutoff=truncation_cutoff,
                             cfg=cfg, save_images=save_images, device=device)
    else:
        for seed in seeds:
            output = os.path.join(outdir, f'{seed}.mp4')
            gen_novel_view_video(G=G, mp4=output, bitrate='10M', grid_dims=grid,
                                 num_keyframes=num_keyframes, w_frames=w_frames,
                                 seeds=[seed], dataset_json=dataset_json,
                                 shuffle_seed=shuffle_seed,
                                 psi=truncation_psi, truncation_cutoff=truncation_cutoff,
                                 cfg=cfg, save_images=save_images, device=device)


if __name__ == '__main__':
    generate_images()
