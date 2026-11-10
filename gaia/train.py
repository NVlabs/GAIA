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

"""Train a GAN using the techniques described in the paper
"Efficient Geometry-aware 3D Generative Adversarial Networks."

Code adapted from
"Alias-Free Generative Adversarial Networks"."""

import os
import click
import re
import json
import tempfile
import torch

import dnnlib

from training import training_loop
from metrics import metric_main
from torch_utils import training_stats
from torch_utils import custom_ops

#----------------------------------------------------------------------------

def subprocess_fn(rank, c, temp_dir):
    dnnlib.util.Logger(file_name=os.path.join(c.run_dir, 'log.txt'), file_mode='a', should_flush=True)

    # Init torch.distributed.
    if c.num_gpus > 1:
        init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
        if os.name == 'nt':
            init_method = 'file:///' + init_file.replace('\\', '/')
            torch.distributed.init_process_group(backend='gloo', init_method=init_method, rank=rank, world_size=c.num_gpus)
        else:
            init_method = f'file://{init_file}'
            torch.distributed.init_process_group(backend='nccl', init_method=init_method, rank=rank, world_size=c.num_gpus)

    # Init torch_utils.
    sync_device = torch.device('cuda', rank) if c.num_gpus > 1 else None
    training_stats.init_multiprocessing(rank=rank, sync_device=sync_device)
    if rank != 0:
        custom_ops.verbosity = 'none'

    # Execute training loop.
    training_loop.training_loop(rank=rank, **c)

#----------------------------------------------------------------------------

def launch_training(c, desc, outdir, dry_run):
    dnnlib.util.Logger(should_flush=True)

    # Pick output directory.
    prev_run_dirs = []
    if os.path.isdir(outdir):
        prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]
    prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    cur_run_id = max(prev_run_ids, default=-1) + 1
    #cur_run_id = max(prev_run_ids, default=0)# + 1
    c.run_dir = os.path.join(outdir, f'{cur_run_id:05d}-{desc}')
    assert not os.path.exists(c.run_dir)

    # Print options.
    print()
    print('Training options:')
    print(json.dumps(c, indent=2))
    print()
    print(f'Output directory:    {c.run_dir}')
    print(f'Number of GPUs:      {c.num_gpus}')
    print(f'Batch size:          {c.batch_size} images')
    print(f'Training duration:   {c.total_kimg} kimg')
    print(f'Dataset path:        {c.training_set_kwargs.path}')
    print(f'Dataset size:        {c.training_set_kwargs.max_size} images')
    print(f'Dataset resolution:  {c.training_set_kwargs.resolution}')
    print(f'Dataset labels:      {c.training_set_kwargs.use_labels}')
    print(f'Dataset x-flips:     {c.training_set_kwargs.xflip}')
    print(f'Use Wandb:           {c.use_wandb}')
    if c.use_wandb:
        print(f'Wandb entity/project: {c.wandb_entity or "(default)"}/{c.wandb_project}')
    print()

    # Dry run?
    if dry_run:
        print('Dry run; exiting.')
        return

    # Create output directory.
    print('Creating output directory...')
    os.makedirs(c.run_dir)
    with open(os.path.join(c.run_dir, 'training_options.json'), 'wt') as f:
        json.dump(c, f, indent=2)

    # copy all training source data as backup
    source_dir = os.path.join(c.run_dir, "source")
    os.makedirs(source_dir)
    command = "cp *.py " + source_dir
    os.system(command)
    command = "cp -r training/ " + os.path.join(source_dir, "training")
    os.system(command)        

    # Launch processes.
    print('Launching processes...')
    torch.multiprocessing.set_start_method('spawn')
    with tempfile.TemporaryDirectory() as temp_dir:
        if c.num_gpus == 1:
            subprocess_fn(rank=0, c=c, temp_dir=temp_dir)
        else:
            torch.multiprocessing.spawn(fn=subprocess_fn, args=(c, temp_dir), nprocs=c.num_gpus)

#----------------------------------------------------------------------------

def init_dataset_kwargs(data, res=None, background_color=[1.0, 1.0, 1.0], use_lanczos=False):
    try:
        #dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=data, use_labels=True, max_size=None, xflip=False, resolution=res)##resolution
        dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=data, use_labels=True, max_size=None, xflip=False, resolution=res, background_color=background_color, use_lanczos=use_lanczos)
        dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs) # Subclass of training.dataset.Dataset.
        dataset_kwargs.resolution = dataset_obj.resolution # Be explicit about resolution.
        dataset_kwargs.use_labels = dataset_obj.has_labels # Be explicit about labels.
        dataset_kwargs.max_size = len(dataset_obj) # Be explicit about dataset size.
        return dataset_kwargs, dataset_obj.name
    except IOError as err:
        raise click.ClickException(f'--data: {err}')

#----------------------------------------------------------------------------

def parse_comma_separated_list(s):
    if isinstance(s, list):
        return s
    if s is None or s.lower() == 'none' or s == '':
        return []
    return s.split(',')

#----------------------------------------------------------------------------

@click.command()

# Required.
@click.option('--outdir',       help='Where to save the results', metavar='DIR',                required=True)
@click.option('--cfg',          help='Base configuration',                                      type=str, required=True)
@click.option('--data',         help='Training data', metavar='[ZIP|DIR]',                      type=str, required=True)
@click.option('--gpus',         help='Number of GPUs to use', metavar='INT',                    type=click.IntRange(min=1), required=True)
@click.option('--batch',        help='Total batch size', metavar='INT',                         type=click.IntRange(min=1), required=True)
@click.option('--gamma',        help='R1 regularization weight', metavar='FLOAT',               type=click.FloatRange(min=0), required=True)
@click.option('--gamma2',        help='R1 regularization weight to D2', metavar='FLOAT',               type=click.FloatRange(min=0), required=False, default=1.0)

@click.option('--use_wandb',    help='Use Wandb to track the training', metavar='BOOL',        type=bool, default=False, show_default=True)
@click.option('--wandb_entity', help='Wandb entity (user or team); default: your Wandb default entity', metavar='STR', type=str, default=None)
@click.option('--wandb_project', help='Wandb project name', metavar='STR',                      type=str, default='GAIA', show_default=True)

# Optional features.
@click.option('--cond',         help='Train conditional model', metavar='BOOL',                 type=bool, default=True, show_default=True)
@click.option('--mirror',       help='Enable dataset x-flips', metavar='BOOL',                  type=bool, default=False, show_default=True)
@click.option('--aug',          help='Augmentation mode',                                       type=click.Choice(['noaug', 'ada', 'fixed']), default='noaug', show_default=True)
@click.option('--resume',       help='Resume from given network pickle', metavar='[PATH|URL]',  type=str)
@click.option('--freezed',      help='Freeze first layers of D', metavar='INT',                 type=click.IntRange(min=0), default=0, show_default=True)

# Misc hyperparameters.
@click.option('--p',            help='Probability for --aug=fixed', metavar='FLOAT',            type=click.FloatRange(min=0, max=1), default=0.2, show_default=True)
@click.option('--target',       help='Target value for --aug=ada', metavar='FLOAT',             type=click.FloatRange(min=0, max=1), default=0.6, show_default=True)
@click.option('--batch-gpu',    help='Limit batch size per GPU', metavar='INT',                 type=click.IntRange(min=1))
@click.option('--cbase',        help='Capacity multiplier', metavar='INT',                      type=click.IntRange(min=1), default=32768, show_default=True)
@click.option('--cmax',         help='Max. feature maps', metavar='INT',                        type=click.IntRange(min=1), default=512, show_default=True)
@click.option('--glr',          help='G learning rate  [default: varies]', metavar='FLOAT',     type=click.FloatRange(min=0))
@click.option('--dlr',          help='D learning rate', metavar='FLOAT',                        type=click.FloatRange(min=0), default=0.002, show_default=True)#0.002
@click.option('--map-depth',    help='Mapping network depth  [default: varies]', metavar='INT', type=click.IntRange(min=1), default=2, show_default=True)
@click.option('--mbstd-group',  help='Minibatch std group size', metavar='INT',                 type=click.IntRange(min=1), default=4, show_default=True)

# Misc settings.
@click.option('--desc',         help='String to include in result dir name', metavar='STR',     type=str)
@click.option('--metrics',      help='Quality metrics', metavar='[NAME|A,B,C|none]',            type=parse_comma_separated_list, default='fid50k_full', show_default=True)
@click.option('--kimg',         help='Total training duration', metavar='KIMG',                 type=click.IntRange(min=1), default=25000, show_default=True)
@click.option('--tick',         help='How often to print progress', metavar='KIMG',             type=click.IntRange(min=1), default=4, show_default=True)
@click.option('--snap',         help='How often to save snapshots', metavar='TICKS',            type=click.IntRange(min=1), default=100, show_default=True)
@click.option('--seed',         help='Random seed', metavar='INT',                              type=click.IntRange(min=0), default=0, show_default=True)
# @click.option('--fp32',         help='Disable mixed-precision', metavar='BOOL',                 type=bool, default=False, show_default=True)
@click.option('--nobench',      help='Disable cuDNN benchmarking', metavar='BOOL',              type=bool, default=False, show_default=True)
@click.option('--workers',      help='DataLoader worker processes', metavar='INT',              type=click.IntRange(min=1), default=3, show_default=True)
@click.option('-n','--dry-run', help='Print training options and exit',                         is_flag=True)

# @click.option('--sr_module',    help='Superresolution module', metavar='STR',  type=str, required=True)
@click.option('--neural_rendering_resolution_initial', help='Resolution to render at', metavar='INT',  type=click.IntRange(min=1), default=64, required=False)
@click.option('--neural_rendering_resolution_final', help='Final resolution to render at, if blending', metavar='INT',  type=click.IntRange(min=1), required=False, default=None)
@click.option('--neural_rendering_resolution_fade_kimg', help='Kimg to blend resolution over', metavar='INT',  type=click.IntRange(min=0), required=False, default=1000, show_default=True)

@click.option('--blur_fade_kimg', help='Blur over how many', metavar='INT',  type=click.IntRange(min=1), required=False, default=200)
@click.option('--gen_pose_cond', help='If true, enable generator pose conditioning.', metavar='BOOL',  type=bool, required=False, default=False)
@click.option('--c-scale', help='Scale factor for generator pose conditioning.', metavar='FLOAT',  type=click.FloatRange(min=0), required=False, default=1)
@click.option('--c-noise', help='Add noise for generator pose conditioning.', metavar='FLOAT',  type=click.FloatRange(min=0), required=False, default=0)
@click.option('--gpc_reg_prob', help='Strength of swapping regularization. None means no generator pose conditioning, i.e. condition with zeros.', metavar='FLOAT',  type=click.FloatRange(min=0), required=False, default=0.5)
@click.option('--gpc_reg_fade_kimg', help='Length of swapping prob fade', metavar='INT',  type=click.IntRange(min=0), required=False, default=1000)
@click.option('--disc_c_noise', help='Strength of discriminator pose conditioning regularization, in standard deviations.', metavar='FLOAT',  type=click.FloatRange(min=0), required=False, default=0)
@click.option('--sr_noise_mode', help='Type of noise for superresolution', metavar='STR',  type=click.Choice(['random', 'none']), required=False, default='none')
@click.option('--resume_blur', help='Enable to blur even on resume', metavar='BOOL',  type=bool, required=False, default=False)
@click.option('--g_num_fp16_res',    help='Number of fp16 layers in generator', metavar='INT', type=click.IntRange(min=0), default=0, required=False, show_default=True)
@click.option('--d_num_fp16_res',    help='Number of fp16 layers in discriminator', metavar='INT', type=click.IntRange(min=0), default=4, required=False, show_default=True)
@click.option('--sr_first_cutoff',    help='First cutoff for AF superresolution', metavar='INT', type=click.IntRange(min=2), default=2, required=False, show_default=True)
@click.option('--sr_first_stopband',    help='First cutoff for AF superresolution', metavar='FLOAT', type=click.FloatRange(min=2), default=2**2.1, required=False, show_default=True)
@click.option('--style_mixing_prob',    help='Style-mixing regularization probability for training.', metavar='FLOAT', type=click.FloatRange(min=0, max=1), default=0, required=False, show_default=True)
@click.option('--sr_module',    help='Superresolution module override', metavar='STR',  type=str, required=False, default=None)
@click.option('--density_reg',    help='Density regularization strength.', metavar='FLOAT', type=click.FloatRange(min=0), default=0, required=False, show_default=True)#0.25 for eg3d
@click.option('--density_reg_every',    help='lazy density reg', metavar='int', type=click.FloatRange(min=1), default=4, required=False, show_default=True)
@click.option('--density_reg_p_dist',    help='density regularization strength.', metavar='FLOAT', type=click.FloatRange(min=0), default=0.004, required=False, show_default=True)
@click.option('--reg_type', help='Type of regularization', metavar='STR',  type=click.Choice(['l1', 'l1-alt', 'monotonic-detach', 'monotonic-fixed', 'total-variation']), required=False, default='l1')
@click.option('--decoder_lr_mul',    help='decoder learning rate multiplier.', metavar='FLOAT', type=click.FloatRange(min=0), default=0.01, required=False, show_default=True)
@click.option('--plane_reg',    help='Triplane regularization strength.', metavar='FLOAT', type=click.FloatRange(min=0), default=0.0, required=False, show_default=True)
@click.option('--triplane_res',    help='Resolution of triplane', metavar='INT', type=click.IntRange(min=0), default=256, required=False, show_default=True)
@click.option('--depth_resolution',    help='depth_resolution', metavar='INT', type=click.IntRange(min=4), default=48, required=False, show_default=True)
@click.option('--depth_resolution_importance',    help='depth_resolution_importance', metavar='INT', type=click.IntRange(min=4), default=48, required=False, show_default=True)

## added for uv plane
@click.option('--num',         help='Training size for debug', metavar='INT', type=click.IntRange(min=1), default=None, show_default=True)
@click.option('--regularization', help='If true, enable regularization loss.', metavar='BOOL',  type=bool, required=False, default=False)
@click.option('--sample_res',         help='Sample resolutsion in uv plane', metavar='INT', type=click.IntRange(min=1), default=512, show_default=True)
@click.option('--pretrained_resolution',         help='For discriminator new layer', metavar='INT', type=click.IntRange(min=1), default=None, show_default=True)
@click.option('--use_new_layer', help='If true, enable use new layer.', metavar='BOOL',  type=bool, required=False, default=False)
@click.option('--max_offset',    help='Max Position Offset.', metavar='FLOAT', type=click.FloatRange(min=0), default=0.25, required=False, show_default=True)
@click.option('--use_high_res_uvmap', help='If true, use high resolution uvmap instead of sampling.', metavar='BOOL',  type=bool, required=False, default=False)
@click.option('--freeze_backbone', help='Freeze the first generator.', metavar='BOOL',  type=bool, required=False, default=True)
@click.option('--use_exp_cond_gen', help='Use exp condition on Generator.', metavar='BOOL',  type=bool, required=False, default=False)
@click.option('--use_face_mask_uv_reg', help='If true use the front face mask.', metavar='BOOL',  type=bool, required=False, default=False)
@click.option('--use_dynamic_featurev2', help='If true use the dynamic featurev2.', metavar='BOOL',  type=bool, required=False, default=False)
@click.option('--direct_512', help='If true directly use the 512 res.', metavar='BOOL',  type=bool, required=False, default=False)
@click.option('--use_front_face_pos_reg', help='If true use pos reg in front face.', metavar='BOOL',  type=bool, required=False, default=False)
@click.option('--black_background', help='If true use black background.', metavar='BOOL',  type=bool, required=False, default=False)
@click.option('--use_eye_uv_reg', help='If true use eye tv reg.', metavar='BOOL',  type=bool, required=False, default=False)
@click.option('--use_upper_lips_constraint', help='If true, cap upper-lips position and scale dynamics during synthesis.', metavar='BOOL',  type=bool, required=False, default=False)
@click.option('--upper_lips_n', help='Number of row-major lips-mask pixels treated as upper lips.', metavar='INT', type=click.IntRange(min=1), default=4500, required=False, show_default=True)
@click.option('--upper_lips_exp_cap', help='Upper-lips expression position offset L2 cap.', metavar='FLOAT', type=click.FloatRange(min=0), default=0.01, required=False, show_default=True)
@click.option('--upper_lips_scale_cap', help='Upper-lips expression scale offset L2 cap.', metavar='FLOAT', type=click.FloatRange(min=0), default=0.1, required=False, show_default=True)
@click.option('--upper_lips_id_cap', help='Upper-lips identity position offset L2 cap.', metavar='FLOAT', type=click.FloatRange(min=0), default=0.60, required=False, show_default=True)

## added for discriminator
@click.option('--only_noise_exp', help='Only add noise to exp.', metavar='BOOL',  type=bool, required=False, default=False)
@click.option('--shape_cond_disc', help='Shape cond disc.', metavar='BOOL',  type=bool, required=False, default=False)
@click.option('--d2_disc_c_noise', help='Strength of discriminator 2 pose conditioning regularization, in standard deviations.', metavar='FLOAT',  type=click.FloatRange(min=0), required=False, default=0)

#use_new_layer
@click.option('--new_layers_disc_start_kimg',         help='new_layers_disc_start_kimg', metavar='INT', type=click.IntRange(min=1), default=6400, show_default=True)
@click.option('--new_layers_disc_blend_kimg',         help='new_layers_disc_blend_kimg', metavar='INT', type=click.IntRange(min=1), default=1000, show_default=True)

## added for loss
@click.option('--use_tv_reg', help='If true, enable tv loss in render uv.', metavar='BOOL',  type=bool, required=False, default=False)
@click.option('--use_tv_uv_gen', help='If true, enable tv loss in gen uv.', metavar='BOOL',  type=bool, required=False, default=False)
@click.option('--use_opacity_reg', help='If true, enable opacity reg.', metavar='BOOL',  type=bool, required=False, default=True)
@click.option('--use_pos_reg', help='If true, enable pos reg.', metavar='BOOL',  type=bool, required=False, default=True)
@click.option('--use_scale_reg', help='If true, enable scale reg.', metavar='BOOL',  type=bool, required=False, default=True)
@click.option('--lambda_raw_gaussian_position',    help='Position regularization strength.', metavar='FLOAT', type=click.FloatRange(min=0), default=0.1, required=False, show_default=True)#0.25 for eg3d
@click.option('--lambda_raw_gaussian_scale',    help='Scale regularization strength.', metavar='FLOAT', type=click.FloatRange(min=0), default=0.05, required=False, show_default=True)#0.25 for eg3d
@click.option('--lambda_tv_genuv_pos',    help='Position regularization strength.', metavar='FLOAT', type=click.FloatRange(min=0), default=100.0, required=False, show_default=True)#0.25 for eg3d
@click.option('--use_fade_exp', help='If true, enable the fade exp on generator.', metavar='BOOL',  type=bool, required=False, default=False)
@click.option('--add_exp_loss', help='If true add a mapping network to fit exp.', metavar='BOOL',  type=bool, required=False, default=False)
@click.option('--only_cam_disc', help='If true only cond on cam disc.', metavar='BOOL',  type=bool, required=False, default=False)
@click.option('--only_jaw_eyelid_disc', help='If true only cond on jaw and eyelid disc.', metavar='BOOL',  type=bool, required=False, default=False)
@click.option('--dual_discrimination', help='If true use dual discriminator.', metavar='BOOL',  type=bool, required=False, default=False)
@click.option('--use_uv_r1', help='If true use rl penalty in discriminator.', metavar='BOOL',  type=bool, required=False, default=False)
@click.option('--lambda_pos_front',    help='Front face position regularization strength.', metavar='FLOAT', type=click.FloatRange(min=0), default=0.1, required=False, show_default=True)#0.25 for eg3d
@click.option('--lambda_tv_eye',    help='Eye uv regularization strength.', metavar='FLOAT', type=click.FloatRange(min=0), default=5.0, required=False, show_default=True)#0.25 for eg3d

@click.option('--weight_d1',    help='D1 Loss strength.', metavar='FLOAT', type=click.FloatRange(min=0), default=0.5, required=False, show_default=True)#0.25 for eg3d
@click.option('--weight_d2',    help='D2 Loss strength.', metavar='FLOAT', type=click.FloatRange(min=0), default=0.5, required=False, show_default=True)#0.25 for eg3d
@click.option('--warmup_weight_d2', help='If true use the warm up for D2 weight', metavar='BOOL',  type=bool, required=False, default=False)
@click.option('--use_d2', help='If true use the second discriminator', metavar='BOOL',  type=bool, required=False, default=False)
@click.option('--init_k',         help='init_k', metavar='INT', type=click.IntRange(min=1), default=23000, show_default=True)
@click.option('--d2_start_weight',    help='d2_start_weight.', metavar='FLOAT', type=click.FloatRange(min=0), default=0, required=False, show_default=True)#0.25 for eg3d
@click.option('--d2_end_weight',    help='d2_end_weight.', metavar='FLOAT', type=click.FloatRange(min=0), default=0.5, required=False, show_default=True)#0.25 for eg3d
@click.option('--use_exp_reg', help='If true, use regularization on uv_exp.', metavar='BOOL',  type=bool, required=False, default=False)
@click.option('--lambda_exp_reg',    help='Exp uv map regularization strength.', metavar='FLOAT', type=click.FloatRange(min=0), default=0.1, required=False, show_default=True)#0.25 for eg3d
@click.option('--use_exp_color_reg', help='If true, regularize expression color dynamics by UV regions.', metavar='BOOL',  type=bool, required=False, default=False)
@click.option('--lambda_exp_color_skin',    help='Expression color regularization strength for non-mouth/eye/forehead face skin.', metavar='FLOAT', type=click.FloatRange(min=0), default=2.0, required=False, show_default=True)
@click.option('--lambda_exp_color_eye',    help='Expression color regularization strength for eye region.', metavar='FLOAT', type=click.FloatRange(min=0), default=1.0, required=False, show_default=True)
@click.option('--lambda_exp_color_forehead',    help='Expression color regularization strength for forehead region.', metavar='FLOAT', type=click.FloatRange(min=0), default=1.0, required=False, show_default=True)
@click.option('--lambda_exp_color_mouth',    help='Expression color regularization strength for mouth and lips region.', metavar='FLOAT', type=click.FloatRange(min=0), default=1.0, required=False, show_default=True)
@click.option('--load_d2_from_d1', help='If true load D2 from D1 when init', metavar='BOOL',  type=bool, required=False, default=True)

# Add for interpolation type option in dataset when stage 0 training
@click.option('--use_lanczos', help='whether use lanczos interpolation when loading images', metavar='BOOL',  type=bool, required=False, default=False)

def main(**kwargs):
    """Train GAIA. See scripts/train_stage_{0,1,2}.sh for the per-stage commands."""

    # Initialize config.
    opts = dnnlib.EasyDict(kwargs) # Command line arguments.
    c = dnnlib.EasyDict() # Main config dict.
    c.G_kwargs = dnnlib.EasyDict(class_name=None, z_dim=512, w_dim=512, mapping_kwargs=dnnlib.EasyDict())
    c.D_kwargs = dnnlib.EasyDict(class_name='training.networks_stylegan2.Discriminator', block_kwargs=dnnlib.EasyDict(), mapping_kwargs=dnnlib.EasyDict(), epilogue_kwargs=dnnlib.EasyDict())
    if opts.use_d2:
        c.D2_kwargs = dnnlib.EasyDict(class_name='training.networks_stylegan2.Discriminator', block_kwargs=dnnlib.EasyDict(), mapping_kwargs=dnnlib.EasyDict(), epilogue_kwargs=dnnlib.EasyDict())
        c.D2_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', betas=[0,0.99], eps=1e-8)
    c.G_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', betas=[0,0.99], eps=1e-8)
    c.D_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', betas=[0,0.99], eps=1e-8)
    c.loss_kwargs = dnnlib.EasyDict(class_name='training.loss.StyleGAN2Loss')
    c.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, prefetch_factor=2)

    # Training set.
    if opts.black_background:
        c.training_set_kwargs, dataset_name = init_dataset_kwargs(data=opts.data, res=opts.sample_res, background_color=[0.0, 0.0, 0.0], use_lanczos=opts.use_lanczos)
    else:
        c.training_set_kwargs, dataset_name = init_dataset_kwargs(data=opts.data, res=opts.sample_res, use_lanczos=opts.use_lanczos)
    if opts.cond and not c.training_set_kwargs.use_labels:
        raise click.ClickException('--cond=True requires labels specified in dataset.json')
    c.training_set_kwargs.use_labels = opts.cond
    c.training_set_kwargs.xflip = opts.mirror

    c.training_set_kwargs.num = opts.num

    # Hyperparameters & settings.
    c.num_gpus = opts.gpus
    c.batch_size = opts.batch
    c.batch_gpu = opts.batch_gpu or opts.batch // opts.gpus
    c.use_wandb = opts.use_wandb
    c.wandb_entity = opts.wandb_entity
    c.wandb_project = opts.wandb_project
    c.G_kwargs.channel_base = c.D_kwargs.channel_base = opts.cbase
    c.G_kwargs.channel_max = c.D_kwargs.channel_max = opts.cmax
    c.G_kwargs.mapping_kwargs.num_layers = opts.map_depth
    c.D_kwargs.block_kwargs.freeze_layers = opts.freezed
    c.D_kwargs.epilogue_kwargs.mbstd_group_size = opts.mbstd_group

    ## add for uv plane
    c.D_kwargs.pretrained_resolution = opts.pretrained_resolution
    c.D_kwargs.use_new_layer = opts.use_new_layer
    c.D_kwargs.only_noise_exp = opts.only_noise_exp
    c.D_kwargs.use_dual_discrimination = opts.dual_discrimination
    c.D_kwargs.shape_cond_disc = opts.shape_cond_disc
    
    if opts.use_d2:
        c.D2_kwargs.block_kwargs.freeze_layers = opts.freezed
        c.D2_kwargs.epilogue_kwargs.mbstd_group_size = opts.mbstd_group
        c.D2_kwargs.pretrained_resolution = opts.pretrained_resolution
        c.D2_kwargs.use_new_layer = opts.use_new_layer
        c.D2_kwargs.c_dim = 80
        

    c.loss_kwargs.r1_gamma = opts.gamma
    c.loss_kwargs.r1_gamma2 = opts.gamma2

    ## add for uv plane
    c.loss_kwargs.regularization = opts.regularization
    c.loss_kwargs.use_tv_reg = opts.use_tv_reg
    c.loss_kwargs.use_tv_uv_gen = opts.use_tv_uv_gen
    c.loss_kwargs.use_opacity_reg = opts.use_opacity_reg
    c.loss_kwargs.use_pos_reg = opts.use_pos_reg
    c.loss_kwargs.use_scale_reg = opts.use_scale_reg
    c.loss_kwargs.lambda_raw_gaussian_position = opts.lambda_raw_gaussian_position
    c.loss_kwargs.lambda_pos_front = opts.lambda_pos_front
    c.loss_kwargs.lambda_raw_gaussian_scale = opts.lambda_raw_gaussian_scale
    c.loss_kwargs.lambda_tv_genuv_pos = opts.lambda_tv_genuv_pos
    c.loss_kwargs.use_fade_exp = opts.use_fade_exp
    c.loss_kwargs.add_exp_loss = opts.add_exp_loss
    c.loss_kwargs.only_cam_disc = opts.only_cam_disc
    c.loss_kwargs.only_jaw_eyelid_disc = opts.only_jaw_eyelid_disc
    c.loss_kwargs.use_face_mask_uv_reg = opts.use_face_mask_uv_reg
    c.loss_kwargs.lambda_tv_eye = opts.lambda_tv_eye
    
    c.loss_kwargs.init_k = opts.init_k
    c.loss_kwargs.d2_start_weight = opts.d2_start_weight
    c.loss_kwargs.d2_end_weight = opts.d2_end_weight
    c.loss_kwargs.weight_d1 = opts.weight_d1
    c.loss_kwargs.weight_d2 = opts.weight_d2
    c.loss_kwargs.warmup_weight_d2 = opts.warmup_weight_d2
    c.loss_kwargs.use_d2 = opts.use_d2
    c.loss_kwargs.use_exp_reg = opts.use_exp_reg
    c.loss_kwargs.lambda_exp_reg = opts.lambda_exp_reg
    c.loss_kwargs.use_exp_color_reg = opts.use_exp_color_reg
    c.loss_kwargs.lambda_exp_color_skin = opts.lambda_exp_color_skin
    c.loss_kwargs.lambda_exp_color_eye = opts.lambda_exp_color_eye
    c.loss_kwargs.lambda_exp_color_forehead = opts.lambda_exp_color_forehead
    c.loss_kwargs.lambda_exp_color_mouth = opts.lambda_exp_color_mouth
    c.loss_kwargs.load_d2_from_d1 = opts.load_d2_from_d1



    c.loss_kwargs.new_layers_disc_start_kimg = opts.new_layers_disc_start_kimg
    c.loss_kwargs.new_layers_disc_blend_kimg = opts.new_layers_disc_blend_kimg
    #new_layers_disc_start_kimg 6600
    #new_layers_disc_blend_kimg 1000

    c.G_opt_kwargs.lr = (0.002 if opts.cfg == 'stylegan2' else 0.0025) if opts.glr is None else opts.glr
    #c.G_opt_kwargs.lr = (1e-4 if opts.cfg == 'stylegan2' else 1e-4) if opts.glr is None else opts.glr

    c.D_opt_kwargs.lr = opts.dlr
    if opts.use_d2:
        c.D2_opt_kwargs.lr = opts.dlr
    c.metrics = opts.metrics
    c.total_kimg = opts.kimg
    c.kimg_per_tick = opts.tick
    c.image_snapshot_ticks = c.network_snapshot_ticks = opts.snap
    c.random_seed = c.training_set_kwargs.random_seed = opts.seed
    c.data_loader_kwargs.num_workers = opts.workers

    # Sanity checks.
    if c.batch_size % c.num_gpus != 0:
        raise click.ClickException('--batch must be a multiple of --gpus')
    if c.batch_size % (c.num_gpus * c.batch_gpu) != 0:
        raise click.ClickException('--batch must be a multiple of --gpus times --batch-gpu')
    if c.batch_gpu < c.D_kwargs.epilogue_kwargs.mbstd_group_size:
        raise click.ClickException('--batch-gpu cannot be smaller than --mbstd')
    if any(not metric_main.is_valid_metric(metric) for metric in c.metrics):
        raise click.ClickException('\n'.join(['--metrics can only contain the following values:'] + metric_main.list_valid_metrics()))

    # Base configuration.
    c.ema_kimg = c.batch_size * 10 / 32
    #c.G_kwargs.class_name = 'training.triplane.TriPlaneGenerator'
    c.G_kwargs.class_name = 'training.uvplane.UVPlaneGenerator'
    c.D_kwargs.class_name = 'training.dual_discriminator.DualDiscriminator'
    if opts.use_d2:
        c.D2_kwargs.class_name = 'training.dual_discriminator.DualDiscriminator'
        c.D2_kwargs.disc_c_noise = opts.d2_disc_c_noise
    #c.D_kwargs.class_name = 'training.dual_discriminator.SingleDiscriminator' #no super resolution
    c.G_kwargs.fused_modconv_default = 'inference_only' # Speed up training by using regular convolutions instead of grouped convolutions.
    c.loss_kwargs.filter_mode = 'antialiased' # Filter mode for raw images ['antialiased', 'none', float [0-1]]
    c.D_kwargs.disc_c_noise = opts.disc_c_noise # Regularization for discriminator pose conditioning
    if c.training_set_kwargs.resolution == 1024:
        sr_module = 'training.superresolution.SuperresolutionHybrid8XDC1024'
    elif c.training_set_kwargs.resolution == 512:
        #sr_module = 'training.superresolution.SuperresolutionHybrid8XDC'
        sr_module = 'training.superresolution.SuperresolutionHybrid4X'
    elif c.training_set_kwargs.resolution == 256:
        sr_module = 'training.superresolution.SuperresolutionHybrid4X'
    elif c.training_set_kwargs.resolution == 128:
        sr_module = 'training.superresolution.SuperresolutionHybrid2X'
    else:
        assert False, f"Unsupported resolution {c.training_set_kwargs.resolution}; make a new superresolution module"
    
    if opts.sr_module != None:
        sr_module = "training.superresolution." + opts.sr_module
    
    rendering_options = {
        'image_resolution': c.training_set_kwargs.resolution,
        'disparity_space_sampling': False,
        'clamp_mode': 'softplus',
        'superresolution_module': sr_module,
        'c_gen_conditioning_zero': not opts.gen_pose_cond, # if true, fill generator pose conditioning label with dummy zero vector
        'gpc_reg_prob': opts.gpc_reg_prob if opts.gen_pose_cond else None,
        'c_scale': opts.c_scale, # mutliplier for generator pose conditioning label
        'superresolution_noise_mode': opts.sr_noise_mode, # [random or none], whether to inject pixel noise into super-resolution layers
        'density_reg': opts.density_reg, # strength of density regularization
        'density_reg_p_dist': opts.density_reg_p_dist, # distance at which to sample perturbed points for density regularization
        'reg_type': opts.reg_type, # for experimenting with variations on density regularization
        'decoder_lr_mul': opts.decoder_lr_mul, # learning rate multiplier for decoder
        'sr_antialias': True,
        'plane_reg': opts.plane_reg, # strength of density regularization
    }

    if opts.cfg == 'ffhq':
        rendering_options.update({
            'depth_resolution': 48, # number of uniform samples to take per ray.
            'depth_resolution_importance': 48, # number of importance samples to take per ray.
            'ray_start': 2.25, # near point along each ray to start taking samples.
            'ray_end': 3.3, # far point along each ray to stop taking samples. 
            'box_warp': 1, # the side-length of the bounding box spanned by the tri-planes; box_warp=1 means [-0.5, -0.5, -0.5] -> [0.5, 0.5, 0.5].
            'avg_camera_radius': 2.7, # camera orbit radius (e.g. gen_* scripts read G.rendering_kwargs['avg_camera_radius']).
            'avg_camera_pivot': [0, 0, 0.2], # center of camera rotation.
        })
    elif opts.cfg == 'ffhq_vc':
        rendering_options.update({
            'depth_resolution': 48, # number of uniform samples to take per ray.
            'depth_resolution_importance': 48, # number of importance samples to take per ray.
            'ray_start': 1.6, # near point along each ray to start taking samples.
            'ray_end': 3.8, # far point along each ray to stop taking samples. 
            'box_warp': 2.2, # the side-length of the bounding box spanned by the tri-planes; box_warp=1 means [-0.5, -0.5, -0.5] -> [0.5, 0.5, 0.5].
            'avg_camera_radius': 2.7, # camera orbit radius (e.g. gen_* scripts read G.rendering_kwargs['avg_camera_radius']).
            'avg_camera_pivot': [0, 0, 0.2], # center of camera rotation.
        })
    elif opts.cfg == 'ffhq_vc_fg':
        rendering_options.update({
            'depth_resolution': 48, # number of uniform samples to take per ray.
            'depth_resolution_importance': 48, # number of importance samples to take per ray.
            'ray_start': 1.6, # near point along each ray to start taking samples.
            'ray_end': 3.8, # far point along each ray to stop taking samples. 
            'box_warp': 2.2, # the side-length of the bounding box spanned by the tri-planes; box_warp=1 means [-0.5, -0.5, -0.5] -> [0.5, 0.5, 0.5].
            'avg_camera_radius': 2.7, # camera orbit radius (e.g. gen_* scripts read G.rendering_kwargs['avg_camera_radius']).
            'avg_camera_pivot': [0, 0, 0.2], # center of camera rotation.
            'white_back': False,
        })
    elif opts.cfg == 'afhq':
        rendering_options.update({
            'depth_resolution': 48,
            'depth_resolution_importance': 48,
            'ray_start': 2.25,
            'ray_end': 3.3,
            'box_warp': 1,
            'avg_camera_radius': 2.7,
            'avg_camera_pivot': [0, 0, -0.06],
        })
    elif opts.cfg == 'shapenet':
        rendering_options.update({
            'depth_resolution': 64,
            'depth_resolution_importance': 64,
            'ray_start': 0.1,
            'ray_end': 2.6,
            'box_warp': 1.6,
            'white_back': True,
            'avg_camera_radius': 1.7,
            'avg_camera_pivot': [0, 0, 0],
        })
    else:
        assert False, "Need to specify config"



    if opts.density_reg > 0:
        c.G_reg_interval = opts.density_reg_every
    # override depth resolution 
    rendering_options['depth_resolution'] = opts.depth_resolution
    rendering_options['depth_resolution_importance'] = opts.depth_resolution_importance
    c.G_kwargs.rendering_kwargs = rendering_options
    c.G_kwargs.num_fp16_res = 0
    c.G_kwargs.triplane_resolution = opts.triplane_res
    c.G_kwargs.sample_res = opts.sample_res ## add for sample res in uv plane
    c.G_kwargs.max_offset = opts.max_offset
    c.G_kwargs.use_high_res_uvmap = opts.use_high_res_uvmap
    c.G_kwargs.freeze_backbone = opts.freeze_backbone
    c.G_kwargs.use_exp_cond_gen = opts.use_exp_cond_gen
    c.G_kwargs.use_dynamic_featurev2 = opts.use_dynamic_featurev2
    c.G_kwargs.direct_512 = opts.direct_512
    c.G_kwargs.black_background = opts.black_background
    c.G_kwargs.use_eye_uv_reg = opts.use_eye_uv_reg
    c.G_kwargs.use_upper_lips_constraint = opts.use_upper_lips_constraint
    c.G_kwargs.upper_lips_n = opts.upper_lips_n
    c.G_kwargs.upper_lips_exp_cap = opts.upper_lips_exp_cap
    c.G_kwargs.upper_lips_scale_cap = opts.upper_lips_scale_cap
    c.G_kwargs.upper_lips_id_cap = opts.upper_lips_id_cap
    
    c.loss_kwargs.blur_init_sigma = 10 # Blur the images seen by the discriminator.
    c.loss_kwargs.blur_fade_kimg = c.batch_size * opts.blur_fade_kimg / 32 # Fade out the blur during the first N kimg.

    c.loss_kwargs.gpc_reg_prob = opts.gpc_reg_prob if opts.gen_pose_cond else None
    c.loss_kwargs.gpc_reg_fade_kimg = opts.gpc_reg_fade_kimg
    #c.loss_kwargs.dual_discrimination = True #False for Gaussian
    c.loss_kwargs.dual_discrimination = opts.dual_discrimination #False for Gaussian
    c.loss_kwargs.neural_rendering_resolution_initial = opts.neural_rendering_resolution_initial
    c.loss_kwargs.neural_rendering_resolution_final = opts.neural_rendering_resolution_final
    c.loss_kwargs.neural_rendering_resolution_fade_kimg = opts.neural_rendering_resolution_fade_kimg
    c.loss_kwargs.use_uv_r1 = opts.use_uv_r1
    c.loss_kwargs.use_front_face_pos_reg = opts.use_front_face_pos_reg
    c.loss_kwargs.use_eye_uv_reg = opts.use_eye_uv_reg

    c.loss_kwargs.style_mixing_prob = opts.style_mixing_prob

    # Augmentation.
    if opts.aug != 'noaug':
        c.augment_kwargs = dnnlib.EasyDict(class_name='training.augment.AugmentPipe', xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1)
        if opts.aug == 'ada':
            c.ada_target = opts.target
        if opts.aug == 'fixed':
            c.augment_p = opts.p

    # Resume.
    if opts.resume is not None:
        c.resume_pkl = opts.resume
        c.ada_kimg = 100 # Make ADA react faster at the beginning.
        c.ema_rampup = None # Disable EMA rampup.
        if not opts.resume_blur:
            c.loss_kwargs.blur_init_sigma = 0 # Disable blur rampup.
            c.loss_kwargs.gpc_reg_fade_kimg = 0 # Disable swapping rampup

    # Performance-related toggles.
    # if opts.fp32:
    #     c.G_kwargs.num_fp16_res = c.D_kwargs.num_fp16_res = 0
    #     c.G_kwargs.conv_clamp = c.D_kwargs.conv_clamp = None
    c.G_kwargs.num_fp16_res = opts.g_num_fp16_res
    c.G_kwargs.conv_clamp = 256 if opts.g_num_fp16_res > 0 else None
    c.D_kwargs.num_fp16_res = opts.d_num_fp16_res
    c.D_kwargs.conv_clamp = 256 if opts.d_num_fp16_res > 0 else None
    if opts.use_d2:
        c.D2_kwargs.num_fp16_res = opts.d_num_fp16_res
        c.D2_kwargs.conv_clamp = 256 if opts.d_num_fp16_res > 0 else None

    if opts.nobench:
        c.cudnn_benchmark = False

    # Description string.
    desc = f'{opts.cfg:s}-{dataset_name:s}-gpus{c.num_gpus:d}-batch{c.batch_size:d}-gamma{c.loss_kwargs.r1_gamma:g}'
    if opts.desc is not None:
        desc += f'-{opts.desc}'

    # Launch.
    launch_training(c=c, desc=desc, outdir=opts.outdir, dry_run=opts.dry_run)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
