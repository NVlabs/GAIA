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

"""Loss functions."""

import numpy as np
import torch
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import upfirdn2d
from training.dual_discriminator import filtered_resizing
from gaia_utils.gaussian_model import _apply_opacity_activation

#----------------------------------------------------------------------------

class Loss:
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg): # to be overridden by subclass
        raise NotImplementedError()

#----------------------------------------------------------------------------

class StyleGAN2Loss(Loss):
    def __init__(self,
                device,
                G,
                D,
                D2=None,
                augment_pipe=None,
                r1_gamma=10,
                style_mixing_prob=0,
                pl_weight=0,
                pl_batch_shrink=2,
                pl_decay=0.01,
                pl_no_weight_grad=False,
                blur_init_sigma=0,
                blur_fade_kimg=0,
                r1_gamma_init=0,
                r1_gamma_fade_kimg=0,
                neural_rendering_resolution_initial=64,
                neural_rendering_resolution_final=None,
                neural_rendering_resolution_fade_kimg=0,
                gpc_reg_fade_kimg=1000,
                gpc_reg_prob=None,
                dual_discrimination=False,
                filter_mode='antialiased',
                regularization=False,
                new_layers_disc_start_kimg=None,
                new_layers_disc_blend_kimg=None,
                use_tv_reg=True,
                use_opacity_reg=True,
                use_pos_reg=True,
                use_scale_reg=True,
                use_tv_uv_gen=False,
                lambda_raw_gaussian_position=0.1,
                lambda_raw_gaussian_scale=0.05,
                lambda_tv_genuv_pos=100.0,
                lambda_tv_genuv_pos_front_face=100.0,
                lambda_tv_genuv_pos_other=5.0,
                use_fade_exp=False,
                use_face_mask_uv_reg=False,
                use_uv_r1=False,
                use_front_face_pos_reg=False,
                lambda_pos_front=0.1,
                use_eye_uv_reg=False,
                lambda_tv_eye=5.0,
                weight_d1=0.5,
                weight_d2=0.5,
                r1_gamma2=1,
                warmup_weight_d2=False,
                use_d2=False,
                load_d2_from_d1=True,
                init_k=23000,
                d2_start_weight=0,
                d2_end_weight=0.5,
                use_exp_reg=False,
                lambda_exp_reg = 0.1,
                use_exp_color_reg=False,
                lambda_exp_color_skin=2.0,
                lambda_exp_color_eye=1.0,
                lambda_exp_color_forehead=1.0,
                lambda_exp_color_mouth=1.0,
                **kwargs
    ):
        super().__init__()
        self.device             = device
        self.G                  = G
        self.D                  = D
        self.D2                 = D2
        self.augment_pipe       = augment_pipe
        self.r1_gamma           = r1_gamma
        self.style_mixing_prob  = style_mixing_prob
        self.pl_weight          = pl_weight
        self.pl_batch_shrink    = pl_batch_shrink
        self.pl_decay           = pl_decay
        self.pl_no_weight_grad  = pl_no_weight_grad
        self.pl_mean            = torch.zeros([], device=device)
        self.blur_init_sigma    = blur_init_sigma
        self.blur_fade_kimg     = blur_fade_kimg
        self.r1_gamma_init      = r1_gamma_init
        self.r1_gamma_fade_kimg = r1_gamma_fade_kimg
        self.neural_rendering_resolution_initial = neural_rendering_resolution_initial
        self.neural_rendering_resolution_final = neural_rendering_resolution_final
        self.neural_rendering_resolution_fade_kimg = neural_rendering_resolution_fade_kimg
        self.gpc_reg_fade_kimg = gpc_reg_fade_kimg
        self.gpc_reg_prob = gpc_reg_prob
        self.dual_discrimination = dual_discrimination
        self.filter_mode = filter_mode
        self.resample_filter = upfirdn2d.setup_filter([1,3,3,1], device=device)
        self.blur_raw_target = True
        self.lambda_tv_eye = lambda_tv_eye
        self.use_eye_uv_reg = use_eye_uv_reg

        self.regularization = regularization
        self.new_layers_disc_start_kimg = new_layers_disc_start_kimg
        self.new_layers_disc_blend_kimg = new_layers_disc_blend_kimg
        self.use_tv_reg = use_tv_reg
        self.use_tv_uv_gen = use_tv_uv_gen
        self.use_opacity_reg = use_opacity_reg
        self.use_pos_reg = use_pos_reg
        self.use_scale_reg = use_scale_reg
        
        self.lambda_raw_gaussian_position = lambda_raw_gaussian_position
        self.lambda_raw_gaussian_scale = lambda_raw_gaussian_scale
        self.lambda_tv_genuv_pos = lambda_tv_genuv_pos
        self.lambda_tv_genuv_pos_front_face = lambda_tv_genuv_pos_front_face
        self.lambda_tv_genuv_pos_other = lambda_tv_genuv_pos_other
        self.use_fade_exp = use_fade_exp
        self.use_face_mask_uv_reg = use_face_mask_uv_reg
        self.use_uv_r1 = use_uv_r1
        self.use_front_face_pos_reg = use_front_face_pos_reg
        self.lambda_pos_front = lambda_pos_front
        self.use_exp_reg = use_exp_reg
        self.lambda_exp_reg = lambda_exp_reg
        self.use_exp_color_reg = use_exp_color_reg
        self.lambda_exp_color_skin = lambda_exp_color_skin
        self.lambda_exp_color_eye = lambda_exp_color_eye
        self.lambda_exp_color_forehead = lambda_exp_color_forehead
        self.lambda_exp_color_mouth = lambda_exp_color_mouth

        self.weight_d1 = weight_d1
        self.weight_d2 = weight_d2
        self.r1_gamma2 = r1_gamma2
        self.warmup_weight_d2 = warmup_weight_d2
        self.use_d2 = use_d2
        self.init_k = init_k
        self.d2_start_weight = d2_start_weight
        self.d2_end_weight = d2_end_weight
        self.load_d2_from_d1 = load_d2_from_d1

        assert self.gpc_reg_prob is None or (0 <= self.gpc_reg_prob <= 1)

    def run_G(self, z, c, swapping_prob, neural_rendering_resolution, zero_exp_prob=None, update_emas=False, cur_nimg=0):
        syn_c = c.clone()
        c_exp = c[:, 325:]
        c_zeroed = c.clone()
        if zero_exp_prob is not None:
            c_zeroed[:, 325:] = torch.where(torch.rand((c.shape[0], 1), device=c.device) < zero_exp_prob, torch.zeros_like(c_exp), c_exp)

        if swapping_prob is not None:
            c_swapped = torch.roll(c_zeroed.clone(), 1, 0)#c[:, :25]#c.clone()
            c_gen_conditioning = torch.where(torch.rand((c_zeroed.shape[0], 1), device=c.device) < swapping_prob, c_swapped, c_zeroed)#c[:, :25]
        else:
            c_gen_conditioning = torch.zeros_like(c)#c[:, :25]

        exp_cond = None
        ws = self.G.mapping(z, c_gen_conditioning, update_emas=update_emas)

        ws_exp = None
        if self.style_mixing_prob > 0:
            with torch.autograd.profiler.record_function('style_mixing'):
                cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), c, update_emas=False)[:, cutoff:] #c[:, :25]
        gen_output = self.G.synthesis(ws, syn_c, neural_rendering_resolution=neural_rendering_resolution, update_emas=update_emas, ws_exp=ws_exp, exp_cond=exp_cond)
        return gen_output, ws

    def run_D(self, img, c, blur_sigma=0, blur_sigma_raw=0, update_emas=False,
              alpha_new_layers_disc=None):

        if self.D.shape_cond_disc:
            c = c[:, :325]

        blur_size = np.floor(blur_sigma * 3)
        if blur_size > 0:
            with torch.autograd.profiler.record_function('blur'):
                f = torch.arange(-blur_size, blur_size + 1, device=img['image'].device).div(blur_sigma).square().neg().exp2()
                img['image'] = upfirdn2d.filter2d(img['image'], f / f.sum())

        if self.augment_pipe is not None:
            augmented_pair = self.augment_pipe(torch.cat([img['image'],
                                                    torch.nn.functional.interpolate(img['image_raw'], size=img['image'].shape[2:], mode='bilinear', antialias=True)],
                                                    dim=1))
            img['image'] = augmented_pair[:, :img['image'].shape[1]]
            img['image_raw'] = torch.nn.functional.interpolate(augmented_pair[:, img['image'].shape[1]:], size=img['image_raw'].shape[2:], mode='bilinear', antialias=True)

        if alpha_new_layers_disc is None:
            logits = self.D(img, c, update_emas=update_emas)#c[:, :25]
        else:
            logits = self.D(img, c, update_emas=update_emas, alpha_new_layers=alpha_new_layers_disc)#c[:, :25]

        return logits

    def run_D2(self, img, c, blur_sigma=0, blur_sigma_raw=0, update_emas=False,
              alpha_new_layers_disc=None):
        c = torch.cat([c[:, :25], c[:, 325:]], dim=1)

        blur_size = np.floor(blur_sigma * 3)
        if blur_size > 0:
            with torch.autograd.profiler.record_function('blur'):
                f = torch.arange(-blur_size, blur_size + 1, device=img['image'].device).div(blur_sigma).square().neg().exp2()
                img['image'] = upfirdn2d.filter2d(img['image'], f / f.sum())

        if self.augment_pipe is not None:
            augmented_pair = self.augment_pipe(torch.cat([img['image'],
                                                    torch.nn.functional.interpolate(img['image_raw'], size=img['image'].shape[2:], mode='bilinear', antialias=True)],
                                                    dim=1))
            img['image'] = augmented_pair[:, :img['image'].shape[1]]
            img['image_raw'] = torch.nn.functional.interpolate(augmented_pair[:, img['image'].shape[1]:], size=img['image_raw'].shape[2:], mode='bilinear', antialias=True)

        #logits = self.D(img, c, update_emas=update_emas)

        if alpha_new_layers_disc is None:
            logits = self.D2(img, c, update_emas=update_emas)#c[:, :25]
        else:
            logits = self.D2(img, c, update_emas=update_emas, alpha_new_layers=alpha_new_layers_disc)#c[:, :25]

        return logits

    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg):
        #phase: #Gboth Dmain Dreg
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth', 'D2main', 'D2reg', 'D2both']
        if self.G.rendering_kwargs.get('density_reg', 0) == 0:
            phase = {'Greg': 'none', 'Gboth': 'Gmain'}.get(phase, phase)
        if self.r1_gamma == 0:
            phase = {'Dreg': 'none', 'Dboth': 'Dmain'}.get(phase, phase)
        blur_sigma = max(1 - cur_nimg / (self.blur_fade_kimg * 1e3), 0) * self.blur_init_sigma if self.blur_fade_kimg > 0 else 0 ##the greater currnimg, the smaller # init: 10  blur_fade_kimg 200
        
        alpha_new_layers_disc = min((cur_nimg - self.new_layers_disc_start_kimg * 1e3) / (self.new_layers_disc_blend_kimg * 1e3),
                                    1) if self.new_layers_disc_start_kimg is not None else None

        r1_gamma = self.r1_gamma
        r1_gamma2 = self.r1_gamma2

        alpha = min(cur_nimg / (self.gpc_reg_fade_kimg * 1e3), 1) if self.gpc_reg_fade_kimg > 0 else 1 ## getting larger till 1
        swapping_prob = (1 - alpha) * 1 + alpha * self.gpc_reg_prob if self.gpc_reg_prob is not None else None ## getting smaller
        zero_exp_prob = None
        if self.use_fade_exp:
            zero_exp_prob = min(cur_nimg / (self.gpc_reg_fade_kimg * 4e3), 1) if self.gpc_reg_fade_kimg > 0 else 0 ## getting larger till 1

        if self.warmup_weight_d2:
            beta = min((cur_nimg-(self.init_k * 1e3)) / (1000 * 1e3), 1) ## getting larger till 1
            beta = max(beta, 0)
            self.weight_d2 = (1 - beta) * self.d2_start_weight + beta * self.d2_end_weight
            self.weight_d1 = 1 - self.weight_d2

        if self.neural_rendering_resolution_final is not None and self.neural_rendering_resolution_fade_kimg > 0:
            alpha = min(cur_nimg / (self.neural_rendering_resolution_fade_kimg * 1e3), 1)
            neural_rendering_resolution = int(np.rint(self.neural_rendering_resolution_initial * (1 - alpha) + self.neural_rendering_resolution_final * alpha))
        else:
            neural_rendering_resolution = self.neural_rendering_resolution_initial

        real_img_raw = filtered_resizing(real_img, size=neural_rendering_resolution, f=self.resample_filter, filter_mode=self.filter_mode)

        if self.blur_raw_target:
            blur_size = np.floor(blur_sigma * 3)
            if blur_size > 0:
                f = torch.arange(-blur_size, blur_size + 1, device=real_img_raw.device).div(blur_sigma).square().neg().exp2()
                real_img_raw = upfirdn2d.filter2d(real_img_raw, f / f.sum())


        real_img = {'image': real_img, 'image_raw': real_img_raw}

        # Gmain: Maximize logits for generated images.
        if phase in ['Gmain', 'Gboth']:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, swapping_prob=swapping_prob, neural_rendering_resolution=neural_rendering_resolution, zero_exp_prob=zero_exp_prob, cur_nimg=cur_nimg)

                gen_logits = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma, alpha_new_layers_disc=alpha_new_layers_disc)
                if self.use_d2:
                    gen_logits2 = self.run_D2(gen_img, gen_c, blur_sigma=blur_sigma, alpha_new_layers_disc=alpha_new_layers_disc)
                    training_stats.report('Loss/scores/fake_exp', gen_logits2)
                    training_stats.report('Loss/signs/fake_exp', gen_logits2.sign())

                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                if self.use_d2:
                    loss_Gmain1 = torch.nn.functional.softplus(-gen_logits)
                    loss_Gmain2 = torch.nn.functional.softplus(-gen_logits2)
                    training_stats.report('Loss/G1/loss', loss_Gmain1)
                    training_stats.report('Loss/G2/loss', loss_Gmain2)
                    loss_Gmain = self.weight_d1 * loss_Gmain1 + self.weight_d2 * loss_Gmain2
                else:
                    loss_Gmain = torch.nn.functional.softplus(-gen_logits)

                training_stats.report('Loss/G/loss', loss_Gmain)

                if self.regularization:
                    loss_Gmain = loss_Gmain.squeeze(1)  # [B, 1] -> [B]
                    raw_gaussian_attributes = gen_img['raw_gaussian_attributes']
                    gaussian_attributes = gen_img['gaussian_attributes']

                    if self.use_pos_reg:

                        reg_raw_gaussian_position = raw_gaussian_attributes['pos_offset'].norm(dim=-1).mean(dim=1)  # [B]
                        training_stats.report('Loss/G/reg_raw_gaussian_position', reg_raw_gaussian_position)
                        loss_Gmain = loss_Gmain + self.lambda_raw_gaussian_position * reg_raw_gaussian_position

                        if self.use_front_face_pos_reg:
                            reg_raw_gaussian_position_front = raw_gaussian_attributes['front_pos_offset'].norm(dim=-1).mean(dim=1)  # [B]
                            training_stats.report('Loss/G/reg_raw_gaussian_position_front', reg_raw_gaussian_position_front)
                            loss_Gmain = loss_Gmain + self.lambda_pos_front * reg_raw_gaussian_position_front
                    
                    if self.use_exp_reg:
                        reg_exp = gen_img['uv_exp'].norm(dim=1).mean(dim=-1).mean(dim=-1)
                        training_stats.report('Loss/G/reg_exp', reg_exp)
                        loss_Gmain = loss_Gmain + self.lambda_exp_reg * reg_exp

                    if self.use_exp_color_reg and gen_img['exp_planes_cache'] is not None:
                        exp_color = gen_img['exp_planes_cache'][:, 10:22]
                        face_mask = gen_img['face_mask']
                        mouth_lips_mask = gen_img['mouth_lips_mask']
                        eye_mask = gen_img['eye_mask']
                        forehead_mask = gen_img['forehead_mask']
                        if eye_mask is None:
                            eye_mask = torch.zeros_like(face_mask)

                        mouth_mask = mouth_lips_mask & face_mask
                        eye_region_mask = eye_mask & face_mask & (~mouth_mask)
                        forehead_region_mask = forehead_mask & face_mask & (~mouth_mask) & (~eye_region_mask)
                        skin_mask = face_mask & (~mouth_mask) & (~eye_region_mask) & (~forehead_region_mask)

                        loss_exp_color = 0
                        if bool(skin_mask.any().item()):
                            reg_skin = exp_color[:, :, skin_mask].norm(dim=1).mean()
                            training_stats.report('Loss/G/reg_exp_color_skin', reg_skin)
                            loss_exp_color = loss_exp_color + self.lambda_exp_color_skin * reg_skin
                        if bool(eye_region_mask.any().item()):
                            reg_eye = exp_color[:, :, eye_region_mask].norm(dim=1).mean()
                            training_stats.report('Loss/G/reg_exp_color_eye', reg_eye)
                            loss_exp_color = loss_exp_color + self.lambda_exp_color_eye * reg_eye
                        if bool(forehead_region_mask.any().item()):
                            reg_forehead = exp_color[:, :, forehead_region_mask].norm(dim=1).mean()
                            training_stats.report('Loss/G/reg_exp_color_forehead', reg_forehead)
                            loss_exp_color = loss_exp_color + self.lambda_exp_color_forehead * reg_forehead
                        if bool(mouth_mask.any().item()):
                            reg_mouth = exp_color[:, :, mouth_mask].norm(dim=1).mean()
                            training_stats.report('Loss/G/reg_exp_color_mouth', reg_mouth)
                            loss_exp_color = loss_exp_color + self.lambda_exp_color_mouth * reg_mouth
                        loss_Gmain = loss_Gmain + loss_exp_color

                    if self.use_scale_reg:
                        reg_raw_gaussian_scale = raw_gaussian_attributes['scale'].norm(dim=-1).mean(dim=1)  # [B]
                        training_stats.report('Loss/G/reg_raw_gaussian_scale', reg_raw_gaussian_scale)
                        loss_Gmain = loss_Gmain + self.lambda_raw_gaussian_scale * reg_raw_gaussian_scale

                    ## opacity regularization
                    if self.use_opacity_reg:
                        opacities = _apply_opacity_activation(gaussian_attributes['opacity'])
                        beta_loss = ((0.1 + opacities).log() + (1.1 - opacities).log() + 2.20727).mean()
                        training_stats.report('Loss/G/beta_loss', beta_loss)
                        loss_Gmain = loss_Gmain + 0.1 * beta_loss

                    ## tv regularization
                    if self.use_tv_reg:
                        uv_renderings = self.G.get_uv_rendering(gen_c, gen_img, include_transparent_gaussians=False)
                        mask = 1 - (uv_renderings[:, [2]] + 1)/2

                        uv_renderings_no_blend = ((uv_renderings - (1 - mask)) / mask)  # Undo effect of alpha blending. Background pixels will be inf
                        background_mask = mask == 0

                        mask_y = (background_mask[:, :, 1:] | background_mask[:, :, :-1]).repeat(1, 3, 1, 1)
                        mask_x = (background_mask[:, :, :, 1:] | background_mask[:, :, :, :-1]).repeat(1, 3, 1, 1)

                        uv_difference_y = uv_renderings_no_blend[:, :, 1:] - uv_renderings_no_blend[:, :, :-1]
                        uv_difference_x = uv_renderings_no_blend[:, :, :, 1:] - uv_renderings_no_blend[:, :, :, :-1]

                        reg_tv_uv_rendering_y = uv_difference_y[~mask_y].abs().mean()
                        reg_tv_uv_rendering_x = uv_difference_x[~mask_x].abs().mean()
                        reg_tv_uv_rendering = reg_tv_uv_rendering_x + reg_tv_uv_rendering_y

                        training_stats.report('Loss/G/reg_tv_uv_rendering', reg_tv_uv_rendering)

                        loss_Gmain = loss_Gmain + 100 * reg_tv_uv_rendering

                    if self.use_tv_uv_gen or self.use_eye_uv_reg:
                        uv_map = gen_img['uv_map']
                        if self.G.direct_512:
                            valid_samples = gen_img['valid_samples']
                        else:
                            valid_samples = gen_img['valid_samples_256']
                        uv_difference_y = uv_map[:, :3, 1:] - uv_map[:, :3, :-1]
                        uv_difference_x = uv_map[:, :3, :, 1:] - uv_map[:, :3, :, :-1]

                        mask_y = valid_samples[1:, :] | valid_samples[:-1, :]
                        mask_x = valid_samples[:, 1:] | valid_samples[:, :-1]
                        if self.use_face_mask_uv_reg:
                            face_mask = gen_img['face_mask']
                            mask_y_front_face = mask_y & (face_mask[1:, :] | face_mask[:-1, :])
                            mask_x_front_face = mask_x & (face_mask[:, 1:] | face_mask[:, :-1])
                            reg_tv_uv_gen_y_front_face = uv_difference_y[:, :, mask_y_front_face].abs().mean()
                            reg_tv_uv_gen_x_front_face = uv_difference_x[:, :, mask_x_front_face].abs().mean()
                            reg_tv_uv_gen_front_face = reg_tv_uv_gen_x_front_face + reg_tv_uv_gen_y_front_face
                            reg_tv_uv_gen_y_other = uv_difference_y[:, :, mask_y & (~mask_y_front_face)].abs().mean()
                            reg_tv_uv_gen_x_other = uv_difference_x[:, :, mask_x & (~mask_x_front_face)].abs().mean()
                            reg_tv_uv_gen_other = reg_tv_uv_gen_x_other + reg_tv_uv_gen_y_other
                            reg_tv_uv_gen = self.lambda_tv_genuv_pos_front_face * reg_tv_uv_gen_front_face + self.lambda_tv_genuv_pos_other * reg_tv_uv_gen_other
                        elif self.use_eye_uv_reg:
                            eye_mask = gen_img['eye_mask']
                            mask_y_eye= mask_y & (eye_mask[1:, :] | eye_mask[:-1, :])
                            mask_x_eye = mask_x & (eye_mask[:, 1:] | eye_mask[:, :-1])
                            reg_tv_uv_gen_y_eye = uv_difference_y[:, :, mask_y_eye].abs().mean()
                            reg_tv_uv_gen_x_eye = uv_difference_x[:, :, mask_x_eye].abs().mean()
                            reg_tv_uv_gen_eye = reg_tv_uv_gen_x_eye + reg_tv_uv_gen_y_eye
                            reg_tv_uv_gen = self.lambda_tv_eye*reg_tv_uv_gen_eye
                        else:
                            reg_tv_uv_gen_y = uv_difference_y[:, :, mask_y].abs().mean()
                            reg_tv_uv_gen_x = uv_difference_x[:, :, mask_x].abs().mean()
                            reg_tv_uv_gen = reg_tv_uv_gen_x + reg_tv_uv_gen_y
                            reg_tv_uv_gen = self.lambda_tv_genuv_pos * reg_tv_uv_gen

                        training_stats.report('Loss/G/reg_tv_uv_gen', reg_tv_uv_gen)

                        loss_Gmain = loss_Gmain + reg_tv_uv_gen#ori 100

            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_Gmain.mean().mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if phase in ['Dmain', 'Dboth']:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, swapping_prob=swapping_prob, neural_rendering_resolution=neural_rendering_resolution, update_emas=True, zero_exp_prob=zero_exp_prob, cur_nimg=cur_nimg)

                gen_logits = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma, update_emas=True, alpha_new_layers_disc=alpha_new_layers_disc)

                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Dgen = torch.nn.functional.softplus(gen_logits)

            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.mean().mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if phase in ['Dmain', 'Dreg', 'Dboth']:
            name = 'Dreal' if phase == 'Dmain' else 'Dr1' if phase == 'Dreg' else 'Dreal_Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp_image = real_img['image'].detach().requires_grad_(phase in ['Dreg', 'Dboth'])
                real_img_tmp_image_raw = real_img['image_raw'].detach().requires_grad_(phase in ['Dreg', 'Dboth'])
                if self.D.use_uv_render_disc:
                    with torch.no_grad():
                        uv_img = self.G.render_uv(real_c)
                    real_img.update({'uv_render': uv_img})
                    real_img_tmp_image_uv_render = real_img['uv_render'].detach().requires_grad_(phase in ['Dreg', 'Dboth'])
                real_img_tmp = {'image': real_img_tmp_image, 'image_raw': real_img_tmp_image_raw}
                if self.D.use_uv_render_disc:
                    real_img_tmp.update({'uv_render': real_img_tmp_image_uv_render})

                real_logits = self.run_D(real_img_tmp, real_c, blur_sigma=blur_sigma, alpha_new_layers_disc=alpha_new_layers_disc)
                
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())

                loss_Dreal = 0
                if phase in ['Dmain', 'Dboth']:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits)
                    training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)
                loss_Dr1 = 0
                if phase in ['Dreg', 'Dboth']:
                    if self.dual_discrimination:
                        with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                            r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp['image'], real_img_tmp['image_raw']], create_graph=True, only_inputs=True)
                            r1_grads_image = r1_grads[0]
                            r1_grads_image_raw = r1_grads[1]
                        r1_penalty = r1_grads_image.square().sum([1,2,3]) + r1_grads_image_raw.square().sum([1,2,3])

                    else: # single discrimination
                        with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                            r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp['image']], create_graph=True, only_inputs=True)
                            r1_grads_image = r1_grads[0]
                        r1_penalty = r1_grads_image.square().sum([1,2,3])
                    loss_Dr1 = r1_penalty * (r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                (loss_Dreal + loss_Dr1).mean().mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen2 = 0
        if phase in ['D2main', 'D2both']:
            with torch.autograd.profiler.record_function('D2gen_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, swapping_prob=swapping_prob, neural_rendering_resolution=neural_rendering_resolution, update_emas=True, zero_exp_prob=zero_exp_prob, cur_nimg=cur_nimg)

                gen_logits = self.run_D2(gen_img, gen_c, blur_sigma=blur_sigma, update_emas=True, alpha_new_layers_disc=alpha_new_layers_disc)

                training_stats.report('Loss/scores/fake_exp', gen_logits)
                training_stats.report('Loss/signs/fake_exp', gen_logits.sign())
                loss_Dgen2 = torch.nn.functional.softplus(gen_logits)

            with torch.autograd.profiler.record_function('D2gen_backward'):
                loss_Dgen2.mean().mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if phase in ['D2main', 'D2reg', 'D2both']:
            name = 'D2real' if phase == 'D2main' else 'D2r1' if phase == 'D2reg' else 'D2real_Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp_image = real_img['image'].detach().requires_grad_(phase in ['D2reg', 'D2both'])
                real_img_tmp_image_raw = real_img['image_raw'].detach().requires_grad_(phase in ['D2reg', 'D2both'])
                if self.D2.use_uv_render_disc:
                    with torch.no_grad():
                        uv_img = self.G.render_uv(real_c)
                    real_img.update({'uv_render': uv_img})
                    real_img_tmp_image_uv_render = real_img['uv_render'].detach().requires_grad_(phase in ['D2reg', 'D2both'])
                real_img_tmp = {'image': real_img_tmp_image, 'image_raw': real_img_tmp_image_raw}
                if self.D2.use_uv_render_disc:
                    real_img_tmp.update({'uv_render': real_img_tmp_image_uv_render})

                real_logits = self.run_D2(real_img_tmp, real_c, blur_sigma=blur_sigma, alpha_new_layers_disc=alpha_new_layers_disc)
                
                training_stats.report('Loss/scores/real_exp', real_logits)
                training_stats.report('Loss/signs/real_exp', real_logits.sign())

                loss_Dreal = 0
                if phase in ['D2main', 'D2both']:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits)
                    training_stats.report('Loss/D2/loss', loss_Dgen2 + loss_Dreal)
                loss_Dr1 = 0
                if phase in ['D2reg', 'D2both']:
                    with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                        if self.D2.use_uv_render_disc:
                            r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp['image'], real_img_tmp['uv_render']], create_graph=True, only_inputs=True)
                            r1_grads_image = r1_grads[0]
                            r1_grads_uv = r1_grads[1]
                        else:
                            r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp['image']], create_graph=True, only_inputs=True)
                            r1_grads_image = r1_grads[0]
                    if self.D2.use_uv_render_disc:
                        r1_penalty = r1_grads_image.square().sum([1,2,3]) + r1_grads_uv.square().sum([1,2,3])
                    else:
                        r1_penalty = r1_grads_image.square().sum([1,2,3])
                    loss_Dr1 = r1_penalty * (r1_gamma2 / 2)
                    training_stats.report('Loss/r1_penalty_exp', r1_penalty)
                    training_stats.report('Loss/D2/reg', loss_Dr1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                (loss_Dreal + loss_Dr1).mean().mul(gain).backward()