import math
import copy
import pickle
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module, ModuleList
from torch.cuda.amp import autocast


from einops import rearrange, reduce, repeat

from tqdm.auto import tqdm

from .utils.diffusion_utils import *

ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])



class GaussianDiffusion(Module):
    def __init__(
        self,
        model,
        *,
        image_size,
        training_timesteps = 1000,
        sampling_timesteps = None,
        objective = 'pred_v',
        beta_schedule = 'cosine',
        schedule_fn_kwargs = dict(),
        ddim_sampling_eta = 0.,
        auto_normalize = True,
        offset_noise_strength = 0.,  # https://www.crosslabs.org/blog/diffusion-with-offset-noise
        min_snr_loss_weight = False, # https://arxiv.org/abs/2303.09556
        min_snr_gamma = 5,
        diffusion_model_params=None,
        ddim_sampling = False,
        beta_schedule_type = '0_to_1',
        rescaled_phi = 0.7,
    ):
        super().__init__()
        assert not (type(self) == GaussianDiffusion and model.channels != model.out_dim)
        assert not hasattr(model, 'random_or_learned_sinusoidal_cond') or not model.random_or_learned_sinusoidal_cond

        self.model = model

        self.diffusion_model_params = diffusion_model_params
        self.unet_type = diffusion_model_params['unet_type']
        assert self.unet_type in ['1D', '2D'], f"unet_type must be either '1D' or '2D', got {self.unet_type}"

        self.channels = self.model.channels

        if self.unet_type == '1D':
            assert isinstance(image_size, (tuple, list)) and len(image_size) == 1, 'for 1D data, image size must be an integer or a tuple/list of one integer'
        elif self.unet_type == '2D':
            assert isinstance(image_size, (tuple, list)) and len(image_size) == 2, 'for 2D data, image size must be an integer or a tuple/list of two integers'
        else:
            raise ValueError(f"Invalid unet type: {self.unet_type}")
        
        self.image_size = image_size

        self.objective = objective

        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        self.beta_schedule_type = beta_schedule_type

        if beta_schedule == 'linear':
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == 'cosine':
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == 'sigmoid':
            beta_schedule_fn = sigmoid_beta_schedule
        elif beta_schedule == 'custom':
            beta_schedule_fn = custom_beta_schedule
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        betas = beta_schedule_fn(training_timesteps, **schedule_fn_kwargs)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        training_timesteps, = betas.shape
        self.training_timesteps = int(training_timesteps)

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps, training_timesteps) # default num sampling timesteps to number of training timesteps

        assert self.sampling_timesteps <= training_timesteps
        self.is_ddim_sampling = self.sampling_timesteps < training_timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # offset noise strength - in blogpost, they claimed 0.1 was ideal

        self.offset_noise_strength = offset_noise_strength

        # derive loss weight
        # snr - signal noise ratio

        snr = alphas_cumprod / (1 - alphas_cumprod)

        # https://arxiv.org/abs/2303.09556

        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max = min_snr_gamma)

        if objective == 'pred_noise':
            register_buffer('loss_weight', maybe_clipped_snr / snr)
        elif objective == 'pred_x0':
            register_buffer('loss_weight', maybe_clipped_snr)
        elif objective == 'pred_v':
            register_buffer('loss_weight', maybe_clipped_snr / (snr + 1))
        
        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity

        self.clip_denoised = True if auto_normalize else False

        self.rescaled_phi = rescaled_phi

        # Solver behavior model (classifier guidance) related parameters
        self.solver_behavior_apply_on_max_diffusion_steps = diffusion_model_params['solver_behavior_apply_on_max_diffusion_steps']
        self.solver_behavior_guidance_step_per_diffusion_step = diffusion_model_params['solver_behavior_guidance_step_per_diffusion_step']
        self.solver_behavior_step_size = diffusion_model_params['solver_behavior_step_size']

    @property
    def device(self):
        return self.betas.device

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, condition_c = None, clip_x_start = False, rederive_pred_noise = False, cond_scale = 6., rescaled_phi = None):
        rescaled_phi = self.rescaled_phi if rescaled_phi is None else rescaled_phi

        model_output = self.model.forward_with_cond_scale(x, t, condition_c, cond_scale = cond_scale, rescaled_phi = rescaled_phi)

        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, condition_c, cond_scale, rescaled_phi):
        rescaled_phi = self.rescaled_phi if rescaled_phi is None else rescaled_phi
        preds = self.model_predictions(x, t, condition_c, cond_scale = cond_scale, rescaled_phi = rescaled_phi)
        x_start = preds.pred_x_start

        if self.clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    def get_mean_from_classifier_guidance(self, mean, variance, t, solver_behavior_condition_c, solver_behavior_model):
        
        for _ in range(self.solver_behavior_guidance_step_per_diffusion_step):
            gradient = solver_behavior_model.solver_behavior_model.get_classifier_guidance_gradient(
                x=mean, c=solver_behavior_condition_c
            )

            # Update mean
            mean = (
                mean.float() - variance * gradient * self.solver_behavior_step_size
            )
            # print(f"gradient: {(variance * gradient)[:3, 0, :3]}")
        return mean

    @torch.inference_mode()
    def p_sample(self, x, t: int, condition_c = None, solver_behavior_condition_c = None, solver_behavior_model=None, cond_scale = 6., rescaled_phi = None):
        rescaled_phi = self.rescaled_phi if rescaled_phi is None else rescaled_phi
        b, *_, device = *x.shape, self.device
        batched_times = torch.full((b,), t, device = device, dtype = torch.long)
        model_mean, variance, model_log_variance, x_start = self.p_mean_variance(x = x, t = batched_times, condition_c = condition_c, cond_scale = cond_scale, rescaled_phi = rescaled_phi)
        
        # Apply solver behavior model as guidance if t <= solver_behavior_apply_on_max_diffusion_steps
        if t <= self.solver_behavior_apply_on_max_diffusion_steps:
            if solver_behavior_condition_c is not None and solver_behavior_model is not None:
                model_mean = self.get_mean_from_classifier_guidance(mean=model_mean,
                                                                    variance=variance,
                                                                    t=batched_times,
                                                                    solver_behavior_condition_c=solver_behavior_condition_c,
                                                                    solver_behavior_model=solver_behavior_model)
            else:
                raise ValueError("solver_behavior_condition_c and solver_behavior_model are not provided")
        
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.inference_mode()
    def p_sample_loop(self, shape, condition_c, solver_behavior_condition_c = None, solver_behavior_model=None, return_all_timesteps = False, cond_scale = 6., rescaled_phi = None):
        rescaled_phi = self.rescaled_phi if rescaled_phi is None else rescaled_phi
        batch, device = shape[0], condition_c.device

        img = torch.randn(shape, device = device)
        imgs = [img]

        x_start = None

        for t in tqdm(reversed(range(0, self.training_timesteps)), desc = 'sampling loop time step', total = self.training_timesteps):
            
            img, x_start = self.p_sample(
                img,
                t,
                condition_c,
                solver_behavior_condition_c,
                solver_behavior_model=solver_behavior_model,
                cond_scale = cond_scale,
                rescaled_phi = rescaled_phi,
            )
            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

        ret = self.unnormalize(ret)
        return ret
    
    @torch.inference_mode()
    def ddim_sample(self, shape, condition_c, return_all_timesteps = False):
        batch, device, total_timesteps, sampling_timesteps, eta = shape[0], condition_c.device, self.training_timesteps, self.sampling_timesteps, self.ddim_sampling_eta

        # Create sampling timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        img = torch.randn(shape, device=device)
        imgs = [img]

        x_start = None

        for time, time_next in tqdm(time_pairs, desc='sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            
            # Get model predictions
            preds = self.model_predictions(img, time_cond, condition_c, clip_x_start=False, rederive_pred_noise=False)
            pred_noise, x_start = preds.pred_noise, preds.pred_x_start

            if time_next < 0:
                img = x_start
                imgs.append(img)
                continue

            # Calculate alphas
            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            # Calculate sigma and c for DDIM step
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            # Add noise
            noise = torch.randn_like(img)

            # DDIM step
            img = x_start * alpha_next.sqrt() + \
                c * pred_noise + \
                sigma * noise

            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim=1)
        return ret

    @torch.inference_mode()
    def sample(self, batch, sample_num, args, solver_behavior_model=None, return_all_timesteps=False, cond_scale=6., rescaled_phi=None):
        rescaled_phi = self.rescaled_phi if rescaled_phi is None else rescaled_phi
        
        # Prepare condition c
        c = batch['encoded_condition_y']

        # Expand c and x_current, x_current by sample_num
        batch_size, c_feature_size = c.shape[0], c.shape[1]
        channels = self.channels

        # Handle different dimensions based on unet_type
        if self.unet_type == '1D':
            h = self.image_size[0]
            shape = (batch_size * sample_num, channels, h)
        else:  # '2D'
            h, w = self.image_size
            shape = (batch_size * sample_num, channels, h, w)

        # Expand c and x_current, x_current by sample_num
        c_expanded = c.unsqueeze(1).expand(batch_size, sample_num, c_feature_size).reshape(-1, c_feature_size)

        # Prepare solver behavior model encoded condition y
        if solver_behavior_model is not None:
            solver_behavior_c = batch['solver_behavior_encoded_condition_y']
            solver_behavior_c_feature_size = solver_behavior_c.shape[1]
            solver_behavior_c_expanded = solver_behavior_c.unsqueeze(1).expand(batch_size, sample_num, solver_behavior_c_feature_size).reshape(-1, solver_behavior_c_feature_size)

        if self.is_ddim_sampling:
            raise ValueError("DDIM sampling is not supported when solver behavior model guidance is used")
            # print("use ddim sampling, no classifier-free guidance")
            # samples = self.ddim_sample(shape=shape, condition_c=c_expanded, return_all_timesteps = return_all_timesteps)
        else:
            print(f"use DDPM sampling, classifier-free guidance, apply solver behavior model as guidance if t <= {self.solver_behavior_apply_on_max_diffusion_steps}")
            samples = self.p_sample_loop(
                shape=shape,
                condition_c=c_expanded,
                solver_behavior_condition_c=solver_behavior_c_expanded,
                solver_behavior_model=solver_behavior_model,
                return_all_timesteps = return_all_timesteps,
                cond_scale = cond_scale,
                rescaled_phi = rescaled_phi,
            )

        # Reshape samples based on unet_type
        if self.unet_type == '1D':
            samples = samples.view(batch_size, sample_num, channels, h)
        else:  # '2D'
            samples = samples.view(batch_size, sample_num, channels, h, w)
        
        return samples

    @autocast(enabled = False)
    def q_sample(self, x_start, t, noise = None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, condition_c, noise = None, offset_noise_strength = None):

        noise = default(noise, lambda: torch.randn_like(x_start))

        x = self.q_sample(x_start = x_start, t = t, noise = noise)

        model_out = self.model(x=x, time=t, c=condition_c)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = F.mse_loss(model_out, target, reduction = 'none')

        loss = reduce(loss, 'b ... -> b', 'mean')
        # apply timestep/SNR weighting
        loss = loss * extract(self.loss_weight, t, loss.shape)

        return loss.mean()

    def diffusion_forward(self, img, condition_c, *args, **kwargs):

        device = img.device
        if self.unet_type == '1D':
            b, c, h = img.shape
            assert h == self.image_size[0], f'length of 1D data must be {self.image_size[0]}, got {h}'
        else:  # '2D'
            b, c, h, w = img.shape
            assert h == self.image_size[0] and w == self.image_size[1], f'height and width of image must be {self.image_size}, got {(h, w)}'
        
        t = torch.randint(0, self.training_timesteps, (b,), device=device).long()

        img = self.normalize(img)
        return self.p_losses(img, t, condition_c, *args, **kwargs)
    
    def forward(self, batch):

        # Prepare UNet data ##############################################################################################
        batch_size = batch['encoded_condition_y'].shape[0]
        device = batch['encoded_condition_y'].device

        # Prepare condition c
        c = batch['encoded_condition_y']

        # prepare data x
        x = batch['data_x_concat']

        # Validate input shape based on unet_type
        if self.unet_type == '1D':
            assert len(x.shape) == 3, f"For 1D UNet, input should have shape (batch, channel, h), got {x.shape}"
        else:  # '2D'
            assert len(x.shape) == 4, f"For 2D UNet, input should have shape (batch, channel, h, w), got {x.shape}"

        p_losses = self.diffusion_forward(img=x, condition_c=c)
        
        return p_losses