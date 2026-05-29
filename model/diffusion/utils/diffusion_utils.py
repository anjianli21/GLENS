import math
import copy
from functools import partial
from einops import rearrange

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module, ModuleList
from torch import einsum


# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def cast_tuple(t, length = 1):
    if isinstance(t, tuple):
        return t
    return ((t,) * length)

def divisible_by(numer, denom):
    return (numer % denom) == 0

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

# gaussian diffusion trainer class

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])

    return torch.clip(betas, 0, 0.999)

def custom_beta_schedule(timesteps, start_beta=1e-7, end_beta=0.999):
    
    first_stage_frac = 0.4
    end_beta_first = 0.02

    second_stage_frac = 0.3
    end_beta_second = 0.5

    third_stage_frac = 0.3
    end_beta_third = end_beta

    betas = torch.zeros(timesteps)
    
    time_steps_first = int(timesteps * first_stage_frac)
    time_steps_second = int(timesteps * second_stage_frac)
    time_steps_third = timesteps - time_steps_first - time_steps_second

    # use linear schedule for the all three stages
    betas[:time_steps_first] = torch.linspace(start_beta, end_beta_first, time_steps_first)
    betas[time_steps_first:time_steps_first + time_steps_second] = torch.linspace(end_beta_first, end_beta_second, time_steps_second)
    betas[time_steps_first + time_steps_second:timesteps] = torch.linspace(end_beta_second, end_beta_third, time_steps_third)

    return betas


def sigmoid_beta_schedule(timesteps, start = -3, end = 3, tau = 1, clamp_min = 1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

# classifier free guidance functions

def uniform(shape, device):
    return torch.zeros(shape, device = device).float().uniform_(0, 1)


if __name__ == "__main__":
    # Create a beta schedule with 50 timesteps
    timesteps = 50
    # betas = cosine_beta_schedule(
    #     timesteps=timesteps,
    # )
    betas = custom_beta_schedule(
        timesteps=timesteps,
    )

    # Plot the beta schedule
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(timesteps), betas.numpy(), marker='o', linestyle='-', markersize=4)
    plt.title('Cosine Beta Schedule')
    plt.xlabel('Timestep')
    plt.ylabel('Beta Value')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Print first and last few values
    print(betas)
    