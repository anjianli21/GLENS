from model.diffusion.diffusion import GaussianDiffusion
from model.utils.unet_1d import Unet1D

def build_diffusion(model_params):
    if model_params.get('unet_type') not in (None, "1D"):
        raise ValueError(f"Only unet_type='1D' is supported (got {model_params['unet_type']!r})")

    unet_model = Unet1D(
        dim=model_params['unet_model_dim'],
        dim_mults=model_params['unet_dim_mults'],
        embed_y_all_dim=model_params['embed_y_all_dim'],
        channels=model_params['data_x_channel'],
        dropout=model_params['dropout'],
        cond_drop_prob=model_params['cond_drop_prob'],
    )

    diffusion_model = GaussianDiffusion(
        model=unet_model,
        training_timesteps=model_params['training_timesteps'],
        sampling_timesteps=model_params['sampling_timesteps'],
        image_size=(model_params['data_x_size_h'],),
        objective='pred_noise',
        beta_schedule=model_params['beta_schedule'],
        beta_schedule_type=model_params['beta_schedule_type'],
        auto_normalize=model_params['auto_normalize'],
        rescaled_phi=model_params['rescaled_phi'],
        diffusion_model_params=model_params,
    )

    return diffusion_model
