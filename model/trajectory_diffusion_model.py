import os
import torch
import torch.nn as nn

from .condition_encoder import build_condition_encoder
from .diffusion import build_diffusion

class TrajectoryDiffusion(nn.Module):

    def __init__(self, model_params):
        super(TrajectoryDiffusion, self).__init__()

        self.model_params = model_params

        self.condition_encoder = build_condition_encoder(model_params=self.model_params)

        self.diffusion_model = build_diffusion(model_params=self.model_params)

    def forward(self, batch, curr_epoch_num=None):

        # Encode condition
        encoded_condition_batch = self.condition_encoder(batch)

        # Diffusion loss
        diffusion_loss = self.diffusion_model(batch=encoded_condition_batch)

        return diffusion_loss.mean()
    
    def get_loss(self):
        loss, tb_dict, disp_dict = self.motion_decoder.get_loss()

        return loss, tb_dict, disp_dict

    # Print parameter counts by layer
    def print_model_parameters(self, model):
        params = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                params.append((name, param.size(), param.numel()))

        # Sort by number of parameters (desc)
        params.sort(key=lambda x: x[2], reverse=True)

        for name, size, num_params in params:
            print(f"Layer: {name} | Size: {size} | Number of parameters: {num_params}")

    @torch.no_grad()
    def sample(self, batch, sample_num, args, solver_behavior_model=None):

        # Encode condition
        encoded_condition_batch = self.condition_encoder(batch)

        # Attach solver behavior condition embedding (for guidance)
        if solver_behavior_model is not None:
            solver_behavior_encoded_condition_batch = solver_behavior_model.solver_behavior_condition_encoder(
                batch=encoded_condition_batch
            )
            solver_behavior_encoded_condition_y = solver_behavior_encoded_condition_batch['solver_behavior_encoded_condition_y']
            encoded_condition_batch['solver_behavior_encoded_condition_y'] = solver_behavior_encoded_condition_y

        # sample_results: (B, S, C, H, W) or (B, S, C, H)
        cond_scale = args.cond_scale
        sample_results = self.diffusion_model.sample(
            batch=encoded_condition_batch,
            sample_num=sample_num,
            cond_scale=cond_scale,
            args=args,
            solver_behavior_model=solver_behavior_model,
        )
        
        return sample_results