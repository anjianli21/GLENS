import torch
import torch.nn as nn
import torch.nn.functional as F

class SolverBehaviorModel(nn.Module):

    def __init__(self, unet_model, solver_behavior_model_params):
        super(SolverBehaviorModel, self).__init__()

        self.unet_model = unet_model
        self.solver_behavior_model_params = solver_behavior_model_params


    def forward(self, batch):

        # Prepare condition c
        c = batch['solver_behavior_encoded_condition_y']

        # prepare data x
        x = batch['data_x_concat']
        
        # Prepare output label
        if self.solver_behavior_model_params['solver_behavior_output_data_type'] == 'radius':
            label = batch['condition_y_radius_concat']
            raise ValueError(f"Radius is currently not supported for solver behavior model output")
        elif self.solver_behavior_model_params['solver_behavior_output_data_type'] == 'radius_square_grad':
            label = batch['condition_y_radius_square_grad_concat']
    
        else:
            raise ValueError(f"Invalid solver behavior model output data type: {self.solver_behavior_model_params['solver_behavior_output_data_type']}")

        loss = self.forward_loss(x=x,
                                 c=c,
                                 label=label)

        return loss

    def forward_loss(self, x, label, c):

        batch_size = x.shape[0]
        device = x.device

        output = self.unet_model(x=x, c=c)

        loss_type = self.solver_behavior_model_params['solver_behavior_loss_type']

        if loss_type == 'mse':
            loss = F.mse_loss(output, label)
        elif loss_type == 'huber':
            loss = F.smooth_l1_loss(output, label, beta=1.0)
        else:
            raise ValueError(f"Invalid solver behavior loss type: {loss_type}")

        return loss
    
    @torch.inference_mode()
    def get_classifier_guidance_gradient(self, x, c):
        if self.solver_behavior_model_params['solver_behavior_output_data_type'] == 'radius_square_grad':
            gradient = self.unet_model(x=x, c=c)
            if self.solver_behavior_model_params['radius_square_grad_data_process_type'] == 'original':
                pass
            elif self.solver_behavior_model_params['radius_square_grad_data_process_type'] == 'scale':
                gradient = gradient / self.solver_behavior_model_params['radius_square_grad_data_scale']
            else:
                raise ValueError(f"Invalid radius square grad data process type: {self.solver_behavior_model_params['radius_square_grad_data_process_type']}")
        else:
            raise ValueError(f"Invalid solver behavior model output data type: {self.solver_behavior_model_params['solver_behavior_output_data_type']}")

        return gradient