import torch
import torch.nn as nn

class SolverBehaviorConditionEncoder(nn.Module):
    def __init__(self, model_params):
        super().__init__()  # Initialize the parent class

        self.solver_behavior_encoder_data_type = model_params['solver_behavior_encoder_data_type']
        
        condition_y_parameter_dim = model_params['solver_behavior_condition_y_parameter_dim']
        embed_y_all_dim = model_params['solver_behavior_embed_y_all_dim']
        self.task_name = model_params['task_name']
        self.mlp_type = model_params['mlp_type']

        if self.mlp_type != 'silu_layer_no_output_norm':
            raise ValueError(
                f"Invalid mlp type: {self.mlp_type}. "
                "Only 'silu_layer_no_output_norm' is supported."
            )

        # Check if condition_encoder_data_type is valid
        valid_types = ['parameter']
        for data_type in self.solver_behavior_encoder_data_type:
            if data_type not in valid_types:
                raise ValueError(f"Invalid solver behavior encoder data type: {data_type}. Valid types: {valid_types}")
        
        self.encode_condition_y_mlp = nn.Sequential(
            nn.Linear(condition_y_parameter_dim, embed_y_all_dim),
            nn.SiLU(),
            nn.Linear(embed_y_all_dim, embed_y_all_dim)
        )

    def forward(self, batch):

        # Process parameter features if parameter is in condition_encoder_data_type
        condition_y_parameter = batch["condition_y_parameter_concat"]

        # Encode the final condition with a single MLP
        encoded_condition_y = self.encode_condition_y_mlp(condition_y_parameter)

        # Add the encoded condition to the batch
        batch["solver_behavior_encoded_condition_y"] = encoded_condition_y

        return batch
        