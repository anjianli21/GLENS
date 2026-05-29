import torch
import torch.nn as nn

class ConditionEncoder(nn.Module):
    def __init__(self, model_params):
        super().__init__()  # Initialize the parent class

        self.condition_encoder_data_type = model_params['condition_encoder_data_type']
        self.parameter_data_process_type = model_params['parameter_data_process_type']
        self.condition_fusion_type = model_params['condition_fusion_type']
        self.radius_data_process_type = model_params['radius_data_process_type']
        self.radius_data_scale = model_params['radius_data_scale']

        self.task_name = model_params['task_name']
        self.mlp_type = model_params['mlp_type']
        
        condition_y_parameter_dim = model_params['condition_y_parameter_dim']
        condition_y_radius_dim = model_params['condition_y_radius_dim']
        embed_y_all_dim = model_params['embed_y_all_dim']

        if self.condition_fusion_type != 'raw_concat':
            raise ValueError(
                f"Invalid condition fusion type: {self.condition_fusion_type}. "
                "Only 'raw_concat' is supported."
            )
        if self.mlp_type != 'silu_layer_no_output_norm':
            raise ValueError(
                f"Invalid mlp type: {self.mlp_type}. "
                "Only 'silu_layer_no_output_norm' is supported."
            )

        # Check if condition_encoder_data_type is valid
        valid_types = ['parameter', 'radius']
        for data_type in self.condition_encoder_data_type:
            if data_type not in valid_types:
                raise ValueError(f"Invalid condition encoder data type: {data_type}. Valid types: {valid_types}")
        
        # Calculate input dimension for the condition encoder MLP.
        combination_input_dim = 0
        if 'parameter' in self.condition_encoder_data_type:
            combination_input_dim += condition_y_parameter_dim
        if 'radius' in self.condition_encoder_data_type:
            combination_input_dim += condition_y_radius_dim
        if combination_input_dim <= 0:
            raise ValueError(
                f"condition_encoder_data_type must include at least one of {valid_types}. "
                f"Got: {self.condition_encoder_data_type}"
            )

        # Encode the concatenated raw condition with a single MLP.
        self.encode_condition_y_mlp = nn.Sequential(
            nn.Linear(combination_input_dim, embed_y_all_dim),
            nn.SiLU(),
            nn.Linear(embed_y_all_dim, embed_y_all_dim)
        )

    def forward(self, batch):
        if self.condition_fusion_type != 'raw_concat':
            raise ValueError(
                f"Invalid condition fusion type: {self.condition_fusion_type}. "
                "Only 'raw_concat' is supported."
            )
        return self.forward_with_raw_concat_condition_encoders(batch)

    def forward_with_raw_concat_condition_encoders(self, batch):
        raw_condition_features = []

        # Process parameter features if parameter is in condition_encoder_data_type
        if 'parameter' in self.condition_encoder_data_type:
            condition_y_parameter = batch["condition_y_parameter_concat"]
            raw_condition_features.append(condition_y_parameter)

        # Process radius features if radius is in condition_encoder_data_type
        if 'radius' in self.condition_encoder_data_type:
            condition_y_radius = batch["condition_y_radius_concat"]
            raw_condition_features.append(condition_y_radius)
        
        # Concatenate the raw condition features to get the final condition
        condition_y_concat = torch.cat(raw_condition_features, dim=-1)

        # Encode the final condition with a single MLP
        encoded_condition_y = self.encode_condition_y_mlp(condition_y_concat)

        # Add the encoded condition to the batch
        batch["encoded_condition_y"] = encoded_condition_y

        return batch