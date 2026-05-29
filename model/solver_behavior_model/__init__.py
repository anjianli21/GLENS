from model.solver_behavior_model.solver_behavior_model import SolverBehaviorModel
from model.utils.unet_1d_without_time import Unet1DWithoutTime

def build_solver_behavior_model(model_params):

    if model_params['solver_behavior_unet_type'] == "1D":
        unet_model = Unet1DWithoutTime(
            dim=model_params['solver_behavior_unet_model_dim'],
            dim_mults=model_params['solver_behavior_unet_dim_mults'],
            embed_y_all_dim=model_params['solver_behavior_embed_y_all_dim'],
            channels=model_params['data_x_channel'],
            dropout=model_params['solver_behavior_dropout'],
            cond_drop_prob=0.0,
        )

        solver_behavior_model = SolverBehaviorModel(
            unet_model=unet_model,
            solver_behavior_model_params=model_params,
        )
    else:
        raise ValueError(f"Invalid unet type: {model_params['unet_type']}")

    return solver_behavior_model
