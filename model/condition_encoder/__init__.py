from model.condition_encoder.condition_encoder import ConditionEncoder
from model.condition_encoder.solver_behavior_condition_encoder import SolverBehaviorConditionEncoder

def build_condition_encoder(model_params):

    if model_params['condition_encoder_type'] == 'MLP':
        return ConditionEncoder(model_params)
    else:
        raise ValueError(f"Invalid condition encoder type: {model_params['condition_encoder_type']}")

def build_solver_behavior_condition_encoder(model_params):
    return SolverBehaviorConditionEncoder(model_params)