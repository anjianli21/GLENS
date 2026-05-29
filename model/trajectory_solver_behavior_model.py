import os
import torch
import torch.nn as nn

from .condition_encoder import build_solver_behavior_condition_encoder
from .solver_behavior_model import build_solver_behavior_model

class TrajectorySolverBehaviorModel(nn.Module):

    def __init__(self, model_params):
        super(TrajectorySolverBehaviorModel, self).__init__()

        self.model_params = model_params

        self.solver_behavior_condition_encoder = build_solver_behavior_condition_encoder(model_params=self.model_params)

        self.solver_behavior_model = build_solver_behavior_model(model_params=self.model_params)

    def forward(self, batch, curr_epoch_num=None):

        # Encode condition
        encoded_condition_batch = self.solver_behavior_condition_encoder(batch)

        # Solver behavior model loss
        loss = self.solver_behavior_model(batch=encoded_condition_batch)

        return loss.mean()