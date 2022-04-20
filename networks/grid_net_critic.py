from abc import ABC

import torch.nn as nn
import torch.nn.functional as F


class StateValueModel(nn.Module, ABC):
    def __init__(self, grid_input_shape: list):
        super(StateValueModel, self).__init__()
        input_shape = grid_input_shape[2] * grid_input_shape[3] * 64
        self.fc = nn.Sequential(
            nn.Linear(in_features=input_shape, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=1),
            nn.ReLU()
        )

    def forward(self, state):
        result = self.fc(state).squeeze()
        return result


class QValueModelDDPG(nn.Module, ABC):
    def __init__(self, grid_input_shape: list, n_agent: int, action_dim: int):
        super(QValueModelDDPG, self).__init__()
        state_input_shape = grid_input_shape[2] * grid_input_shape[3] * 64
        self.actions_input_shape = n_agent * action_dim
        self.fc_state = nn.Sequential(
            nn.Linear(in_features=state_input_shape, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=128),
            nn.ReLU(),
        )
        self.fc_actions = nn.Sequential(
            nn.Linear(in_features=self.actions_input_shape, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=128),
            nn.ReLU(),
        )
        self.output_final = nn.Linear(128, 1)

    def forward(self, state, actions):
        state_output = self.fc_state(state)
        actions = actions.view(-1, self.actions_input_shape)
        action_output = self.fc_actions(actions)
        result = self.output_final(F.relu(state_output + action_output))
        return result
