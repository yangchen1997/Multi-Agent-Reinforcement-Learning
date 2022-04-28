from abc import ABC

import torch
import torch.nn as nn


class PPOActor(nn.Module, ABC):
    """
        ppo算法属于on_policy算法，使用rnn网络来记录之前的经验。
        centralized_ppo 属于 centralized算法，需要通过搜集所有agent的观测值来给出动作
    """

    def __init__(self, input_shape: int, action_dim: int, n_agents: int, rnn_hidden_dim: int):
        super(PPOActor, self).__init__()
        self.rnn_hidden_dim = rnn_hidden_dim
        self.n_agents = n_agents
        self.action_dim = action_dim
        self.fc1 = nn.Linear(input_shape, self.rnn_hidden_dim)
        self.rnn = nn.GRUCell(self.rnn_hidden_dim, self.rnn_hidden_dim)
        self.fc2 = nn.Linear(self.rnn_hidden_dim, self.action_dim * self.n_agents)

    def forward(self, obs, hidden_state):
        fc1_out = torch.relu(self.fc1(obs))
        h_in = hidden_state.reshape(-1, self.rnn_hidden_dim)
        rnn_out = self.rnn(fc1_out, h_in)
        fc2_out = torch.sigmoid(self.fc2(rnn_out))
        fc2_out = fc2_out.view(-1, self.n_agents, self.action_dim)
        return fc2_out, rnn_out


class PPOCritic(nn.Module, ABC):
    def __init__(self, state_dim: int):
        super(PPOCritic, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features=state_dim, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=1),
            nn.ReLU()
        )

    def forward(self, state):
        result = self.fc(state).squeeze()
        return result
