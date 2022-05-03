from abc import ABC

import torch
import torch.nn as nn


class CentralizedPPOActor(nn.Module, ABC):
    """
        ppo算法属于on_policy算法，使用rnn网络来记录之前的经验。
        centralized_ppo 属于 centralized算法，需要通过搜集所有agent的观测值来给出动作
    """

    def __init__(self, input_shape: int, action_dim: int, n_agents: int, rnn_hidden_dim: int):
        super(CentralizedPPOActor, self).__init__()
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
        # 将动作空间映射到[0,1]
        fc2_out = torch.sigmoid(self.fc2(rnn_out))
        fc2_out = fc2_out.view(-1, self.n_agents, self.action_dim)
        return fc2_out, rnn_out


class CentralizedPPOCritic(nn.Module, ABC):
    """
        centralized_ppo的Critic将全局的state作为输入
    """

    def __init__(self, state_dim: int):
        super(CentralizedPPOCritic, self).__init__()
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


class IndependentPPOActor(nn.Module, ABC):
    """
        ppo算法属于on_policy算法，使用rnn网络来记录之前的经验。
        independent_ppo 属于 independent算法，只需要收集单个agent的观测值就能给出动作
    """

    def __init__(self, obs_dim: int, action_dim: int, rnn_hidden_dim: int):
        super(IndependentPPOActor, self).__init__()
        self.rnn_hidden_dim = rnn_hidden_dim
        self.action_dim = action_dim
        self.fc1 = nn.Linear(obs_dim, self.rnn_hidden_dim)
        self.rnn = nn.GRUCell(self.rnn_hidden_dim, self.rnn_hidden_dim)
        self.fc2 = nn.Linear(self.rnn_hidden_dim, self.action_dim)

    def forward(self, obs, hidden_state):
        fc1_out = torch.relu(self.fc1(obs))
        rnn_out = self.rnn(fc1_out, hidden_state)
        # 将动作空间映射到[0,1]
        fc2_out = torch.sigmoid(self.fc2(rnn_out))
        return fc2_out, rnn_out


class IndependentPPOCritic(nn.Module, ABC):
    """
        centralized_ppo的Critic将每个agent的obs作为输入
    """

    def __init__(self, obs_dim: int):
        super(IndependentPPOCritic, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features=obs_dim, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=1),
            nn.ReLU()
        )

    def forward(self, state):
        result = self.fc(state).squeeze()
        return result
