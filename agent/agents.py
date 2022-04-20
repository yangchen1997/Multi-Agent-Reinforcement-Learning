import random
import numpy as np
import torch
from torch import Tensor
from torch.distributions import Categorical

from policy.grid_wise_control import GridWiseControl
from policy.grid_wise_control_ddpg import GridWiseControlDDPG
from policy.qmix import QMix
from utils.config_utils import ConfigObjectFactory


class MyAgents:
    def __init__(self, env_info: dict):
        self.env_info = env_info
        self.train_config = ConfigObjectFactory.get_train_config()
        self.env_config = ConfigObjectFactory.get_environment_config()
        self.n_agents = self.env_info['n_agents']

        if self.train_config.cuda:
            torch.cuda.empty_cache()
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')

        if self.env_config.learn_policy == "grid_wise_control":
            self.n_actions = self.env_info['n_actions']
            self.policy = GridWiseControl(self.env_info)
        elif self.env_config.learn_policy == "grid_wise_control+ddpg":
            self.action_space = self.env_info['action_space']
            self.policy = GridWiseControlDDPG(self.env_info)
        elif self.env_config.learn_policy == "qmix":
            self.n_actions = self.env_info['n_actions']
            self.policy = QMix(self.env_info)
        else:
            raise ValueError("learn_policy error, just support grid_wise_control, grid_wise_control+ddpg, qmix")

    def learn(self, batch_data: dict, episode_num: int):
        self.policy.learn(batch_data, episode_num)

    def choose_actions_in_grid(self, unit_pos: list, grid_input: Tensor) -> tuple:
        actions_with_name = {}
        actions = []
        action_map = None
        if self.train_config.cuda:
            grid_input = grid_input.to(self.device)
        if isinstance(self.policy, GridWiseControl) or isinstance(self.policy, GridWiseControlDDPG):
            action_map = self.policy.get_action_map(grid_input)
        for agent_name, pos in zip(self.env_info['agents_name'], unit_pos):
            pos_x = pos[0]
            pos_y = pos[1]
            action_prop = action_map[0, :, pos_y, pos_x]
            if self.env_config.learn_policy == "grid_wise_control":
                action = Categorical(action_prop).sample().int()
                actions_with_name[agent_name] = (int(action))
                actions.append(int(action))
            else:
                action_with_noise = np.clip(
                    np.random.normal(action_prop.cpu().numpy(), self.train_config.var), self.action_space.low,
                    self.action_space.high).astype(dtype=np.float32)
                actions_with_name[agent_name] = action_with_noise
                actions.append(action_with_noise)
        return actions_with_name, actions

    def choose_actions(self, obs: dict) -> tuple:
        actions_with_name = {}
        actions = []
        obs = torch.stack([torch.Tensor(value) for value in obs.values()], dim=0)
        self.policy.init_hidden(1)
        actions_ind = [i for i in range(self.n_actions)]
        for i, agent in enumerate(self.env_info['agents_name']):
            inputs = list()
            inputs.append(obs[i, :])
            inputs.append(torch.zeros(self.n_actions))
            agent_id = torch.zeros(self.n_agents)
            agent_id[i] = 1
            inputs.append(agent_id)
            inputs = torch.cat(inputs).unsqueeze(dim=0).to(self.device)
            hidden_state = self.policy.eval_hidden[:, i, :]
            q_value, _ = self.policy.rnn_eval(inputs, hidden_state)
            if random.uniform(0, 1) > self.train_config.epsilon:
                action = random.sample(actions_ind, 1)[0]
            else:
                action = int(torch.argmax(q_value.squeeze()))
            actions_with_name[agent] = action
            actions.append(action)
        return actions_with_name, actions

    def save_model(self):
        self.policy.save_model()

    def load_model(self):
        self.policy.load_model()

    def del_model(self):
        self.policy.del_model()

    def is_saved_model(self) -> bool:
        return self.policy.is_saved_model()

    def get_results_path(self):
        return self.policy.result_path