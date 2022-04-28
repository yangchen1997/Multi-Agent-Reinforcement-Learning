import os

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import MultivariateNormal

from networks.grid_net_actor import AutoEncoderContinuousActions
from networks.grid_net_critic import StateValueModel
from policy.base_policy import BasePolicy
from utils.config_utils import ConfigObjectFactory
from utils.train_utils import weight_init


class GridWiseControlPPO(BasePolicy):

    def __init__(self, env_info: dict):
        # 读取配置
        self.train_config = ConfigObjectFactory.get_train_config()
        self.env_config = ConfigObjectFactory.get_environment_config()
        self.action_dim = env_info['action_dim']
        self.grid_input_shape = env_info['grid_input_shape']
        # 初始化网络
        self.auto_encoder = AutoEncoderContinuousActions(self.grid_input_shape)
        self.state_value_network = StateValueModel(self.grid_input_shape)
        self.optimizer_actor = torch.optim.Adam(params=self.auto_encoder.parameters(),
                                                lr=self.train_config.lr_actor)
        self.optimizer_critic = torch.optim.Adam(params=self.state_value_network.parameters(),
                                                 lr=self.train_config.lr_critic)

        # 初始化路径
        self.model_path = os.path.join(self.train_config.model_dir, self.env_config.learn_policy)
        self.result_path = os.path.join(self.train_config.result_dir, self.env_config.learn_policy)
        self.init_path(self.model_path, self.result_path)
        self.state_value_network_path = os.path.join(self.model_path, "grid_wise_control_ppo_state_value.pth")
        self.auto_encoder_path = os.path.join(self.model_path, "grid_wise_control_ppo_auto_encoder.pth")

        # 是否使用GPU加速
        if self.train_config.cuda:
            torch.cuda.empty_cache()
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')
        self.auto_encoder.to(self.device)
        self.state_value_network.to(self.device)
        self.init_wight()

        # 初始化动作的协方差矩阵，以便动作取样
        self.cov_var = torch.full(size=(self.action_dim,), fill_value=0.1)
        self.cov_mat = torch.diag(self.cov_var).to(self.device)

    def init_wight(self):
        self.auto_encoder.apply(weight_init)
        self.state_value_network.apply(weight_init)

    def learn(self, batch_data: dict, episode_num: int):
        # 从batch data中提取数据
        grid_inputs = batch_data['grid_inputs'].to(self.device)
        unit_pos = batch_data['unit_pos'].to(self.device)
        actions = batch_data['actions'].to(self.device)
        log_probs = batch_data['log_probs'].to(self.device)
        rewards = batch_data['rewards']
        discount_reward = self.get_discount_reward(rewards).to(self.device)
        # 计算状态价值
        with torch.no_grad():
            action_map, encoder_output = self.auto_encoder(grid_inputs)
            state_value = self.state_value_network(encoder_output)
            advantage_function = discount_reward - state_value
            # 标准化advantage_function减少环境不确定带来的影响
            advantage_function = ((advantage_function - advantage_function.mean()) / (
                    advantage_function.std() + 1e-10)).unsqueeze(dim=-1)

        # 开始学习，重度重采样。
        for i in range(self.train_config.learn_num):
            curr_action_map, curr_encoder_output = self.auto_encoder(grid_inputs)
            curr_log_probs = self.get_action_log_probs(curr_action_map, actions, unit_pos)
            # 计算loss
            ratios = torch.exp(curr_log_probs - log_probs)
            surr1 = ratios * advantage_function
            surr2 = torch.clamp(ratios, 1 - self.train_config.ppo_loss_clip,
                                1 + self.train_config.ppo_loss_clip) * advantage_function

            # actor_loss:取两个函数的最小值
            actor_loss = (-torch.min(surr1, surr2)).mean()
            self.optimizer_actor.zero_grad()
            actor_loss.backward()
            self.optimizer_actor.step()

            # critic_loss: td_error的均方误差
            curr_state_value = self.state_value_network(curr_encoder_output)
            critic_loss = nn.MSELoss()(curr_state_value, discount_reward)
            self.optimizer_critic.zero_grad()
            critic_loss.backward()
            self.optimizer_critic.step()

    def get_action_log_probs(self, action_map: Tensor, actions: Tensor, unit_pos: Tensor) -> Tensor:
        action_means = []
        for i, agents_pos in enumerate(unit_pos):
            one_step_mean = []
            for agent_num, pos in enumerate(agents_pos):
                x = int(pos[0])
                y = int(pos[1])
                action_mean = action_map[i, :, y, x]
                one_step_mean.append(action_mean)
            action_means.append(torch.stack(one_step_mean, dim=0))
        action_means = torch.stack(action_means, dim=0)
        dist = MultivariateNormal(action_means, self.cov_mat)
        log_probs = dist.log_prob(actions)
        return log_probs

    def get_discount_reward(self, batch_reward: list) -> Tensor:
        discount_rewards = []
        for reward in reversed(batch_reward):
            discounted_reward = 0
            for one_reward in reversed(reward):
                discounted_reward = one_reward + discounted_reward * self.train_config.gamma
                discount_rewards.insert(0, discounted_reward)
        return torch.Tensor(discount_rewards)

    def get_cov_mat(self) -> Tensor:
        return self.cov_mat

    def get_action_map(self, grid_input: Tensor) -> Tensor:
        with torch.no_grad():
            action_map, _ = self.auto_encoder(grid_input)
        return action_map

    def save_model(self):
        torch.save(self.auto_encoder.state_dict(), self.auto_encoder_path)
        torch.save(self.state_value_network.state_dict(), self.state_value_network_path)

    def load_model(self):
        self.auto_encoder.load_state_dict(torch.load(self.auto_encoder_path))
        self.state_value_network.load_state_dict(torch.load(self.state_value_network_path))

    def del_model(self):
        file_list = os.listdir(self.model_path)
        for file in file_list:
            os.remove(os.path.join(self.model_path, file))

    def is_saved_model(self) -> bool:
        return os.path.exists(self.auto_encoder_path) and os.path.exists(self.state_value_network_path)
