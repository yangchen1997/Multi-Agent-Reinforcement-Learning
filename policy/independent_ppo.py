import os

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import MultivariateNormal

from networks.ppo_net import IndependentPPOActor, IndependentPPOCritic
from policy.base_policy import BasePolicy
from utils.config_utils import ConfigObjectFactory
from utils.train_utils import weight_init


class IndependentPPO(BasePolicy):
    def __init__(self, env_info: dict):
        # 读取配置
        self.train_config = ConfigObjectFactory.get_train_config()
        self.env_config = ConfigObjectFactory.get_environment_config()
        self.n_agents = env_info['n_agents']
        self.action_dim = env_info['action_dim']

        # 初始化网络
        self.rnn_hidden_dim = 64
        self.ppo_actor = IndependentPPOActor(env_info['obs_space'], self.action_dim, self.rnn_hidden_dim)
        self.ppo_critic = IndependentPPOCritic(env_info['obs_space'])
        self.optimizer_actor = torch.optim.Adam(params=self.ppo_actor.parameters(),
                                                lr=self.train_config.lr_actor)
        self.optimizer_critic = torch.optim.Adam(params=self.ppo_critic.parameters(),
                                                 lr=self.train_config.lr_critic)

        # 初始化路径
        self.model_path = os.path.join(self.train_config.model_dir, self.env_config.learn_policy)
        self.result_path = os.path.join(self.train_config.result_dir, self.env_config.learn_policy)
        self.init_path(self.model_path, self.result_path)
        self.ppo_actor_path = os.path.join(self.model_path, "ppo_actor.pth")
        self.ppo_critic_path = os.path.join(self.model_path, "ppo_critic.pth")

        # 是否使用GPU加速
        if self.train_config.cuda:
            torch.cuda.empty_cache()
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')

        self.ppo_actor.to(self.device)
        self.ppo_critic.to(self.device)

        # 初始化动作的协方差矩阵，以便动作取样
        self.cov_var = torch.full(size=(self.action_dim,), fill_value=0.05)
        self.cov_mat = torch.diag(self.cov_var).to(self.device)
        self.init_wight()

    def init_wight(self):
        self.ppo_actor.apply(weight_init)
        self.ppo_critic.apply(weight_init)

    def init_hidden(self, batch_size: int):
        # 把 hidden做成一个字典，不用tenseor保存
        self.rnn_hidden = {}
        for i in range(self.n_agents):
            self.rnn_hidden[i] = torch.zeros((batch_size, self.rnn_hidden_dim)).to(self.device)

    def learn(self, batch_data: dict, episode_num: int):
        # 从batch data中提取数据
        obs = batch_data['obs'].to(self.device).detach()
        actions = batch_data['actions'].to(self.device)
        log_probs = batch_data['log_probs'].to(self.device)
        batch_size = sum(batch_data['per_episode_len'])
        rewards = batch_data['rewards']
        discount_reward = self.get_discount_reward(rewards).to(self.device)
        self.init_hidden(batch_size)
        # 计算状态价值和优势函数
        with torch.no_grad():
            state_values = []
            for i in range(self.n_agents):
                one_state_value = self.ppo_critic(obs[:, i])
                state_values.append(one_state_value)
            state_values = torch.stack(state_values, dim=0)
            advantage_function = discount_reward - state_values
            # 标准化advantage_function减少环境不确定带来的影响
            advantage_function = ((advantage_function - advantage_function.mean()) / (
                    advantage_function.std() + 1e-10)).unsqueeze(dim=-1)
        # 开始学习，重度重采样。
        for i in range(self.train_config.learn_num):
            curr_log_probs = []
            curr_state_values = []
            for agent_num in range(self.n_agents):
                one_action_mean, self.rnn_hidden[i] = self.ppo_actor(obs[:, i], self.rnn_hidden[i])
                curr_state_value = self.ppo_critic(obs[:, i])
                dist = MultivariateNormal(one_action_mean, self.cov_mat)
                curr_log_prob = dist.log_prob(actions[:, i])
                curr_log_probs.append(curr_log_prob)
                curr_state_values.append(curr_state_value)
            curr_log_probs = torch.stack(curr_log_probs, dim=1)
            curr_state_values = torch.stack(curr_state_values, dim=0)
            # 计算loss
            ratios = torch.exp(curr_log_probs - log_probs)
            surr1 = ratios * advantage_function
            surr2 = torch.clamp(ratios, 1 - self.train_config.ppo_loss_clip,
                                1 + self.train_config.ppo_loss_clip) * advantage_function
            actor_loss = (-torch.min(surr1, surr2)).mean()

            # actor_loss:取两个函数的最小值
            self.optimizer_actor.zero_grad()
            actor_loss.backward()
            self.optimizer_actor.step()

            # critic_loss: td_error的均方误差
            critic_loss = nn.MSELoss()(curr_state_values, discount_reward)
            self.optimizer_critic.zero_grad()
            critic_loss.backward()
            self.optimizer_critic.step()

    def get_cov_mat(self) -> Tensor:
        return self.cov_mat

    def get_discount_reward(self, batch_reward: list) -> Tensor:
        discount_rewards = []
        for reward in reversed(batch_reward):
            discounted_reward = 0
            for one_reward in reversed(reward):
                discounted_reward = one_reward + discounted_reward * self.train_config.gamma
                discount_rewards.insert(0, discounted_reward)
        return torch.Tensor(discount_rewards)

    def save_model(self):
        torch.save(self.ppo_actor.state_dict(), self.ppo_actor_path)
        torch.save(self.ppo_critic.state_dict(), self.ppo_critic_path)

    def load_model(self):
        self.ppo_actor.load_state_dict(torch.load(self.ppo_actor_path))
        self.ppo_critic.load_state_dict(torch.load(self.ppo_critic_path))

    def del_model(self):
        file_list = os.listdir(self.model_path)
        for file in file_list:
            os.remove(os.path.join(self.model_path, file))

    def is_saved_model(self) -> bool:
        return os.path.exists(self.ppo_actor_path) and os.path.exists(self.ppo_critic_path)
