import os

import torch
from torch import Tensor

from networks.qmix_net import RNN, QMixNet
from policy.base_policy import BasePolicy
from utils.config_utils import ConfigObjectFactory
from utils.train_utils import weight_init


class QMix(BasePolicy):

    def __init__(self, env_info: dict):
        self.train_config = ConfigObjectFactory.get_train_config()
        self.env_config = ConfigObjectFactory.get_environment_config()
        self.n_agents = env_info['n_agents']
        self.n_actions = env_info['n_actions']
        input_shape = env_info['obs_space'] + self.n_agents + self.n_actions
        state_space = env_info['state_space']

        # 神经网络
        self.rnn_hidden_dim = 64
        # 每个agent选动作的网络
        self.rnn_eval = RNN(input_shape, self.n_actions, self.rnn_hidden_dim)
        self.rnn_target = RNN(input_shape, self.n_actions, self.rnn_hidden_dim)
        # 把agentsQ值加起来的网络
        self.qmix_net_eval = QMixNet(self.n_agents, state_space)
        self.qmix_net_target = QMixNet(self.n_agents, state_space)
        self.init_wight()
        self.eval_parameters = list(self.qmix_net_eval.parameters()) + list(self.rnn_eval.parameters())
        self.optimizer = torch.optim.RMSprop(self.eval_parameters, lr=self.train_config.lr_critic)

        # 初始化路径
        self.model_path = os.path.join(self.train_config.model_dir, self.env_config.learn_policy)
        self.result_path = os.path.join(self.train_config.result_dir, self.env_config.learn_policy)
        self.init_path(self.model_path, self.result_path)
        self.rnn_eval_path = os.path.join(self.model_path, "rnn_eval.pth")
        self.rnn_target_path = os.path.join(self.model_path, "rnn_target.pth")
        self.qmix_net_eval_path = os.path.join(self.model_path, "qmix_net_eval.pth")
        self.qmix_net_target_path = os.path.join(self.model_path, "qmix_net_target.pth")

        # 是否使用GPU加速
        if self.train_config.cuda:
            torch.cuda.empty_cache()
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')
        self.rnn_eval.to(self.device)
        self.rnn_target.to(self.device)
        self.qmix_net_eval.to(self.device)
        self.qmix_net_target.to(self.device)

    def init_wight(self):
        self.rnn_eval.apply(weight_init)
        self.rnn_target.apply(weight_init)
        self.qmix_net_eval.apply(weight_init)
        self.qmix_net_target.apply(weight_init)

    def learn(self, batch_data: dict, episode_num: int):
        obs = batch_data['obs'].to(self.device)
        obs_next = batch_data['obs_next'].to(self.device)
        state = batch_data['state'].to(self.device)
        state_next = batch_data['state_next'].to(self.device)
        rewards = batch_data['rewards'].unsqueeze(dim=-1).to(self.device)
        actions = batch_data['actions'].long().to(self.device)
        actions_onehot = batch_data['actions_onehot'].to(self.device)
        terminated = batch_data['terminated'].unsqueeze(dim=-1).to(self.device)

        q_evals, q_targets = [], []
        batch_size = batch_data['sample_size']
        self.init_hidden(batch_size)
        for i in range(batch_data['max_step']):
            inputs, inputs_next = self._get_inputs(batch_size, i, obs[:, i], obs_next[:, i],
                                                   actions_onehot)
            q_eval, self.eval_hidden = self.rnn_eval(inputs, self.eval_hidden)
            q_target, self.target_hidden = self.rnn_target(inputs_next, self.target_hidden)
            # 将q值reshape, 以n_agents分开
            q_eval = q_eval.view(batch_size, self.n_agents, -1)
            q_target = q_target.view(batch_size, self.n_agents, -1)
            q_evals.append(q_eval)
            q_targets.append(q_target)
        # 将上面的到的q值聚合
        q_evals = torch.stack(q_evals, dim=1)
        q_targets = torch.stack(q_targets, dim=1)
        # 找出所选动作对应的q值
        q_evals = torch.gather(q_evals, dim=3, index=actions).squeeze(3)
        q_targets = q_targets.max(dim=3)[0]
        # 将q值和state输入mix网络
        q_total_eval = self.qmix_net_eval(q_evals, state)
        q_total_target = self.qmix_net_target(q_targets, state_next)

        targets = rewards + self.train_config.gamma * q_total_target * terminated
        td_error = (q_total_eval - targets.detach())
        # 抹掉填充的经验的td_error
        masked_td_error = terminated * td_error
        # 计算损失函数
        loss = (masked_td_error ** 2).sum() / terminated.sum()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.eval_parameters, self.train_config.grad_norm_clip)
        self.optimizer.step()
        if episode_num > 0 and episode_num % self.train_config.target_update_cycle == 0:
            self.rnn_target.load_state_dict(self.rnn_eval.state_dict())
            self.qmix_net_target.load_state_dict(self.qmix_net_eval.state_dict())

    def _get_inputs(self, batch_size: int, batch_index: int, obs: Tensor, obs_next: Tensor,
                    actions_onehot: Tensor) -> tuple:
        """
            获取q网络的输入值, 将动作放入obs中
            :return:
        """
        inputs, inputs_next = [], []
        inputs.append(obs)
        inputs_next.append(obs_next)
        if batch_index == 0:
            inputs.append(torch.zeros_like(actions_onehot[:, batch_index]))
        else:
            inputs.append(actions_onehot[:, batch_index - 1])
        inputs_next.append(actions_onehot[:, batch_index])
        inputs.append(torch.eye(self.n_agents).unsqueeze(0).expand(batch_size, -1, -1).to(self.device))
        inputs_next.append(torch.eye(self.n_agents).unsqueeze(0).expand(batch_size, -1, -1).to(self.device))
        inputs = torch.cat([x.reshape(batch_size * self.n_agents, -1) for x in inputs], dim=1)
        inputs_next = torch.cat([x.reshape(batch_size * self.n_agents, -1) for x in inputs_next], dim=1)
        return inputs, inputs_next

    def init_hidden(self, batch_size):
        # 为每个episode中的每个agent都初始化一个eval_hidden、target_hidden
        self.eval_hidden = torch.zeros((batch_size, self.n_agents, self.rnn_hidden_dim)).to(self.device)
        self.target_hidden = torch.zeros((batch_size, self.n_agents, self.rnn_hidden_dim)).to(self.device)

    def save_model(self):
        torch.save(self.rnn_eval.state_dict(), self.rnn_eval_path)
        torch.save(self.rnn_target.state_dict(), self.rnn_target_path)
        torch.save(self.qmix_net_eval.state_dict(), self.qmix_net_eval_path)
        torch.save(self.qmix_net_target.state_dict(), self.qmix_net_target_path)

    def load_model(self):
        self.rnn_eval.load_state_dict(torch.load(self.rnn_eval_path))
        self.rnn_target.load_state_dict(torch.load(self.rnn_target_path))
        self.qmix_net_eval.load_state_dict(torch.load(self.qmix_net_eval_path))
        self.qmix_net_target.load_state_dict(torch.load(self.qmix_net_target_path))

    def del_model(self):
        file_list = os.listdir(self.model_path)
        for file in file_list:
            os.remove(os.path.join(self.model_path, file))

    def is_saved_model(self) -> bool:
        return os.path.exists(self.rnn_eval_path) and os.path.exists(
            self.rnn_target_path) and os.path.exists(self.qmix_net_eval_path) and os.path.exists(
            self.qmix_net_target_path)
