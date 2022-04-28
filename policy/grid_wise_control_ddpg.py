import os

import torch
from torch import Tensor

from networks.grid_net_actor import AutoEncoderContinuousActions
from networks.grid_net_critic import QValueModelDDPG
from policy.base_policy import BasePolicy
from utils.config_utils import ConfigObjectFactory
from utils.train_utils import weight_init


class GridWiseControlDDPG(BasePolicy):
    def __init__(self, env_info: dict):
        # 读取配置
        self.train_config = ConfigObjectFactory.get_train_config()
        self.env_config = ConfigObjectFactory.get_environment_config()
        self.n_agents = env_info["n_agents"]
        self.action_dim = env_info["action_dim"]
        # 初始化各种网络
        self.auto_encoder_eval = AutoEncoderContinuousActions(env_info["grid_input_shape"])
        self.auto_encoder_target = AutoEncoderContinuousActions(env_info["grid_input_shape"]).requires_grad_(False)
        self.q_value_network_eval = QValueModelDDPG(env_info["grid_input_shape"], self.n_agents,
                                                    self.action_dim)
        self.q_value_network_target = QValueModelDDPG(env_info["grid_input_shape"],
                                                      self.n_agents, self.action_dim).requires_grad_(False)
        self.optimizer_actor = torch.optim.RMSprop(params=self.auto_encoder_eval.parameters(),
                                                   lr=self.train_config.lr_actor)
        self.optimizer_critic = torch.optim.RMSprop(params=self.q_value_network_eval.parameters(),
                                                    lr=self.train_config.lr_critic)
        # 初始化路径
        self.model_path = os.path.join(self.train_config.model_dir, self.env_config.learn_policy)
        self.result_path = os.path.join(self.train_config.result_dir, self.env_config.learn_policy)
        self.init_path(self.model_path, self.result_path)
        self.q_value_network_eval_path = os.path.join(self.model_path,
                                                      "grid_wise_control_ddpg_q_value_eval.pth")
        self.q_value_network_target_path = os.path.join(self.model_path,
                                                        "grid_wise_control_ddpg_q_value_target.pth")
        self.auto_encoder_eval_path = os.path.join(self.model_path, "grid_wise_control_ddpg_auto_encoder_eval.pth")
        self.auto_encoder_target_path = os.path.join(self.model_path, "grid_wise_control_ddpg_auto_encoder_target.pth")

        # 是否使用GPU加速
        if self.train_config.cuda:
            torch.cuda.empty_cache()
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')
        self.auto_encoder_eval.to(self.device)
        self.auto_encoder_target.to(self.device)
        self.q_value_network_eval.to(self.device)
        self.q_value_network_target.to(self.device)
        self.init_wight()

    def init_wight(self):
        self.auto_encoder_eval.apply(weight_init)
        self.auto_encoder_target.apply(weight_init)
        self.q_value_network_eval.apply(weight_init)
        self.q_value_network_target.apply(weight_init)

    def learn(self, batch_data: dict, episode_num: int):
        grid_inputs = batch_data['grid_inputs'].to(self.device)
        grid_inputs_next = batch_data['grid_inputs_next'].to(self.device)
        unit_pos = batch_data['unit_pos'].to(self.device)
        reward = batch_data['reward'].to(self.device)
        actions = batch_data['actions'].to(self.device).squeeze()
        terminated = batch_data['terminated'].to(self.device)
        q_eval = []
        q_target = []
        for i in range(batch_data['max_step']):
            one_grid_input = grid_inputs[:, i]
            one_unit_pos = unit_pos[:, i]
            one_action_map, one_encoder_out = self.auto_encoder_eval(one_grid_input)
            one_actions_output = self.get_actions_output(one_action_map, one_unit_pos).to(self.device)
            one_q_eval = self.q_value_network_eval(one_encoder_out, one_actions_output)
            with torch.no_grad():
                one_grid_input_next = grid_inputs_next[:, i]
                one_action_map_next, one_encoder_out_next = self.auto_encoder_target(one_grid_input_next)
                one_actions_output_next = self.get_actions_output(one_action_map_next, one_unit_pos).to(self.device)
                one_q_target = self.q_value_network_target(one_encoder_out_next, one_actions_output_next)

            q_eval.append(one_q_eval)
            q_target.append(one_q_target)
        # 获取动作价值
        q_eval = torch.stack(q_eval, dim=1).squeeze()
        q_target = torch.stack(q_target, dim=1).squeeze().detach()
        # 计算td-error，再除去填充的部分
        targets = reward + self.train_config.gamma * q_target
        td_error = (q_eval - targets.detach())
        masked_td_error = td_error * terminated
        # 优化critic
        loss_critic = (masked_td_error ** 2).sum() / terminated.sum()
        self.optimizer_critic.zero_grad()
        loss_critic.backward()
        # 梯度截断
        torch.nn.utils.clip_grad_norm_(list(self.q_value_network_eval.parameters()),
                                       self.train_config.grad_norm_clip)
        self.optimizer_critic.step()

        # 获取actor的q值
        q_value = []
        for i in range(batch_data['max_step']):
            one_grid_input = grid_inputs[:, i]
            one_action = actions[:, i]
            _, one_encoder_out = self.auto_encoder_eval(one_grid_input)
            one_q_value = self.q_value_network_eval(one_encoder_out, one_action)
            q_value.append(one_q_value)
        q_value = torch.stack(q_value, dim=1).squeeze()

        # 优化actor
        loss_actor = - (q_value * terminated).sum() / terminated.sum()
        self.optimizer_actor.zero_grad()
        loss_actor.backward()
        self.optimizer_actor.step()
        # 参数截断, 防止梯度爆炸
        for parm in self.auto_encoder_eval.parameters():
            parm.data.clamp_(-10, 10)
        # 到一定回合数时，target加载eval的最新网络参数
        if episode_num > 0 and episode_num % self.train_config.target_update_cycle == 0:
            self.auto_encoder_target.load_state_dict(self.auto_encoder_eval.state_dict())
            self.q_value_network_target.load_state_dict(self.q_value_network_eval.state_dict())

    def get_action_map(self, grid_input: Tensor) -> Tensor:
        with torch.no_grad():
            action_map, _ = self.auto_encoder_eval(grid_input)
        return action_map

    @staticmethod
    def get_actions_output(action_map: Tensor, unit_pos: Tensor) -> Tensor:
        actions_output = []
        for batch_num, pos in enumerate(unit_pos):
            batch_actions_output = []
            for agent_num, one_agent_pos in enumerate(pos):
                batch_actions_output.append(action_map[batch_num, :, int(one_agent_pos[1]), int(one_agent_pos[0])])
            actions_output.append(torch.stack(batch_actions_output, dim=0))
        actions_outputs = torch.stack(actions_output, dim=0)
        return actions_outputs

    def save_model(self):
        torch.save(self.q_value_network_eval.state_dict(), self.q_value_network_eval_path)
        torch.save(self.q_value_network_target.state_dict(), self.q_value_network_target_path)
        torch.save(self.auto_encoder_eval.state_dict(), self.auto_encoder_eval_path)
        torch.save(self.auto_encoder_target.state_dict(), self.auto_encoder_target_path)

    def load_model(self):
        self.q_value_network_eval.load_state_dict(torch.load(self.q_value_network_eval_path))
        self.q_value_network_target.load_state_dict(torch.load(self.q_value_network_target_path))
        self.auto_encoder_eval.load_state_dict(torch.load(self.auto_encoder_eval_path))
        self.auto_encoder_target.load_state_dict(torch.load(self.auto_encoder_target_path))

    def del_model(self):
        file_list = os.listdir(self.model_path)
        for file in file_list:
            os.remove(os.path.join(self.model_path, file))

    def is_saved_model(self) -> bool:
        return os.path.exists(self.auto_encoder_eval_path) and os.path.exists(
            self.auto_encoder_target_path) and os.path.exists(self.q_value_network_eval_path) and os.path.exists(
            self.q_value_network_target_path)
