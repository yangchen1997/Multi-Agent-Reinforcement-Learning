import os

import torch
from torch import Tensor

from networks.grid_net_actor import AutoEncoder
from networks.grid_net_critic import StateValueModel
from policy.base_policy import BasePolicy
from utils.config_utils import ConfigObjectFactory
from utils.train_utils import weight_init


class GridWiseControl(BasePolicy):

    def __init__(self, env_info: dict):
        self.train_config = ConfigObjectFactory.get_train_config()
        self.env_config = ConfigObjectFactory.get_environment_config()
        self.n_agents = env_info["n_agents"]
        self.n_actions = env_info["n_actions"]
        # 初始化模型可优化器
        self.auto_encoder = AutoEncoder(self.n_actions, env_info["grid_input_shape"])
        self.state_value_network_eval = StateValueModel(env_info["grid_input_shape"])
        self.state_value_network_target = StateValueModel(env_info["grid_input_shape"]).requires_grad_(False)
        self.optimizer_actor = torch.optim.RMSprop(params=self.auto_encoder.parameters(), lr=self.train_config.lr_actor)
        self.optimizer_critic = torch.optim.RMSprop(params=self.state_value_network_eval.parameters(),
                                                    lr=self.train_config.lr_critic)

        # 初始化路径
        self.model_path = os.path.join(self.train_config.model_dir, self.env_config.learn_policy)
        self.result_path = os.path.join(self.train_config.result_dir, self.env_config.learn_policy)
        self.init_path(self.model_path, self.result_path)
        self.state_value_network_eval_path = os.path.join(self.model_path, "grid_wise_control_state_value_eval.pth")
        self.state_value_network_target_path = os.path.join(self.model_path, "grid_wise_control_state_value_target.pth")
        self.auto_encoder_path = os.path.join(self.model_path, "grid_wise_control_auto_encoder.pth")

        # 是否使用GPU加速
        if self.train_config.cuda:
            torch.cuda.empty_cache()
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')
        self.auto_encoder.to(self.device)
        self.state_value_network_eval.to(self.device)
        self.state_value_network_target.to(self.device)
        self.init_wight()

    def init_wight(self):
        self.auto_encoder.apply(weight_init)
        self.state_value_network_eval.apply(weight_init)
        self.state_value_network_target.apply(weight_init)

    def learn(self, batch_data: dict, episode_num: int):
        grid_inputs = batch_data['grid_inputs'].to(self.device)
        grid_inputs_next = batch_data['grid_inputs_next'].to(self.device)
        unit_pos = batch_data['unit_pos'].to(self.device)
        reward = batch_data['reward'].to(self.device)
        actions = batch_data['actions'].long().to(self.device)
        terminated = batch_data['terminated'].to(self.device)
        mask = terminated.unsqueeze(dim=-1).repeat(1, 1, self.n_agents).to(self.device)

        state_values = []
        state_values_next = []
        for i in range(batch_data['max_step']):
            one_grid_input = grid_inputs[:, i]
            _, one_encoder_out = self.auto_encoder(one_grid_input)
            state_value = self.state_value_network_eval(one_encoder_out)
            with torch.no_grad():
                one_grid_input_next = grid_inputs_next[:, i]
                _, one_encoder_out_next = self.auto_encoder(one_grid_input_next)
                state_value_next = self.state_value_network_target(one_encoder_out_next)
            state_values.append(state_value)
            state_values_next.append(state_value_next)
        # 获取状态价值
        state_values = torch.stack(state_values, dim=1)
        state_values_next = torch.stack(state_values_next, dim=1)
        # 计算td-error，再除去填充的部分
        targets = reward + self.train_config.gamma * state_values_next
        td_error = (state_values - targets.detach())
        masked_td_error = td_error * terminated

        # 优化critic
        loss_critic = (masked_td_error ** 2).sum() / terminated.sum()
        self.optimizer_critic.zero_grad()
        loss_critic.backward()
        # 梯度截断
        torch.nn.utils.clip_grad_norm_(list(self.state_value_network_eval.parameters()),
                                       self.train_config.grad_norm_clip)
        self.optimizer_critic.step()

        actions_probes = []
        for i in range(batch_data['max_step']):
            one_grid_input = grid_inputs[:, i]
            one_unit_pos = unit_pos[:, i]
            one_action_map, _ = self.auto_encoder(one_grid_input)
            one_actions_prob = self.get_actions_prob(one_action_map, one_unit_pos).to(self.device)
            actions_probes.append(one_actions_prob)
        # 获取动作概率
        actions_probes = torch.stack(actions_probes, dim=1)
        # 取每个动作对应的概率
        pi_taken = torch.gather(actions_probes, dim=3, index=actions).squeeze()
        # 因为要取对数，对于那些填充的经验，所有概率都为0，取了log就是负无穷了，所以让它们变成1
        pi_taken[mask == 0] = 1.0
        log_pi_taken = torch.log(pi_taken)
        advantage = masked_td_error.detach().unsqueeze(dim=-1)
        # 优化actor
        loss_actor = - ((advantage * log_pi_taken) * mask).sum() / mask.sum()
        self.optimizer_actor.zero_grad()
        loss_actor.backward()
        self.optimizer_actor.step()
        # 参数截断, 防止梯度爆炸
        for parm in self.auto_encoder.parameters():
            parm.data.clamp_(-10, 10)
        # 到一定回合数时，target加载eval的最新网络参数
        if episode_num > 0 and episode_num % self.train_config.target_update_cycle == 0:
            self.state_value_network_target.load_state_dict(self.state_value_network_eval.state_dict())

    def get_action_map(self, grid_input: Tensor) -> Tensor:
        with torch.no_grad():
            action_map, _ = self.auto_encoder(grid_input)
        return action_map

    @staticmethod
    def get_actions_prob(action_map: Tensor, unit_pos: Tensor) -> Tensor:
        actions_prob = []
        for batch_num, pos in enumerate(unit_pos):
            batch_actions_prob = []
            for agent_num, one_agent_pos in enumerate(pos):
                batch_actions_prob.append(action_map[batch_num, :, int(one_agent_pos[1]), int(one_agent_pos[0])])
            actions_prob.append(torch.stack(batch_actions_prob, dim=0))
        actions_probes = torch.stack(actions_prob, dim=0)
        # 进行归一化
        actions_probes_final = actions_probes / actions_probes.sum(dim=-1, keepdim=True)
        return actions_probes_final

    def save_model(self):
        torch.save(self.state_value_network_eval.state_dict(), self.state_value_network_eval_path)
        torch.save(self.state_value_network_target.state_dict(), self.state_value_network_target_path)
        torch.save(self.auto_encoder.state_dict(), self.auto_encoder_path)

    def load_model(self):
        self.state_value_network_eval.load_state_dict(torch.load(self.state_value_network_eval_path))
        self.state_value_network_target.load_state_dict(torch.load(self.state_value_network_target_path))
        self.auto_encoder.load_state_dict(torch.load(self.auto_encoder_path))

    def del_model(self):
        file_list = os.listdir(self.model_path)
        for file in file_list:
            os.remove(os.path.join(self.model_path, file))

    def is_saved_model(self) -> bool:
        return os.path.exists(self.auto_encoder_path) and os.path.exists(
            self.state_value_network_eval_path) and os.path.exists(self.state_value_network_target_path)
