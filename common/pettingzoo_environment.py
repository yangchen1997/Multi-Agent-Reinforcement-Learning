import matplotlib.pyplot as plt
import numpy as np
import torch
from pettingzoo.mpe import simple_spread_v3
from pettingzoo.utils import aec_to_parallel
from torch import Tensor

from utils.env_utils import *


class SimpleSpreadEnv(object):
    def __init__(self):
        self.env_config = ConfigObjectFactory.get_environment_config()
        self.continuous_actions = "ddpg" in self.env_config.learn_policy or "ppo" in self.env_config.learn_policy
        self.env = simple_spread_v3.env(N=self.env_config.n_agents, local_ratio=0.5,
                                        max_cycles=self.env_config.max_cycles,
                                        continuous_actions=self.continuous_actions)
        self.parallel_env = aec_to_parallel(self.env)
        self.world = self.parallel_env.unwrapped.world
        self.agents_name = self.parallel_env.possible_agents
        self.agents = self.world.agents
        self.grid_input_features = ["density", "mass", "size"]
        self.grid_size = self.env_config.grid_size

    def render(self, mode="human"):
        return self.parallel_env.render(mode=mode)

    def reset(self):
        return self.parallel_env.reset(seed=self.env_config.seed)

    def close(self):
        return self.parallel_env.close()

    def state(self):
        return self.parallel_env.state()

    def get_env_info(self) -> dict:
        map_info = {
            'grid_input_shape': [0, len(self.grid_input_features), self.grid_size, self.grid_size],
            'n_agents': self.env_config.n_agents,
            'agents_name': self.agents_name,
            'obs_space': sum(self.parallel_env.observation_spaces[self.agents_name[0]].shape),
            'state_space': sum(self.parallel_env.state_space.shape)
        }
        if not self.continuous_actions:
            map_info['n_actions'] = 5
        else:
            map_info['action_dim'] = 5
            map_info['action_space'] = self.parallel_env.action_space(self.agents_name[0])
        return map_info

    def step(self, actions: dict):
        # self.env.render()
        # time.sleep(0.05)
        observations, rewards, dones, _, infos = self.parallel_env.step(actions)
        # 所有agent都结束游戏整局游戏才算结束
        finish_game = not (False in dones.values())
        rewards = sum(rewards.values()) / self.env_config.n_agents
        return observations, rewards, finish_game, infos

    def get_agents_approximate_pos(self) -> list:
        """
             初始化grid_input,先将坐标映射，然后再离散化取整
             如果当前格子存在agent，将后来的智能体进行移位
            :return:
        """
        position_dict = {}
        approximate_poses = []
        for agent in self.agents:
            if agent.movable:
                map_pos = map_value(agent.state.p_pos)
                approximate_pos = get_approximate_pos(map_pos)
                # 当前近似坐标已存在重新计算近似坐标
                if approximate_pos in position_dict.keys():
                    approximate_pos = recomput_approximate_pos(position_dict, approximate_pos, map_pos)
                position_dict[approximate_pos] = map_pos
                approximate_poses.append(approximate_pos)
        return approximate_poses

    def get_grid_input(self) -> Tensor:
        """
            根据当前状态初始化grid_input
            :return:
        """
        approximate_pos_list = self.get_agents_approximate_pos()
        grid_input = np.zeros((1, len(self.grid_input_features), self.grid_size, self.grid_size),
                              dtype=np.float32)
        for agent_id, pos in enumerate(approximate_pos_list):
            agent = self.agents[agent_id]
            for feature_num, feature_name in enumerate(self.grid_input_features):
                grid_input[0, feature_num, pos[1], pos[0]] = getattr(agent, feature_name, float(0))
        tensor = torch.from_numpy(grid_input)
        return tensor

    def draw_maps(self):
        """
            画出每个agent的位置，以便进行验证
            :return:
        """
        agents = self.world.agents
        agent_x = []
        agent_y = []
        for agent in agents:
            agent_x.append(agent.state.p_pos[0])
            agent_y.append(agent.state.p_pos[1])
        plt.scatter(agent_x, agent_y, c='red')
        plt.title("real_pos")
        plt.xlim((-3, 3))
        plt.ylim((-3, 3))
        plt.show()

        approximate_pos_list = self.get_agents_approximate_pos()
        plt.scatter([pos[0] for pos in approximate_pos_list], [pos[1] for pos in approximate_pos_list], c='red')
        plt.title("map_pos")
        plt.xlim((0, self.env_config.grid_size - 1))
        plt.ylim((0, self.env_config.grid_size - 1))
        plt.show()
