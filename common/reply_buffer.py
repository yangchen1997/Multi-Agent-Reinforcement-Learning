import threading
from random import sample

import torch
from numpy import ndarray
from torch import Tensor

from utils.config_utils import *

LOCK_MEMORY = threading.Lock()


class OneEpisodeMemory(object):
    """
        每一局游戏的临时存储单元，
        一局游戏完毕后将临时数据放入Memory中，
        并清空数据。
    """

    def __init__(self):
        self.grid_inputs = []
        self.grid_inputs_next = []
        self.rewards = []
        self.unit_poses = []
        self.unit_actions = []
        self.n_step = 0

    def store_one_episode(self, grid_input: Tensor, grid_input_next: Tensor,
                          unit_pos: list, action: list, reward: float):
        self.grid_inputs.append(grid_input)
        self.grid_inputs_next.append(grid_input_next)
        self.unit_poses.append(unit_pos)
        self.unit_actions.append(action)
        self.rewards.append(reward)
        self.n_step += 1

    def clear_memories(self):
        self.grid_inputs.clear()
        self.grid_inputs_next.clear()
        self.unit_poses.clear()
        self.unit_actions.clear()
        self.rewards.clear()
        self.n_step = 0


class Memory(object):
    """
        记忆单元，保存了之前所有局游戏的数据
    """

    def __init__(self):
        self.train_config = ConfigObjectFactory.get_train_config()
        self.memory_size = self.train_config.memory_size
        self.current_idx = 0
        self.memory = []

    def store_episode(self, one_episode_memory: OneEpisodeMemory):
        with LOCK_MEMORY:
            grid_inputs = torch.cat(one_episode_memory.grid_inputs, dim=0)
            grid_inputs_next = torch.cat(one_episode_memory.grid_inputs_next, dim=0)
            unit_pos = torch.Tensor(one_episode_memory.unit_poses)
            actions = torch.Tensor(one_episode_memory.unit_actions)
            reward = torch.Tensor(one_episode_memory.rewards)
            data = {
                'grid_inputs': grid_inputs,
                'grid_inputs_next': grid_inputs_next,
                'unit_pos': unit_pos,
                'reward': reward,
                'actions': actions,
                'n_step': one_episode_memory.n_step
            }
            if len(self.memory) < self.memory_size:
                self.memory.append(data)
            else:
                self.memory[self.current_idx % self.memory_size] = data
            self.current_idx += 1

    def sample(self, batch_size) -> dict:
        """
            从记忆单元中随机抽样，但是每一局游戏的step不同，找出这个batch中
            最大的那个，将其他游戏的数据补齐
            :param batch_size: 一个batch的大小
            :return: 一个batch的数据
        """
        sample_size = min(len(self.memory), batch_size)
        sample_list = sample(self.memory, sample_size)
        n_step = torch.Tensor([one_data['n_step'] for one_data in sample_list])
        max_step = int(torch.max(n_step))

        grid_inputs = torch.stack(
            [torch.cat([one_data['grid_inputs'],
                        torch.zeros([max_step - one_data['grid_inputs'].shape[0]] +
                                    list(one_data['grid_inputs'].shape[1:]))
                        ])
             for one_data in sample_list], dim=0).detach()

        grid_inputs_next = torch.stack(
            [torch.cat([one_data['grid_inputs_next'],
                        torch.zeros(size=[max_step - one_data['grid_inputs_next'].shape[0]] +
                                         list(one_data['grid_inputs_next'].shape[1:]))])
             for one_data in sample_list], dim=0).detach()

        unit_pos = torch.stack(
            [torch.cat([one_data['unit_pos'],
                        torch.zeros([max_step - one_data['unit_pos'].shape[0]] +
                                    list(one_data['unit_pos'].shape[1:]))])
             for one_data in sample_list], dim=0).detach()

        reward = torch.stack(
            [torch.cat([one_data['reward'],
                        torch.zeros([max_step - one_data['reward'].shape[0]] +
                                    list(one_data['reward'].shape[1:]))])
             for one_data in sample_list], dim=0).detach()

        actions = torch.stack(
            [torch.cat([one_data['actions'],
                        torch.zeros([max_step - one_data['actions'].shape[0]] +
                                    list(one_data['actions'].shape[1:]))])
             for one_data in sample_list], dim=0).unsqueeze(dim=-1).detach()

        terminated = torch.stack(
            [torch.cat([torch.ones(one_data['n_step']), torch.zeros(max_step - one_data['n_step'])])
             for one_data in sample_list], dim=0).detach()

        batch_data = {
            'grid_inputs': grid_inputs,
            'grid_inputs_next': grid_inputs_next,
            'unit_pos': unit_pos,
            'reward': reward,
            'actions': actions,
            'max_step': max_step,
            'sample_size': sample_size,
            'terminated': terminated
        }
        return batch_data

    def get_memory_real_size(self):
        return len(self.memory)


class CommOneEpisodeMemory(object):
    """
        存储每局游戏的记忆单元, 适用于常规marl算法(grid_net除外)
    """

    def __init__(self, n_actions: int, n_agents: int):
        self.obs = []
        self.obs_next = []
        self.state = []
        self.state_next = []
        self.rewards = []
        self.unit_actions = []
        self.unit_actions_onehot = []
        self.n_actions = n_actions
        self.n_agents = n_agents
        self.n_step = 0

    def store_one_episode(self, one_obs: dict, one_obs_next: dict, one_state: ndarray, one_state_next: ndarray,
                          action: list, reward: float):
        one_obs = torch.stack([torch.Tensor(value) for value in one_obs.values()], dim=0)
        one_obs_next = torch.stack([torch.Tensor(value) for value in one_obs_next.values()], dim=0)
        self.obs.append(one_obs)
        self.obs_next.append(one_obs_next)
        self.state.append(torch.Tensor(one_state))
        self.state_next.append(torch.Tensor(one_state_next))
        self.rewards.append(reward)
        self.unit_actions.append(action)
        self.unit_actions_onehot.append(
            torch.zeros(self.n_agents, self.n_actions).scatter_(1, torch.LongTensor(action).unsqueeze(dim=-1), 1))
        self.n_step += 1

    def clear_memories(self):
        self.obs.clear()
        self.obs_next.clear()
        self.state.clear()
        self.state_next.clear()
        self.rewards.clear()
        self.unit_actions.clear()
        self.unit_actions_onehot.clear()
        self.n_step = 0


class CommMemory(object):
    """
        存储所有游戏的记忆单元, 适用于常规marl算法(grid_net除外)
    """

    def __init__(self):
        self.train_config = ConfigObjectFactory.get_train_config()
        self.memory_size = self.train_config.memory_size
        self.current_idx = 0
        self.memory = []

    def store_episode(self, one_episode_memory: CommOneEpisodeMemory):
        with LOCK_MEMORY:
            obs = torch.stack(one_episode_memory.obs, dim=0)
            obs_next = torch.stack(one_episode_memory.obs_next, dim=0)
            state = torch.stack(one_episode_memory.state, dim=0)
            state_next = torch.stack(one_episode_memory.state_next, dim=0)
            actions = torch.Tensor(one_episode_memory.unit_actions)
            actions_onehot = torch.stack(one_episode_memory.unit_actions_onehot, dim=0)
            reward = torch.Tensor(one_episode_memory.rewards)
            data = {
                'obs': obs,
                'obs_next': obs_next,
                'state': state,
                'state_next': state_next,
                'rewards': reward,
                'actions': actions,
                'actions_onehot': actions_onehot,
                'n_step': one_episode_memory.n_step
            }
            if len(self.memory) < self.memory_size:
                self.memory.append(data)
            else:
                self.memory[self.current_idx % self.memory_size] = data
            self.current_idx += 1

    def sample(self, batch_size) -> dict:
        """
            从记忆单元中随机抽样，但是每一局游戏的step不同，找出这个batch中
            最大的那个，将其他游戏的数据补齐
            :param batch_size: 一个batch的大小
            :return: 一个batch的数据
        """
        sample_size = min(len(self.memory), batch_size)
        sample_list = sample(self.memory, sample_size)
        n_step = torch.Tensor([one_data['n_step'] for one_data in sample_list])
        max_step = int(torch.max(n_step))

        obs = torch.stack(
            [torch.cat([one_data['obs'],
                        torch.zeros([max_step - one_data['obs'].shape[0]] +
                                    list(one_data['obs'].shape[1:]))
                        ])
             for one_data in sample_list], dim=0).detach()

        obs_next = torch.stack(
            [torch.cat([one_data['obs_next'],
                        torch.zeros(size=[max_step - one_data['obs_next'].shape[0]] +
                                         list(one_data['obs_next'].shape[1:]))])
             for one_data in sample_list], dim=0).detach()

        state = torch.stack(
            [torch.cat([one_data['state'],
                        torch.zeros([max_step - one_data['state'].shape[0]] +
                                    list(one_data['state'].shape[1:]))])
             for one_data in sample_list], dim=0).detach()

        state_next = torch.stack(
            [torch.cat([one_data['state_next'],
                        torch.zeros([max_step - one_data['state_next'].shape[0]] +
                                    list(one_data['state_next'].shape[1:]))])
             for one_data in sample_list], dim=0).detach()

        rewards = torch.stack(
            [torch.cat([one_data['rewards'],
                        torch.zeros([max_step - one_data['rewards'].shape[0]] +
                                    list(one_data['rewards'].shape[1:]))])
             for one_data in sample_list], dim=0).detach()

        actions = torch.stack(
            [torch.cat([one_data['actions'],
                        torch.zeros([max_step - one_data['actions'].shape[0]] +
                                    list(one_data['actions'].shape[1:]))])
             for one_data in sample_list], dim=0).unsqueeze(dim=-1).detach()

        actions_onehot = torch.stack(
            [torch.cat([one_data['actions_onehot'],
                        torch.zeros([max_step - one_data['actions_onehot'].shape[0]] +
                                    list(one_data['actions_onehot'].shape[1:]))])
             for one_data in sample_list], dim=0).detach()

        terminated = torch.stack(
            [torch.cat([torch.ones(one_data['n_step']), torch.zeros(max_step - one_data['n_step'])])
             for one_data in sample_list], dim=0).detach()

        batch_data = {
            'obs': obs,
            'obs_next': obs_next,
            'state': state,
            'state_next': state_next,
            'rewards': rewards,
            'actions': actions,
            'actions_onehot': actions_onehot,
            'max_step': max_step,
            'sample_size': sample_size,
            'terminated': terminated
        }
        return batch_data

    def get_memory_real_size(self):
        return len(self.memory)
