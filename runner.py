import csv
import os
import pickle

from agent.agents import MyAgents
from common.pettingzoo_environment import SimpleSpreadEnv
from common.reply_buffer import *


class RunnerSimpleSpreadEnv(object):

    def __init__(self, env: SimpleSpreadEnv):
        self.env = env
        self.train_config = ConfigObjectFactory.get_train_config()
        self.env_config = ConfigObjectFactory.get_environment_config()
        self.current_epoch = 0
        self.result_buffer = []
        self.env_info = self.env.get_env_info()
        self.agents = MyAgents(self.env_info)
        # 初始化reply_buffer
        if "grid_wise_control" in self.env_config.learn_policy:
            self.memory = Memory()
            self.one_episode_memory = OneEpisodeMemory()
        else:
            self.memory = CommMemory()
            self.one_episode_memory = CommOneEpisodeMemory(self.env_info['n_actions'], self.env_info['n_agents'])
        self.lock = threading.Lock()
        # 初始化路径
        self.results_path = self.agents.get_results_path()
        self.memory_path = os.path.join(self.results_path, "memory.txt")
        self.result_path = os.path.join(self.results_path, "result.csv")

    def run_marl(self):
        self.init_saved_model()
        for epoch in range(self.current_epoch, self.train_config.epochs + 1):
            # 在正式开始训练之前做一些动作并将信息存进记忆单元中
            # grid_wise_control系列算法和常规marl算法不同, 是以格子作为观测空间。
            if "grid_wise_control" in self.env_config.learn_policy and isinstance(self.one_episode_memory, OneEpisodeMemory):
                self.env.reset()
                finish_game = False
                cycle = 0
                while not finish_game and cycle < self.env_config.max_cycles:
                    grid_input = self.env.get_grid_input()
                    unit_pos = self.env.get_agents_approximate_pos()
                    actions_with_name, actions = self.agents.choose_actions_in_grid(unit_pos=unit_pos,
                                                                                    grid_input=grid_input)
                    observations, rewards, finish_game, infos = self.env.step(actions_with_name)
                    grid_input_next = self.env.get_grid_input()
                    self.one_episode_memory.store_one_episode(grid_input, grid_input_next, unit_pos,
                                                              actions, rewards)
                    cycle += 1
            elif isinstance(self.one_episode_memory, CommOneEpisodeMemory):
                obs = self.env.reset()
                finish_game = False
                cycle = 0
                while not finish_game and cycle < self.env_config.max_cycles:
                    state = self.env.state()
                    actions_with_name, actions = self.agents.choose_actions(obs)
                    obs_next, rewards, finish_game, infos = self.env.step(actions_with_name)
                    state_next = self.env.state()
                    self.one_episode_memory.store_one_episode(obs, obs_next, state, state_next,
                                                              actions, rewards)
                    obs = obs_next
                    cycle += 1
            self.memory.store_episode(self.one_episode_memory)
            self.one_episode_memory.clear_memories()
            if self.memory.get_memory_real_size() >= 100:
                for i in range(self.train_config.learn_num):
                    batch = self.memory.sample(self.train_config.memory_batch)
                    self.agents.learn(batch, epoch)
            avg_reward = self.evaluate()
            one_result_buffer = [avg_reward]
            self.result_buffer.append(one_result_buffer)
            if epoch % self.train_config.save_epoch == 0 and epoch != 0:
                self.save_model_and_result(epoch)
            print("episode_{} over,avg_reward {}".format(epoch, avg_reward))

    def init_saved_model(self):
        if os.path.exists(self.result_path) and os.path.exists(self.memory_path) and self.agents.is_saved_model():
            with open(self.memory_path, 'rb') as f:
                self.memory = pickle.load(f)
                self.current_epoch = self.memory.episode + 1
                self.result_buffer.clear()
            self.agents.load_model()
        else:
            self.agents.del_model()
            file_list = os.listdir(self.results_path)
            for file in file_list:
                os.remove(os.path.join(self.results_path, file))

    def save_model_and_result(self, episode: int):
        self.agents.save_model()
        with open(self.result_path, 'a', newline='') as f:
            f_csv = csv.writer(f)
            f_csv.writerows(self.result_buffer)
            self.result_buffer.clear()
        with open(self.memory_path, 'wb') as f:
            self.memory.episode = episode
            pickle.dump(self.memory, f)

    def evaluate(self):
        total_rewards = 0
        for i in range(self.train_config.evaluate_epoch):
            if "grid_wise_control" in self.env_config.learn_policy:
                self.env.reset()
                terminated = False
                cycle = 0
                while not terminated and cycle < self.env_config.max_cycles:
                    grid_input = self.env.get_grid_input()
                    unit_pos = self.env.get_agents_approximate_pos()
                    actions_with_name, _ = self.agents.choose_actions_in_grid(unit_pos=unit_pos, grid_input=grid_input)
                    _, rewards, finish_game, _ = self.env.step(actions_with_name)
                    total_rewards += rewards
                    cycle += 1
            else:
                obs = self.env.reset()
                finish_game = False
                cycle = 0
                while not finish_game and cycle < self.env_config.max_cycles:
                    actions_with_name, actions = self.agents.choose_actions(obs)
                    obs_next, rewards, finish_game, _ = self.env.step(actions_with_name)
                    total_rewards += rewards
                    obs = obs_next
                    cycle += 1
        return total_rewards / self.train_config.evaluate_epoch
