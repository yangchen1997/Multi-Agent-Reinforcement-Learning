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
        if "grid_wise_control+ppo" == self.env_config.learn_policy:
            # 需要注意的是ppo算法是每局游戏内进行更新的，因此只需要搜集每局游戏的数据，并不需要把所有局的游戏整合在一起再采样
            self.memory = None
            self.batch_episode_memory = GridBatchEpisodeMemory()
        elif "grid_wise_control" in self.env_config.learn_policy:
            self.memory = GridMemory()
            self.batch_episode_memory = GridBatchEpisodeMemory()
        elif "centralized_ppo" == self.env_config.learn_policy or "independent_ppo" == self.env_config.learn_policy:
            self.memory = None
            self.batch_episode_memory = CommBatchEpisodeMemory(continuous_actions=True)
        else:
            self.memory = CommMemory()
            self.batch_episode_memory = CommBatchEpisodeMemory(continuous_actions=False,
                                                               n_actions=self.env_info['n_actions'],
                                                               n_agents=self.env_info['n_agents'])
        self.lock = threading.Lock()
        # 初始化路径
        self.results_path = self.agents.get_results_path()
        self.memory_path = os.path.join(self.results_path, "memory.txt")
        self.result_path = os.path.join(self.results_path, "result.csv")

    def run_marl(self):
        self.init_saved_model()
        run_episode = self.train_config.run_episode_before_train if "ppo" in self.env_config.learn_policy else 1
        for epoch in range(self.current_epoch, self.train_config.epochs + 1):
            # 在正式开始训练之前做一些动作并将信息存进记忆单元中
            # grid_wise_control系列算法和常规marl算法不同, 是以格子作为观测空间。
            # ppo 属于on policy算法，训练数据要是同策略的
            total_reward = 0
            if "grid_wise_control" in self.env_config.learn_policy and isinstance(self.batch_episode_memory,
                                                                                  GridBatchEpisodeMemory):
                for i in range(run_episode):
                    self.env.reset()
                    finish_game = False
                    cycle = 0
                    while not finish_game and cycle < self.env_config.max_cycles:
                        grid_input = self.env.get_grid_input()
                        unit_pos = self.env.get_agents_approximate_pos()
                        actions_with_name, actions, log_probs = self.agents.choose_actions_in_grid(unit_pos=unit_pos,
                                                                                                   grid_input=grid_input)
                        observations, rewards, finish_game, infos = self.env.step(actions_with_name)
                        grid_input_next = self.env.get_grid_input()
                        self.batch_episode_memory.store_one_episode(grid_input, grid_input_next, unit_pos,
                                                                    actions, rewards, log_probs)
                        total_reward += rewards
                        cycle += 1
                    self.batch_episode_memory.set_per_episode_len(cycle)
            elif isinstance(self.batch_episode_memory, CommBatchEpisodeMemory):

                for i in range(run_episode):
                    obs = self.env.reset()
                    finish_game = False
                    cycle = 0
                    while not finish_game and cycle < self.env_config.max_cycles:
                        state = self.env.state()
                        actions_with_name, actions, log_probs = self.agents.choose_actions(obs)
                        obs_next, rewards, finish_game, infos = self.env.step(actions_with_name)
                        state_next = self.env.state()
                        if "ppo" in self.env_config.learn_policy:
                            self.batch_episode_memory.store_one_episode(one_obs=obs, one_state=state, action=actions,
                                                                        reward=rewards, log_probs=log_probs)
                        else:
                            self.batch_episode_memory.store_one_episode(one_obs=obs, one_state=state, action=actions,
                                                                        reward=rewards, one_obs_next=obs_next,
                                                                        one_state_next=state_next)
                        total_reward += rewards
                        obs = obs_next
                        cycle += 1
                    self.batch_episode_memory.set_per_episode_len(cycle)
            if "ppo" in self.env_config.learn_policy:
                # 可以用一个policy跑一个batch的数据来收集，由于性能问题假设batch=1，后续来优化
                batch_data = self.batch_episode_memory.get_batch_data()
                self.agents.learn(batch_data)
                self.batch_episode_memory.clear_memories()
            else:
                self.memory.store_episode(self.batch_episode_memory)
                self.batch_episode_memory.clear_memories()
                if self.memory.get_memory_real_size() >= 10:
                    for i in range(self.train_config.learn_num):
                        batch = self.memory.sample(self.train_config.memory_batch)
                        self.agents.learn(batch, epoch)
            # avg_reward = self.evaluate()
            avg_reward = total_reward / run_episode
            one_result_buffer = [avg_reward]
            self.result_buffer.append(one_result_buffer)
            if epoch % self.train_config.save_epoch == 0 and epoch != 0:
                self.save_model_and_result(epoch)
            print("episode_{} over,avg_reward {}".format(epoch, avg_reward))

    def init_saved_model(self):
        if os.path.exists(self.result_path) and (
                os.path.exists(self.memory_path) or "ppo" in self.env_config.learn_policy) \
                and self.agents.is_saved_model():
            if "ppo" not in self.env_config.learn_policy:
                with open(self.memory_path, 'rb') as f:
                    self.memory = pickle.load(f)
                    self.current_epoch = self.memory.episode + 1
                self.result_buffer.clear()
            else:
                with open(self.result_path, 'r') as f:
                    count = 0
                    for _ in csv.reader(f):
                        count += 1
                    self.current_epoch = count
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
        if "ppo" not in self.env_config.learn_policy:
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
                    actions_with_name, _, _ = self.agents.choose_actions_in_grid(unit_pos=unit_pos,
                                                                                 grid_input=grid_input)
                    _, rewards, finish_game, _ = self.env.step(actions_with_name)
                    total_rewards += rewards
                    cycle += 1
            else:
                obs = self.env.reset()
                finish_game = False
                cycle = 0
                while not finish_game and cycle < self.env_config.max_cycles:
                    actions_with_name, actions, _ = self.agents.choose_actions(obs)
                    obs_next, rewards, finish_game, _ = self.env.step(actions_with_name)
                    total_rewards += rewards
                    obs = obs_next
                    cycle += 1
        return total_rewards / self.train_config.evaluate_epoch
