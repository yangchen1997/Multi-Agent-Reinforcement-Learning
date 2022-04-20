import csv
import os

import matplotlib.pyplot as plt
from absl import app

from common.pettingzoo_environment import SimpleSpreadEnv
from runner import RunnerSimpleSpreadEnv
from utils.config_utils import ConfigObjectFactory


def main(args):
    print(args)
    env = SimpleSpreadEnv()
    try:
        runner = RunnerSimpleSpreadEnv(env)
        runner.run_marl()
    finally:
        env.close()


def evaluate():
    train_config = ConfigObjectFactory.get_train_config()
    env_config = ConfigObjectFactory.get_environment_config()
    csv_filename = os.path.join(train_config.result_dir, env_config.learn_policy, "result.csv")
    rewards = []
    total_rewards = []
    avg_rewards = 0
    len_csv = 0
    with open(csv_filename, 'r') as f:
        r_csv = csv.reader(f)
        for data in r_csv:
            total_rewards.append(round(float(data[0]), 2))
            avg_rewards += float(data[0])
            if len_csv % train_config.show_evaluate_epoch == 0 and len_csv > 0:
                rewards.append(round(avg_rewards / train_config.show_evaluate_epoch, 2))
                avg_rewards = 0
            len_csv += 1

    plt.plot([i * train_config.show_evaluate_epoch for i in range(len_csv // train_config.show_evaluate_epoch)],
             rewards)
    plt.plot([i for i in range(len_csv)], total_rewards, alpha=0.3)
    plt.title("rewards")
    plt.show()


if __name__ == "__main__":
    app.run(main)
    evaluate()
