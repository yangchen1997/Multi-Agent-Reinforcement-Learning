class EnvironmentConfig:
    def __init__(self):
        self.seed = 8
        self.n_agents = 3
        self.grid_size = 16
        self.max_cycles = 25
        self.learn_policy = "grid_wise_control"


class TrainConfig:
    def __init__(self):
        self.epochs = 100000
        self.evaluate_epoch = 2
        self.show_evaluate_epoch = 20
        self.memory_batch = 32
        self.memory_size = 5000
        self.learn_num = 3
        self.lr_actor = 1e-4
        self.lr_critic = 1e-3
        self.gamma = 0.99  # 衰减因子
        self.var = 0.05  # ddpg选择动作添加的噪声点,以输出为均值var为方差进行探索
        self.epsilon = 0.7
        self.grad_norm_clip = 10
        self.target_update_cycle = 100
        self.save_epoch = 1000
        self.model_dir = r"./models"
        self.result_dir = r"./results"
        self.cuda = True
