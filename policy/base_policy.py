import os


class BasePolicy(object):

    @staticmethod
    def init_path(*args):
        for path in args:
            if not os.path.exists(path):
                os.mkdir(path)

    # 初始化权重
    def init_wight(self):
        raise NotImplementedError

    # 学习的方法，以一个batch的数据作为输入(封装成字典形式)，
    # episode_num表示当前是第几次迭代，用于double类型的算法
    def learn(self, batch_data: dict, episode_num: int):
        raise NotImplementedError

    # 保存模型
    def save_model(self):
        raise NotImplementedError

    # 加载模型
    def load_model(self):
        raise NotImplementedError

    # 删除模型
    def del_model(self):
        raise NotImplementedError

    # 判断是否保存过模型
    def is_saved_model(self) -> bool:
        raise NotImplementedError
