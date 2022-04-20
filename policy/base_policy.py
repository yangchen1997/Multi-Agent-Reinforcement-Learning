import os


class BasePolicy(object):

    @staticmethod
    def init_path(*args):
        for path in args:
            if not os.path.exists(path):
                os.mkdir(path)

    def init_wight(self):
        raise NotImplementedError

    def learn(self, batch_data: dict, episode_num: int):
        raise NotImplementedError

    def save_model(self):
        raise NotImplementedError

    def load_model(self):
        raise NotImplementedError

    def del_model(self):
        raise NotImplementedError

    def is_saved_model(self) -> bool:
        raise NotImplementedError
