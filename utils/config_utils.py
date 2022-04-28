import yaml

from utils.config_objects import *


class ConfigName:
    environment_config_name = "environment"
    train_config_name = "train"


class ConfigObjectFactory(object):
    yaml_path = r"./config.yaml"
    yaml_config = None
    environment_config = None
    train_config = None

    @classmethod
    def init_yaml_config(cls):
        if not cls.yaml_config:
            with open(cls.yaml_path, 'rb') as file:
                cls.yaml_config = yaml.safe_load(file)

    @staticmethod
    def init_config_object_attr(instance: object, attrs: dict):
        if not instance or not attrs:
            return
        for name, value in attrs.items():
            if hasattr(instance, name):
                setattr(instance, name, value)

    @classmethod
    def get_environment_config(cls) -> EnvironmentConfig:
        if cls.environment_config is None:
            cls.init_yaml_config()
            cls.environment_config = EnvironmentConfig()
            if ConfigName.environment_config_name in cls.yaml_config:
                cls.init_config_object_attr(cls.environment_config, cls.yaml_config[ConfigName.environment_config_name])
        return cls.environment_config

    @classmethod
    def get_train_config(cls) -> TrainConfig:
        if cls.train_config is None:
            cls.init_yaml_config()
            cls.train_config = TrainConfig()
            if ConfigName.train_config_name in cls.yaml_config:
                cls.init_config_object_attr(cls.train_config, cls.yaml_config[ConfigName.train_config_name])
        return cls.train_config


