import tomllib as toml
from collections.abc import MetaABC, abstractmethod

class Algo(metaclass=MetaABC):
    def __init__(self, env, device="cpu"):
        self.env = env
        self.device = device
        with open("config/config.toml", "rb") as f:
            config = toml.load(f)
        config = config[self.env.__name__][self.__class__.__name__]
        for key, value in config.items():
            setattr(self, key, value)

    @abstractmethod
    def select_action(self, state):
        pass

    @abstractmethod
    def train(self, episodes=10000):
        pass
        