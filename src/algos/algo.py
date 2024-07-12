import tomllib as toml
from abc import ABCMeta, abstractmethod
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter

class Algo(metaclass=ABCMeta):
    def __init__(self, env_name, device="cpu"):
        self.env = gym.make(env_name)
        self.device = device
        with open("config/config.toml", "rb") as f:
            config = toml.load(f)
        config = config[env_name][self.__class__.__name__]
        for key, value in config.items():
            setattr(self, key, value)
        self.model_path = f"models/{env_name}/{self.__class__.__name__}.pt"
        self.writer = SummaryWriter()

    @abstractmethod
    def select_action(self, state):
        pass

    @abstractmethod
    def train(self, episodes=10000):
        pass
        