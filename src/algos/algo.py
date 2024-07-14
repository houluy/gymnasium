import tomllib as toml
from abc import ABCMeta, abstractmethod
from typing import List
import gymnasium as gym
import torch
from tqdm import tqdm
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
    def select_action(self, state, train=True):
        pass

    @abstractmethod
    def evaluate(self, episodes=100) -> List[float]:
        rewards = []
        for episode in tqdm(range(episodes), desc="Evaluation"):
            state, _ = self.env.reset()
            done = False
            episodic_reward = 0
            while not done:
                action = self.select_action(torch.from_numpy(state).to(self.device), train=False)
                next_state, reward, done, truncated, info = self.env.step(action)
                episodic_reward += reward
                state = next_state
                done = done or truncated
            self.writer.add_scalar("Evaluation Reward", episodic_reward, episode)
            rewards.append(episodic_reward)
        return rewards

    @abstractmethod
    def train(self, episodes=10000):
        pass
        