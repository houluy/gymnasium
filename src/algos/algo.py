import tomllib as toml
from abc import ABCMeta, abstractmethod
from collections.abc import Callable
from typing import List, TypeVar
import gymnasium as gym
from gymnasium.spaces import Space
import numpy as np
from numpy.typing import ArrayLike
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

ObsType = TypeVar("ObsType")


def sigmoid(x: ArrayLike) -> ArrayLike:
    return 1 / (1 + np.exp(-x))


def make_state_normalize(observation_space: Space[ObsType]):
    def normalizer(state: ArrayLike) -> ArrayLike:
        if observation_space.is_bounded:
            return (state - observation_space.low) / (observation_space.high - observation_space.low)
        else:
            return sigmoid(state)
    return normalizer


class Algo(metaclass=ABCMeta):
    def __init__(self, env_name=None, continuous=False, device=torch.device("cpu")):#, state_normalize: Callable[[ArrayLike], ArrayLike] = lambda x: x):
        self.env = gym.make(env_name)
        
        self.device = device
        self.continuous = continuous
        self.state_dim = self.env.observation_space.shape[0]
        if self.continuous:
            self.action_dim = self.env.action_space.shape[0]
        else:
            self.action_dim = self.env.action_space.n
        self.select_action = self.select_action_continuous if self.continuous else self.select_action_discrete
        with open("config/config.toml", "rb") as f:
            config = toml.load(f)
        config = config[env_name][self.__class__.__name__]
        for key, value in config.items():
            setattr(self, key, value)
        self.model_path = f"models/{env_name}/{self.__class__.__name__}.pt"
        self.state_normalize = make_state_normalize(self.env.observation_space)
        self.writer = SummaryWriter()

    @abstractmethod
    def select_action_continuous(self, state, train=True):
        pass

    @abstractmethod
    def select_action_discrete(self, state, train=True):
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
                action = action.cpu()
                next_state, reward, done, truncation, info = self.env.step(action)
                episodic_reward += reward
                state = next_state
                done = done or truncation
            self.writer.add_scalar("evaluation/episodic_reward", episodic_reward, episode)
            rewards.append(episodic_reward)
        return rewards

    @abstractmethod
    def train(self, episodes=10000) -> None:
        pass
