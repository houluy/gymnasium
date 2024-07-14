import torch
import torch.optim as optim
import torch.nn.functional as F
from src.algos.algo import Algo
from src.utils import ReplayBuffer
from src.networks import DuelingDiscreteQ as Q
from src.algos.dqn2 import DQN
import numpy as np
import random
from tqdm import tqdm


class DuelingDQN(DQN):
    def __init__(self, env_name, device=torch.device("cpu")):
        super().__init__(env_name, device)
        self.q = Q(self.env.observation_space.shape[0], self.env.action_space.n).to(device)
        self.target_network = Q(self.env.observation_space.shape[0], self.env.action_space.n).to(device)
        self.optimizer = optim.Adam(self.q.parameters(), lr=self.lr)

    def select_action(self, state, train=True):
        if train and (random.random() < self.epsilon):
            return self.env.action_space.sample()
        else:
            return self.q(state.unsqueeze(0)).argmax().item()


