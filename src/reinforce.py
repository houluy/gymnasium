import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm

import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter

from collections import deque
import random
import numpy as np


class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)

    def __len__(self):
        return len(self.buffer)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        sample = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.stack, zip(*sample))
        return torch.tensor(states), torch.tensor(actions), torch.tensor(rewards), torch.tensor(next_states), torch.tensor(dones)


class EpisodeBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []

    def add(self, state, action, reward, next_state, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)

    def get(self):
        return torch.tensor(self.states), torch.tensor(self.actions), torch.tensor(self.rewards), torch.tensor(self.next_states), torch.tensor(self.dones)


# REINFORCE

class DiscretePolicy(nn.Module):
    def __init__(self, state_dim, action_num, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_num)
        self.model = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            self.fc2,
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        y = self.model(x)
        return y


class Value(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.model = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            self.fc2
        )

    def forward(self, x):
        y = self.model(x)
        return y    


class REINFORCE:
    def __init__(self, state_dim, action_num,
        hidden_dim=64,
        policy_lr=0.001,
        value_lr=0.001,
        gamma=0.9,
        batch_size=64,
        buffer_size=20000,
    ):
        self.state_dim = state_dim
        self.action_num = action_num
        self.hidden_dim = hidden_dim
        self.policy = DiscretePolicy(state_dim, action_num, hidden_dim)
        self.value = Value(state_dim, hidden_dim)
        self.policy_lr = policy_lr
        self.value_lr = value_lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.buffer = ReplayBuffer(buffer_size)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=policy_lr)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=value_lr)
        
        # Tensorboard writer
        self.writer = SummaryWriter()


    def train(self, env, episodes=1000):
        for episode in tqdm(range(episodes), desc='Training Reinforce'):
            state, _ = env.reset()
            done = False
            episodic_reward = 0
            while not done:
                action_dist = self.policy(state)

            



if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    reinforce = REINFORCE(env.observation_space.shape[0], env.action_space.n)
    reinforce.train(env, episodes=10000)

