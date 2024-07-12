import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch.distributions import Categorical
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

from collections import deque
import random
import numpy as np
import tomllib as toml
from src.utils import ReplayBuffer, EpisodeBuffer
from src.networks import DiscretePolicy as Policy
from src.algos.algo import Algo


class Reinforce(Algo):
    def __init__(self,
        env_name,
        device
    ):
        super().__init__(env_name, device)

        self.episode_buffer = EpisodeBuffer(self.gamma, self.device)

        # Policy network
        self.policy = Policy(self.env.observation_space.shape[0], self.env.action_space.n).to(self.device)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=self.policy_lr)

    def select_action(self, state):
        with torch.no_grad():
            action_probs = self.policy(state)
            sample = torch.multinomial(action_probs, 1)
        return sample[0].item()

    def update_policy(self, batch):
        states, actions, rewards, next_states, dones, rewards_to_go = batch
        for t, reward in enumerate(rewards):
            prob = self.policy(states[t])
            self.policy_optimizer.zero_grad()
            m = Categorical(prob)
            performance = torch.sum(-m.log_prob(actions[t]) * rewards_to_go[t])
            performance.backward()
            self.policy_optimizer.step()
        return performance, rewards_to_go

    def train(self, episodes=50000):
        for episode in tqdm(range(episodes)):
            state, _ = self.env.reset()
            done = False
            step = 0
            episodic_reward = 0
            while not done:
                action = self.select_action(torch.from_numpy(state).to(self.device))
                next_state, reward, done, truncation, _ = self.env.step(action)
                episodic_reward += reward
                self.episode_buffer.add(state, action, reward, next_state, 1 if done else 0)
                state = next_state
                done = done or truncation
                step += 1
            exp = self.episode_buffer.get()
            performance, rewards_to_go = self.update_policy(exp)
            self.episode_buffer.clear()
            #self.writer.add_scalar('training/policy_delta', delta, episode)
            self.writer.add_scalar('training/performance', performance, episode)
            if episode % self.info_step == 0:
                tqdm.write(f'Episode {episode}, Performance: {performance}, G(s0): {rewards_to_go[0]}')
            self.writer.add_scalar('training/episodic_reward', episodic_reward, episode)