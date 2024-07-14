from src.algos.algo import Algo
import torch
import torch.optim as optim
import torch.nn.functional as F
from src.networks import ContinuousPolicy as Policy, ContinuousQ as Q
from src.utils import ReplayBuffer
from tqdm import tqdm
import numpy as np


class DeepDeterministicPolicyGradient(Algo):
    """Deep Deterministic Policy Gradient (DDPG) algorithm."""
    def __init__(self, env_name, device=torch.device("cpu")):
        super().__init__(env_name, device)

        self.buffer = ReplayBuffer(self.buffer_size, self.device, action_dtype="continuous")

        self.policy = Policy(self.env.observation_space.shape[0], self.env.action_space.shape[0]).to(device)
        self.target_policy = Policy(self.env.observation_space.shape[0], self.env.action_space.shape[0]).to(device)
        self.q = Q(self.env.observation_space.shape[0] + self.env.action_space.shape[0]).to(device)
        self.target_q = Q(self.env.observation_space.shape[0] + self.env.action_space.shape[0]).to(device)
        self.target_network_soft_update()
        
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=self.policy_lr)
        self.q_optimizer = optim.Adam(self.q.parameters(), lr=self.critic_lr)

        self.noise_std = self.start_noise_std
    
    def decay_noise(self):
        self.noise_std *= self.noise_std_decay_factor
        return self.noise_std

    def select_action(self, state):
        with torch.no_grad():
            action = self.policy(state)
            action += torch.randn_like(action) * self.noise_std

        return action.cpu()

    def target_network_soft_update(self):
        with torch.no_grad():
            for target_param, param in zip(self.target_q.parameters(), self.q.parameters()):
                target_param.data.copy_(param.data * self.polyak + target_param.data * (1.0 - self.polyak))
            for target_param, param in zip(self.target_policy.parameters(), self.policy.parameters()):
                target_param.data.copy_(param.data * self.polyak + target_param.data * (1.0 - self.polyak))

    def update_q(self, batch):
        states, actions, rewards, next_states, dones = batch
        next_actions = self.target_policy(next_states)
        y = rewards.unsqueeze(1) + self.gamma * (1 - dones).unsqueeze(1) * self.target_q(torch.concat([next_states, next_actions], dim=1))
        q = self.q(torch.concat([states, actions], dim=1))
        loss = F.mse_loss(q, y)

        self.q_optimizer.zero_grad()
        loss.backward()
        self.q_optimizer.step()
        return loss

    def update_policy(self, batch):
        states, actions, rewards, next_states, dones = batch
        q = self.q(torch.concat([states, actions], dim=1))
        for p in self.q.parameters():
            p.requires_grad = False
        self.policy_optimizer.zero_grad()
        performance = -q.mean()
        performance.backward()
        self.policy_optimizer.step()
        for p in self.q.parameters():
            p.requires_grad = True
        return performance

    def evaluate(self, episodes=100):
        rewards = super().evaluate(episodes)
        return rewards

    def train(self, episodes=10000):
        training_step = -self.training_start
        for episode in tqdm(range(episodes), desc="Training"):
            state, _ = self.env.reset()
            done = False
            episodic_reward = 0
            step = 0
            while not done:
                action = self.select_action(torch.from_numpy(state).to(self.device))
                next_state, reward, done, truncation, _ = self.env.step(action)
                episodic_reward += reward
                self.buffer.add(state, action, reward, next_state, 1 if done else 0)
                done = done or truncation
                step += 1
                training_step += 1
                if training_step > 0:
                    batch = self.buffer.sample(self.batch_size)
                    loss = self.update_q(batch)
                    performance = self.update_policy(batch)
                    self.writer.add_scalar("training/loss_value", loss, training_step)
                    self.writer.add_scalar("training/q_value", -performance, training_step)
                    self.writer.add_scalar("training/Gaussian_noise_std", self.noise_std, training_step)
                    if training_step % self.noise_std_decay_freq == 0:
                        self.decay_noise()
                    if training_step % self.info_step == 0:
                        tqdm.write(f"Episode: {episode}, Training step: {training_step}, Episodic Reward: {episodic_reward}")
                    self.target_network_soft_update()
            self.writer.add_scalar("training/episodic_reward", episodic_reward, episode)

