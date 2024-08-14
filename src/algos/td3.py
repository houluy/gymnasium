from src.algos.algo import Algo
import torch
import torch.optim as optim
import torch.nn.functional as F
from src.networks import ContinuousPolicy as Policy, ContinuousQ as Q
from src.utils import ReplayBuffer
from tqdm import tqdm
import numpy as np


class TwinDelayedDDPG(Algo):
    """Twin Delayed Deep Deterministic Policy Gradient (DDPG) algorithm."""
    def __init__(self, env_name, continuous=True, device=torch.device("cpu")):
        super().__init__(env_name, continuous, device)
        self.buffer = ReplayBuffer(self.buffer_size, self.device, action_type="continuous")
        self.policy = Policy(self.state_dim, self.action_dim).to(device)
        self.target_policy = Policy(self.state_dim, self.action_dim).to(device)
        self.q1 = Q(self.state_dim + self.action_dim).to(device)
        self.q2 = Q(self.state_dim + self.action_dim).to(device)

        self.qs = (self.q1, self.q2)

        self.target_q1 = Q(self.state_dim + self.action_dim).to(device)
        self.target_q2 = Q(self.state_dim + self.action_dim).to(device)

        self.target_qs = (self.target_q1, self.target_q2)

        self.target_network_soft_update()

        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=self.policy_lr)
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=self.critic_lr)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=self.critic_lr)
        self.noise_std = self.start_noise_std

    def target_network_soft_update(self):
        with torch.no_grad():
            for target_q, q in zip(self.target_qs, self.qs):
                for target_param, param in zip(target_q.parameters(), q.parameters()):
                    target_param.data.copy_(param.data * self.polyak + target_param.data * (1.0 - self.polyak))
            for target_param, param in zip(self.target_policy.parameters(), self.policy.parameters()):
                target_param.data.copy_(param.data * self.polyak + target_param.data * (1.0 - self.polyak))

    def select_action_continuous(self, state, policy, target=False, train=True):
        with torch.no_grad():
            action = policy(state)
            if train:
                random_v = torch.randn_like(action) * self.noise_std
                if target:
                    random_v = torch.clamp(random_v, min=self.target_policy_threshold_min, max=self.target_policy_threshold_max)
                action += random_v
                action = torch.clamp(action, min=self.action_min, max=self.action_max)
        return action

    def select_action_discrete(self, state, train=True):
        return None#super().select_action_discrete(state, train)

    def evaluate(self, episodes=100) -> torch.List[float]:
        return super().evaluate(episodes)

    def no_grad(self, model):
        for p in model.parameters():
            p.requires_grad = False
    
    def yes_grad(self, model):
        for p in model.parameters():
            p.requires_grad = True

    def decay_noise(self):
        self.noise_std *= self.noise_std_decay_factor
        return self.noise_std

    def update_qs(self, batch):
        states, actions, rewards, next_states, dones = batch
        next_actions = self.select_action_continuous(next_states, self.target_policy, target=True)
        state_actions = torch.concat([states, actions], dim=1)
        next_state_actions = torch.concat([next_states, next_actions], dim=1)
        target_q_1 = self.target_q1(next_state_actions)
        target_q_2 = self.target_q2(next_state_actions)
        y = rewards.unsqueeze(1) + self.gamma * (1 - dones).unsqueeze(1) * torch.min(target_q_1, target_q_2)
        q1 = self.q1(state_actions)
        q2 = self.q2(state_actions)
        loss_1 = F.mse_loss(q1, y.float())
        loss_2 = F.mse_loss(q2, y.float())
        self.q1_optimizer.zero_grad()
        loss_1.backward(retain_graph=True)
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        loss_2.backward()
        self.q2_optimizer.step()

        return loss_1, loss_2

    def update_policy(self, batch):
        states, actions, rewards, next_states, dones = batch
        state_actions = torch.concat([states, actions], dim=1)
        q = self.q1(state_actions)
        self.no_grad(self.q1)
        self.no_grad(self.q2)
        self.policy_optimizer.zero_grad()
        performance = -q.mean()
        performance.backward()
        self.policy_optimizer.step()
        self.yes_grad(self.q1)
        self.yes_grad(self.q2)
        return performance

    def train(self, episodes=10000):
        if self.continuous:
            action_dist = []

        training_step = -self.training_start
        for episode in tqdm(range(episodes), desc="Training"):
            state, _ = self.env.reset()
            done = False
            episodic_reward = 0
            step = 0
            while not done:
                action = self.select_action_continuous(torch.from_numpy(state).to(self.device), self.policy, target=False)
                action = action.cpu()
                if self.continuous:
                    action_dist.append(action)
                next_state, reward, done, truncation, _ = self.env.step(action)
                episodic_reward += reward
                self.buffer.add(state, action, reward, next_state, 1 if done else 0)
                done = done or truncation
                step += 1
                training_step += 1
                if training_step > 0:
                    batch = self.buffer.sample(self.batch_size)
                    loss_1, loss_2 = self.update_qs(batch)
                    if training_step % self.policy_update_freq == 0:
                        performance = self.update_policy(batch)
                        self.writer.add_scalar("training/q_value", -performance, training_step // self.policy_update_freq)
                    self.writer.add_scalar("training/loss_value_1", loss_1, training_step)
                    self.writer.add_scalar("training/loss_value_2", loss_2, training_step)
                    self.writer.add_scalar("training/Gaussian_noise_std", self.noise_std, training_step)
                    if training_step % self.noise_std_decay_freq == 0:
                        self.decay_noise()
                    if training_step % self.info_step == 0:
                        tqdm.write(f"Episode: {episode}, Training step: {training_step}, Episodic Reward: {episodic_reward}, Loss 1: {loss_1}, Loss 2: {loss_2}")
                    self.target_network_soft_update()
            self.writer.add_scalar("training/episode_length", step, episode)
            self.writer.add_scalar("training/episodic_reward", episodic_reward, episode)
        if self.continuous:
            self.writer.add_histogram("training/action_dist", np.array(action_dist), 1, bins="auto")

