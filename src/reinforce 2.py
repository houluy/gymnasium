import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical
from tqdm import tqdm
import gymnasium as gym
import tomllib
import random

from src.utils.replay_buffer import ReplayBuffer

torch.autograd.set_detect_anomaly(True)
device = torch.device("cuda:0" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")) 

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
        return torch.from_numpy(np.array(self.states, dtype=np.float32)).to(device),\
            torch.from_numpy(np.array(self.actions, dtype=np.float32)).to(device),\
            torch.from_numpy(np.array(self.rewards, dtype=np.float32)).to(device),\
            torch.from_numpy(np.array(self.next_states, dtype=np.float32)).to(device),\
            torch.from_numpy(np.array(self.dones, dtype=np.float32)).to(device)

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []


class Policy(nn.Module):
    def __init__(self, input_dim, output_num, hidden_size=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_num)
        self.model = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            self.fc2,
            nn.ReLU(),
            self.fc3,
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.model(x)


class Value(nn.Module):
    def __init__(self, input_dim, hidden_size=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.model = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            self.fc2,
            nn.ReLU(),
            self.fc3
        )

    def forward(self, x):
        return self.model(x)


class Q(nn.Module):
    def __init__(self, state_dim, action_num, hidden_size=128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_num)
        self.model = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            self.fc2,
            nn.ReLU(),
            self.fc3
        )

    def forward(self, x):
        return self.model(x)


class Reinforce:
    def __init__(self,
        env,
    ):
        with open("config/config.toml", 'rb') as f:
            config = tomllib.load(f)
        a2c = config["cartpole"]["a2c"]
        for key, value in a2c.items():
            setattr(self, key, value)
        self.env = env
        self.replay_buffer = ReplayBuffer(self.buffer_size, device=device)
        self.episode_buffer = EpisodeBuffer()

        # Value network
        self.value = Value(self.env.observation_space.shape[0]).to(device)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=self.value_lr)

        # Policy network
        self.policy = Policy(self.env.observation_space.shape[0], self.env.action_space.n).to(device)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=self.policy_lr)

        # Tensorboard Writer
        self.writer = SummaryWriter()

    def select_action(self, state):
        action_probs = self.policy(state)
        sample = torch.multinomial(action_probs, 1)
        return sample[0].item()

    def update_value(self, batch):
        states, actions, rewards, next_states, dones = batch

        # Calculate value function
        with torch.no_grad():
            target_values = rewards.unsqueeze(1) + self.gamma * self.value(next_states) * ((1 - dones).unsqueeze(1))
        values = self.value(states)

        # Calculate REINFORCE value with baseline
        #with torch.no_grad():
        #    delta = target_values - values

        # Calculate loss
        loss = F.smooth_l1_loss(values, target_values)

        # Update value network
        self.value_optimizer.zero_grad()
        loss.backward()
        self.value_optimizer.step()
        return loss

    def update_policy(self, batch):
        states, actions, rewards, next_states, dones = batch
        delta = rewards.unsqueeze(1) + self.gamma * self.value(next_states) * ((1 - dones).unsqueeze(1)) - self.value(states)
        probs = self.policy(states)
        self.policy_optimizer.zero_grad()
        m = Categorical(probs)
        performance = torch.mean(-m.log_prob(actions).unsqueeze(1) * delta)
        performance.backward()
        self.policy_optimizer.step()
        return performance, delta

    def train(self, episodes=50000):
        for episode in tqdm(range(episodes)):
            state, _ = self.env.reset()
            done = False
            step = 0
            episodic_reward = 0
            while not done:
                action = self.select_action(torch.from_numpy(state).to(device))
                next_state, reward, done, truncation, _ = self.env.step(action)
                episodic_reward += reward
                self.episode_buffer.add(state, action, reward, next_state, 1 if done else 0)
                self.replay_buffer.add(state, action, reward, next_state, 1 if done else 0)
                state = next_state
                done = done or truncation
                step += 1
            exp = self.episode_buffer.get()
            performance, delta = self.update_policy(exp)
            if len(self.replay_buffer) > self.batch_size:
                batch = self.replay_buffer.sample(self.batch_size)
                loss = self.update_value(batch)
                self.writer.add_scalar('training/loss_value', loss, episode)
            else:
                loss = None
            self.episode_buffer.clear()
            self.writer.add_scalar('training/policy_delta', delta.mean(0), episode)
            self.writer.add_scalar('training/performance', performance, episode)
            if episode % self.info_step == 0:
                tqdm.write(f'Episode {episode}, Step {step}, Loss: {loss}, Performance: {performance}')
            self.writer.add_scalar('training/episodic_reward', episodic_reward, episode)


class DQN:
    def __init__(self, env):
        with open("config/config.toml", 'rb') as f:
            config = tomllib.load(f)
        agent_config = config["cartpole"]["dqn"]
        for attr, value in agent_config.items():
            setattr(self, attr, value)
        self.env = env
        self.epsilon = self.start_epsilon = 1
        self.end_epsilon = 0.1
        self.replay_buffer = ReplayBuffer(self.buffer_size, device=device)
        self.q = Q(env.observation_space.shape[0], env.action_space.n).to(device)
        self.target_network = Q(env.observation_space.shape[0], env.action_space.n).to(device)
        self.optimizer = optim.Adam(self.q.parameters(), lr=self.lr)
        self.model_path = "models/cartpole-dqn.pth"
        self.writer = SummaryWriter()

    def epsilon_decay(self):
        self.epsilon *= self.epsilon_decay_factor
        return self.epsilon

    def save(self):
        torch.save(self.q.state_dict(), self.model_path)

    def load(self):
        self.q.load_state_dict(torch.load(self.model_path))

    def select_action(self, state):
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return self.q(state).argmax().item()

    def update_value(self, batch):
        states, actions, rewards, next_states, dones = batch
        y = rewards + self.gamma * self.target_network(next_states).max(1)[0] * (1 - dones)
        q = self.q(states).gather(1, actions.unsqueeze(1))
        loss = F.mse_loss(q, y.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def train(self, episodes=10000):
        training_step = -self.training_start
        for episode in tqdm(range(episodes), desc="Training"):
            state, _ = self.env.reset()
            done = False
            step = 0
            episodic_reward = 0
            while not done:
                action = self.select_action(torch.from_numpy(state).to(device))
                next_state, reward, done, truncation, _ = self.env.step(action)
                episodic_reward += reward
                self.replay_buffer.add(state, action, reward, next_state, 1 if done else 0)
                state = next_state
                done = done or truncation
                step += 1
                training_step += 1
                if training_step > 0:
                    loss = self.update_value(self.replay_buffer.sample(self.batch_size))
                    self.writer.add_scalar("training/loss_value", loss, training_step)
                    self.writer.add_scalar("training/epsilon", self.epsilon, training_step)
                    if training_step % self.epsilon_decay_step == 0:
                        self.epsilon_decay()
                    if training_step % self.info_step == 0:
                        tqdm.write(f"Episode {episode}, Step {step}, Loss: {loss}")
                    if training_step % self.target_update_step == 0:
                        self.target_network.load_state_dict(self.q.state_dict())
            self.writer.add_scalar("training/episodic_reward", episodic_reward, episode)


def main():
    env = gym.make('CartPole-v1')
    agent = DQN(env)
    #agent = Reinforce(env)
    agent.train()
