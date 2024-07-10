import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical
from tqdm import tqdm
import gymnasium as gym
from collections import deque
import random
import tomllib as toml

# from src.utils.replay_buffer import ReplayBuffer


torch.autograd.set_detect_anomaly(True)
device = torch.device("cuda:0" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")) 

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.stack, zip(*batch))
        return torch.from_numpy(states).to(device),\
            torch.from_numpy(actions).to(device),\
            torch.from_numpy(rewards).to(device),\
            torch.from_numpy(next_states).to(device),\
            torch.from_numpy(dones).to(device)

    def __len__(self):
        return len(self.buffer)


class EpisodeBuffer:
    def __init__(self, gamma):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.next_states = []
        self.rewards_to_go = []
        self.gamma = gamma

    def add(self, state, action, reward, next_state, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)

    def get(self):
        # Here an episode ends
        self.rewards_to_go = [0 for _ in range(len(self.rewards))]
        for i in range(-1, -len(self.rewards) - 1, -1):
            if i == -1:
                self.rewards_to_go[i] = self.rewards[i]
            else:
                self.rewards_to_go[i] = self.rewards[i] + self.gamma * self.rewards_to_go[i + 1]
        return torch.from_numpy(np.array(self.states)).to(device),\
            torch.from_numpy(np.array(self.actions)).to(device),\
            torch.from_numpy(np.array(self.rewards)).to(device),\
            torch.from_numpy(np.array(self.next_states)).to(device),\
            torch.from_numpy(np.array(self.dones)).to(device),\
            torch.from_numpy(np.array(self.rewards_to_go)).to(device)

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.next_states = []


class Policy(nn.Module):
    def __init__(self, input_dim, output_num, hidden_size=32):
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
    def __init__(self, input_dim, hidden_size=32):
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


class VanillaActorCritic:
    def __init__(self,
        env,
        gamma=0.7,
        value_lr=0.001,
        policy_lr=0.001,
        buffer_size=10000,
        batch_size=32,
        training_start=10000,
        info_step=1000,
    ):
        self.env = env
        self.gamma = gamma
        self.value_lr = value_lr
        self.policy_lr = policy_lr
        self.buffer = EpisodeBuffer(self.gamma)
        #self.replay_buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.training_start = training_start
        self.info_step = info_step

        # Value network
        self.value = Value(self.env.observation_space.shape[0]).to(device)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=value_lr)

        # Policy network
        self.policy = Policy(self.env.observation_space.shape[0], self.env.action_space.n).to(device)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=policy_lr)

        # Tensorboard Writer
        self.writer = SummaryWriter()

    def select_action(self, state):
        action_probs = self.policy(state)
        sample = torch.multinomial(action_probs, 1)
        return sample[0].item()

    def update_value(self, batch):
        states, actions, rewards, next_states, dones, rewards_to_go = batch

        # Calculate value function
        #with torch.no_grad():
        #    target_values = rewards.unsqueeze(1) + self.gamma * self.value(next_states) * ((1 - dones).unsqueeze(1))
        values = self.value(states)
        ## Calculate REINFORCE value with baseline
        #with torch.no_grad():
        #    delta = target_values - values

        # Calculate loss
        loss = F.smooth_l1_loss(values, rewards_to_go.unsqueeze(1))

        # Update value network
        self.value_optimizer.zero_grad()
        loss.backward()
        self.value_optimizer.step()
        return loss

    def update_policy(self, batch):
        states, actions, rewards, next_states, dones, rewards_to_go = batch
        delta = rewards.unsqueeze(1) + self.gamma * self.value(next_states) * ((1 - dones).unsqueeze(1)) - self.value(states)
        probs = self.policy(states)
        self.policy_optimizer.zero_grad()
        m = Categorical(probs)
        performance = torch.mean(-m.log_prob(actions).unsqueeze(1) * delta)
        performance.backward()
        self.policy_optimizer.step()
        return performance, delta

    def train(self, episodes=10000):
        #training_step = -self.training_start
        for episode in tqdm(range(episodes), desc="Training episode"):
            state, _ = self.env.reset()
            done = False
            step = 0
            episodic_reward = 0
            while not done:
                action = self.select_action(torch.from_numpy(state).to(device))
                next_state, reward, done, truncation, _ = self.env.step(action)
                episodic_reward += reward
                self.buffer.add(state, action, reward, next_state, 1 if done else 0)
                #self.replay_buffer.add(state, action, reward, next_state, 1 if done else 0)
                state = next_state
                done = done or truncation
                step += 1
                #if training_step > 0:
            batch = self.buffer.get()
            performance, delta = self.update_policy(batch)
            loss = self.update_value(batch)
            self.writer.add_scalar('training/loss_value', loss, episode)
            self.writer.add_scalar('training/policy_performance', performance, episode)
            self.writer.add_scalar('training/delta', torch.mean(delta), episode)
            self.writer.add_scalar('training/episodic_reward', episodic_reward, episode)
            if episode % self.info_step == 0:
                tqdm.write(f'Episode {episode}, episode reward: {episodic_reward}, loss {loss}, delta {torch.mean(delta).item()}')
            self.buffer.clear()


class ActorCritic:
    def __init__(self, env):
        self.env = env
        with open("config/config.toml", "rb") as f:
            config = toml.load(f)
        config = config["cartpole"]["actor_critic"]
        for key, value in config.items():
            setattr(self, key, value)

        self.policy = Policy(self.env.observation_space.shape[0], self.env.action_space.n).to(device)
        self.critic = Value(self.env.observation_space.shape[0]).to(device)

        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        self.writer = SummaryWriter()

    def select_action(self, state):
        with torch.no_grad():
            action_probs = self.policy(state)
            sample = torch.multinomial(action_probs, 1)
        return sample[0].item()

    def update_policy(self, exp, delta):
        state, action, reward, done, next_state = exp
        self.policy_optimizer.zero_grad()
        probs = self.policy(state)
        m = Categorical(probs)
        performance = -m.log_prob(action) * delta
        performance.backward()
        self.policy_optimizer.step() 
        return performance

    def update_value(self, exp):
        state, action, reward, done, next_state = exp
        self.critic_optimizer.zero_grad()
        value = self.critic(state)
        next_value = self.critic(next_state)
        target = reward + self.gamma * next_value * (1 - done)
        delta = target - value
        loss = F.smooth_l1_loss(value, target)
        loss.backward()
        self.critic_optimizer.step()
        return loss, delta.item()

    def train(self, episodes=10000):
        training_step = 0
        for episode in tqdm(range(episodes), desc="Training"):
            state, _ = self.env.reset()
            episodic_reward = 0
            step = 0
            done = False
            while not done:
                action = self.select_action(torch.from_numpy(state).to(device))
                next_state, reward, done, truncation, _ = self.env.step(action)
                episodic_reward += reward
                exp = (torch.from_numpy(state).to(device),
                        torch.tensor(action).to(device),
                        torch.tensor(reward).to(device),
                        torch.tensor(1 if done else 0).to(device),
                        torch.from_numpy(next_state).to(device))
                loss, delta = self.update_value(exp)
                performance = self.update_policy(exp, delta)
                training_step += 1
                step += 1
                done = done or truncation
            self.writer.add_scalar("training/loss_value", loss, episode)
            self.writer.add_scalar("training/performance", performance, episode)
            self.writer.add_scalar("training/delta", delta, episode)
            self.writer.add_scalar("training/episodic_reward", episodic_reward, episode)
            if episode % self.info_step == 0:
                tqdm.write(f"Episode: {episode} | Reward: {episodic_reward} | Step: {step} | Loss: {loss} | Performance: {performance} | Delta: {delta}") 


def main():
    env = gym.make('CartPole-v1')
    #agent = VanillaActorCritic(env)
    agent = ActorCritic(env)
    agent.train()
