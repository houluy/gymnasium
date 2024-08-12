import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal
from tqdm import tqdm
import gymnasium as gym
import numpy as np
from src.algos.algo import Algo
from src.networks import DiscretePolicy, ContinuousPolicy, Value

# device = torch.device("cuda:0" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")) 


class ActorCritic(Algo):
    def __init__(self, env, continuous=False, device=torch.device("cpu")):
        super().__init__(env, continuous, device)
        Policy = ContinuousPolicy if self.continuous else DiscretePolicy
        self.policy = Policy(self.state_dim, self.action_dim).to(self.device)
        self.critic = Value(self.state_dim).to(self.device)

        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        self.noise_std = torch.from_numpy(np.ones((self.action_dim, )) * self.start_noise_std).to(self.device)

    def select_action_continuous(self, state, train=True):
        with torch.no_grad():
            action_probs = self.policy(state)
            sample = torch.normal(action_probs, self.noise_std)
        return sample.cpu()

    def select_action_discrete(self, state, train=True):
        with torch.no_grad():
            action_probs = self.policy(state)
            sample = torch.multinomial(action_probs, 1)
        return sample[0].item()

    def decay_noise(self):
        self.noise_std *= self.noise_std_decay_factor
        return self.noise_std

    def evaluate(self, episodes=100):
        rewards = super().evaluate(episodes)
        return rewards

    def update_policy(self, exp):
        state, action, reward, done, next_state = exp
        for p in self.critic.parameters():
            p.requires_grad = False
        self.policy_optimizer.zero_grad()
        value = self.critic(state)
        next_value = self.critic(next_state)
        delta = reward + self.gamma * next_value * (1 - done) - value
        probs = self.policy(state)
        if self.continuous:
            m = Normal(probs, self.noise_std)
        else:
            m = Categorical(probs)
        performance = -m.log_prob(action) * delta
        performance.backward()
        self.policy_optimizer.step() 
        for p in self.critic.parameters():
            p.requires_grad = True
        return performance, delta

    def update_value(self, exp):
        state, action, reward, done, next_state = exp
        self.critic_optimizer.zero_grad()
        value = self.critic(state)
        next_value = self.critic(next_state)
        target = reward + self.gamma * next_value * (1 - done)
        loss = F.smooth_l1_loss(value, target)
        loss.backward()
        self.critic_optimizer.step()
        return loss

    def train(self, episodes=50000):
        training_step = 0
        if self.continuous:
            action_dist = []
        for episode in tqdm(range(episodes), desc="Training"):
            state, _ = self.env.reset()
            episodic_reward = 0
            step = 0
            done = False
            while not done:
                action = self.select_action(torch.from_numpy(state).to(self.device))
                if self.continuous:
                    action_dist.append(action)
                next_state, reward, done, truncation, _ = self.env.step(action)
                episodic_reward += reward
                exp = (torch.from_numpy(state).to(self.device),
                        torch.tensor(action).to(self.device),
                        torch.tensor(reward).to(self.device),
                        torch.tensor(1 if done else 0).to(self.device),
                        torch.from_numpy(next_state).to(self.device))
                performance, delta = self.update_policy(exp)
                loss = self.update_value(exp)
                training_step += 1
                step += 1
                done = done or truncation
            self.writer.add_scalar("training/loss_value", loss, episode)
            self.writer.add_scalar("training/performance", performance, episode)
            self.writer.add_scalar("training/delta", delta, episode)
            self.writer.add_scalar("training/episodic_reward", episodic_reward, episode)
            self.writer.add_scalar("training/episode_length", step, episode)
            self.writer.add_scalar("training/noise_std", self.noise_std, episode)
            if episode % self.info_episode == 0:
                tqdm.write(f"Episode: {episode} | Reward: {episodic_reward} | Step: {step} | Loss: {loss} | Performance: {performance.item()} | Delta: {delta.item()}") 
            if episode % self.noise_std_decay_freq_episode == 0:
                self.decay_noise()
        if self.continuous:
            self.writer.add_histogram("training/action_dist", np.array(action_dist), 1, bins="auto")
