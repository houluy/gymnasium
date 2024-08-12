from tqdm import tqdm

from src.networks import Value, DiscretePolicy, ContinuousPolicy, ContinuousPolicyWithStd
from src.utils import EpisodeBuffer

import torch
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical, Normal
from torch.functional import F
from src.algos.algo import Algo


class VanillaActorCritic(Algo):
    def __init__(self, env_name, continuous=False, device=torch.device("cpu")):
        super().__init__(env_name, continuous, device)
        
        self.buffer = EpisodeBuffer(self.gamma, device=self.device)

        # Value network
        self.value = Value(self.state_dim).to(self.device)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=self.value_lr)

        # Policy network
        Policy = DiscretePolicy if not self.continuous else (ContinuousPolicy if not self.policy_with_std else ContinuousPolicyWithStd)
        if not self.policy_with_std:
            self.policy = Policy(self.state_dim, self.action_dim).to(self.device)
            self.noise_std = torch.from_numpy(np.ones((self.action_dim,)) * self.start_noise_std).to(self.device)
        else:
            self.policy = Policy(self.state_dim, self.action_dim).to(self.device)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=self.policy_lr)

    def select_action_discrete(self, state, train=True):
        with torch.no_grad():
            action_probs = self.policy(state)
            sample = torch.multinomial(action_probs, 1)
        return sample[0].item()

    def select_action_continuous(self, state, train=True):
        with torch.no_grad():
            action_probs = self.policy(state.unsqueeze(0))
            if self.policy_with_std:
                mean, std = action_probs[:, :self.action_dim], action_probs[:, self.action_dim:]
            else:
                mean, std = action_probs, self.noise_std
            sample = torch.normal(mean, std)
        return sample.squeeze(0).cpu()

    def evaluate(self, episodes=100):
        rewards = super().evaluate(episodes)
        return rewards

    def decay_noise(self):
        self.noise_std *= self.noise_std_decay_factor
        return self.noise_std

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
        for p in self.value.parameters():
            p.requires_grad = False
        delta = rewards.unsqueeze(1) + self.gamma * self.value(next_states) * ((1 - dones).unsqueeze(1)) - self.value(states)
        probs = self.policy(states)
        self.policy_optimizer.zero_grad()
        if self.continuous:
            if self.policy_with_std:
                mean = probs[:, :self.action_dim]
                std = probs[:, self.action_dim:]
            else:
                mean = probs
                std = self.noise_std
            m = Normal(mean, scale=std)
        else:
            m = Categorical(probs)
        performance = torch.mean(-m.log_prob(actions).unsqueeze(1) * delta)
        performance.backward()
        self.policy_optimizer.step()
        for p in self.value.parameters():
            p.requires_grad = True
        return performance, delta

    def train(self, episodes=10000):
        #training_step = -self.training_start
        if self.continuous:
            action_dist = []
        for episode in tqdm(range(episodes), desc="Training episode"):
            state, _ = self.env.reset()
            done = False
            step = 0
            episodic_reward = 0
            while not done:
                action = self.select_action(torch.from_numpy(state).to(self.device))
                if self.continuous:
                    action_dist.append(action)
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
            self.writer.add_scalar("training/episode_length", step, episode)
            if self.continuous:
                if not self.policy_with_std:
                    self.writer.add_scalar("training/noise_std", self.noise_std, episode)
                    if episode and (episode % self.noise_std_decay_freq_episode == 0):
                        self.decay_noise()
            if episode % self.info_step == 0:
                tqdm.write(f'Episode {episode}, episode reward: {episodic_reward}, loss {loss}, delta {torch.mean(delta).item()}')
            self.buffer.clear()
        self.writer.add_histogram("training/action_dist", np.array(action_dist), 1, bins="auto")



class VanillaActorCritic(Algo):
    def __init__(self, env_name, continuous=False, device=torch.device("cpu")):
        super().__init__(env_name, continuous, device)


    def select_action_continuous(self, state, train=True):
         
        return 

