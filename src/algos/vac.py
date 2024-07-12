from tqdm import tqdm

from src.networks import Value, DiscretePolicy as Policy
from src.utils import EpisodeBuffer

import torch
import torch.optim as optim
from torch.distributions import Categorical
from torch.functional import F
from src.algos.algo import Algo


class VanillaActorCritic(Algo):
    def __init__(self, env_name, device="cpu"):
        super().__init__(env_name, device)
        
        self.buffer = EpisodeBuffer(self.gamma, device=self.device)

        # Value network
        self.value = Value(self.env.observation_space.shape[0]).to(self.device)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=self.value_lr)

        # Policy network
        self.policy = Policy(self.env.observation_space.shape[0], self.env.action_space.n).to(self.device)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=self.policy_lr)

    def select_action(self, state):
        with torch.no_grad():
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
                action = self.select_action(torch.from_numpy(state).to(self.device))
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
