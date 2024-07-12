from src.algos.algo import Algo
from src.utils import ReplayBuffer
from src.networks import DiscreteQ as Q
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch
import random
from tqdm import tqdm
from torch.functional import F


class DQN(Algo):
    def __init__(self, env_name, device="cpu"):
        super().__init__(env_name, device)
        self.epsilon = self.start_epsilon = 1
        self.end_epsilon = 0.1
        self.replay_buffer = ReplayBuffer(self.buffer_size, device=device)
        self.q = Q(self.env.observation_space.shape[0], self.env.action_space.n, hidden_size=256).to(device)
        self.target_network = Q(self.env.observation_space.shape[0], self.env.action_space.n, hidden_size=256).to(device)
        self.optimizer = optim.Adam(self.q.parameters(), lr=self.lr)

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
        loss = F.mse_loss(q, y.unsqueeze(1).float())
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
                action = self.select_action(torch.from_numpy(state).to(self.device))
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
                        tqdm.write(f"Episode {episode}, Training step {training_step}, Loss: {loss}")
                    if training_step % self.target_update_step == 0:
                        self.target_network.load_state_dict(self.q.state_dict())
            self.writer.add_scalar("training/episodic_reward", episodic_reward, episode)