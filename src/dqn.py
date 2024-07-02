import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque
from tqdm import tqdm


class QValue(nn.Module):
    def __init__(self, state_dim, action_num, hidden_dim=64):
        super().__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_num)
        self.model = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            self.fc2,
            nn.ReLU(),
            self.fc3
        )

    def forward(self, x):
        y = self.model(x)
        return y


class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)

    def add(self, state, action, reward, next_state, terminate):
        experience = (state, action, reward, next_state, terminate)
        self.buffer.append(experience)

    def sample(self, batch_size):
        sample = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, terminates = zip(*sample)
        return torch.tensor(states), torch.tensor(actions), torch.tensor(rewards), torch.tensor(next_states), torch.tensor(terminates)


class DQN:
    def __init__(self, state_dim, action_num, hidden_dim=64, lr=0.001, gamma=0.99, epsilon=0.1, batch_size=32, buffer_size=10000, target_update_freq=100, info_freq=1000):
        self.state_dim = state_dim
        self.action_num = action_num
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.target_update_freq = target_update_freq
        self.info_freq = info_freq
        # Replay Buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        # Q-Network
        self.q_network = QValue(state_dim, action_num, hidden_dim)
        # Target Q-Network
        self.target_q_network = QValue(state_dim, action_num, hidden_dim)
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

    def select_action(self, state):
        # Epsilon-Greedy
        if random.random() < self.epsilon:
            return random.randint(0, self.action_num - 1)
        else:
            with torch.no_grad():
                state = torch.tensor(state, dtype=torch.float32)
                q_values = self.q_network(state)
                return torch.argmax(q_values).item()

    def update(self):
        # Sample from replay buffer
        states, actions, rewards, next_states, terminates = self.replay_buffer.sample(self.batch_size)
        # Compute target Q-values
        with torch.no_grad():
            target_Q = self.target_q_network(next_states).max(1)[0].unsqueeze(1)
            target_Q = rewards + self.gamma * target_Q * (1 - terminates)
        # Compute Q-values
        q_values = self.q_network(states).gather(1, actions)
        # Compute loss
        loss = F.smooth_l1_loss(q_values, target_Q)
        # Update Q-network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        # Update target Q-network
        self.target_q_network.load_state_dict(self.q_network.state_dict())

    def train(self, env, episodes, start_training_steps=10000):
        # CTDE
        rewards = []
        training_step = -start_training_steps
        for episode in tqdm(range(episodes), desc="Episode", position=0):
            env.reset()
            termination = False
            step = 0
            # Metrics
            episodic_reward = 0
            agent_rewards = dict.fromkeys(env.possible_agents, 0)
            # Collect experience
            while not termination:
                for agent in env.agent_iter():
                    observation, reward, termination, truncation, info = env.last()
                    agent_rewards[agent] += reward
                    episodic_reward += reward
                    if termination or truncation:
                        action = None
                    else:
                        action = self.select_action(observation)
                    env.step(action)
                    next_state = env.observe(agent)
                    termination = termination or truncation
                    # Add transition to replay buffer
                    self.replay_buffer.add(observation, action, reward, next_state, termination)
                    step += 1
                    training_step += 1
                    if training_step > 0:
                        self.update()
                        if training_step % self.target_update_freq == 0:
                            self.update_target()
                    if training_step % self.info_freq == 0:
                        tqdm.write(f"Training Step: {training_step}, Epsilon: {self.epsilon}, Reward: {episodic_reward}")
    


if __name__ == "__main__":
    from pettingzoo.mpe import simple_spread_v3
    env = simple_spread_v3.env(render_mode=None)
    agent_num = env.possible_agents
    state_dim = env.observation_space(env.possible_agents[0]).shape[0]
    action_num = env.action_space(env.possible_agents[0]).n

    dqn = DQN(state_dim=state_dim, action_num=action_num)
    dqn.train(env, episodes=1000)