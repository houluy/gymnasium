from src.algos.algo import Algo
from src.utils import ReplayBuffer
from src.networks import DiscreteQ as Q
from src.environments.simple_spread import SimpleSpread
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch
import random
from tqdm import tqdm
from torch.functional import F
import itertools
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


class DQN(Algo):
    def __init__(self, env_name=None, device="cpu", continuous=False):
        super().__init__(env_name, device=device, continuous=continuous)
        self.epsilon = self.start_epsilon = 1
        self.end_epsilon = 0.1
        self.replay_buffer = ReplayBuffer(self.buffer_size, device=device)
        self.q = Q(self.state_dim, self.action_dim, self.hidden_size).to(device)
        self.target_network = Q(self.state_dim, self.action_dim, self.hidden_size).to(device)
        self.optimizer = optim.Adam(self.q.parameters(), lr=self.lr, weight_decay=0.01)
        self.update_target_network()

    def epsilon_decay(self):
        self.epsilon *= self.epsilon_decay_factor
        return self.epsilon

    def save(self):
        torch.save(self.q.state_dict(), self.model_path)

    def load(self):
        self.q.load_state_dict(torch.load(self.model_path))

    def select_action_discrete(self, state, train=True, agent=None):
        if (train) and (random.random() < self.epsilon):
            if agent is None:
                action_space = self.env.action_space
            else:
                action_space = self.env.action_space(agent)
            return action_space.sample()
        else:
            return self.q(state).argmax().item()

    select_action_continuous = select_action_discrete

    def update_target_network(self):
        self.target_network.load_state_dict(self.q.state_dict())

    def update_value(self, batch):
        states, actions, rewards, next_states, dones = batch
        y = rewards + self.gamma * self.target_network(next_states).max(1)[0] * (1 - dones)
        q = self.q(states).gather(1, actions.unsqueeze(1))
        loss = F.mse_loss(q, y.unsqueeze(1).float())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def evaluate(self, episodes=100):
        rewards = super().evaluate(episodes)
        return rewards

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
                        self.update_target_network()
            self.writer.add_scalar("training/episodic_reward", episodic_reward, episode)



class IDQN:
    # Independent Q Networks
    def __init__(self, env: SimpleSpread, env_name, state_dim, action_dim, buffer_size, lr, gamma, device):
        self.agent_list = env.agent_list
        self.agent_num = env.agent_num
        self.env = env
        self.qs = {agent:Q(state_dim, action_dim, hidden_size=128).to(device) for agent in self.agent_list}
        self.target_networks = {agent:Q(state_dim, action_dim, hidden_size=128).to(device) for agent in self.agent_list}
        self.replay_buffer = {agent:ReplayBuffer(buffer_size, device=device) for agent in self.agent_list}
        self.optimizer = {agent:torch.optim.Adam(self.qs[agent].parameters(), lr=lr) for agent in self.agent_list}
        self.gamma = gamma
        self.epsilon = self.start_epsilon = 1
        self.end_epsilon = 0.1
        self.batch_size = 32
        self.info_step = 1000
        self.target_update_step = 10
        self.epsilon_decay_step = 200
        self.hidden_size = 128
        self.training_start = 2000
        self.training_interval = 2  # Update the model every training_interval steps
        self.epsilon_decay_factor = 0.999
        self.device = device
        self.lr = lr
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.buffer_size = buffer_size
        self.writer = SummaryWriter()
        self.model_path = f"models/{env_name}/{self.__class__.__name__}.pt"
        self.update_target_network()

    def epsilon_decay(self):
        self.epsilon *= self.epsilon_decay_factor
        self.epsilon = max(self.epsilon, self.end_epsilon)
        return max(self.epsilon, self.end_epsilon)

    def save(self, agent):
        torch.save(self.qs[agent].state_dict(), self.model_path)

    def load(self, agent):
        self.qs[agent].load_state_dict(torch.load(self.model_path))

    def select_action(self, state, agent: str, train=True):
        if (train) and (random.random() < self.epsilon):
            return random.randint(0, self.action_dim - 1)#self.env.action_space(agent)
        else:
            return self.qs[agent](state).argmax().item()

    def update_target_network(self):
        for agent in self.agent_list:
            self.target_networks[agent].load_state_dict(self.qs[agent].state_dict())

    def update_value(self, batch, agent: str):
        states, actions, rewards, next_states, dones = batch
        for oagent in self.agent_list:
            if oagent != agent:
                for p in self.qs[oagent].parameters():
                    p.requires_grad = False
        y = rewards + self.gamma * self.target_networks[agent](next_states).max(1)[0] * (1 - dones)
        q = self.qs[agent](states).gather(1, actions.unsqueeze(1))
        loss = F.mse_loss(q, y.unsqueeze(1).float())
        self.optimizer[agent].zero_grad()
        loss.backward()
        self.optimizer[agent].step()
        for oagent in self.agent_list:
            if oagent != agent:
                for p in self.qs[oagent].parameters():
                    p.requires_grad = True
        return loss

    def train(self, episodes=10000):
        training_step = -self.training_start
        real_training_step = 0
        total_occupied_landmarks = 0
        total_collisions = 0
        for episode in tqdm(range(episodes), desc="Training"):
            observations, infos = self.env.reset()
            for agent in self.agent_list:
                observation = self.env._remove_comm_feature(observations[agent])
                observations[agent] = observation
            step = 0
            episodic_reward = 0
            dones = {agent: False for agent in self.agent_list}
            episodic_reward_per_agent = dict.fromkeys(self.agent_list, 0)
            while self.env.agents:
                actions = dict.fromkeys(self.agent_list, 0)
                for agent in self.agent_list:
                    action = self.select_action(torch.from_numpy(observations[agent]).to(self.device), agent=agent)
                    actions[agent] = action
                next_states, rewards, terminations, truncations, infos = self.env.step(actions)
                for agent in self.agent_list:
                    next_state = self.env._remove_comm_feature(next_states[agent])
                    next_states[agent] = next_state
                    reward = rewards[agent]
                    done = terminations[agent] or truncations[agent]
                    episodic_reward_per_agent[agent] += reward
                    self.replay_buffer[agent].add(observations[agent], actions[agent], reward, next_state, 1 if done else 0)
                average_dist, occupied_landmarks, collisions = self.env.benchmark_data(next_states)
                total_occupied_landmarks += occupied_landmarks
                total_collisions += collisions
                #next_states, *_ = self.env.local2global_observation_simple(next_states)
                episodic_reward += sum(rewards.values())/self.agent_num
                done = any(terminations) or any(truncations)
                observations = next_states
                step += 1
                training_step += 1
                if training_step >= 0 and (training_step % self.training_interval == 0):
                    losses = dict.fromkeys(self.agent_list, 0)
                    for agent in self.agent_list:
                        loss = self.update_value(self.replay_buffer[agent].sample(self.batch_size), agent=agent)
                        loss = loss.cpu()
                        losses[agent] = loss.item()
                        self.writer.add_scalar(f"training/loss_value/{agent}", loss, real_training_step)
                    self.writer.add_scalar("training/epsilon", self.epsilon, real_training_step)
                    self.writer.add_scalar("training/benchmark/average_dist", average_dist, real_training_step)
                    if real_training_step % self.epsilon_decay_step == 0:
                        self.epsilon_decay()
                    #if real_training_step % self.info_step == 0:
                    #    tqdm.write(f"Episode {episode}, Training step {real_training_step}, Loss: {loss}")
                    
                    #if real_training_step % self.target_update_step == 0:
                    #    self.update_target_network()
                    real_training_step += 1
            self.writer.add_scalar("training/episodic_reward", episodic_reward, episode)
            self.writer.add_scalar("training/episode_length", step, episode)
        #self.writer.add_histogram("training/action_dist/global", torch.tensor(action_dist))
        #for agent, reward_dist in agent_reward_dist.items():
        #    self.writer.add_histogram(f"training/action_dist/{agent}", torch.tensor(action_dist))
        #    self.writer.add_histogram(f"training/reward_dist/{agent}", torch.tensor(reward_dist))
        print(total_occupied_landmarks, total_collisions, training_step + self.training_start)

# MARL
class MADQN:
    def __init__(self, env: SimpleSpread, env_name, state_dim, action_dim, buffer_size, lr, gamma, device):
        self.env = env
        self.env_name = env_name
        self.epsilon = self.start_epsilon = 1
        self.end_epsilon = 1
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.buffer_size = buffer_size
        self.lr = lr
        self.gamma = gamma
        self.batch_size = 32
        self.info_step = 1000
        self.target_update_step = 10
        self.epsilon_decay_step = 200
        self.hidden_size = 128
        self.training_start = 20000
        self.training_interval = 2  # Update the model every training_interval steps
        self.epsilon_decay_factor = 0.999
        self.communication_feature_num = 4  # MPE Spread Env Only
        self.agent_lists = self.env.agents[:]
        self.agent_num = self.env.agent_num

        self.device = device
        self.replay_buffer = ReplayBuffer(self.buffer_size, device=self.device)
        self.writer = SummaryWriter()
        self.model_path = f"models/{env_name}/{self.__class__.__name__}.pt"
        self.q = Q(self.state_dim, self.action_dim, self.hidden_size).to(self.device)
        self.target_network = Q(self.state_dim, self.action_dim, self.hidden_size).to(self.device)
        self.optimizer = optim.Adam(self.q.parameters(), lr=self.lr, weight_decay=0.01)
        self.update_target_network()

    def epsilon_decay(self):
        self.epsilon *= self.epsilon_decay_factor
        self.epsilon = max(self.epsilon, self.end_epsilon)
        return max(self.epsilon, self.end_epsilon)

    def save(self):
        torch.save(self.q.state_dict(), self.model_path)

    def load(self):
        self.q.load_state_dict(torch.load(self.model_path))

    def select_action(self, state, train=True):
        if (train) and (random.random() < self.epsilon):
            return random.randint(0, self.action_dim - 1)#self.env.action_space(agent)
        else:
            return self.q(state).argmax().item()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q.state_dict())

    def update_value(self, batch):
        states, actions, rewards, next_states, dones = batch
        y = rewards + self.gamma * self.target_network(next_states).max(1)[0] * (1 - dones)
        q = self.q(states).gather(1, actions.unsqueeze(1))
        loss = F.mse_loss(q, y.unsqueeze(1).float())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def train_ctde(self, episodes=10000):
        training_step = -self.training_start
        for episode in tqdm(range(episodes), desc="Training"):
            self.env.reset()
            agent_counter = 0 # Equal to step * agent_num
            step = 0 # Equal to agent_counter // agent_num
            episodic_reward = 0 # Globally shared reward
            agent_num = self.env.num_agents
            last_dic = {}  # Store the last info of each agent: (last_state, last_action), Current state add a next_state, a reward and a done in it to form the experience item
            # 1. initial state from env.last()
            # 2. select action from policy network
            # 3. next step, add reward, termination or truncation, observation to the experience pool
            # 4. if termination or truncation is True, then action is None
            for agent in self.env.agent_iter():
                observation, reward, termination, truncation, info = self.env.last()
                # Remove the communication features in observation
                observation = observation[:-self.communication_feature_num]
                episodic_reward += reward
                agent_done = termination or truncation
                if agent_done:
                    action = None
                else:
                    action = self.select_action(torch.from_numpy(observation).to(self.device), agent=agent)
                if last_dic.get(agent) is not None:
                    last_dic[agent].extend([reward, observation, 1 if agent_done else 0])
                    self.replay_buffer.add(*last_dic[agent])
                last_dic[agent] = [observation, action]
                self.env.step(action)
                agent_counter += 1
                step += agent_counter % agent_num
                training_step += agent_counter % agent_num
                if training_step >= 0:
                    #loss = self.update_value(self.replay_buffer.sample(self.batch_size))
                    #self.writer.add_scalar("training/loss_value", loss, training_step)
                    self.writer.add_scalar("training/epsilon", self.epsilon, training_step)
                    if training_step % self.epsilon_decay_step == 0:
                        self.epsilon_decay()
                    if training_step % self.info_step == 0:
                        tqdm.write(f"Episode {episode}, Training step {training_step}, Loss: {loss}")
                    #if training_step % self.target_update_step == 0:
                    #    self.update_target_network()
            self.writer.add_scalar("training/episodic_reward", episodic_reward, episode)

    def _observation_process(self, observation):
        # MPE Simple Spread: Remove the communication features in observation
        return observation[:-self.communication_feature_num]

    def _global_state(self, observations):
        return torch.cat([torch.from_numpy(self._observation_process(observation)) for agent, observation in observations.items()])

    def train_centralized(self, episodes=10000):
        training_step = -self.training_start
        real_training_step = 0
        action_dist = []
        agent_action_dist = dict.fromkeys(self.agent_lists, [])
        agent_reward_dist = dict.fromkeys(self.agent_lists, [])
        total_occupied_landmarks = 0
        total_collsions = 0
        for episode in tqdm(range(episodes), desc="Training"):
            observations, infos = self.env.reset()
            observations, *_ = self.env.local2global_observation_simple(observations)
            step = 0
            episodic_reward = 0
            dones = {agent: False for agent in self.agent_lists}
            while self.env.agents:
                raw_action = self.select_action(torch.from_numpy(observations).to(self.device))
                action_dist.append(raw_action)
                actions = self.env.discrete_action_assignment(
                    raw_action
                )
                #for agent, action in actions.items():
                #    agent_action_dist[agent].append(action)
                next_states, rewards, terminations, truncations, infos = self.env.step(actions)
                for agent, reward in rewards.items():
                    agent_reward_dist[agent].append(reward)
                average_dist, occupied_landmarks, collisions = self.env.benchmark_data(next_states)
                total_occupied_landmarks += occupied_landmarks
                total_collsions += collisions
                next_states, *_ = self.env.local2global_observation_simple(next_states)
                reward = sum(rewards.values())/self.agent_num
                episodic_reward += reward
                done = any(terminations) or any(truncations)
                self.replay_buffer.add(observations, raw_action, reward, next_states, 1 if done else 0)
                observations = next_states
                step += 1
                training_step += 1
                if training_step >= 0 and (training_step % self.training_interval == 0):
                    #loss = self.update_value(self.replay_buffer.sample(self.batch_size))
                    #loss = loss.cpu()
                    #self.writer.add_scalar("training/loss_value", loss, real_training_step)
                    self.writer.add_scalar("training/epsilon", self.epsilon, real_training_step)
                    self.writer.add_scalar("training/benchmark/average_dist", average_dist, real_training_step)
                    if real_training_step % self.epsilon_decay_step == 0:
                        self.epsilon_decay()
                    #if real_training_step % self.info_step == 0:
                    #    tqdm.write(f"Episode {episode}, Training step {real_training_step}, Loss: {loss}")
                    
                    #if real_training_step % self.target_update_step == 0:
                    #    self.update_target_network()
                    real_training_step += 1
            self.writer.add_scalar("training/episodic_reward", episodic_reward, episode)
            self.writer.add_scalar("training/episode_length", step, episode)
        #self.writer.add_histogram("training/action_dist/global", torch.tensor(action_dist))
        for agent, reward_dist in agent_reward_dist.items():
        #    self.writer.add_histogram(f"training/action_dist/{agent}", torch.tensor(action_dist))
            self.writer.add_histogram(f"training/reward_dist/{agent}", torch.tensor(reward_dist))
        print(total_occupied_landmarks, total_collsions, training_step + self.training_start)

