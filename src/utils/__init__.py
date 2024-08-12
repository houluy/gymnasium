from collections import deque
import random
import torch
import numpy as np


class ReplayBuffer:
    def __init__(self, buffer_size, device=torch.device("cpu"), action_type="discrete"):
        self.buffer_size = buffer_size
        self.device = device
        self.action_dtype = np.int64 if action_type == "discrete" else np.float32
        self.buffer = deque(maxlen=buffer_size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        #for i in range(len(batch)):
        #    print(i, len(batch[i]), end=" ")
        #    for item in batch[i]:
        #        try:
        #            print(item.shape, end=" ")
        #        except:
        #            print(item, end=" ")
        #    print()
        states, actions, rewards, next_states, dones = map(np.stack, zip(*batch))
        return torch.from_numpy(states).to(self.device),\
            torch.from_numpy(np.array(actions, dtype=self.action_dtype)).to(self.device),\
            torch.from_numpy(rewards).to(self.device),\
            torch.from_numpy(next_states).to(self.device),\
            torch.from_numpy(dones).to(self.device)

    def __len__(self):
        return len(self.buffer)


class EpisodeBuffer:
    def __init__(self, gamma, device="cpu"):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.next_states = []
        self.rewards_to_go = []
        self.gamma = gamma
        self.device = device

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
        return torch.from_numpy(np.array(self.states)).to(self.device),\
            torch.from_numpy(np.array(self.actions)).to(self.device),\
            torch.from_numpy(np.array(self.rewards)).to(self.device),\
            torch.from_numpy(np.array(self.next_states)).to(self.device),\
            torch.from_numpy(np.array(self.dones)).to(self.device),\
            torch.from_numpy(np.array(self.rewards_to_go)).to(self.device)

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.next_states = []

