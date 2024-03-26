import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from collections import namedtuple
from typing import List
#from tensorboard import SummaryWriter


env_pool_sample = namedtuple("env_pool_sample", ["state", "action", "reward"])
#writer = SummaryWriter("./logs")


class ParamPi:
    def __init__(self, state_space):
        self.state_space = state_space
        self.action_space = 2
        self.all_actions_onehot = np.array([[1, 0], [0, 1]])
        self.all_actions = np.array([0, 1])
        self.noise = 0.1
        self.alpha = 0.01
        self.gamma = 1

        # Feature matrix c
        self.k = self.state_space + self.action_space
        self.n = 1
        self.feature_dim = (self.n + 1) ** self.k
        self.parameters = np.random.random((self.feature_dim, ))
        self.c = feature_mat_c(self.k, self.n)


    def feature_eng(self, s, a):
        '''xi = cos(π<s, a> ci)
            i ∈ [0, (n + 1)^k]
        '''
        sa = np.concatenate((s, a))
        assert sa.shape == (self.k, )
        feature = np.array([
            np.cos(np.pi * np.dot(sa, cij)) for cij in self.c
        ])
        assert feature.shape == (self.feature_dim, )
        return feature
    
    def action_dist(self, state):
        all_as = []
        for a in self.all_actions_onehot:
            fsa = self.feature_eng(state, a)
            all_as.append(np.dot(fsa, self.parameters))
        return self.softmax(all_as)

    def action(self, state):
        action_props = self.action_dist(state)
        return np.random.choice(self.all_actions, p=action_props)
    
    def action_by_norm(self, mu, sigma):
        return 

    def action_noise(self, state):
        return self.action(state) + np.random.normal(0, self.noise, (self.action_space, 1))

    def nabla(self, state, action):
        return np.array([state[0], state[1]]) / action
    
    def update(self, env_pool: List[env_pool_sample]):
        last_gt = env_pool[-1].reward
        T = len(env_pool)
        for ind in range(T - 1):
            # 0 ~ length - 2  ->  -1 ~ -length + 1
            current_t = T - ind - 1
            current_env_sample = env_pool[current_t]
            Gt = current_env_sample.reward
            state = current_env_sample.state
            action = self.all_actions_onehot[current_env_sample.action]
            last_gt = Gt + self.gamma * last_gt
            gradient = self.nabla_softmax(state, action)
            self.parameters = self.parameters + self.alpha * (self.gamma ** current_t) * last_gt * gradient
    
    def softmax(self, x):
        y = np.exp(x)
        f_x = y / np.sum(np.exp(x))
        return f_x

    def nabla_softmax(self, s, a):
        fsa = self.feature_eng(s, a)
        action_props = self.action_dist(s)

        for ind, b in enumerate(self.all_actions_onehot):
            fsb = action_props[ind] * self.feature_eng(s, b)
            fsa -= fsb
        return fsa


def feature_mat_c(k, n):
    max_i = (n + 1) ** k
    feature_lst = []
    for i in range(max_i):
        arr = [0 for _ in range(k)]
        ind = - k
        while i > 0:
            arr[ind] = i % (n + 1)
            i = i // (n + 1)
            ind += 1
        feature_lst.append(arr)
    feature_mat = np.array(feature_lst)
    return feature_mat


def action_adapter(input_action):
    input_action = min(max(input_action, -1), 1)
    return np.array([input_action])


def train():
    # env_name = "MountainCarContinuous-v0"
    env_name = "CartPole-v1"
    env = gym.make(env_name, render_mode=None)
    policy = ParamPi(env.observation_space.shape[0])
    
    episode = 100000
    episodic_reward = np.zeros((episode, 1))
    for e in range(episode):
        done = False
        truncated = False
        observation = env.reset()
        state = observation[0]
        env.render()
        env_pool = []
        epoch = 0
        total_reward = 0
        while (not done) and (not truncated):
            action = policy.action(state)
            next_state, reward, done, truncated, info = env.step(action)
            env_pool.append(env_pool_sample(state, action, reward))
            total_reward += reward
            env.render()
            state = next_state
            epoch += 1
        #writer.add_scalar("Episodic_reward", total_reward, e)
        episodic_reward[e] = total_reward
        policy.update(env_pool)
    
    fig = plt.figure()
    plt.plot(episodic_reward)
    plt.show()

    env = gym.make(env_name, render_mode="human")
    ev_episodes = 10
    for e in range(ev_episodes):
        done = False
        truncated = False
        observation = env.reset()
        state = observation[0]
        env.render()
        epoch = 0
        total_reward = 0
        while (not done) and (not truncated):
            action = policy.action(state)
            next_state, reward, done, truncated, info = env.step(action)
            total_reward += reward
            env.render()
            state = next_state
            epoch += 1

 
train()



