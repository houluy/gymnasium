from src.algos.algo import Algo
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.normal import Normal
from src.networks import ContinuousPolicy as Policy, ContinuousQ as Q
from src.utils import ReplayBuffer
from tqdm import tqdm
import numpy as np

LOG_STD_MAX = 2
LOG_STD_MIN = -20


class SoftActorCritic(Algo):
    """Soft Actor Critic (SAC) algorithm."""
    def __init__(self, env_name, auto_temperature_tuning=True, continuous=True, device=torch.device("cpu")):
        super().__init__(env_name, continuous, device)
        self.buffer = ReplayBuffer(self.buffer_size, self.device, action_type="continuous")
        self.policy = Policy(self.state_dim, 2 * self.action_dim).to(device)
        #self.target_policy = Policy(self.state_dim, 2 * self.action_dim).to(device)
        self.q1 = Q(self.state_dim + self.action_dim).to(device)
        self.q2 = Q(self.state_dim + self.action_dim).to(device)

        self.qs = (self.q1, self.q2)

        self.target_q1 = Q(self.state_dim + self.action_dim).to(device)
        self.target_q2 = Q(self.state_dim + self.action_dim).to(device)

        self.target_qs = (self.target_q1, self.target_q2)

        self.target_network_soft_update()
        self.auto_temperature_tuning = auto_temperature_tuning
        self.target_entropy = - torch.prod(torch.Tensor(self.action_dim))
        if self.auto_temperature_tuning:
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.policy_lr)
        else:
            self.log_alpha = 0
            self.alpha_optimizer = None
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=self.policy_lr)
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=self.critic_lr)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=self.critic_lr)
        #self.noise_std = self.start_noise_std

    def target_network_soft_update(self):
        with torch.no_grad():
            for target_q, q in zip(self.target_qs, self.qs):
                for target_param, param in zip(target_q.parameters(), q.parameters()):
                    target_param.data.copy_(param.data * self.polyak + target_param.data * (1.0 - self.polyak))
            #for target_param, param in zip(self.target_policy.parameters(), self.policy.parameters()):
            #    target_param.data.copy_(param.data * self.polyak + target_param.data * (1.0 - self.polyak))

    def select_action_continuous(self, state, train=True, with_logprob=True):
        with torch.no_grad():
            # Sample by Reparameterization trick
            action_layer = self.policy(state)
            action_layer_shape = action_layer.shape
            if len(action_layer_shape) == 1:
                batch_size = 1
                half_ind = action_layer.shape[0] // 2
                action_mu = action_layer[:half_ind]
                action_log_std = action_layer[half_ind:]
            else:
                batch_size = action_layer_shape[0]   
                half_ind = action_layer.shape[1] // 2
                action_mu = action_layer[:, :half_ind]
                action_log_std = action_layer[:, half_ind:]
            action_log_std = torch.clamp(action_log_std, LOG_STD_MIN, LOG_STD_MAX)
            action_std = torch.exp(action_log_std)
            m = Normal(loc=action_mu, scale=action_std)
            if not train:
                pi_action = action_mu
                return pi_action
            else:
                pi_action = action_mu
                if with_logprob:
                    logp_pi = m.log_prob(pi_action).sum(axis=-1)
                    logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(axis=-1)
                else:
                    logp_pi = None
                pi_action = torch.clamp(torch.tanh(pi_action), min=self.action_min, max=self.action_max)
                if len(action_layer_shape) == 1:
                    return pi_action.reshape((self.action_dim,)), logp_pi.reshape((1,))
                else:
                    return pi_action.reshape((self.batch_size, self.action_dim)), logp_pi.reshape((self.batch_size, 1))

    def select_action_discrete(self, state, train=True):
        return None#super().select_action_discrete(state, train)

    def evaluate(self, episodes=100) -> torch.List[float]:
        return super().evaluate(episodes)

    def no_grad(self, model):
        for p in model.parameters():
            p.requires_grad = False
    
    def yes_grad(self, model):
        for p in model.parameters():
            p.requires_grad = True

    def update_qs(self, batch, alpha):
        states, actions, rewards, next_states, dones = batch
        next_actions, next_actions_logprob = self.select_action_continuous(next_states)
        state_actions = torch.concat([states, actions], dim=1)
        next_state_actions = torch.concat([next_states, next_actions], dim=1)
        target_q_1 = self.target_q1(next_state_actions)
        target_q_2 = self.target_q2(next_state_actions)
        y = rewards.unsqueeze(1) + self.gamma * (1 - dones).unsqueeze(1) * (torch.min(target_q_1, target_q_2) - alpha * next_actions_logprob)
        q1 = self.q1(state_actions)
        q2 = self.q2(state_actions)
        loss_1 = F.mse_loss(q1, y.float())
        loss_2 = F.mse_loss(q2, y.float())
        self.q1_optimizer.zero_grad()
        loss_1.backward(retain_graph=True)
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        loss_2.backward()
        self.q2_optimizer.step()

        return loss_1, loss_2

    def update_policy(self, batch):
        states, actions, rewards, next_states, dones = batch
        actions, action_logprobs = self.select_action_continuous(states)
        if self.auto_temperature_tuning:
            alpha_loss = -(self.log_alpha * (action_logprobs + self.target_entropy).detach()).mean()
            alpha = self.log_alpha.exp()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
        else:
            alpha_loss = 0
            alpha = 1
        state_actions = torch.concat([states, actions], dim=1)
        q = torch.min(self.q1(state_actions), self.q2(state_actions))
        self.no_grad(self.q1)
        self.no_grad(self.q2)
        self.policy_optimizer.zero_grad()
        performance = (alpha * action_logprobs - q).mean()
        performance.backward(retain_graph=True)
        self.policy_optimizer.step()
        self.yes_grad(self.q1)
        self.yes_grad(self.q2)
        return performance, q.mean(), alpha, alpha_loss

    def train(self, episodes=10000):
        if self.continuous:
            action_dist = []

        training_step = -self.training_start
        for episode in tqdm(range(episodes), desc="Training Episodes", position=0):
            state, _ = self.env.reset()
            done = False
            episodic_reward = 0
            step = 0
            while not done:
                action, action_logprob = self.select_action_continuous(torch.from_numpy(state).to(self.device))
                action = action.cpu()
                if self.continuous:
                    action_dist.append(action)
                next_state, reward, done, truncation, _ = self.env.step(action)
                episodic_reward += reward
                self.buffer.add(state, action, reward, next_state, 1 if done else 0)
                done = done or truncation
                step += 1
                training_step += 1
                if training_step > 0:
                    #for _ in tqdm(range(self.update_num_per_step), desc="Updating models", position=1):
                    for _ in range(self.update_num_per_step):
                        batch = self.buffer.sample(self.batch_size)
                        performance, min_q_mean, alpha, alpha_loss = self.update_policy(batch)
                        loss_1, loss_2 = self.update_qs(batch, alpha)
                        self.writer.add_scalar("training/performance", -performance, training_step)
                        self.writer.add_scalar("training/min_q_mean", min_q_mean, training_step)
                        self.writer.add_scalar("training/alpha", alpha, training_step)
                        self.writer.add_scalar("training/alpha_loss", alpha_loss, training_step)
                        self.writer.add_scalar("training/loss_value_1", loss_1, training_step)
                        self.writer.add_scalar("training/loss_value_2", loss_2, training_step)
                        if training_step % self.info_step == 0:
                            tqdm.write(f"Episode: {episode}, Training step: {training_step}, Episodic Reward: {episodic_reward}, Loss 1: {loss_1}, Loss 2: {loss_2}, Min Q: {min_q_mean}")
                        self.target_network_soft_update()
                        training_step += 1
                    training_step -= 1
            self.writer.add_scalar("training/episode_length", step, episode)
            self.writer.add_scalar("training/episodic_reward", episodic_reward, episode)
        if self.continuous:
            self.writer.add_histogram("training/action_dist", np.array(action_dist), 1, bins="auto")

