import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical
from tqdm import tqdm
import gymnasium as gym
from src.algos.algo import Algo
from src.networks import Policy, Value

# device = torch.device("cuda:0" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")) 


class ActorCritic(Algo):
    def __init__(self, env, device="cpu"):
        super().__init__(env, device)

        self.policy = Policy(self.env.observation_space.shape[0], self.env.action_space.n).to(self.device)
        self.critic = Value(self.env.observation_space.shape[0]).to(self.device)

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

    def train(self, episodes=50000):
        training_step = 0
        for episode in tqdm(range(episodes), desc="Training"):
            state, _ = self.env.reset()
            episodic_reward = 0
            step = 0
            done = False
            while not done:
                action = self.select_action(torch.from_numpy(state).to(self.device))
                next_state, reward, done, truncation, _ = self.env.step(action)
                episodic_reward += reward
                exp = (torch.from_numpy(state).to(self.device),
                        torch.tensor(action).to(self.device),
                        torch.tensor(reward).to(self.device),
                        torch.tensor(1 if done else 0).to(self.device),
                        torch.from_numpy(next_state).to(self.device))
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
    agent = ActorCritic(env)
    agent.train()