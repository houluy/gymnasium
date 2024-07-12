from src.algos.ac import ActorCritic
from src.algos.dqn2 import DQN
from src.algos.vac import VanillaActorCritic
from src.algos.reinforce import Reinforce
from src.algos.ddpg import DeepDeterministicPolicyGradient
import gymnasium as gym
import torch


device = torch.device("cuda:0" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")) 
#env_name = "CartPole-v1"
env_name = "Pendulum-v1"

#env = gym.make("CartPole-v1")
#agent = ActorCritic(env_name, device)
#agent = DQN(env_name, device)
#agent = VanillaActorCritic(env_name, device)
#agent = Reinforce(env_name, device)
agent = DeepDeterministicPolicyGradient(env_name, device)
agent.train()
