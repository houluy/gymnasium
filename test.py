from src.algos.ac import ActorCritic
from src.algos.dqn2 import DQN, MADQN, IDQN
from src.algos.vac import VanillaActorCritic
from src.algos.ddpg import DeepDeterministicPolicyGradient
from src.algos.doubledqn import DoubleDQN
from src.algos.duelingdqn import DuelingDQN
from src.environments.simple_spread import SimpleSpread
import gymnasium as gym
import torch
from pettingzoo.mpe import simple_spread_v3


device = torch.device("cuda:0" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")) 
#env_name = "CartPole-v1"
#env_name = "Pendulum-v1"
#env_name = "Acrobot-v1"
#env_name = "MountainCar-v0"
#env_name = "MountainCarContinuous-v0"


#agent = ActorCritic(env_name, continuous=True, device=device)
#agent = DQN(env_name, device)
#agent = VanillaActorCritic(env_name, continuous=True, device=device)
#agent = Reinforce(env_name, device)
#agent = DeepDeterministicPolicyGradient(env_name, continuous=True, device=device)
#agent = DoubleDQN(env_name, device)
#agent = DuelingDQN(env_name, device)
env = simple_spread_v3.parallel_env(render_mode=None)
wrapped_env = SimpleSpread(env)
#state_dim = (env.observation_space("agent_0").shape[0] - communication_feature_dim) * env.num_agents
#action_num = env.action_space("agent_0").n * env.num_agents
#agent = DQN_CTDE(env=env, env_name="simple_spread_v3", state_dim=state_dim, action_dim=action_num, lr=0.001, buffer_size=100000, gamma=0.9, device=device)
agent = IDQN(env=wrapped_env, env_name="simple_spread_v3", state_dim=wrapped_env.local_observation_feature_num, action_dim=wrapped_env.local_action_num, lr=0.0001, buffer_size=100000, gamma=0.9, device=device)
agent.train(500000)
#agent.evaluate(100)
