import gymnasium as gym
import torch.nn as nn
import torch


def copy_net(net, target):
    target.load_state_dict(net.state_dict())
    target.eval()


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


class SQL(nn.Module):
    input_size: int
    hidden_size: int
    output_size: int
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.linear_relu = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        output = self.linear_relu(x)
        return output


class SQL_Agent:
    def __init__(self, input_size, hidden_size, output_size):
        # Soft Q value
        self.Q = SQL(input_size, hidden_size, output_size).to(device)
        self.Q_target = copy_net(self.Q)
        # Sample action from random input
        noise_size = 10
        hidden_size = 32
        self.f = SQL(input_size=noise_size, hidden_size=hidden_size, output_size=output_size)
        self.q_optimizer = torch.optim.Adam(self.Q.parameters(), lr=0.001)
        self.f_optimizer = torch.optim.Adam(self.f.parameters(), lr=0.001)

        # Soft Q Loss
        self.q_loss = nn.MSELoss()


    def select_action(self, state):
        noise = torch.randn(self.f.input_size)
        action = self.f(noise) 
        return action


def train():
    env = gym.make('CartPole-v1')
    input_size = env.observation_space.shape[0]
    sql_agent = None

    state = env.reset()
