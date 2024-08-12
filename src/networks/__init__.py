import torch.nn as nn
import torch
import torch.nn.functional as F


class DiscretePolicy(nn.Module):
    def __init__(self, input_dim, output_num, hidden_size=32):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_num)
        self.model = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            self.fc2,
            nn.ReLU(),
            self.fc3,
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.model(x)


class ContinuousPolicy(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=32):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_dim)
        self.model = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            self.fc2,
            nn.ReLU(),
            self.fc3
        )

    def forward(self, x):
        return self.model(x)


class ContinuousPolicyWithStd(nn.Module):
    """Policy network with learnable standard deviation"""
    def __init__(self, input_dim, output_dim, hidden_size=32):
        super().__init__()
        self.output_dim = output_dim
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_dim * 2)
        self.model = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            self.fc2,
            nn.ReLU(),
            self.fc3
        )

    def forward(self, x):
        y = self.model(x)
        y = torch.concatenate([y[:, :self.output_dim], F.sigmoid(y[:, self.output_dim:])], dim=1)
        return y


class Value(nn.Module):
    def __init__(self, input_dim, hidden_size=32):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.model = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            self.fc2,
            nn.ReLU(),
            self.fc3
        )

    def forward(self, x):
        return self.model(x)


class DiscreteQ(nn.Module):
    def __init__(self, state_dim, action_num, hidden_size=64):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, action_num)
        self.model = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            self.fc2,
            nn.ReLU(),
            self.fc3,
            nn.ReLU(),
            self.fc4
        )

    def forward(self, x):
        return self.model(x)


class ContinuousQ(nn.Module):
    def __init__(self, input_dim, hidden_size=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.model = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            self.fc2,
            nn.ReLU(),
            self.fc3
        )

    def forward(self, x):
        return self.model(x)


class DuelingDiscreteQ(nn.Module):
    def __init__(self, state_dim, action_num, hidden_size=64):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        # Advantage output
        self.fc3 = nn.Linear(hidden_size, action_num)
        # Value output
        self.fc4 = nn.Linear(hidden_size, 1)

        # Q = V + A(averaged)

    def forward(self, x):
        hidden1 = F.relu(self.fc1(x))
        hidden2 = F.relu(self.fc2(hidden1))
        value_output = self.fc4(hidden2)
        advantage_output = self.fc3(hidden2)
        q = value_output + advantage_output - advantage_output.mean()
        return q


