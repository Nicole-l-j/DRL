import torch
import torch.nn as nn
from torch.autograd import Variable


class ConvDqn(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(ConvDqn, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc_input_dim = self.fc_features_dim()

        self.conv_net = nn.Sequential(
            nn.Conv2d(self.input_dim, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, out_channels=64,  kernel_size=3, stride=1)
        )

        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, self.output_dim)
        )

    def fc_features_dim(self):
        return self.conv_net(Variable(torch.zeros(self.input_dim, self.output_dim))).view(1, -1).size(1)

    def forward(self, state):
        features = self.conv_net(state)
        features = features.view(features.size(0), -1)
        qvals = self.fc(features)
        return qvals


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, self.output_dim)
        )

    def forward(self, state):
        qvals = self.fc(state)
        return qvals