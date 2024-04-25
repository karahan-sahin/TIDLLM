import torch
from torch import nn

"""
Feed-Forward Encoder Decoder Networks
"""

class FFNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FFNEncoder, self).__init__()
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.LeakyReLU(0.02)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        return x

class FFNDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FFNDecoder, self).__init__()
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.LeakyReLU(0.02)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        return x