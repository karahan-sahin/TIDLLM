import torch
import torch.nn as nn
import torch.nn.functional as F

class Activation(nn.Module):
    def __init__(self, activation):
        super(Activation, self).__init__()
        self.activation = activation

    def forward(self, x):
        if self.activation == "relu":
            return F.relu(x)
        elif self.activation == "sigmoid":
            return F.sigmoid(x)
        elif self.activation == "tanh":
            return F.tanh(x)
        elif self.activation == "softmax":
            return F.softmax(x)
        elif self.activation == "leaky_relu":
            return F.leaky_relu(x)
        elif self.activation == "elu":
            return F.elu(x)
        elif self.activation == "gelu":
            return F.gelu(x)
        else:
            raise ValueError("Activation function not supported")