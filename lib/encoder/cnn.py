import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNEncoder(nn.Module):
    def __init__(self, input_channels):
        super(CNNEncoder, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=input_channels, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 18 * 18, 768)

    def forward(self, x):
        # print('** Encoder **')
        # print(f'Encoder input shape: {x.shape}')
        x = F.relu(self.conv1(x))
        # print(f'After conv1: {x.shape}')
        x = self.pool(x)
        # print(f'After pool: {x.shape}')
        x = F.relu(self.conv2(x))
        # print(f'After conv2: {x.shape}')
        x = self.pool(x)
        # print(f'After pool: {x.shape}')
        x = x.contiguous().view(x.size(0), -1)
        # print(f'After view: {x.shape}')
        x = F.relu(self.fc1(x))
        # print(f'After fc1: {x.shape}')
        return x


class CNNDecoder(nn.Module):
    def __init__(self, output_channels):
        super(CNNDecoder, self).__init__()
        # Assuming the output of the encoder before the fully connected layer was (64, D/4, H/4, W/4)
        # We start reversing from the output of the fully connected layer
        self.fc1 = nn.Linear(768, 64 * 18 * 18)
        
        self.conv1 = nn.Conv3d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(32, 16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv3d(16, 3, kernel_size=3, stride=1, padding=1)
        self.upsample = nn.Upsample((5, 75, 75))
        self.relu = nn.ReLU()


    def forward(self, x):
        # Reverse the fully connected layer
        # print('** Decoder **')
        # print(f'Decoder input shape: {x.shape}')
        x = F.relu(self.fc1(x))
        # print(f'After fc1: {x.shape}')
        x = x.view(-1, 64, 1, 18, 18)  # Reshape to match the pre-pooling size, adjust dimensions as necessary
        # print(f'After view: {x.shape}')
        # Apply transposed convolutions, reversing the pooling and convolution operations
        x = self.relu(self.conv1(x))
        # print(f'After conv1: {x.shape}')
        x = self.relu(self.conv2(x))
        # print(f'After conv2: {x.shape}')
        x = self.relu(self.conv3(x))
        # print(f'After conv3: {x.shape}')
        x = self.upsample(x)
        # print(f'After upsample: {x.shape}')
        return x