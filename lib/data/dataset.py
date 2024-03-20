import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class PoseDataset(Dataset):

    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        sample = torch.tensor(self.data[idx], dtype=torch.float32)

        if self.transform: sample = self.transform(sample)

        return sample