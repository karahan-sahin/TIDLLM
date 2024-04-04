import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class PoseDataset(Dataset):

    def __init__(self, data, transform=None):
        self.paths = data
        self.tokens = []
        self.frames = []
        self.load_data(self.paths)
        self.transform = transform

    def load_data(self, paths):
        data = []
        for path in tqdm(paths):
            with open(path, 'rb') as f:
                array = np.load(f, allow_pickle=True)
                # replace nan with 0 
                array = np.nan_to_num(array)
                data.append(array)

            self.tokens += [ path.split('/')[-1] ] * array.shape[0]
            self.frames += list(range(array.shape[0]))

        self.data = np.vstack(data)
        self.data = self.data.astype(np.float32)
        self.data = np.nan_to_num(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            'array': self.data[idx,:],
            'token': self.tokens[idx],
            'frame': self.frames[idx]
        }