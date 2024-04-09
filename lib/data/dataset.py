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


class PoseDistanceDataset(Dataset):

    def __init__(self, 
                 paths, 
                 transform=None,
                 window=5):
        
        self.data = []
        self.paths = paths
        self.window = window
        self.generate_instances()

    def generate_instances(self):
        for path in tqdm(self.paths):
            with open(path, 'rb') as f:
                array = np.load(f, allow_pickle=True)
                # create set of windows for each video
                for start_idx in range(self.window, array.shape[0] - self.window):
                    self.data.append((path, start_idx, start_idx + self.window))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        path, start_idx, end_idx = self.data[idx]

        with open(path, 'rb') as f:
            array = np.load(f)
            array = array[start_idx:end_idx]

        return {
            'array': array,
            'token': path.split('/')[-1].replace('.npy', ''),
            'start_idx': start_idx,
            'end_idx': end_idx
        }

    @staticmethod
    def collate_fn(batch):
        # concatenate tensors into new axis
        data = torch.stack([torch.tensor(item['array']) for item in batch]).float().permute(0, 4, 1, 2, 3)
        tokens = [item['token'] for item in batch]
        start_idx = [item['start_idx'] for item in batch]
        end_idx = [item['end_idx'] for item in batch]
        return {
            'array': data,
            'tokens': tokens,
            'start_idx': start_idx,
            'end_idx': end_idx
        }
        