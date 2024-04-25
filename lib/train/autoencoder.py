import os
import torch
import numpy as np
import pandas as pd
import pickle as pkl
from tqdm import tqdm
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

class AutoTrainer():

    def __init__(self,
                 model,
                 train_dataset,
                 eval_dataset,
                 batch_size=8,
                 epochs=100,
                 learning_rate=0.001,
                 step_size=10,
                 gamma=0.1,
                 device='cuda',
                 start_epoch=0,
                 num_codebooks=8,
                 model_path='drive/MyDrive/Graph-Quant/model/model-0.pt',
                 log_dir='drive/MyDrive/Graph-Quant/logs/logs.json'
                 ):


        self.device = device
        self.model = model.to(self.device)

        self.num_codebooks = num_codebooks
        self.start_epoch = start_epoch
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate

        self.model_path = model_path

        self.optimizer = torch.optim.Adam(
           self.model.parameters(), 
           lr=self.learning_rate
        )
        self.step_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
           self.optimizer, T_max=step_size, eta_min=0
        )

        self.loss_fn = nn.MSELoss()

        self.train_dataloader = DataLoader(
            train_dataset, batch_size=self.batch_size, 
            shuffle=True, collate_fn=train_dataset.collate_fn
        )

        self.eval_dataloader = DataLoader(
            eval_dataset, batch_size=self.batch_size, shuffle=True, 
            collate_fn=eval_dataset.collate_fn
        )


        self.logs = {
            'train': {
                'vocab':  { e: [] for e in range(self.epochs)},
                'start_idx':  { e: [] for e in range(self.epochs)},
                'end_idx': { e: [] for e in range(self.epochs)},
                'commit-loss': { e: [] for e in range(self.epochs)},
                'recon-loss': { e: [] for e in range(self.epochs)},
                'loss': { e: [] for e in range(self.epochs) },
                'quantization': { e: {f'Code_{idx}': [] for idx in range(self.num_codebooks)}
                 for e in range(self.epochs)},
            },
            'validation': {
                'vocab':  { e: [] for e in range(self.epochs)},
                'start_idx':  { e: [] for e in range(self.epochs)},
                'end_idx': { e: [] for e in range(self.epochs)},
                'commit-loss': { e: [] for e in range(self.epochs)},
                'recon-loss': { e: [] for e in range(self.epochs)},
                'loss': { e: [] for e in range(self.epochs) },
                'quantization': { e: {f'Code_{idx}': [] for idx in range(self.num_codebooks)}
                  for e in range(self.epochs) },
            }
        }

    def train(self):

        for epoch in tqdm(range(self.epochs)):

            print(f"Epoch: {epoch+self.start_epoch}")
            self.model.train()
            for batch in tqdm(self.train_dataloader):

                self.optimizer.zero_grad()

                data = batch['array'].to(self.device)

                self.logs['train']['vocab'][epoch+self.start_epoch].extend(batch['tokens'])
                self.logs['train']['start_idx'][epoch+self.start_epoch].extend(batch['start_idx'])
                self.logs['train']['end_idx'][epoch+self.start_epoch].extend(batch['end_idx'])

                output, indices, commit_loss = self.model(data)

                indexes = indices.cpu().detach().numpy().reshape(-1, self.num_codebooks)

                commit_loss = commit_loss.mean()

                for i in range(self.num_codebooks):
                  self.logs['train']['quantization'][epoch+self.start_epoch][f'Code_{i}'].extend(indexes[:, i])

                loss = self.loss_fn(data, output)
                self.logs['train']['recon-loss'][epoch+self.start_epoch].append(loss.detach().item())
                self.logs['train']['commit-loss'][epoch+self.start_epoch].append(commit_loss.detach().item())

                total_loss = loss + commit_loss
                self.logs['train']['loss'][epoch+self.start_epoch].append(total_loss.detach().item())

                total_loss.backward()
                self.optimizer.step()
                self.step_scheduler.step()

            self.model.eval()
            for batch in tqdm(self.eval_dataloader):

                data = batch['array'].to(self.device)

                self.logs['validation']['vocab'][epoch+self.start_epoch].extend(batch['tokens'])
                self.logs['validation']['start_idx'][epoch+self.start_epoch].extend(batch['start_idx'])
                self.logs['validation']['end_idx'][epoch+self.start_epoch].extend(batch['end_idx'])


                output, indices, commit_loss = self.model(data)
                indexes = indices.cpu().detach().numpy().reshape(-1, self.num_codebooks)
                for i in range(self.num_codebooks):
                    self.logs['validation']['quantization'][epoch+self.start_epoch][f'Code_{i}'].extend(indexes[:, i])

                commit_loss = commit_loss.mean()

                loss = self.loss_fn(data, output)

                self.logs['validation']['recon-loss'][epoch+self.start_epoch].append(loss.item())
                self.logs['validation']['commit-loss'][epoch+self.start_epoch].append(commit_loss.item())

                loss = loss + commit_loss
                self.logs['validation']['loss'][epoch+self.start_epoch].append(loss.item())

            torch.save(self.model.state_dict(), self.model_path)

            self.write_logs(epoch)

            print(f"Train Loss: {np.sum(self.logs['train']['loss'][epoch+self.start_epoch])}")
            print(f"Validation Loss: {np.sum(self.logs['validation']['loss'][epoch+self.start_epoch])}")
            print(f'epoch+self.start_epoch {epoch+self.start_epoch+1} - Training Quantization')
            for i in range(self.num_codebooks):
              print(f'Code_{i}')
              print(pd.DataFrame(pd.Series(self.logs['train']['quantization'][epoch+self.start_epoch][f'Code_{i}']).value_counts()).T.to_markdown())
              print('-'*100)
            print(f'Epoch {epoch+self.start_epoch+1} - Validation Quantization')
            for i in range(self.num_codebooks):
              print(f'Code_{i}')
              print(pd.DataFrame(pd.Series(self.logs['validation']['quantization'][epoch+self.start_epoch][f'Code_{i}']).value_counts()).T.to_markdown())
              print('-'*100)
            print('*'*100)

    def write_logs(self, epoch):
      """Save logs to json file"""

      with open(self.log_dir, 'w') as f:
        import json
        json.dump(self.logs, f)