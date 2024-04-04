import os
import torch
import numpy as np
import pandas as pd
import pickle as pkl
from tqdm import tqdm
from torch import nn, optim
import torch.nn.functional as F

# def codebook_diversity_loss(embedding_weights):
#     normed_weights = F.normalize(embedding_weights, p=2, dim=1)
#     similarity = torch.matmul(normed_weights, normed_weights.T)
#     # Remove self-similarity by setting diagonal to zero
#     similarity.fill_diagonal_(0)
#     diversity_loss = torch.mean(similarity)
#     return diversity_loss

class AutoencoderTrainer:
    def __init__(self, 
                 model, 
                 train_dataloader,
                 val_dataloader,
                 num_epochs,
                 learning_rate, 
                 device='cuda'):
        
        self.model = model
        self.device = device
        self.model.to(device)

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        self.num_epochs = num_epochs
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.losses = { 'train': {}, 'validation': {} }

        self.losses['train']['reconstruction'] = { e:[] for e in range(num_epochs) }
        self.losses['train']['commitment'] = { e:[] for e in range(num_epochs) }
        self.losses['train']['quantization'] = { e:[] for e in range(num_epochs) }

        self.losses['validation']['reconstruction'] = { e:[] for e in range(num_epochs) }
        self.losses['validation']['commitment'] = { e:[] for e in range(num_epochs) }
        self.losses['validation']['quantization'] = { e:[] for e in range(num_epochs) }


    def train(self):

        self.model.train()
        for epoch in tqdm(range(self.num_epochs)):
            for batch in self.train_dataloader:
                data = batch['array'].to(self.device)
                data = data.float()
                self.optimizer.zero_grad()
                
                quantized, indices, commitment_loss = self.model(data)
                reconstruction_loss = self.criterion(quantized, data)

                loss = commitment_loss + reconstruction_loss

                self.losses['train']['reconstruction'][epoch].append(reconstruction_loss.item())
                self.losses['train']['commitment'][epoch].append(commitment_loss.item())
                self.losses['train']['quantization'][epoch].extend(list(indices.detach().cpu().numpy().reshape(-1)))

                loss.backward()
                self.optimizer.step()

            for batch in self.val_dataloader:
                data = batch['array'].to(self.device)
                data = data.float()
                
                quantized, indices, commitment_loss = self.model(data)
                reconstruction_loss = self.criterion(quantized, data)

                loss = commitment_loss + reconstruction_loss

                self.losses['validation']['reconstruction'][epoch].append(reconstruction_loss.item())
                self.losses['validation']['commitment'][epoch].append(commitment_loss.item())
                self.losses['validation']['quantization'][epoch].extend(list(indices.detach().cpu().numpy().reshape(-1)))

            
            print(f"""
***
Epoch:{epoch+1}, 
Train Commitment Loss: {np.array(self.losses['train']['commitment'][epoch]).mean()}, 
Train Reconstruction Loss: {np.array(self.losses['train']['reconstruction'][epoch]).mean()}, 
Train Quantization Tokens:
{pd.DataFrame(pd.Series(self.losses['train']['quantization'][epoch]).value_counts()).T.to_markdown()}, 

Validation Commitment Loss: {np.array(self.losses['validation']['commitment'][epoch]).mean()},
Validation Reconstruction Loss: {np.array(self.losses['validation']['reconstruction'][epoch]).mean()},
Validation Quantization Tokens:
{pd.DataFrame(pd.Series(self.losses['validation']['quantization'][epoch]).value_counts()).T.to_markdown()}, 
***
            """)

        # with open('loss.pkl', 'wb') as f:
        #   print(f"loss.pkl is saved to {os.getcwd()}")
        #   pkl.dump(LOSS, f)

    def test(self, dataloader):
        self.model.eval()
        with torch.no_grad():
            for data in dataloader:
                outputs = self.model(data).to(self.device)
                loss = self.criterion(outputs, data)
            print(f'Test Loss:{loss.item()}')

    def test(self, dataloader):
        self.model.eval()
        with torch.no_grad():
            for data in dataloader:
                outputs = self.model(data)
                loss = self.criterion(outputs, data)
            print(f'Test Loss:{loss.item()}')
