import os
import torch
import cv2
import numpy as np
import pandas as pd
import pickle as pkl
from tqdm import tqdm
from torch import nn, optim
import torch.nn.functional as F
import random
from lib.config import *

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

        # self.experiment = wandb.init(
        #     project="TIDLLM",
        #     config=GLOBAL_CONFIG,
        #     settings=wandb.Settings(start_method="fork"),
        # )

    def train(self):
        RECON_TRAIN_LOSS = []
        COMMIT_TRAIN_LOSS = []

        RECON_VAL_LOSS = []
        COMMIT_VAL_LOSS = []

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
    Train Commitment Loss: {np.array(self.losses['train']['commitment'][epoch]).sum()}, 
    Train Reconstruction Loss: {np.array(self.losses['train']['reconstruction'][epoch]).sum()}, 
    Train Quantization Tokens:
    {pd.DataFrame(pd.Series(self.losses['train']['quantization'][epoch]).value_counts()).T.to_markdown()}, 

    Validation Commitment Loss: {np.array(self.losses['validation']['commitment'][epoch]).sum()},
    Validation Reconstruction Loss: {np.array(self.losses['validation']['reconstruction'][epoch]).sum()},
    Validation Quantization Tokens:
    {pd.DataFrame(pd.Series(self.losses['validation']['quantization'][epoch]).value_counts()).T.to_markdown()}, 
    ***
            """)

            RECON_TRAIN_LOSS.append(np.array(self.losses['train']['reconstruction'][epoch]).sum() / len(self.train_dataloader))
            COMMIT_TRAIN_LOSS.append(np.array(self.losses['train']['commitment'][epoch]).sum() / len(self.train_dataloader))

            RECON_VAL_LOSS.append(np.array(self.losses['validation']['reconstruction'][epoch]).sum() / len(self.val_dataloader))
            COMMIT_VAL_LOSS.append(np.array(self.losses['validation']['commitment'][epoch]).sum() / len(self.val_dataloader))

        return RECON_TRAIN_LOSS, COMMIT_TRAIN_LOSS, RECON_VAL_LOSS, COMMIT_VAL_LOSS
    # with open('loss.pkl', 'wb') as f:
    #   print(f"loss.pkl is saved to {os.getcwd()}")
    #   pkl.dump(LOSS, f)
