import os
import torch
import numpy as np
import pandas as pd
import pickle as pkl
from tqdm import tqdm
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import cv2
import wandb
import random
from lib.config import *
from moviepy.editor import ImageSequenceClip


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

        self.log_dir = log_dir
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

        self.experiment = wandb.init(
            project="TIDLLM",
            config=GLOBAL_CONFIG,
            settings=wandb.Settings(start_method="fork"),
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

    def evaluate_model(self, epoch):
      self.model.eval()

      dfs = []
      for sample in self.train_dataloader:
          with torch.no_grad():
              quantized, indices, commitment_loss = self.model(
                  sample["array"].to(self.device).float()
              )
              dfs.append(
                  pd.DataFrame(
                      {
                          "videos": sample["tokens"],
                          "labels": indices.detach().cpu().numpy().reshape(-1),
                          "start_idx": sample["start_idx"],
                          "end_idx": sample["end_idx"],
                      }
                  )
              )

      df = pd.concat(dfs)
      sorted_id_list = (
          pd.DataFrame(df["labels"].value_counts())
          .reset_index()
          .query("count > 3")["labels"]
          .to_list()
      )
      random_id = random.choice(sorted_id_list)
      print("RANDOM_ID", random_id)
      videos = []
      vid_names = []
      imgs = []
      for rec in df[df["labels"] == random_id].to_dict(orient="records"):
          # save frame video to disk
          video = rec["videos"].split(".")[0]
          video_path = f"dataset/corpus/{video}.mp4"
          start_idx = rec["start_idx"]
          end_idx = rec["end_idx"]
          label = rec["labels"]

          cap = cv2.VideoCapture(video_path)

          if not os.path.exists(f"analyze/quantization/{label}"):
              os.mkdir(f"analyze/quantization/{label}")

          FRAMES = []
          MAX = max(df[df["labels"] == random_id]["end_idx"])
          for i in range(MAX):
              ret, frame = cap.read()
              if i >= start_idx and i <= end_idx:
                  h, w, c = frame.shape
                  frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                  FRAMES.append(frame)
                  if (end_idx + start_idx) // 2 == i:
                      imgs.append((frame, video))
          vid_names.append(video)
          videos.append(FRAMES)
          if len(videos) >= 3:
              break

      videos = np.array(videos)
      vid_frames = []
      for i in range(videos.shape[1]):
          img1 = videos[0][i]
          img2 = videos[1][i]
          img3 = videos[2][i]

          stacked_imgs = np.hstack([img1, img2, img3])
          vid_frames.append(stacked_imgs)
      self.create_video_from_frames(
          vid_frames, f"analyze/quantization/{random_id}/{epoch}.mp4"
      )

      wandb_img = wandb.Image(
          np.hstack([img for img, vid in imgs]),
          caption=f"{random_id}, {[vid for img, vid in imgs]}",
      )
      self.experiment.log({"Inference Id": wandb_img})

    def create_video_from_frames(self, frames, output_file):
      clip = ImageSequenceClip(frames, fps=5)
      clip.write_videofile(output_file)

    def save_to_wandb(self, epoch):
      df = pd.DataFrame(
          pd.Series(self.losses["train"]["quantization"][epoch]).value_counts()
      )
      transposed_df = df.reset_index()
      transposed_df.columns = ["id", "count"]

      self.experiment.log(
          {
              "train_commitment_loss": np.array(
                  self.losses["train"]["commitment"][epoch]
              ).mean(),
              "train_reconstruction_loss": np.array(
                  self.losses["train"]["reconstruction"][epoch]
              ).mean(),
              "train_quantization_token_count": wandb.Table(
                  data=transposed_df, columns=transposed_df.columns.to_list()
              ),
          }
      )

      df = pd.DataFrame(
          pd.Series(self.losses["validation"]["quantization"][epoch]).value_counts()
      )
      transposed_df = df.reset_index()
      transposed_df.columns = ["id", "count"]

      self.experiment.log(
          {
              "val_commitment_loss": np.array(
                  self.losses["validation"]["commitment"][epoch]
              ).mean(),
              "val_reconstruction_loss": np.array(
                  self.losses["validation"]["reconstruction"][epoch]
              ).mean(),
              "val_quantization_token_count": wandb.Table(
                  data=transposed_df, columns=transposed_df.columns.to_list()
              ),
          }
      )

      self.evaluate_model(epoch)

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

            self.save_to_wandb(epoch)

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