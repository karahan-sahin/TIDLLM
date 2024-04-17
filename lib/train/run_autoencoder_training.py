import os
import torch
import numpy as np
import pandas as pd
import pickle as pkl
from tqdm import tqdm
from torch import nn, optim
import torch.nn.functional as F
import wandb
from lib.config import *

class AutoencoderTrainer:
    def __init__(
        self,
        model,
        train_dataloader,
        val_dataloader,
        num_epochs,
        learning_rate,
        device="cuda",
    ):

        self.model = model
        self.device = device
        self.model.to(device)

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        self.num_epochs = num_epochs
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.losses = {"train": {}, "validation": {}}

        self.losses["train"]["reconstruction"] = {e: [] for e in range(num_epochs)}
        self.losses["train"]["commitment"] = {e: [] for e in range(num_epochs)}
        self.losses["train"]["quantization"] = {e: [] for e in range(num_epochs)}

        self.losses["validation"]["reconstruction"] = {e: [] for e in range(num_epochs)}
        self.losses["validation"]["commitment"] = {e: [] for e in range(num_epochs)}
        self.losses["validation"]["quantization"] = {e: [] for e in range(num_epochs)}

        self.experiment = wandb.init(
            project="TIDLLM",
            config=GLOBAL_CONFIG,
            settings=wandb.Settings(start_method="fork"),
        )

    def evaluate_model(self):
        self.model.eval()

        dfs = []
        for val_sample in tqdm(self.train_dataloader):
            with torch.no_grad():
                quantized, indices, commitment_loss = self.model(
                    val_sample["array"].float()
                )
                dfs.append(
                    pd.DataFrame(
                        {
                            "videos": val_sample["token"],
                            "labels": indices.detach().cpu().numpy().reshape(-1),
                            "frame": val_sample["frame"]
                            .detach()
                            .cpu()
                            .numpy()
                            .reshape(-1),
                        }
                    )
                )

        df = pd.concat(dfs)
        sorted_id_list = (
            pd.DataFrame(df["labels"].value_counts()).reset_index()["labels"].to_list()
        )
        random_ids = random.sample(sorted_id_list[: len(sorted_id_list) // 3], 1)

        for id in random_ids:
            imgs = []
            for rec in tqdm(df[df["labels"] == id].to_dict(orient="records")):
                # save frame video to disk
                video = rec["videos"].split(".")[0]
                video_path = f"dataset/corpus/{video}.mp4"
                frame_idx = rec["frame"]
                label = rec["labels"]

                if video in [vid for img, vid in imgs]:
                    continue

                cap = cv2.VideoCapture(video_path)
                ret, frame = cap.read()

                for i in range(frame_idx):
                    ret, frame = cap.read()
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    if i == frame_idx - 1:
                        imgs.append((frame, video))
                if len(imgs) >= 3:
                    break

            wandb_images = wandb.Image(
                np.hstack([img for img, vid in imgs]),
                caption=f"{id}, {[vid for img, vid in imgs]}",
            )
            self.experiment.log({"Inference Imgs": wandb_images})

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

        self.evaluate_model()

    def train(self):

        self.model.train()
        for epoch in tqdm(range(self.num_epochs)):
            for batch in self.train_dataloader:
                data = batch["array"].to(self.device)
                data = data.float()
                self.optimizer.zero_grad()

                quantized, indices, commitment_loss = self.model(data)
                reconstruction_loss = self.criterion(quantized, data)

                loss = commitment_loss + reconstruction_loss

                self.losses["train"]["reconstruction"][epoch].append(
                    reconstruction_loss.item()
                )
                self.losses["train"]["commitment"][epoch].append(commitment_loss.item())
                self.losses["train"]["quantization"][epoch].extend(
                    list(indices.detach().cpu().numpy().reshape(-1))
                )

                loss.backward()
                self.optimizer.step()

            for batch in self.val_dataloader:
                data = batch["array"].to(self.device)
                data = data.float()

                quantized, indices, commitment_loss = self.model(data)
                reconstruction_loss = self.criterion(quantized, data)

                loss = commitment_loss + reconstruction_loss

                self.losses["validation"]["reconstruction"][epoch].append(
                    reconstruction_loss.item()
                )
                self.losses["validation"]["commitment"][epoch].append(
                    commitment_loss.item()
                )
                self.losses["validation"]["quantization"][epoch].extend(
                    list(indices.detach().cpu().numpy().reshape(-1))
                )

            self.save_to_wandb(epoch)
            
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


        # with open('loss.pkl', 'wb') as f:
        #   print(f"loss.pkl is saved to {os.getcwd()}")
        #   pkl.dump(LOSS, f)

    def test(self, dataloader):
        self.model.eval()
        with torch.no_grad():
            for data in dataloader:
                outputs = self.model(data).to(self.device)
                loss = self.criterion(outputs, data)
            print(f"Test Loss:{loss.item()}")

    def test(self, dataloader):
        self.model.eval()
        with torch.no_grad():
            for data in dataloader:
                outputs = self.model(data)
                loss = self.criterion(outputs, data)
            print(f"Test Loss:{loss.item()}")
