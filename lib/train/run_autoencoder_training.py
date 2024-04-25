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
import random
import cv2
from moviepy.editor import ImageSequenceClip

# def codebook_diversity_loss(embedding_weights):
#     normed_weights = F.normalize(embedding_weights, p=2, dim=1)
#     similarity = torch.matmul(normed_weights, normed_weights.T)
#     # Remove self-similarity by setting diagonal to zero
#     similarity.fill_diagonal_(0)
#     diversity_loss = torch.mean(similarity)
#     return diversity_loss


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

            print(
                f"""
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
            """
            )
        self.experiment.finish()

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
