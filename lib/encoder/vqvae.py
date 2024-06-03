import torch
from torch import nn
import torch.nn.functional as F

class VQVAE_POSE(nn.Module):
    def __init__(self,
                 encoder,
                 decoder,
                 vq_vae,
                 ):
        super(VQVAE_POSE, self).__init__()
        self.encoder = encoder
        self.vq_vae = vq_vae
        self.decoder = decoder

    def forward(self, x):
        z = self.encoder(x)
        quantized, indices, commit_loss = self.vq_vae(z)
        output = self.decoder(quantized)
        return output, indices, commit_loss