import torch
from torch import nn

"""
Vector Quantized Variational AutoEncoder 
"""

class VQVAE(nn.Module):
    def __init__(self, encoder, decoder, vq):
        super(VQVAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.vq = vq

    def encode(self, x):
        return self.encoder(x)

    def quantize(self, x):
        z = self.encoder(x)
        _, indices, _ = self.vq(z)
        return z

    def forward(self, x):
        z = self.encoder(x)
        quantized, indices, commitment_loss = self.vq(z)
        x_recon = self.decoder(quantized)

        return x_recon, indices, commitment_loss