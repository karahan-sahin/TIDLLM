import torch
from torch import nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)
        self.commitment_cost = commitment_cost

    def forward(self, z):

        # Flatten z to fit with embedding shape
        z_flattened = z.view(-1, self.embedding_dim)
        # Calculate distances between z and embeddings
        distances = (torch.sum(z_flattened**2, dim=1, keepdim=True) 
                     + torch.sum(self.embedding.weight**2, dim=1)
                     - 2 * torch.matmul(z_flattened, self.embedding.weight.t()))
        
        # Find closest embeddings indices for each item in batch
        min_distances_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        z_q = self.embedding(min_distances_indices).view(z.shape)
        
        # Commitment loss
        loss = F.mse_loss(z, z_q.detach()) * self.commitment_cost
        # Add embedding gradients
        z_q = z + (z_q - z).detach()
        
        return loss, z_q

class VQVAE(nn.Module):
    def __init__(self, encoder, decoder, vq):
        super(VQVAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.vq = vq

    def forward(self, x):
        z = self.encoder(x)
        loss, quantized = self.vq(z)
        x_recon = self.decoder(quantized)

        return loss, x_recon