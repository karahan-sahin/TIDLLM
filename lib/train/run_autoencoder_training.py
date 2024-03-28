import torch
from torch import nn, optim
from tqdm import tqdm

class AutoencoderTrainer:
    def __init__(self, model, learning_rate, device='cuda'):
        self.model = model
        self.model.to(device)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def train(self, dataloader, num_epochs):
        self.model.train()
        for epoch in tqdm(range(num_epochs)):
            for batch in dataloader:
                data = batch['array']
                data = data.float()
                self.optimizer.zero_grad()
                commitment_loss, x_recon, indices, quantized = self.model(data)
                reconsctruction_loss = self.criterion(x_recon, data)
                loss = commitment_loss + reconsctruction_loss
                loss.backward()
                self.optimizer.step()
            print(f"Epoch:{epoch+1}, Loss:{loss.item()}")

    def test(self, dataloader):
        self.model.eval()
        with torch.no_grad():
            for data in dataloader:
                outputs = self.model(data)
                loss = self.criterion(outputs, data)
            print(f'Test Loss:{loss.item()}')
