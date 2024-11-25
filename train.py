import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from model import *

batch_size = 32
learning_rate = 1e-3
epochs = 5000

def load_data(file_path):
    data = pd.read_csv(file_path, header=None)  # Assuming no header row
    return torch.tensor(data.values, dtype=torch.float32)
data = load_data("data.csv")
dataset = TensorDataset(data, data)  # Autoencoder-style training (input == output)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

vae = VAE(input_dim, hidden_dim, latent_dim)
optimizer = torch.optim.Adam(vae.parameters(), lr=learning_rate)

vae.train()
for epoch in range(epochs):
    total_loss = 0
    for batch in data_loader:  # Assume data_loader is defined
        x, _ = batch
        recon_x, mean, logvar = vae(x)
        loss = vae_loss(recon_x, x, mean, logvar)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    if epoch % 50 == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

print("Training complete!")

torch.save(vae.state_dict(), "mocap_vae.pth")
print("Model saved!")