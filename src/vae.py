import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, input_size):
        super(VAE, self).__init__()

        self.input_size = input_size
        self.latent_dim = 20
        self.hidden_dim = 400

        self.fc1 = nn.Linear(input_size, self.hidden_dim)
        self.fc2_mean = nn.Linear(self.hidden_dim, self.latent_dim)
        self.fc2_logvar = nn.Linear(self.hidden_dim, self.latent_dim)

        self.fc3 = nn.Linear(self.latent_dim, self.hidden_dim)
        self.fc4 = nn.Linear(self.hidden_dim, input_size)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        mu = self.fc2_mean(h)
        log_var = self.fc2_logvar(h)
        return mu, log_var

    def reparameterize(self, mu, logvar):
        std = torch.exp(logvar / 2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.fc3(z))
        out = torch.sigmoid(self.fc4(h))
        return out

    def forward(self, x):
        mu, log_var = self.encode(x.view(-1, self.input_size))
        z = self.reparameterize(mu, log_var)
        reconstructed = self.decode(z)
        return reconstructed, mu, log_var
