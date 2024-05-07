import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import json


with open('code/utils/hyperparameters.json') as f:
    hyperparameters = json.load(f)

BATCH_SIZE = hyperparameters['BATCH_SIZE']
DROPOUT_RATE = hyperparameters['DROPOU_RATE']
beta = hyperparameters['BETA']


class model(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(model, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2*latent_dim)  # Output mean and variance for each latent dimension
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Tanh(),
        )
        self.latent_dim = latent_dim

    def encode(self, x):
        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=-1)  # Split the output into mean and log variance
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

    def loss_function(self, x, x_recon, mu, logvar):
        if torch.isnan(x).any() or torch.isnan(x_recon).any():
            raise ValueError("Input or Output data contains NaN values")
        # Reconstruction loss using Mean Squared Error
        recon_loss = F.mse_loss(x_recon, x, reduction='sum')

        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # Total loss
        return recon_loss + kl_loss



class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(in_features=input_dim, out_features=512)
        self.fc21 = nn.Linear(in_features=512, out_features=latent_dim)
        self.fc22 = nn.Linear(in_features=512, out_features=latent_dim)
        self.fc3 = nn.Linear(in_features=latent_dim, out_features=512)
        self.fc4 = nn.Linear(in_features=512, out_features=input_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(DROPOUT_RATE)


    def encode(self, x):
        h1 = self.dropout(self.relu(self.fc1(x)))
        return self.dropout(self.fc21(h1)), self.dropout(self.fc22(h1))

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return mu + eps*std
        else:
            return mu

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        return self.tanh(self.dropout(self.fc4(h3)))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        recon_loss = torch.mean(torch.pow(recon_x - x, 2)) #/ BATCH_SIZE
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp()) #/ BATCH_SIZE
        return (1-beta)*recon_loss + beta*kl_loss
