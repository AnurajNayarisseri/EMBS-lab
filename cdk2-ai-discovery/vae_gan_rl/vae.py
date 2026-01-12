import torch
import torch.nn as nn
import torch.nn.functional as F

#############################################
# Molecular Variational Autoencoder (VAE)
#############################################

class MolecularVAE(nn.Module):
    """
    Graph-based molecular VAE operating on Morgan fingerprints
    Maps molecules → latent space → reconstructs molecules
    """

    def __init__(self, input_dim=2048, latent_dim=128):
        super().__init__()

        # Encoder
        self.fc1 = nn.Linear(input_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.mu = nn.Linear(512, latent_dim)
        self.logvar = nn.Linear(512, latent_dim)

        # Decoder
        self.fc3 = nn.Linear(latent_dim, 512)
        self.fc4 = nn.Linear(512, 1024)
        self.out = nn.Linear(1024, input_dim)

    #########################################
    # Encode fingerprint → latent
    #########################################
    def encode(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.mu(h), self.logvar(h)

    #########################################
    # Reparameterization trick
    #########################################
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    #########################################
    # Decode latent → fingerprint
    #########################################
    def decode(self, z):
        h = F.relu(self.fc3(z))
        h = F.relu(self.fc4(h))
        return torch.sigmoid(self.out(h))

    #########################################
    # Forward
    #########################################
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


#############################################
# VAE Loss
#############################################

def vae_loss(x, recon, mu, logvar):
    recon_loss = F.binary_cross_entropy(recon, x, reduction="sum")
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl
