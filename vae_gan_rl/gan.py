import torch
import torch.nn as nn
import torch.nn.functional as F

#############################################
# Generator (latent → molecular fingerprint)
#############################################

class MolGenerator(nn.Module):
    """
    GAN generator that maps latent vectors
    to molecular fingerprints
    """

    def __init__(self, latent_dim=128, fp_dim=2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, fp_dim),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.net(z)


#############################################
# Discriminator (fingerprint → real/fake)
#############################################

class MolDiscriminator(nn.Module):
    """
    Discriminator distinguishing
    real CDK2 inhibitors vs generated molecules
    """

    def __init__(self, fp_dim=2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(fp_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


#############################################
# Wasserstein loss (for stability)
#############################################

def gan_loss(pred_real, pred_fake):
    return torch.mean(pred_fake) - torch.mean(pred_real)
