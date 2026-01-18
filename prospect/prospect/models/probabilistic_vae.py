"""
probabilistic_vae.py
--------------------

Probabilistic Variational Autoencoder (VAE) for PROSPECT.

This module models continuous cellular transcriptional states
as probability distributions in latent space rather than
discrete clusters.

Key principles:
- Each cell is represented by a latent distribution
- No hard clustering
- Uncertainty is explicitly retained
"""

import logging
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence


# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Encoder network
# ------------------------------------------------------------------
class Encoder(nn.Module):
    """
    Encoder network mapping gene expression to latent distributions.
    """

    def __init__(
        self,
        n_genes: int,
        latent_dim: int = 20,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(n_genes, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.mu_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : torch.Tensor
            Expression matrix (cells x genes).

        Returns
        -------
        mu : torch.Tensor
            Latent mean.
        logvar : torch.Tensor
            Latent log-variance.
        """
        h = self.network(x)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar


# ------------------------------------------------------------------
# Decoder network
# ------------------------------------------------------------------
class Decoder(nn.Module):
    """
    Decoder network mapping latent variables to expression space.

    Outputs expected expression intensity (not imputed counts).
    """

    def __init__(
        self,
        latent_dim: int,
        n_genes: int,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_genes),
        )

    def forward(
        self,
        z: torch.Tensor,
    ) -> torch.Tensor:
        return self.network(z)


# ------------------------------------------------------------------
# Probabilistic VAE
# ------------------------------------------------------------------
class ProbabilisticVAE(nn.Module):
    """
    Probabilistic Variational Autoencoder.

    Latent prior: N(0, I)
    Posterior: N(mu, sigma)
    """

    def __init__(
        self,
        n_genes: int,
        latent_dim: int = 20,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.encoder = Encoder(
            n_genes=n_genes,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        self.decoder = Decoder(
            latent_dim=latent_dim,
            n_genes=n_genes,
            hidden_dim=hidden_dim,
        )

        self.latent_dim = latent_dim

        logger.info(
            f"ProbabilisticVAE initialized "
            f"(genes={n_genes}, latent_dim={latent_dim})"
        )

    # --------------------------------------------------------------
    # Reparameterization trick
    # --------------------------------------------------------------
    @staticmethod
    def reparameterize(
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sample latent variables using reparameterization.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    # --------------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : torch.Tensor
            Expression matrix (cells x genes).

        Returns
        -------
        recon : torch.Tensor
            Reconstructed expression intensity.
        mu : torch.Tensor
            Latent mean.
        logvar : torch.Tensor
            Latent log-variance.
        """
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

    # --------------------------------------------------------------
    # Loss function (ELBO)
    # --------------------------------------------------------------
    def loss(
        self,
        x: torch.Tensor,
        recon: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        kl_weight: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute Evidence Lower Bound (ELBO).

        Returns
        -------
        total_loss : torch.Tensor
        recon_loss : torch.Tensor
        kl_div : torch.Tensor
        """

        # Reconstruction loss (conservative, avoids hallucination)
        recon_loss = F.mse_loss(recon, x, reduction="sum")

        # KL divergence
        q = Normal(mu, torch.exp(0.5 * logvar))
        p = Normal(
            torch.zeros_like(mu),
            torch.ones_like(mu),
        )

        kl = kl_divergence(q, p).sum()

        total_loss = recon_loss + kl_weight * kl

        return total_loss, recon_loss, kl


# ------------------------------------------------------------------
# Training utility
# ------------------------------------------------------------------
def train_vae(
    model: ProbabilisticVAE,
    data: torch.Tensor,
    n_epochs: int = 500,
    lr: float = 1e-3,
    kl_weight: float = 1.0,
    verbose: bool = True,
) -> ProbabilisticVAE:
    """
    Train the probabilistic VAE.

    Parameters
    ----------
    model : ProbabilisticVAE
    data : torch.Tensor
        Expression matrix (cells x genes).
    """

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(n_epochs):
        optimizer.zero_grad()

        recon, mu, logvar = model(data)
        loss, recon_loss, kl = model.loss(
            data,
            recon,
            mu,
            logvar,
            kl_weight=kl_weight,
        )

        loss.backward()
        optimizer.step()

        if verbose and epoch % 50 == 0:
            logger.info(
                f"[VAE] Epoch {epoch:04d} | "
                f"Total: {loss.item():.2f} | "
                f"Recon: {recon_loss.item():.2f} | "
                f"KL: {kl.item():.2f}"
            )

    return model
