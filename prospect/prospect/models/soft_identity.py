"""
soft_identity.py
----------------

Soft (probabilistic) cell identity modeling for PROSPECT.

This module represents cell identity as a probability distribution
over latent biological programs rather than discrete clusters.

Key principles:
- Continuous cell states
- Mixed identities allowed
- Explicit uncertainty via entropy
- No hard labels enforced
"""

import logging
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Soft identity model
# ------------------------------------------------------------------
class SoftIdentity(nn.Module):
    """
    Soft probabilistic identity assignment.

    Each cell is assigned a probability distribution over
    K latent biological programs.
    """

    def __init__(
        self,
        latent_dim: int,
        n_programs: int = 10,
        temperature: float = 1.0,
    ):
        """
        Parameters
        ----------
        latent_dim : int
            Dimensionality of latent space.
        n_programs : int
            Number of latent programs.
        temperature : float
            Softmax temperature (lower = sharper, higher = smoother).
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.n_programs = n_programs
        self.temperature = temperature

        # Learnable program centroids in latent space
        self.program_centroids = nn.Parameter(
            torch.randn(n_programs, latent_dim) * 0.05
        )

        logger.info(
            f"SoftIdentity initialized "
            f"(programs={n_programs}, latent_dim={latent_dim})"
        )

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(
        self,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute soft identity probabilities.

        Parameters
        ----------
        z : torch.Tensor
            Latent representations (cells x latent_dim).

        Returns
        -------
        torch.Tensor
            Identity probability matrix (cells x n_programs).
        """

        # Squared Euclidean distance to each program centroid
        # shape: (cells, programs)
        dists = torch.cdist(z, self.program_centroids, p=2) ** 2

        # Convert distances to similarity scores
        logits = -dists / self.temperature

        probs = F.softmax(logits, dim=1)

        return probs

    # ------------------------------------------------------------------
    # Identity uncertainty (entropy)
    # ------------------------------------------------------------------
    @staticmethod
    def entropy(
        probs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute entropy of identity distributions.

        High entropy indicates ambiguous or mixed identity.

        Parameters
        ----------
        probs : torch.Tensor
            Identity probabilities (cells x programs).

        Returns
        -------
        torch.Tensor
            Entropy per cell.
        """
        eps = 1e-8
        return -torch.sum(probs * torch.log(probs + eps), dim=1)

    # ------------------------------------------------------------------
    # Regularization
    # ------------------------------------------------------------------
    def regularization(
        self,
        weight: float = 1e-3,
    ) -> torch.Tensor:
        """
        Regularize program centroids to avoid collapse.

        Parameters
        ----------
        weight : float
            Regularization strength.

        Returns
        -------
        torch.Tensor
            Regularization penalty.
        """
        return weight * torch.sum(self.program_centroids ** 2)


# ------------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------------
def most_likely_program(
    probs: torch.Tensor,
) -> torch.Tensor:
    """
    Identify the most probable program per cell.

    NOTE:
    This is provided for visualization only and should not be
    interpreted as a definitive cell-type assignment.
    """
    return torch.argmax(probs, dim=1)


def identity_summary(
    probs: torch.Tensor,
    confidence_threshold: float = 0.6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Summarize confident versus ambiguous identities.

    Parameters
    ----------
    probs : torch.Tensor
        Identity probability matrix.
    confidence_threshold : float
        Threshold for confident identity.

    Returns
    -------
    confident : torch.Tensor
        Boolean mask of confident cells.
    max_prob : torch.Tensor
        Maximum identity probability per cell.
    """
    max_prob, _ = torch.max(probs, dim=1)
    confident = max_prob >= confidence_threshold
    return confident, max_prob
