"""
soft_identity.py
----------------

Soft (probabilistic) cell identity modeling for PROSPECT.

Key principles:
- Cell identity is continuous, not discrete
- Each cell is represented as a probability distribution
  over latent biological programs
- No hard clustering or forced labels
"""

import logging
from typing import Optional, Tuple

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
    Soft identity assignment using probabilistic program mixtures.

    Each cell is assigned probabilities over K latent programs.
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
        n_programs : int, default=10
            Number of latent biological programs.
        temperature : float, default=1.0
            Softmax temperature (lower = sharper, higher = smoother).
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.n_programs = n_programs
        self.temperature = temperature

        # Learnable program centroids in latent space
        self.programs = nn.Parameter(
            torch.randn(n_programs, latent_dim) * 0.1
        )

        logger.info(
            f"SoftIdentity initialized "
            f"(programs={n_programs}, latent_dim={latent_dim})"
        )

    # ------------------------------------------------------------------
    # Forward
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
            Soft identity probabilities (cells x n_programs).
        """

        # Similarity via negative squared distance
        # shape: (cells, programs)
        dists = torch.cdist(z, self.programs, p=2) ** 2
        logits = -dists / self.temperature

        probs = F.softmax(logits, dim=1)

        return probs

    # ------------------------------------------------------------------
    # Entropy (uncertainty measure)
    # ------------------------------------------------------------------
    @staticmethod
    def entropy(probs: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy of soft identity distributions.

        High entropy = ambiguous identity.

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
    def regularization(self, weight: float = 1e-3) -> torch.Tensor:
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
        return weight * torch.sum(self.programs**2)


# ------------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------------
def most_likely_program(
    probs: torch.Tensor,
) -> torch.Tensor:
    """
    Get most probable program per cell
    (for visualization only; NOT a hard label).

    Parameters
    ----------
    probs : torch.Tensor
        Soft identity probabilities.

    Returns
    -------
    torch.Tensor
        Program indices.
    """
    return torch.argmax(probs, dim=1)


def identity_summary(
    probs: torch.Tensor,
    threshold: float = 0.6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Summarize confident vs ambiguous identities.

    Parameters
    ----------
    probs : torch.Tensor
        Soft identity probabilities.
    threshold : float
        Probability threshold for confident assignment.

    Returns
    -------
    confident : torch.Tensor
        Boolean mask of confident cells.
    max_prob : torch.Tensor
        Maximum probability per cell.
    """
    max_prob, _ = torch.max(probs, dim=1)
    confident = max_prob >= threshold
    return confident, max_prob
