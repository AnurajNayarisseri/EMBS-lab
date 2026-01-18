"""
batch_latent.py
---------------

Latent batch-effect modeling for PROSPECT.

Batch effects are treated as explicit latent variables that are
jointly modeled with biological variation, rather than removed
through deterministic correction or alignment.

Key principles:
- Batch is a source of variation, not a nuisance to erase
- Biology and batch are allowed to coexist
- Uncertainty is preserved when batch and biology are confounded
"""

import logging
from typing import Tuple, Dict

import torch
import torch.nn as nn


# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Batch latent model
# ------------------------------------------------------------------
class BatchLatent(nn.Module):
    """
    Latent batch-effect embedding.

    Each batch is represented by a learnable embedding that can be
    combined with biological latent variables.
    """

    def __init__(
        self,
        n_batches: int,
        latent_dim: int,
        mode: str = "additive",
    ):
        """
        Parameters
        ----------
        n_batches : int
            Number of unique batches.
        latent_dim : int
            Dimensionality of the biological latent space.
        mode : str, default="additive"
            How batch embeddings interact with biological latents:
              - "additive": z_total = z_bio + z_batch
              - "concat":   z_total = concat(z_bio, z_batch)
        """
        super().__init__()

        if mode not in {"additive", "concat"}:
            raise ValueError("mode must be 'additive' or 'concat'")

        self.n_batches = n_batches
        self.latent_dim = latent_dim
        self.mode = mode

        self.embedding = nn.Embedding(
            num_embeddings=n_batches,
            embedding_dim=latent_dim,
        )

        # Initialize batch embeddings conservatively
        nn.init.zeros_(self.embedding.weight)

        logger.info(
            f"BatchLatent initialized "
            f"(batches={n_batches}, latent_dim={latent_dim}, mode={mode})"
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        z_bio: torch.Tensor,
        batch_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Combine biological and batch latent variables.

        Parameters
        ----------
        z_bio : torch.Tensor
            Biological latent representation (cells x latent_dim).
        batch_ids : torch.Tensor
            Batch indices (cells,).

        Returns
        -------
        torch.Tensor
            Combined latent representation.
        """

        z_batch = self.embedding(batch_ids)

        if self.mode == "additive":
            z = z_bio + z_batch
        else:  # concat
            z = torch.cat([z_bio, z_batch], dim=1)

        return z

    # ------------------------------------------------------------------
    # Regularization
    # ------------------------------------------------------------------
    def regularization(
        self,
        weight: float = 1e-3,
    ) -> torch.Tensor:
        """
        Regularize batch embeddings to avoid dominance.

        Parameters
        ----------
        weight : float
            Regularization strength.

        Returns
        -------
        torch.Tensor
            Regularization penalty.
        """
        return weight * torch.sum(self.embedding.weight ** 2)


# ------------------------------------------------------------------
# Utility: encode batch labels
# ------------------------------------------------------------------
def encode_batches(
    batch_labels,
) -> Tuple[torch.Tensor, Dict]:
    """
    Encode batch labels as integer IDs.

    Parameters
    ----------
    batch_labels : array-like
        Batch labels from adata.obs.

    Returns
    -------
    batch_ids : torch.Tensor
        Encoded batch IDs.
    mapping : dict
        Mapping from original labels to integers.
    """

    unique_batches = sorted(set(batch_labels))
    mapping = {b: i for i, b in enumerate(unique_batches)}

    batch_ids = torch.tensor(
        [mapping[b] for b in batch_labels],
        dtype=torch.long,
    )

    logger.info(f"Encoded {len(unique_batches)} batches")

    return batch_ids, mapping
