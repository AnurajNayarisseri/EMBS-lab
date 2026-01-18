"""
spatial_constraints.py
----------------------

Spatial constraint modeling for PROSPECT.

This module incorporates spatial transcriptomics information
as soft constraints on cellular identity rather than enforcing
deterministic spatial assignments.

Key principles:
- Spatial proximity informs plausibility, not certainty
- No spatial deconvolution or hard labeling
- Uncertainty increases where spatial signals conflict
"""

import logging
from typing import Optional

import numpy as np
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
# Spatial constraint model
# ------------------------------------------------------------------
class SpatialConstraint(nn.Module):
    """
    Spatial constraint model based on neighborhood consistency.

    Cells or spots that are spatially close are expected to have
    more similar identity distributions, but violations are allowed
    and reflected as increased uncertainty.
    """

    def __init__(
        self,
        coords: np.ndarray,
        radius: float = 50.0,
        kernel: str = "gaussian",
        sigma: Optional[float] = None,
    ):
        """
        Parameters
        ----------
        coords : np.ndarray
            Spatial coordinates (n_spots x 2).
        radius : float
            Neighborhood radius (in spatial units).
        kernel : str, default="gaussian"
            Weighting kernel: "gaussian" or "binary".
        sigma : float, optional
            Length scale for gaussian kernel.
        """
        super().__init__()

        if kernel not in {"gaussian", "binary"}:
            raise ValueError("kernel must be 'gaussian' or 'binary'")

        self.coords = torch.tensor(coords, dtype=torch.float32)
        self.radius = radius
        self.kernel = kernel
        self.sigma = sigma if sigma is not None else radius / 2.0

        logger.info(
            f"SpatialConstraint initialized "
            f"(spots={coords.shape[0]}, radius={radius}, kernel={kernel})"
        )

        # Precompute neighborhood graph
        self.neighbors, self.weights = self._build_graph()

    # ------------------------------------------------------------------
    # Build neighborhood graph
    # ------------------------------------------------------------------
    def _build_graph(self):
        """
        Construct neighborhood relationships and weights.

        Returns
        -------
        neighbors : list of torch.Tensor
            Indices of neighbors for each spot.
        weights : list of torch.Tensor
            Corresponding weights.
        """
        coords = self.coords
        n = coords.shape[0]

        # Pairwise distances
        dists = torch.cdist(coords, coords, p=2)

        neighbors = []
        weights = []

        for i in range(n):
            mask = (dists[i] > 0) & (dists[i] <= self.radius)
            idx = torch.where(mask)[0]

            if self.kernel == "binary":
                w = torch.ones(len(idx))
            else:
                w = torch.exp(
                    - (dists[i, idx] ** 2) / (2 * self.sigma ** 2)
                )

            # Normalize weights
            if len(w) > 0:
                w = w / w.sum()

            neighbors.append(idx)
            weights.append(w)

        return neighbors, weights

    # ------------------------------------------------------------------
    # Spatial consistency loss
    # ------------------------------------------------------------------
    def consistency_loss(
        self,
        identity_probs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Penalize strong disagreement between neighboring spots.

        Parameters
        ----------
        identity_probs : torch.Tensor
            Soft identity probabilities (spots x programs).

        Returns
        -------
        torch.Tensor
            Scalar spatial consistency loss.
        """
        loss = 0.0
        n = identity_probs.shape[0]

        for i in range(n):
            if len(self.neighbors[i]) == 0:
                continue

            neigh_probs = identity_probs[self.neighbors[i]]
            w = self.weights[i].to(identity_probs.device)

            diff = identity_probs[i] - neigh_probs
            sq = (diff ** 2).sum(dim=1)

            loss += torch.sum(w * sq)

        return loss / n

    # ------------------------------------------------------------------
    # Spatial uncertainty estimation
    # ------------------------------------------------------------------
    def uncertainty(
        self,
        identity_probs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Estimate spatial uncertainty for each spot.

        Uncertainty increases when a spot's identity distribution
        differs strongly from its neighbors.

        Parameters
        ----------
        identity_probs : torch.Tensor
            Soft identity probabilities (spots x programs).

        Returns
        -------
        torch.Tensor
            Uncertainty score per spot (normalized to [0, 1]).
        """
        n = identity_probs.shape[0]
        scores = torch.zeros(n)

        for i in range(n):
            if len(self.neighbors[i]) == 0:
                scores[i] = 0.0
                continue

            neigh_probs = identity_probs[self.neighbors[i]]
            w = self.weights[i].to(identity_probs.device)

            diff = identity_probs[i] - neigh_probs
            sq = (diff ** 2).sum(dim=1)

            scores[i] = torch.sum(w * sq)

        # Normalize to [0, 1]
        if scores.max() > scores.min():
            scores = (scores - scores.min()) / (scores.max() - scores.min())

        return scores
