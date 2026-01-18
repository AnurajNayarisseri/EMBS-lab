"""
detection_model.py
------------------

Probabilistic transcript detection model for PROSPECT.

This module models the probability that a transcript is detected
given that it is expressed, explicitly separating detection
uncertainty from biological expression.

No imputation is performed.
"""

import logging
from typing import Optional

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
# Detection probability model
# ------------------------------------------------------------------
class DetectionModel(nn.Module):
    """
    Models gene-specific and cell-specific detection probabilities.

    Detection is modeled as a Bernoulli process:
        observed_count > 0 ~ Bernoulli(p_detect)

    This does NOT infer true expression.
    """

    def __init__(
        self,
        n_genes: int,
        cell_specific: bool = True,
        init_logit: float = -2.0,
    ):
        """
        Parameters
        ----------
        n_genes : int
            Number of genes.
        cell_specific : bool, default=True
            Whether to include cell-specific detection effects.
        init_logit : float, default=-2.0
            Initial logit value for detection probabilities
            (corresponds to ~12% detection).
        """
        super().__init__()

        self.n_genes = n_genes
        self.cell_specific = cell_specific

        # Gene-specific detection logits
        self.gene_logits = nn.Parameter(
            torch.full((n_genes,), init_logit)
        )

        # Optional cell-specific detection bias
        if self.cell_specific:
            self.cell_bias = nn.Parameter(torch.zeros(1))
        else:
            self.register_parameter("cell_bias", None)

        logger.info(
            f"DetectionModel initialized "
            f"(genes={n_genes}, cell_specific={cell_specific})"
        )

    # --------------------------------------------------------------
    # Detection probability
    # --------------------------------------------------------------
    def detection_probability(
        self,
        n_cells: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Compute detection probabilities.

        Parameters
        ----------
        n_cells : int, optional
            Number of cells (required if cell_specific=True).

        Returns
        -------
        torch.Tensor
            Detection probabilities of shape (cells, genes)
            or (genes,) if cell_specific=False.
        """

        gene_p = torch.sigmoid(self.gene_logits)

        if self.cell_specific:
            if n_cells is None:
                raise ValueError(
                    "n_cells must be provided when cell_specific=True"
                )
            cell_bias = torch.sigmoid(self.cell_bias)
            p = gene_p.unsqueeze(0) * cell_bias
            p = p.expand(n_cells, self.n_genes)
        else:
            p = gene_p

        return torch.clamp(p, 1e-5, 1.0 - 1e-5)

    # --------------------------------------------------------------
    # Likelihood
    # --------------------------------------------------------------
    def log_likelihood(
        self,
        observed_counts: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Bernoulli log-likelihood of detection events.

        Parameters
        ----------
        observed_counts : torch.Tensor
            Raw count matrix of shape (cells, genes).

        Returns
        -------
        torch.Tensor
            Scalar log-likelihood.
        """

        detected = (observed_counts > 0).float()
        n_cells, _ = detected.shape

        p = self.detection_probability(n_cells=n_cells)

        ll = (
            detected * torch.log(p)
            + (1.0 - detected) * torch.log(1.0 - p)
        )

        return ll.sum()

    # --------------------------------------------------------------
    # Regularization
    # --------------------------------------------------------------
    def regularization(self, weight: float = 1e-3) -> torch.Tensor:
        """
        Regularize detection logits to avoid extreme probabilities.

        Parameters
        ----------
        weight : float
            Regularization strength.

        Returns
        -------
        torch.Tensor
            Regularization penalty.
        """
        reg = torch.sum(self.gene_logits**2)
        if self.cell_specific:
            reg = reg + torch.sum(self.cell_bias**2)
        return weight * reg

    # --------------------------------------------------------------
    # Full loss
    # --------------------------------------------------------------
    def loss(
        self,
        observed_counts: torch.Tensor,
        reg_weight: float = 1e-3,
    ) -> torch.Tensor:
        """
        Total loss = -log-likelihood + regularization.
        """
        nll = -self.log_likelihood(observed_counts)
        reg = self.regularization(weight=reg_weight)
        return nll + reg


# ------------------------------------------------------------------
# Utility function
# ------------------------------------------------------------------
def estimate_detection_model(
    counts: torch.Tensor,
    n_epochs: int = 500,
    lr: float = 1e-2,
    verbose: bool = True,
) -> DetectionModel:
    """
    Fit detection model to observed counts.

    Parameters
    ----------
    counts : torch.Tensor
        Raw count matrix (cells x genes).
    n_epochs : int
        Number of optimization steps.
    lr : float
        Learning rate.
    verbose : bool
        Whether to print progress.

    Returns
    -------
    DetectionModel
        Trained detection model.
    """

    n_cells, n_genes = counts.shape

    model = DetectionModel(n_genes=n_genes)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        loss = model.loss(counts)
        loss.backward()
        optimizer.step()

        if verbose and epoch % 50 == 0:
            logger.info(
                f"Epoch {epoch:04d} | Loss: {loss.item():.3f}"
            )

    return model
