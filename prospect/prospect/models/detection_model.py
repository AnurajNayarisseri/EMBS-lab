"""
detection_model.py
------------------

Probabilistic transcript detection model for PROSPECT.

This module explicitly models transcript detection uncertainty,
separating technical non-detection from biological absence.

Key design principles:
- Detection is probabilistic (Bernoulli)
- No imputation of expression values
- Gene-specific (and optional cell-specific) detection probabilities
"""

import logging
from typing import Optional

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
# Detection model
# ------------------------------------------------------------------
class DetectionModel(nn.Module):
    """
    Probabilistic model of transcript detection.

    Detection is modeled as:
        detected ~ Bernoulli(p_detect)

    where p_detect is learned from data.
    """

    def __init__(
        self,
        n_genes: int,
        cell_specific: bool = False,
        init_logit: float = -2.0,
    ):
        """
        Parameters
        ----------
        n_genes : int
            Number of genes.
        cell_specific : bool, default=False
            Whether to include a global cell-level detection bias.
        init_logit : float, default=-2.0
            Initial logit for detection probability (~12%).
        """
        super().__init__()

        self.n_genes = n_genes
        self.cell_specific = cell_specific

        # Gene-specific detection logits
        self.gene_logits = nn.Parameter(
            torch.full((n_genes,), init_logit)
        )

        # Optional global cell detection bias
        if cell_specific:
            self.cell_logit = nn.Parameter(torch.tensor(0.0))
        else:
            self.register_parameter("cell_logit", None)

        logger.info(
            f"DetectionModel initialized "
            f"(genes={n_genes}, cell_specific={cell_specific})"
        )

    # ------------------------------------------------------------------
    # Detection probabilities
    # ------------------------------------------------------------------
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
            Detection probability matrix (cells x genes) or
            vector (genes,) if cell_specific=False.
        """

        gene_p = torch.sigmoid(self.gene_logits)

        if self.cell_specific:
            if n_cells is None:
                raise ValueError(
                    "n_cells must be provided when cell_specific=True"
                )
            cell_p = torch.sigmoid(self.cell_logit)
            p = gene_p.unsqueeze(0) * cell_p
            p = p.expand(n_cells, self.n_genes)
        else:
            p = gene_p

        return torch.clamp(p, 1e-6, 1.0 - 1e-6)

    # ------------------------------------------------------------------
    # Log-likelihood
    # ------------------------------------------------------------------
    def log_likelihood(
        self,
        counts: torch.Tensor,
    ) -> torch.Tensor:
        """
        Bernoulli log-likelihood of observed detection events.

        Parameters
        ----------
        counts : torch.Tensor
            Raw count matrix (cells x genes).

        Returns
        -------
        torch.Tensor
            Scalar log-likelihood.
        """

        detected = (counts > 0).float()
        n_cells, _ = detected.shape

        p = self.detection_probability(n_cells=n_cells)

        ll = (
            detected * torch.log(p)
            + (1.0 - detected) * torch.log(1.0 - p)
        )

        return ll.sum()

    # ------------------------------------------------------------------
    # Regularization
    # ------------------------------------------------------------------
    def regularization(
        self,
        weight: float = 1e-3,
    ) -> torch.Tensor:
        """
        L2 regularization to prevent extreme probabilities.

        Parameters
        ----------
        weight : float
            Regularization strength.

        Returns
        -------
        torch.Tensor
            Regularization penalty.
        """

        reg = torch.sum(self.gene_logits ** 2)

        if self.cell_specific:
            reg = reg + self.cell_logit ** 2

        return weight * reg

    # ------------------------------------------------------------------
    # Total loss
    # ------------------------------------------------------------------
    def loss(
        self,
        counts: torch.Tensor,
        reg_weight: float = 1e-3,
    ) -> torch.Tensor:
        """
        Total loss = negative log-likelihood + regularization.
        """

        nll = -self.log_likelihood(counts)
        reg = self.regularization(reg_weight)
        return nll + reg


# ------------------------------------------------------------------
# Training utility
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
        Whether to log progress.

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
                f"[DetectionModel] Epoch {epoch:04d} | "
                f"Loss: {loss.item():.3f}"
            )

    return model
