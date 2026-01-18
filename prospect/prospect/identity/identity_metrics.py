"""
identity_metrics.py
-------------------

Quantitative metrics for probabilistic cell identity in PROSPECT.

This module provides interpretable, model-agnostic metrics derived
from soft identity probability distributions.

Design principles:
- Identity is probabilistic, not categorical
- Uncertainty is explicitly quantified
- Metrics are simple, transparent, and reproducible
"""

import numpy as np


# ------------------------------------------------------------------
# Identity entropy
# ------------------------------------------------------------------
def identity_entropy(
    probs: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Compute Shannon entropy of identity distributions.

    High entropy indicates ambiguous or mixed identity,
    while low entropy indicates a more concentrated identity.

    Parameters
    ----------
    probs : np.ndarray
        Identity probability matrix (cells x programs).
    eps : float
        Small constant to ensure numerical stability.

    Returns
    -------
    np.ndarray
        Entropy per cell.
    """

    if probs.ndim != 2:
        raise ValueError("probs must be a 2D array (cells x programs)")

    return -np.sum(probs * np.log(probs + eps), axis=1)


# ------------------------------------------------------------------
# Identity purity
# ------------------------------------------------------------------
def identity_purity(
    probs: np.ndarray,
) -> np.ndarray:
    """
    Compute identity purity for each cell.

    Purity is defined as the maximum identity probability and
    reflects how strongly a cell is associated with a single
    latent biological program.

    Parameters
    ----------
    probs : np.ndarray
        Identity probability matrix (cells x programs).

    Returns
    -------
    np.ndarray
        Purity score per cell.
    """

    if probs.ndim != 2:
        raise ValueError("probs must be a 2D array (cells x programs)")

    return np.max(probs, axis=1)


# ------------------------------------------------------------------
# Identity dominance
# ------------------------------------------------------------------
def identity_dominance(
    probs: np.ndarray,
) -> np.ndarray:
    """
    Compute identity dominance score.

    Dominance measures the difference between the highest and
    second-highest identity probabilities.

    This helps distinguish:
    - confident single-program identities
    - borderline mixed identities

    Parameters
    ----------
    probs : np.ndarray
        Identity probability matrix (cells x programs).

    Returns
    -------
    np.ndarray
        Dominance score per cell.
    """

    if probs.ndim != 2:
        raise ValueError("probs must be a 2D array (cells x programs)")

    sorted_probs = np.sort(probs, axis=1)
    return sorted_probs[:, -1] - sorted_probs[:, -2]


# ------------------------------------------------------------------
# Identity uncertainty (normalized)
# ------------------------------------------------------------------
def identity_uncertainty(
    probs: np.ndarray,
) -> np.ndarray:
    """
    Compute normalized identity uncertainty score.

    This score rescales entropy to [0, 1] for easier comparison
    and visualization.

    Parameters
    ----------
    probs : np.ndarray
        Identity probability matrix (cells x programs).

    Returns
    -------
    np.ndarray
        Normalized uncertainty per cell.
    """

    entropy = identity_entropy(probs)

    if entropy.max() > entropy.min():
        entropy = (entropy - entropy.min()) / (entropy.max() - entropy.min())
    else:
        entropy = np.zeros_like(entropy)

    return entropy
