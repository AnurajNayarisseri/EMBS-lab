"""
identity_summary.py
-------------------

High-level summaries for probabilistic cell identity in PROSPECT.

This module provides interpretable summaries derived from soft
identity probability distributions, enabling transparent reporting
of identity confidence and uncertainty.

Design principles:
- Identity summaries are descriptive, not inferential
- No hard cell-type assignment is enforced
- Metrics are simple, explicit, and reproducible
"""

import numpy as np
import pandas as pd

from .identity_metrics import (
    identity_entropy,
    identity_purity,
    identity_dominance,
    identity_uncertainty,
)


# ------------------------------------------------------------------
# Main identity summary
# ------------------------------------------------------------------
def summarize_identities(
    probs: np.ndarray,
    purity_threshold: float = 0.6,
    dominance_threshold: float = 0.2,
) -> pd.DataFrame:
    """
    Summarize probabilistic cell identities.

    Parameters
    ----------
    probs : np.ndarray
        Identity probability matrix (cells x programs).
    purity_threshold : float, default=0.6
        Minimum purity to consider an identity confident.
    dominance_threshold : float, default=0.2
        Minimum dominance to distinguish a primary identity
        from secondary identities.

    Returns
    -------
    pd.DataFrame
        Per-cell identity summary table with the following columns:
        - identity_purity
        - identity_entropy
        - identity_dominance
        - identity_uncertainty
        - confident_identity (boolean)
    """

    if probs.ndim != 2:
        raise ValueError("probs must be a 2D array (cells x programs)")

    purity = identity_purity(probs)
    entropy = identity_entropy(probs)
    dominance = identity_dominance(probs)
    uncertainty = identity_uncertainty(probs)

    confident = (purity >= purity_threshold) & (
        dominance >= dominance_threshold
    )

    summary_df = pd.DataFrame(
        {
            "identity_purity": purity,
            "identity_entropy": entropy,
            "identity_dominance": dominance,
            "identity_uncertainty": uncertainty,
            "confident_identity": confident,
        }
    )

    return summary_df


# ------------------------------------------------------------------
# Program-level summary
# ------------------------------------------------------------------
def summarize_programs(
    probs: np.ndarray,
) -> pd.DataFrame:
    """
    Summarize latent biological programs across cells.

    Parameters
    ----------
    probs : np.ndarray
        Identity probability matrix (cells x programs).

    Returns
    -------
    pd.DataFrame
        Program-level summary including:
        - mean_probability
        - median_probability
        - fraction_dominant
    """

    if probs.ndim != 2:
        raise ValueError("probs must be a 2D array (cells x programs)")

    dominant_program = np.argmax(probs, axis=1)

    summaries = []

    for k in range(probs.shape[1]):
        summaries.append(
            {
                "program": k,
                "mean_probability": float(np.mean(probs[:, k])),
                "median_probability": float(np.median(probs[:, k])),
                "fraction_dominant": float(
                    np.mean(dominant_program == k)
                ),
            }
        )

    return pd.DataFrame(summaries)
