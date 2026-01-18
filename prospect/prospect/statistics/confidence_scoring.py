"""
confidence_scoring.py
---------------------

Confidence and uncertainty scoring for PROSPECT.

This module integrates multiple evidence streams to produce
interpretable confidence scores for inferred biological results.

Design principles:
- Explicit uncertainty
- Conservative aggregation of evidence
- Separation of confidence from statistical significance
"""

import logging
from typing import Dict, Optional, Set

import numpy as np
import pandas as pd


# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Core confidence scoring
# ------------------------------------------------------------------
def compute_confidence_score(
    replicated: bool,
    multimodal_consistent: bool,
    effect_size: float,
    uncertainty: Optional[float] = None,
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """
    Compute an overall confidence score for a biological result.

    Parameters
    ----------
    replicated : bool
        Whether the result is supported across biological replicates.
    multimodal_consistent : bool
        Whether the result is consistent across modalities
        (e.g., scRNA + spatial).
    effect_size : float
        Magnitude of biological effect (e.g., |log2FC|).
    uncertainty : float, optional
        Estimated uncertainty (higher = more uncertain), normalized to [0, 1].
    weights : dict, optional
        Weights for each evidence component.

    Returns
    -------
    float
        Confidence score between 0 and 1.
    """

    if weights is None:
        weights = {
            "replication": 0.4,
            "multimodal": 0.3,
            "effect": 0.2,
            "uncertainty": 0.1,
        }

    score = 0.0

    # Replication support
    if replicated:
        score += weights["replication"]

    # Multimodal agreement
    if multimodal_consistent:
        score += weights["multimodal"]

    # Effect size contribution (scaled, capped)
    effect_component = min(abs(effect_size) / 2.0, 1.0)
    score += weights["effect"] * effect_component

    # Uncertainty penalty (higher uncertainty reduces confidence)
    if uncertainty is not None:
        uncertainty_component = max(1.0 - uncertainty, 0.0)
        score += weights["uncertainty"] * uncertainty_component

    return float(min(score, 1.0))


# ------------------------------------------------------------------
# Confidence categorization
# ------------------------------------------------------------------
def confidence_category(score: float) -> str:
    """
    Categorize confidence score.

    Parameters
    ----------
    score : float
        Confidence score in [0, 1].

    Returns
    -------
    str
        Confidence category.
    """

    if score >= 0.75:
        return "High confidence"
    elif score >= 0.4:
        return "Moderate confidence"
    else:
        return "Hypothesis-generating"


# ------------------------------------------------------------------
# Batch scoring for differential expression results
# ------------------------------------------------------------------
def score_de_results(
    de_df: pd.DataFrame,
    replicated_genes: Optional[Set[str]] = None,
    multimodal_genes: Optional[Set[str]] = None,
    uncertainty_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Apply confidence scoring to differential expression results.

    Parameters
    ----------
    de_df : pd.DataFrame
        Differential expression results. Must include 'gene' and 'log2FC'.
    replicated_genes : set, optional
        Genes supported across biological replicates.
    multimodal_genes : set, optional
        Genes supported by multiple modalities.
    uncertainty_col : str, optional
        Column containing per-gene uncertainty estimates (0–1).

    Returns
    -------
    pd.DataFrame
        DE results with confidence scores and categories.
    """

    replicated_genes = replicated_genes or set()
    multimodal_genes = multimodal_genes or set()

    scores = []
    categories = []

    for _, row in de_df.iterrows():
        gene = row["gene"]
        effect_size = row["log2FC"]

        replicated = gene in replicated_genes
        multimodal = gene in multimodal_genes

        uncertainty = (
            row[uncertainty_col]
            if uncertainty_col is not None and uncertainty_col in de_df.columns
            else None
        )

        score = compute_confidence_score(
            replicated=replicated,
            multimodal_consistent=multimodal,
            effect_size=effect_size,
            uncertainty=uncertainty,
        )

        scores.append(score)
        categories.append(confidence_category(score))

    out = de_df.copy()
    out["confidence_score"] = scores
    out["confidence_category"] = categories

    logger.info("Applied confidence scoring to DE results")

    return out


# ------------------------------------------------------------------
# Utility: uncertainty normalization
# ------------------------------------------------------------------
def normalize_uncertainty(
    uncertainty_values: np.ndarray,
) -> np.ndarray:
    """
    Normalize uncertainty values to the range [0, 1].

    Parameters
    ----------
    uncertainty_values : np.ndarray
        Raw uncertainty measurements.

    Returns
    -------
    np.ndarray
        Normalized uncertainty values.
    """

    uncertainty_values = np.asarray(uncertainty_values)

    min_u = np.min(uncertainty_values)
    max_u = np.max(uncertainty_values)

    if max_u == min_u:
        return np.zeros_like(uncertainty_values)

    return (uncertainty_values - min_u) / (max_u - min_u)
