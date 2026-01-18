"""
Statistics module for PROSPECT.

This subpackage contains statistically principled methods for
sample-aware differential expression and confidence scoring,
with explicit modeling of uncertainty and biological replication.
"""

from .sample_aware_de import sample_aware_de
from .confidence_scoring import (
    compute_confidence_score,
    confidence_category,
    score_de_results,
)

__all__ = [
    "sample_aware_de",
    "compute_confidence_score",
    "confidence_category",
    "score_de_results",
]
