"""
Identity module for PROSPECT.

This subpackage contains utilities for interpreting and summarizing
probabilistic cell identity representations, including uncertainty
and confidence metrics derived from soft identity distributions.
"""

from .identity_metrics import (
    identity_entropy,
    identity_purity,
)
from .identity_summary import summarize_identities

__all__ = [
    "identity_entropy",
    "identity_purity",
    "summarize_identities",
]
