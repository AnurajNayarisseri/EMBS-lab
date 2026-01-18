"""
confidence_scoring.py
---------------------

Confidence and uncertainty scoring for PROSPECT.

This module integrates multiple evidence streams to produce
interpretable confidence scores for inferred biological results.

Principles:
- Explicit uncertainty
- Conservative aggregation
- Transparent thresholds
"""

import logging
from typing import Dict, Optional

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
    Compute an ove
