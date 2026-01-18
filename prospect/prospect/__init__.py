"""
PROSPECT
========

PRObabilistic Single-cell and Spatial Epistemic Context Toolkit

An uncertainty-aware probabilistic framework for the analysis of
single-cell RNA sequencing and spatial transcriptomics data.

Developed at Eminent Biosciences (EMBS).
"""

__author__ = "Dr. Anuraj Nayarisseri"
__email__ = "contact@eminentbiosciences.com"
__version__ = "0.1.0"

# Public API
from .preprocessing.qc import run_qc

from .models.detection_model import DetectionModel, estimate_detection_model
from .models.probabilistic_vae import ProbabilisticVAE, train_vae
from .models.batch_latent import BatchLatent, encode_batches
from .models.soft_identity import SoftIdentity

from .spatial.spatial_constraints import SpatialConstraint

from .statistics.sample_aware_de import sample_aware_de
from .statistics.confidence_scoring import (
    compute_confidence_score,
    confidence_category,
)

__all__ = [
    "run_qc",
    "DetectionModel",
    "estimate_detection_model",
    "ProbabilisticVAE",
    "train_vae",
    "BatchLatent",
    "encode_batches",
    "SoftIdentity",
    "SpatialConstraint",
    "sample_aware_de",
    "compute_confidence_score",
    "confidence_category",
]
