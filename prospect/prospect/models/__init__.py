"""
Models module for PROSPECT.

This subpackage contains probabilistic models used to represent
transcript detection uncertainty, latent cellular states,
batch-associated variation, and continuous cell identity.
"""

from .detection_model import DetectionModel, estimate_detection_model
from .probabilistic_vae import ProbabilisticVAE, train_vae
from .batch_latent import BatchLatent, encode_batches
from .soft_identity import SoftIdentity

__all__ = [
    "DetectionModel",
    "estimate_detection_model",
    "ProbabilisticVAE",
    "train_vae",
    "BatchLatent",
    "encode_batches",
    "SoftIdentity",
]
