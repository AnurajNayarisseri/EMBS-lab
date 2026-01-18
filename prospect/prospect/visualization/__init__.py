"""
Visualization module for PROSPECT.

This subpackage contains publication-quality, uncertainty-aware
visualization utilities for single-cell and spatial transcriptomics
analysis. Plots are designed to emphasize probabilistic structure
and confidence rather than discrete clustering.
"""

from .plots import (
    plot_latent_space,
    plot_soft_identity_heatmap,
    plot_detection_probabilities,
    plot_sample_aware_volcano,
    plot_spatial_uncertainty,
)

__all__ = [
    "plot_latent_space",
    "plot_soft_identity_heatmap",
    "plot_detection_probabilities",
    "plot_sample_aware_volcano",
    "plot_spatial_uncertainty",
]
