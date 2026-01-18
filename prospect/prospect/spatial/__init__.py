"""
Spatial module for PROSPECT.

This subpackage contains spatially-aware constraints and
uncertainty modeling utilities for spatial transcriptomics
data. Spatial information is used to constrain plausible
cellular configurations without enforcing deterministic
assignments.
"""

from .spatial_constraints import SpatialConstraint

__all__ = [
    "SpatialConstraint",
]
