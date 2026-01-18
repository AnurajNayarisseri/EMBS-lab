"""
Preprocessing module for PROSPECT.

This subpackage contains conservative preprocessing steps
designed to clean single-cell and spatial transcriptomics data
without imposing deterministic modeling assumptions.
"""

from .qc import run_qc

__all__ = [
    "run_qc",
]
