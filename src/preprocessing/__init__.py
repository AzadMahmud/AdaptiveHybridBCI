"""
Package initialization file for the preprocessing module.

This file makes the preprocessing directory a Python package and can be used
to define what gets imported with 'from preprocessing import *'.
"""

# Import key functions for easier access
from .artifact_removal import apply_hwt_to_epochs, apply_ssa_to_epochs

__all__ = [
    'apply_hwt_to_epochs',
    'apply_ssa_to_epochs'
]