"""
Machine learning module for EEG data.

This module provides functions for applying machine learning to EEG data, including:
- Feature extraction
- Classification
- Model evaluation
"""

from . import features
from . import classification

__all__ = [
    'features',
    'classification',
]
