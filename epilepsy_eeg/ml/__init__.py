"""
Machine learning package for EEG data analysis.

This package provides modules for applying machine learning techniques to EEG data,
including feature extraction, classification, and model evaluation.
"""

from . import features
from . import classification

__all__ = ['features', 'classification']
