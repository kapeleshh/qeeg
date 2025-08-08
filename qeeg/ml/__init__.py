"""
Machine learning module for EEG data.

This module provides functions for applying machine learning to EEG data, including:
- Feature extraction
- Classification
- Model evaluation
"""

# Define __all__ to control what is imported with "from qeeg.ml import *"
__all__ = [
    'features',
    'classification',
]

# Avoid circular imports by not importing submodules here
# Users will need to import them explicitly:
# from qeeg.ml import features
# from qeeg.ml import classification
