"""
Preprocessing module for EEG data.

This module provides functions for preprocessing EEG data, including:
- Filtering (bandpass, notch, etc.)
- Artifact removal (ICA, regression, etc.)
- Montage operations (referencing, etc.)
"""

from . import filtering
from . import artifacts
from . import montage

__all__ = [
    'filtering',
    'artifacts',
    'montage',
]
