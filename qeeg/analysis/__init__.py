"""
Analysis module for EEG data.

This module provides functions for analyzing EEG data, including:
- Spectral analysis (power spectral density, frequency bands)
- Asymmetry analysis
- Epileptiform activity detection
- Brodmann area analysis
"""

from . import spectral
from . import asymmetry
from . import epileptiform
from . import brodmann

__all__ = [
    'spectral',
    'asymmetry',
    'epileptiform',
    'brodmann',
]
