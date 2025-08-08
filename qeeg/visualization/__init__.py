"""
Visualization module for EEG data.

This module provides functions for visualizing EEG data, including:
- Topographic maps
- Spectral plots
- Brain activation visualization
- Report generation
"""

from . import topomaps
from . import spectra
from . import brain_activation
from . import reports

__all__ = [
    'topomaps',
    'spectra',
    'brain_activation',
    'reports',
]
