"""
Visualization package for EEG data.

This package provides modules for visualizing EEG data, including:
- Topomaps: Topographic mapping of EEG data
- Spectra: Spectral visualization of EEG data
"""

from . import topomaps
from . import spectra

__all__ = ['topomaps', 'spectra']
