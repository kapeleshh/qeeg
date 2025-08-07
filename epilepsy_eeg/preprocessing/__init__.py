"""
Preprocessing package for EEG signal processing.

This package provides modules for preprocessing EEG signals, including:
- Filtering: Bandpass, notch, and other filtering operations
- Artifacts: ICA-based artifact removal and automatic rejection of noisy epochs
- Montage: EEG montage setup, channel selection, and reference setting
"""

from . import filtering
from . import artifacts
from . import montage

__all__ = ['filtering', 'artifacts', 'montage']
