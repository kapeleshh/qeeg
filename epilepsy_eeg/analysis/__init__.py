"""
Analysis package for EEG signal processing.

This package provides modules for analyzing EEG signals, including:
- Spectral: Power spectral density calculation and frequency band analysis
- Asymmetry: Left-right hemisphere power asymmetry detection
- Brodmann: Brodmann area analysis
- Epileptiform: Epileptiform activity detection (OIRDA, FIRDA, spikes)
"""

from . import spectral
from . import asymmetry
from . import brodmann
from . import epileptiform

__all__ = ['spectral', 'asymmetry', 'brodmann', 'epileptiform']
