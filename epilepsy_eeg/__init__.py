"""
Epilepsy-EEG: A Python package for EEG analysis focused on epilepsy detection and neurological condition assessment.

This package provides modules for:
- Preprocessing: Filtering, artifact removal, and montage setup
- Analysis: Spectral analysis, asymmetry analysis, Brodmann area analysis, and epileptiform activity detection
- Conditions: Assessment of neurological conditions (ADHD, depression, anxiety, etc.)
- Visualization: EEG visualization tools
- Utils: Utility functions for EEG analysis
"""

from . import preprocessing
from . import analysis
from . import conditions
from . import visualization
from . import utils

__version__ = '0.1.0'
__author__ = 'EEG Analysis Team'
__email__ = 'example@example.com'
__license__ = 'MIT'

__all__ = ['preprocessing', 'analysis', 'conditions', 'visualization', 'utils']
