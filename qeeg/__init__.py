"""
Qeeg: A Python package for quantitative EEG analysis and neurological condition assessment.

This package provides tools for EEG data analysis, including:
- Preprocessing (filtering, artifact removal)
- Analysis (spectral, asymmetry, epileptiform activity detection)
- Visualization (topomaps, spectra)
- Machine learning (feature extraction, classification)
- Neurological condition assessment
"""

__version__ = '0.1.0'

# Define __all__ to control what is imported with "from qeeg import *"
__all__ = [
    'preprocessing',
    'analysis',
    'visualization',
    'ml',
    'utils',
    'conditions',
    'cli',
]

# Import submodules - these are at the end to avoid circular imports
from . import preprocessing
from . import utils
from . import analysis
from . import visualization
from . import conditions
from . import cli
from . import ml
