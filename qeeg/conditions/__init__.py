"""
Neurological conditions module for EEG data analysis.

This module provides functions for analyzing EEG data in the context of specific
neurological conditions, including:
- Epilepsy
- ADHD
- Anxiety
- Autism
- Depression
- Other neurological conditions
"""

from . import adhd
from . import anxiety
from . import autism
from . import depression
from . import other_conditions

__all__ = [
    'adhd',
    'anxiety',
    'autism',
    'depression',
    'other_conditions',
]
