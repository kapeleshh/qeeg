"""
Tests for the utils package initialization.
"""

import pytest
import importlib


def test_utils_imports():
    """Test that all utility modules can be imported from the utils package."""
    # Import the utils package
    import epilepsy_eeg.utils
    
    # Reload to ensure we get a fresh import
    importlib.reload(epilepsy_eeg.utils)
    
    # Test that the package has the expected modules
    assert hasattr(epilepsy_eeg.utils, 'exceptions')
    assert hasattr(epilepsy_eeg.utils, 'logging')
    assert hasattr(epilepsy_eeg.utils, 'memory')
    assert hasattr(epilepsy_eeg.utils, 'compat')
    assert hasattr(epilepsy_eeg.utils, 'benchmark')
    assert hasattr(epilepsy_eeg.utils, 'validation')
    assert hasattr(epilepsy_eeg.utils, 'cache')
    assert hasattr(epilepsy_eeg.utils, 'parallel')


def test_direct_imports():
    """Test that utility classes and functions can be imported directly from the utils package."""
    # Test importing from exceptions
    from epilepsy_eeg.utils import EEGError, ValidationError
    
    # Test importing from logging
    from epilepsy_eeg.utils import setup_logger, get_logger
    
    # Test importing from memory
    from epilepsy_eeg.utils import process_in_chunks, estimate_memory_usage
    
    # Test importing from compat
    from epilepsy_eeg.utils import check_dependencies, get_mne_version
    
    # Test importing from benchmark
    from epilepsy_eeg.utils import timeit, benchmark_function
    
    # Test importing from validation
    from epilepsy_eeg.utils import validate_type, validate_range
    
    # Test importing from cache
    from epilepsy_eeg.utils import cache_result, memoize
    
    # Test importing from parallel
    from epilepsy_eeg.utils import parallel_process


def test_module_docstrings():
    """Test that all utility modules have docstrings."""
    import epilepsy_eeg.utils.exceptions
    import epilepsy_eeg.utils.logging
    import epilepsy_eeg.utils.memory
    import epilepsy_eeg.utils.compat
    import epilepsy_eeg.utils.benchmark
    import epilepsy_eeg.utils.validation
    import epilepsy_eeg.utils.cache
    import epilepsy_eeg.utils.parallel
    
    assert epilepsy_eeg.utils.exceptions.__doc__ is not None
    assert epilepsy_eeg.utils.logging.__doc__ is not None
    assert epilepsy_eeg.utils.memory.__doc__ is not None
    assert epilepsy_eeg.utils.compat.__doc__ is not None
    assert epilepsy_eeg.utils.benchmark.__doc__ is not None
    assert epilepsy_eeg.utils.validation.__doc__ is not None
    assert epilepsy_eeg.utils.cache.__doc__ is not None
    assert epilepsy_eeg.utils.parallel.__doc__ is not None
