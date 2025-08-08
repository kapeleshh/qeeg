"""
Utility package for EEG data analysis.

This package provides utility modules for EEG data analysis, including:
- Exceptions: Custom exception classes for better error handling
- Logging: Logging utilities for debugging and monitoring
- Memory: Memory-efficient processing tools for large datasets
- Compatibility: Dependency management and version checking
- Benchmark: Performance benchmarking tools
- Validation: Input validation utilities
- Cache: Caching utilities for expensive computations
- Parallel: Parallel processing utilities for computationally intensive operations
"""

# Import all utility modules
from . import exceptions
from . import logging
from . import memory
from . import compat
from . import benchmark
from . import validation
from . import cache
from . import parallel

# Import key classes and functions from exceptions
from .exceptions import (
    QEEGError,
    DataQualityError,
    ProcessingError,
    ValidationError,
    ConfigurationError,
    DependencyError
)

# Import key functions from logging
from .logging import setup_logger, get_logger

# Import key functions from memory
from .memory import (
    get_memory_usage,
    memory_usage,
    estimate_memory_requirement,
    check_memory_available,
    process_large_eeg,
    reduce_data_resolution,
    MemoryMonitor
)

# Import key functions from compat
from .compat import (
    check_dependencies,
    get_mne_version,
    check_mne_version,
    get_sklearn_version,
    check_sklearn_version,
    print_dependency_status
)

# Import key functions from benchmark
from .benchmark import (
    timeit,
    benchmark_function,
    plot_benchmark_results,
    compare_functions,
    benchmark_preprocessing_pipeline,
    benchmark_memory_usage,
    create_benchmark_report
)

# Import key functions from validation
from .validation import (
    validate_type,
    validate_range,
    validate_options,
    validate_raw,
    validate_epochs,
    validate_array,
    validate_function_params,
    validate_frequency_bands,
    validate_picks,
    input_validator
)

# Import key functions from cache
from .cache import (
    get_cache_dir,
    clear_cache,
    get_cache_size,
    cache_result,
    memoize,
    LRUCache,
    lru_cache
)

# Import key functions from parallel
from .parallel import parallel_process

# Define __all__ to control what is imported with "from qeeg.utils import *"
__all__ = [
    # Modules
    'exceptions',
    'logging',
    'memory',
    'compat',
    'benchmark',
    'validation',
    'cache',
    'parallel',
    
    # Exception classes
    'QEEGError',
    'DataQualityError',
    'ProcessingError',
    'ValidationError',
    'ConfigurationError',
    'DependencyError',
    
    # Logging functions
    'setup_logger',
    'get_logger',
    
    # Memory functions
    'get_memory_usage',
    'memory_usage',
    'estimate_memory_requirement',
    'check_memory_available',
    'process_large_eeg',
    'reduce_data_resolution',
    'MemoryMonitor',
    
    # Compatibility functions
    'check_dependencies',
    'get_mne_version',
    'check_mne_version',
    'get_sklearn_version',
    'check_sklearn_version',
    'print_dependency_status',
    
    # Benchmark functions
    'timeit',
    'benchmark_function',
    'plot_benchmark_results',
    'compare_functions',
    'benchmark_preprocessing_pipeline',
    'benchmark_memory_usage',
    'create_benchmark_report',
    
    # Validation functions
    'validate_type',
    'validate_range',
    'validate_options',
    'validate_raw',
    'validate_epochs',
    'validate_array',
    'validate_function_params',
    'validate_frequency_bands',
    'validate_picks',
    'input_validator',
    
    # Cache functions and classes
    'get_cache_dir',
    'clear_cache',
    'get_cache_size',
    'cache_result',
    'memoize',
    'LRUCache',
    'lru_cache',
    
    # Parallel functions
    'parallel_process'
]
