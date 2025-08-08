"""
Memory management utilities for EEG data processing.

This module provides functions and classes for efficient memory management
when processing large EEG datasets.
"""

import os
import numpy as np
import mne
import psutil
import warnings
from functools import wraps
from typing import Callable, List, Dict, Any, Optional, Union, Tuple

from qeeg.utils.exceptions import ProcessingError


def get_memory_usage():
    """
    Get the current memory usage of the process.
    
    Returns
    -------
    dict
        Dictionary with memory usage information in MB
    """
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    
    return {
        'rss': mem_info.rss / (1024 * 1024),  # Resident Set Size in MB
        'vms': mem_info.vms / (1024 * 1024),  # Virtual Memory Size in MB
        'percent': process.memory_percent(),   # Percentage of system memory
        'available': psutil.virtual_memory().available / (1024 * 1024)  # Available system memory in MB
    }


def memory_usage(func):
    """
    Decorator to measure memory usage of a function.
    
    Parameters
    ----------
    func : callable
        The function to measure memory usage for
        
    Returns
    -------
    callable
        Wrapped function that prints memory usage information
        
    Examples
    --------
    >>> @memory_usage
    >>> def process_data(raw):
    >>>     # Process data
    >>>     return processed_data
    >>>
    >>> # Call the function
    >>> result = process_data(raw)  # Will print memory usage information
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Get memory usage before function call
        mem_before = get_memory_usage()
        
        # Call the function
        result = func(*args, **kwargs)
        
        # Get memory usage after function call
        mem_after = get_memory_usage()
        
        # Calculate the difference
        mem_diff = {
            'rss': mem_after['rss'] - mem_before['rss'],
            'vms': mem_after['vms'] - mem_before['vms'],
            'percent': mem_after['percent'] - mem_before['percent']
        }
        
        # Print memory usage information
        print(f"Memory usage for {func.__name__}:")
        print(f"  Before: {mem_before['rss']:.2f} MB (RSS), {mem_before['percent']:.2f}% of system memory")
        print(f"  After:  {mem_after['rss']:.2f} MB (RSS), {mem_after['percent']:.2f}% of system memory")
        print(f"  Diff:   {mem_diff['rss']:.2f} MB (RSS), {mem_diff['percent']:.2f}% of system memory")
        
        return result
    
    return wrapper


def estimate_memory_requirement(raw: mne.io.Raw, operation: str = 'copy') -> float:
    """
    Estimate the memory requirement for an operation on the given raw data.
    
    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data
    operation : str, optional
        The operation to estimate memory for, by default 'copy'
        Options: 'copy', 'filter', 'ica', 'epoch'
        
    Returns
    -------
    float
        Estimated memory requirement in MB
        
    Notes
    -----
    This is a rough estimate and actual memory usage may vary.
    """
    # Get data dimensions
    n_channels = len(raw.ch_names)
    n_times = raw.n_times
    dtype_size = 8  # Assuming float64, 8 bytes per value
    
    # Base memory for the raw data (channels x time points x dtype size)
    base_memory = n_channels * n_times * dtype_size / (1024 * 1024)  # Convert to MB
    
    # Estimate based on operation
    if operation == 'copy':
        # A copy requires approximately the same amount of memory as the original
        return base_memory
    elif operation == 'filter':
        # Filtering typically requires 2-3x the original memory
        return 2.5 * base_memory
    elif operation == 'ica':
        # ICA can require 3-5x the original memory
        return 4 * base_memory
    elif operation == 'epoch':
        # Epoching depends on the number and size of epochs, but typically 1-2x
        return 1.5 * base_memory
    else:
        # Default to a conservative estimate
        return 3 * base_memory


def check_memory_available(required_mb: float, threshold_ratio: float = 0.8) -> bool:
    """
    Check if there is enough memory available for an operation.
    
    Parameters
    ----------
    required_mb : float
        Required memory in MB
    threshold_ratio : float, optional
        Threshold ratio of available memory to use, by default 0.8
        
    Returns
    -------
    bool
        True if enough memory is available, False otherwise
    """
    available_mb = psutil.virtual_memory().available / (1024 * 1024)
    return required_mb <= available_mb * threshold_ratio


def process_large_eeg(
    raw: mne.io.Raw,
    processing_func: Callable,
    chunk_duration: float = 10.0,
    overlap: float = 1.0,
    **kwargs
) -> mne.io.Raw:
    """
    Process large EEG files in chunks to reduce memory usage.
    
    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data
    processing_func : callable
        Function to apply to each chunk
    chunk_duration : float, optional
        Size of each chunk in seconds, by default 10.0
    overlap : float, optional
        Overlap between chunks in seconds, by default 1.0
    **kwargs
        Additional keyword arguments to pass to processing_func
        
    Returns
    -------
    mne.io.Raw
        Processed EEG data
        
    Notes
    -----
    This function is useful for processing large EEG files that don't fit in memory.
    It processes the data in chunks and then concatenates the results.
    
    Examples
    --------
    >>> import mne
    >>> from qeeg.preprocessing import filtering
    >>> from qeeg.utils.memory import process_large_eeg
    >>> 
    >>> # Load a large EEG file
    >>> raw = mne.io.read_raw_edf("large_eeg.edf", preload=True)
    >>> 
    >>> # Process it in chunks
    >>> filtered_raw = process_large_eeg(
    >>>     raw,
    >>>     filtering.bandpass_filter,
    >>>     chunk_duration=30.0,
    >>>     overlap=2.0,
    >>>     l_freq=1.0,
    >>>     h_freq=40.0
    >>> )
    """
    # Check if the data is loaded
    if not raw.preload:
        raise ProcessingError(
            "Raw data must be preloaded before processing in chunks.",
            function="process_large_eeg"
        )
    
    # Get data dimensions
    sfreq = raw.info['sfreq']
    n_times = raw.n_times
    duration = n_times / sfreq
    
    # Calculate chunk size in samples
    chunk_samples = int(chunk_duration * sfreq)
    overlap_samples = int(overlap * sfreq)
    
    # Check if chunking is necessary
    if duration <= chunk_duration:
        # Process the entire file at once
        return processing_func(raw, **kwargs)
    
    # Calculate the number of chunks
    step_samples = chunk_samples - overlap_samples
    n_chunks = int(np.ceil((n_times - overlap_samples) / step_samples))
    
    # Process each chunk
    processed_chunks = []
    
    for i in range(n_chunks):
        # Calculate chunk start and end
        start_sample = i * step_samples
        end_sample = min(start_sample + chunk_samples, n_times)
        
        # Extract chunk
        chunk = raw.copy().crop(
            start_sample / sfreq,
            end_sample / sfreq,
            include_tmax=False
        )
        
        # Process chunk
        try:
            processed_chunk = processing_func(chunk, **kwargs)
            processed_chunks.append(processed_chunk)
        except Exception as e:
            raise ProcessingError(
                f"Error processing chunk {i+1}/{n_chunks}: {str(e)}",
                function="process_large_eeg"
            ) from e
    
    # Concatenate processed chunks
    if len(processed_chunks) == 1:
        return processed_chunks[0]
    else:
        try:
            return mne.concatenate_raws(processed_chunks)
        except Exception as e:
            raise ProcessingError(
                f"Error concatenating processed chunks: {str(e)}",
                function="process_large_eeg"
            ) from e


def reduce_data_resolution(
    raw: mne.io.Raw,
    factor: int = 2,
    method: str = 'decimate'
) -> mne.io.Raw:
    """
    Reduce the temporal resolution of EEG data to save memory.
    
    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data
    factor : int, optional
        Downsampling factor, by default 2
    method : str, optional
        Downsampling method, by default 'decimate'
        Options: 'decimate', 'resample'
        
    Returns
    -------
    mne.io.Raw
        Downsampled raw data
        
    Notes
    -----
    'decimate' is faster but less accurate than 'resample'.
    'resample' applies an anti-aliasing filter before downsampling.
    
    Examples
    --------
    >>> import mne
    >>> from qeeg.utils.memory import reduce_data_resolution
    >>> 
    >>> # Load EEG data
    >>> raw = mne.io.read_raw_edf("sample.edf", preload=True)
    >>> print(f"Original sampling rate: {raw.info['sfreq']} Hz")
    >>> 
    >>> # Reduce resolution by a factor of 4
    >>> raw_downsampled = reduce_data_resolution(raw, factor=4)
    >>> print(f"New sampling rate: {raw_downsampled.info['sfreq']} Hz")
    """
    # Check if the data is loaded
    if not raw.preload:
        raise ProcessingError(
            "Raw data must be preloaded before reducing resolution.",
            function="reduce_data_resolution"
        )
    
    # Create a copy of the raw data
    raw_copy = raw.copy()
    
    # Get the original sampling rate
    orig_sfreq = raw_copy.info['sfreq']
    new_sfreq = orig_sfreq / factor
    
    # Check if the new sampling rate is too low
    if new_sfreq < 2 * raw_copy.info['highpass'] or new_sfreq < 2 * raw_copy.info['lowpass']:
        warnings.warn(
            f"New sampling rate ({new_sfreq} Hz) may be too low for the data's "
            f"frequency content (highpass: {raw_copy.info['highpass']} Hz, "
            f"lowpass: {raw_copy.info['lowpass']} Hz)."
        )
    
    # Downsample the data
    try:
        if method == 'decimate':
            raw_copy.decimate(factor)
        elif method == 'resample':
            raw_copy.resample(new_sfreq)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'decimate' or 'resample'.")
    except Exception as e:
        raise ProcessingError(
            f"Error reducing data resolution: {str(e)}",
            function="reduce_data_resolution"
        ) from e
    
    return raw_copy


class MemoryMonitor:
    """
    Monitor memory usage during processing.
    
    This class provides methods to track memory usage during processing
    and raise warnings or errors if memory usage exceeds thresholds.
    
    Parameters
    ----------
    warning_threshold : float, optional
        Memory usage threshold (percentage) for warnings, by default 70.0
    error_threshold : float, optional
        Memory usage threshold (percentage) for errors, by default 90.0
    check_interval : float, optional
        Interval (in seconds) between memory checks, by default 5.0
        
    Examples
    --------
    >>> from qeeg.utils.memory import MemoryMonitor
    >>> 
    >>> # Create a memory monitor
    >>> monitor = MemoryMonitor(warning_threshold=70.0, error_threshold=90.0)
    >>> 
    >>> # Start monitoring
    >>> monitor.start()
    >>> 
    >>> # Perform memory-intensive operations
    >>> # ...
    >>> 
    >>> # Check memory usage
    >>> usage = monitor.check()
    >>> print(f"Current memory usage: {usage['percent']:.2f}%")
    >>> 
    >>> # Stop monitoring
    >>> monitor.stop()
    """
    def __init__(
        self,
        warning_threshold: float = 70.0,
        error_threshold: float = 90.0,
        check_interval: float = 5.0
    ):
        self.warning_threshold = warning_threshold
        self.error_threshold = error_threshold
        self.check_interval = check_interval
        self.monitoring = False
        self._thread = None
    
    def start(self):
        """Start monitoring memory usage."""
        if self.monitoring:
            return
        
        self.monitoring = True
        
        # Start monitoring in a separate thread
        import threading
        import time
        
        def _monitor():
            while self.monitoring:
                usage = self.check()
                time.sleep(self.check_interval)
        
        self._thread = threading.Thread(target=_monitor)
        self._thread.daemon = True
        self._thread.start()
    
    def stop(self):
        """Stop monitoring memory usage."""
        self.monitoring = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
    
    def check(self) -> Dict[str, float]:
        """
        Check current memory usage.
        
        Returns
        -------
        dict
            Dictionary with memory usage information
            
        Raises
        ------
        MemoryError
            If memory usage exceeds the error threshold
        """
        usage = psutil.virtual_memory()
        percent = usage.percent
        
        if percent >= self.error_threshold:
            self.stop()
            raise MemoryError(
                f"Memory usage ({percent:.2f}%) exceeds error threshold ({self.error_threshold:.2f}%)."
            )
        elif percent >= self.warning_threshold:
            warnings.warn(
                f"Memory usage ({percent:.2f}%) exceeds warning threshold ({self.warning_threshold:.2f}%)."
            )
        
        return {
            'total': usage.total / (1024 * 1024),  # MB
            'available': usage.available / (1024 * 1024),  # MB
            'used': usage.used / (1024 * 1024),  # MB
            'percent': usage.percent
        }
    
    def __enter__(self):
        """Start monitoring when entering a context."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop monitoring when exiting a context."""
        self.stop()
