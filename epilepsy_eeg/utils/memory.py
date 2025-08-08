"""
Memory-efficient processing module for EEG data.

This module provides functions for memory-efficient processing of EEG data,
particularly for large datasets that may not fit entirely in memory.
"""

import numpy as np
from typing import Callable, List, Optional, Any, Dict, Union
import mne
from epilepsy_eeg.utils.logging import get_logger
from epilepsy_eeg.utils.exceptions import EEGError

# Get logger
logger = get_logger(__name__)


def process_in_chunks(
    raw: mne.io.Raw,
    chunk_size_seconds: float = 60.0,
    func: Optional[Callable] = None,
    overlap_seconds: float = 0.0,
    combine_results: bool = True,
    **kwargs
) -> Union[List[Any], Any]:
    """
    Process a raw object in chunks to reduce memory usage.
    
    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data.
    chunk_size_seconds : float, optional
        Size of each chunk in seconds, by default 60.0
    func : callable or None, optional
        Function to apply to each chunk, by default None
    overlap_seconds : float, optional
        Overlap between consecutive chunks in seconds, by default 0.0
    combine_results : bool, optional
        Whether to combine results from all chunks, by default True
    **kwargs
        Additional keyword arguments to pass to func
    
    Returns
    -------
    list or any
        If combine_results is True, returns the combined result.
        Otherwise, returns a list of results from processing each chunk.
    
    Examples
    --------
    >>> import mne
    >>> from epilepsy_eeg.utils.memory import process_in_chunks
    >>> from epilepsy_eeg.analysis.spectral import compute_band_powers
    >>> raw = mne.io.read_raw_edf("sample.edf", preload=True)
    >>> results = process_in_chunks(raw, chunk_size_seconds=30, func=compute_band_powers)
    """
    if not isinstance(raw, mne.io.Raw):
        raise TypeError("raw must be an instance of mne.io.Raw")
    
    if chunk_size_seconds <= 0:
        raise ValueError("chunk_size_seconds must be positive")
    
    if overlap_seconds < 0:
        raise ValueError("overlap_seconds must be non-negative")
    
    if overlap_seconds >= chunk_size_seconds:
        raise ValueError("overlap_seconds must be less than chunk_size_seconds")
    
    # Calculate chunk size in samples
    chunk_size = int(chunk_size_seconds * raw.info['sfreq'])
    overlap_size = int(overlap_seconds * raw.info['sfreq'])
    
    # Calculate number of chunks
    effective_chunk_size = chunk_size - overlap_size
    if effective_chunk_size <= 0:
        raise ValueError("Effective chunk size is non-positive. Reduce overlap_seconds.")
    
    n_chunks = int(np.ceil(raw.n_times / effective_chunk_size))
    
    logger.info(f"Processing {raw.n_times} samples in {n_chunks} chunks of {chunk_size} samples each")
    
    # Process each chunk
    results = []
    for i in range(n_chunks):
        # Calculate chunk boundaries
        start_sample = i * effective_chunk_size
        end_sample = min(start_sample + chunk_size, raw.n_times)
        
        # Convert to seconds
        start_time = start_sample / raw.info['sfreq']
        end_time = end_sample / raw.info['sfreq']
        
        logger.debug(f"Processing chunk {i+1}/{n_chunks}: {start_time:.2f}s - {end_time:.2f}s")
        
        # Extract chunk
        chunk = raw.copy().crop(tmin=start_time, tmax=end_time)
        
        # Apply function if provided
        if func is not None:
            try:
                result = func(chunk, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing chunk {i+1}/{n_chunks}: {str(e)}")
                raise EEGError(f"Error processing chunk {i+1}/{n_chunks}") from e
    
    # Combine results if requested
    if combine_results and results:
        if isinstance(results[0], dict):
            # Combine dictionaries
            combined_result = {}
            for key in results[0].keys():
                if isinstance(results[0][key], np.ndarray):
                    # Average arrays
                    combined_result[key] = np.mean([r[key] for r in results], axis=0)
                else:
                    # Use the first value for non-array values
                    combined_result[key] = results[0][key]
            return combined_result
        elif isinstance(results[0], np.ndarray):
            # Average arrays
            return np.mean(results, axis=0)
        else:
            # Return list of results
            return results
    
    return results


def estimate_memory_usage(
    raw: mne.io.Raw,
    operation: str = 'filter',
    **kwargs
) -> Dict[str, float]:
    """
    Estimate memory usage for a given operation on EEG data.
    
    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data.
    operation : str, optional
        Operation to estimate memory usage for, by default 'filter'
        Options: 'filter', 'ica', 'psd', 'epochs'
    **kwargs
        Additional keyword arguments for the operation
    
    Returns
    -------
    Dict[str, float]
        Dictionary with memory usage estimates in MB
    
    Examples
    --------
    >>> import mne
    >>> from epilepsy_eeg.utils.memory import estimate_memory_usage
    >>> raw = mne.io.read_raw_edf("sample.edf", preload=True)
    >>> memory_usage = estimate_memory_usage(raw, operation='filter')
    >>> print(f"Estimated memory usage: {memory_usage['total']:.2f} MB")
    """
    if not isinstance(raw, mne.io.Raw):
        raise TypeError("raw must be an instance of mne.io.Raw")
    
    # Calculate base memory usage
    n_channels = len(raw.ch_names)
    n_samples = raw.n_times
    bytes_per_value = 8  # float64
    
    # Raw data memory usage
    raw_data_mb = (n_channels * n_samples * bytes_per_value) / (1024 * 1024)
    
    # Operation-specific memory usage
    operation_data_mb = 0
    
    if operation == 'filter':
        # Filtering typically requires 2-3x the raw data size
        operation_data_mb = 2 * raw_data_mb
    elif operation == 'ica':
        # ICA can require significant memory for the mixing matrix and components
        n_components = kwargs.get('n_components', min(n_channels, 30))
        operation_data_mb = (n_components * n_samples * bytes_per_value) / (1024 * 1024)
        operation_data_mb += (n_components * n_channels * bytes_per_value) / (1024 * 1024)
    elif operation == 'psd':
        # PSD computation requires memory for the FFT and output
        n_fft = kwargs.get('n_fft', 256)
        operation_data_mb = (n_channels * n_fft * bytes_per_value) / (1024 * 1024)
    elif operation == 'epochs':
        # Epochs can duplicate data
        n_epochs = kwargs.get('n_epochs', 10)
        epoch_duration = kwargs.get('epoch_duration', 2.0)
        samples_per_epoch = int(epoch_duration * raw.info['sfreq'])
        operation_data_mb = (n_channels * samples_per_epoch * n_epochs * bytes_per_value) / (1024 * 1024)
    else:
        raise ValueError(f"Unknown operation: {operation}")
    
    # Total memory usage
    total_mb = raw_data_mb + operation_data_mb
    
    # Add overhead (20%)
    total_mb *= 1.2
    
    return {
        'raw_data_mb': raw_data_mb,
        'operation_data_mb': operation_data_mb,
        'total': total_mb
    }


def check_memory_limit(
    memory_usage: float,
    limit_mb: Optional[float] = None
) -> bool:
    """
    Check if estimated memory usage exceeds a limit.
    
    Parameters
    ----------
    memory_usage : float
        Estimated memory usage in MB
    limit_mb : float or None, optional
        Memory limit in MB, by default None (use 80% of available memory)
    
    Returns
    -------
    bool
        True if memory usage is within limit, False otherwise
    
    Examples
    --------
    >>> from epilepsy_eeg.utils.memory import estimate_memory_usage, check_memory_limit
    >>> import mne
    >>> raw = mne.io.read_raw_edf("sample.edf", preload=True)
    >>> memory_usage = estimate_memory_usage(raw, operation='filter')
    >>> if check_memory_limit(memory_usage['total']):
    ...     print("Memory usage is within limit")
    ... else:
    ...     print("Memory usage exceeds limit")
    """
    if limit_mb is None:
        # Use 80% of available memory
        import psutil
        available_mb = psutil.virtual_memory().available / (1024 * 1024)
        limit_mb = 0.8 * available_mb
    
    return memory_usage <= limit_mb


def suggest_chunk_size(
    raw: mne.io.Raw,
    operation: str = 'filter',
    memory_limit_mb: Optional[float] = None,
    **kwargs
) -> float:
    """
    Suggest a chunk size for memory-efficient processing.
    
    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data.
    operation : str, optional
        Operation to estimate memory usage for, by default 'filter'
    memory_limit_mb : float or None, optional
        Memory limit in MB, by default None (use 80% of available memory)
    **kwargs
        Additional keyword arguments for the operation
    
    Returns
    -------
    float
        Suggested chunk size in seconds
    
    Examples
    --------
    >>> import mne
    >>> from epilepsy_eeg.utils.memory import suggest_chunk_size
    >>> raw = mne.io.read_raw_edf("sample.edf", preload=True)
    >>> chunk_size = suggest_chunk_size(raw, operation='filter')
    >>> print(f"Suggested chunk size: {chunk_size:.2f} seconds")
    """
    if not isinstance(raw, mne.io.Raw):
        raise TypeError("raw must be an instance of mne.io.Raw")
    
    # Set memory limit if not provided
    if memory_limit_mb is None:
        import psutil
        available_mb = psutil.virtual_memory().available / (1024 * 1024)
        memory_limit_mb = 0.8 * available_mb
    
    # Calculate memory usage per second
    n_channels = len(raw.ch_names)
    sfreq = raw.info['sfreq']
    bytes_per_value = 8  # float64
    
    # Memory usage per second of data
    mb_per_second = (n_channels * sfreq * bytes_per_value) / (1024 * 1024)
    
    # Adjust for operation
    if operation == 'filter':
        mb_per_second *= 3  # Filtering requires more memory
    elif operation == 'ica':
        mb_per_second *= 4  # ICA requires even more memory
    elif operation == 'psd':
        mb_per_second *= 2  # PSD computation requires additional memory
    elif operation == 'epochs':
        mb_per_second *= 2  # Epochs can duplicate data
    
    # Add overhead (20%)
    mb_per_second *= 1.2
    
    # Calculate chunk size
    chunk_size_seconds = memory_limit_mb / mb_per_second
    
    # Ensure chunk size is at least 1 second
    chunk_size_seconds = max(1.0, chunk_size_seconds)
    
    # Ensure chunk size is not larger than the entire recording
    chunk_size_seconds = min(chunk_size_seconds, raw.times[-1])
    
    return chunk_size_seconds
