"""
Tests for the memory module.
"""

import pytest
import numpy as np
import mne
import psutil
import time
from unittest.mock import patch, MagicMock

from qeeg.utils.memory import (
    get_memory_usage,
    memory_usage,
    estimate_memory_requirement,
    check_memory_available,
    process_large_eeg,
    reduce_data_resolution,
    MemoryMonitor
)
from qeeg.utils.exceptions import ProcessingError


def create_test_raw(n_channels=5, n_samples=1000, sfreq=100):
    """Create a test MNE Raw object with simulated EEG data."""
    # Create simulated data
    data = np.random.randn(n_channels, n_samples)
    
    # Create MNE Raw object
    ch_names = [f'CH{i}' for i in range(n_channels)]
    ch_types = ['eeg'] * n_channels
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(data, info)
    
    return raw


def test_get_memory_usage():
    """Test the get_memory_usage function."""
    # Get memory usage
    usage = get_memory_usage()
    
    # Check that the result is a dictionary
    assert isinstance(usage, dict)
    
    # Check that the dictionary contains the expected keys
    assert 'rss' in usage
    assert 'vms' in usage
    assert 'percent' in usage
    assert 'available' in usage
    
    # Check that the values are of the expected types
    assert isinstance(usage['rss'], float)
    assert isinstance(usage['vms'], float)
    assert isinstance(usage['percent'], float)
    assert isinstance(usage['available'], float)
    
    # Check that the values are reasonable
    assert usage['rss'] > 0
    assert usage['vms'] > 0
    assert 0 <= usage['percent'] <= 100
    assert usage['available'] > 0


def test_memory_usage_decorator():
    """Test the memory_usage decorator."""
    # Define a function with the decorator
    @memory_usage
    def test_func():
        # Allocate some memory
        x = [0] * 1000000
        return x
    
    # Capture stdout to check the output
    with patch('builtins.print') as mock_print:
        # Call the function
        result = test_func()
        
        # Check that the function returned the expected result
        assert len(result) == 1000000
        
        # Check that print was called with memory usage information
        assert mock_print.call_count >= 3
        
        # Check that the output contains the expected information
        for call in mock_print.call_args_list:
            args, _ = call
            arg = args[0]
            if "Memory usage for test_func" in arg:
                break
        else:
            assert False, "Memory usage information not printed"


def test_estimate_memory_requirement():
    """Test the estimate_memory_requirement function."""
    # Create a test raw object
    raw = create_test_raw(n_channels=10, n_samples=10000)
    
    # Estimate memory requirement for different operations
    copy_mem = estimate_memory_requirement(raw, 'copy')
    filter_mem = estimate_memory_requirement(raw, 'filter')
    ica_mem = estimate_memory_requirement(raw, 'ica')
    epoch_mem = estimate_memory_requirement(raw, 'epoch')
    
    # Check that the estimates are reasonable
    assert copy_mem > 0
    assert filter_mem > copy_mem  # Filtering should require more memory than copying
    assert ica_mem > copy_mem  # ICA should require more memory than copying
    assert epoch_mem > 0
    
    # Check that the default operation is handled
    default_mem = estimate_memory_requirement(raw, 'unknown')
    assert default_mem > 0


def test_check_memory_available():
    """Test the check_memory_available function."""
    # Get available memory
    available_mb = psutil.virtual_memory().available / (1024 * 1024)
    
    # Check with a small requirement
    assert check_memory_available(1.0)
    
    # Check with a requirement just below available memory
    assert check_memory_available(available_mb * 0.5)
    
    # Check with a requirement above available memory
    assert not check_memory_available(available_mb * 2.0)
    
    # Check with different threshold ratios
    assert check_memory_available(available_mb * 0.9, threshold_ratio=1.0)
    assert not check_memory_available(available_mb * 0.9, threshold_ratio=0.5)


def test_process_large_eeg():
    """Test the process_large_eeg function."""
    # Create a test raw object
    raw = create_test_raw(n_channels=5, n_samples=10000, sfreq=100)
    
    # Define a simple processing function
    def multiply_data(raw, factor=2.0):
        data = raw.get_data() * factor
        info = raw.info.copy()
        return mne.io.RawArray(data, info)
    
    # Process the data
    processed = process_large_eeg(raw, multiply_data, chunk_duration=0.5, overlap=0.1, factor=2.0)
    
    # Check that the output is a Raw object
    assert isinstance(processed, mne.io.Raw)
    
    # Check that the data has been modified
    assert not np.array_equal(raw.get_data(), processed.get_data())
    
    # Check that the data has been multiplied by 2
    np.testing.assert_allclose(processed.get_data(), raw.get_data() * 2.0)
    
    # Check that the function works with a single chunk
    processed = process_large_eeg(raw, multiply_data, chunk_duration=200.0, factor=3.0)
    np.testing.assert_allclose(processed.get_data(), raw.get_data() * 3.0)
    
    # Check that the function raises an error if the data is not preloaded
    raw_copy = raw.copy()
    raw_copy.preload = False
    with pytest.raises(ProcessingError):
        process_large_eeg(raw_copy, multiply_data)
    
    # Check that the function handles errors in the processing function
    def failing_func(raw):
        raise ValueError("Test error")
    
    with pytest.raises(ProcessingError):
        process_large_eeg(raw, failing_func)


def test_reduce_data_resolution():
    """Test the reduce_data_resolution function."""
    # Create a test raw object
    raw = create_test_raw(n_channels=5, n_samples=10000, sfreq=100)
    
    # Reduce resolution using decimate
    raw_decimated = reduce_data_resolution(raw, factor=2, method='decimate')
    
    # Check that the output is a Raw object
    assert isinstance(raw_decimated, mne.io.Raw)
    
    # Check that the sampling rate has been reduced
    assert raw_decimated.info['sfreq'] == raw.info['sfreq'] / 2
    
    # Check that the number of samples has been reduced
    assert raw_decimated.n_times == raw.n_times / 2
    
    # Reduce resolution using resample
    raw_resampled = reduce_data_resolution(raw, factor=4, method='resample')
    
    # Check that the sampling rate has been reduced
    assert raw_resampled.info['sfreq'] == raw.info['sfreq'] / 4
    
    # Check that the function raises an error if the data is not preloaded
    raw_copy = raw.copy()
    raw_copy.preload = False
    with pytest.raises(ProcessingError):
        reduce_data_resolution(raw_copy)
    
    # Check that the function raises an error for unknown methods
    with pytest.raises(ProcessingError):
        reduce_data_resolution(raw, method='unknown')


def test_memory_monitor():
    """Test the MemoryMonitor class."""
    # Create a memory monitor
    monitor = MemoryMonitor(warning_threshold=0.0, error_threshold=100.0)
    
    # Check initial state
    assert not monitor.monitoring
    assert monitor._thread is None
    
    # Start monitoring
    monitor.start()
    
    # Check that monitoring has started
    assert monitor.monitoring
    assert monitor._thread is not None
    
    # Check memory usage
    usage = monitor.check()
    
    # Check that the result is a dictionary
    assert isinstance(usage, dict)
    
    # Check that the dictionary contains the expected keys
    assert 'total' in usage
    assert 'available' in usage
    assert 'used' in usage
    assert 'percent' in usage
    
    # Stop monitoring
    monitor.stop()
    
    # Check that monitoring has stopped
    assert not monitor.monitoring
    
    # Test context manager
    with MemoryMonitor() as m:
        assert m.monitoring
    
    assert not m.monitoring


def test_memory_monitor_warnings():
    """Test that MemoryMonitor raises warnings when memory usage exceeds thresholds."""
    # Create a memory monitor with a low warning threshold
    monitor = MemoryMonitor(warning_threshold=0.0, error_threshold=100.0)
    
    # Check should raise a warning
    with pytest.warns(UserWarning):
        monitor.check()


def test_memory_monitor_errors():
    """Test that MemoryMonitor raises errors when memory usage exceeds thresholds."""
    # Mock psutil.virtual_memory to return high memory usage
    mock_vm = MagicMock()
    mock_vm.percent = 95.0
    mock_vm.total = 16 * 1024 * 1024 * 1024  # 16 GB
    mock_vm.available = 1 * 1024 * 1024 * 1024  # 1 GB
    mock_vm.used = 15 * 1024 * 1024 * 1024  # 15 GB
    
    # Create a memory monitor with a low error threshold
    monitor = MemoryMonitor(warning_threshold=0.0, error_threshold=90.0)
    
    # Check should raise an error
    with patch('psutil.virtual_memory', return_value=mock_vm):
        with pytest.raises(MemoryError):
            monitor.check()
