"""
Tests for the memory module.
"""

import pytest
import numpy as np
import mne
from unittest.mock import patch, MagicMock

from epilepsy_eeg.utils.memory import (
    process_in_chunks,
    estimate_memory_usage,
    check_memory_limit,
    suggest_chunk_size
)
from epilepsy_eeg.utils.exceptions import EEGError


def create_test_raw():
    """Create a test Raw object for testing."""
    data = np.random.randn(2, 1000)
    info = mne.create_info(['ch1', 'ch2'], 100, 'eeg')
    return mne.io.RawArray(data, info)


def test_process_in_chunks():
    # Create a test Raw object
    raw = create_test_raw()
    
    # Define a test function to apply to chunks
    def test_func(raw_chunk):
        return raw_chunk.get_data().mean()
    
    # Test processing in chunks
    results = process_in_chunks(raw, chunk_size_seconds=0.5, func=test_func)
    
    # Should return a list of results
    assert isinstance(results, float)  # Combined result
    
    # Test processing in chunks without combining results
    results = process_in_chunks(raw, chunk_size_seconds=0.5, func=test_func, combine_results=False)
    
    # Should return a list of results
    assert isinstance(results, list)
    assert len(results) > 1
    
    # Test with overlap
    results = process_in_chunks(raw, chunk_size_seconds=0.5, overlap_seconds=0.1, func=test_func)
    assert isinstance(results, float)
    
    # Test with invalid parameters
    with pytest.raises(ValueError):
        process_in_chunks(raw, chunk_size_seconds=0, func=test_func)
    
    with pytest.raises(ValueError):
        process_in_chunks(raw, chunk_size_seconds=0.5, overlap_seconds=0.5, func=test_func)
    
    with pytest.raises(TypeError):
        process_in_chunks("not a raw object", chunk_size_seconds=0.5, func=test_func)
    
    # Test with a function that raises an exception
    def failing_func(raw_chunk):
        raise ValueError("Test error")
    
    with pytest.raises(EEGError):
        process_in_chunks(raw, chunk_size_seconds=0.5, func=failing_func)


def test_process_in_chunks_with_dict_results():
    # Create a test Raw object
    raw = create_test_raw()
    
    # Define a test function that returns a dictionary
    def test_func(raw_chunk):
        data = raw_chunk.get_data()
        return {
            'mean': data.mean(),
            'std': data.std()
        }
    
    # Test processing in chunks with dictionary results
    results = process_in_chunks(raw, chunk_size_seconds=0.5, func=test_func)
    
    # Should return a combined dictionary
    assert isinstance(results, dict)
    assert 'mean' in results
    assert 'std' in results
    
    # Test without combining results
    results = process_in_chunks(raw, chunk_size_seconds=0.5, func=test_func, combine_results=False)
    
    # Should return a list of dictionaries
    assert isinstance(results, list)
    assert all(isinstance(r, dict) for r in results)


def test_estimate_memory_usage():
    # Create a test Raw object
    raw = create_test_raw()
    
    # Test estimating memory usage for different operations
    filter_usage = estimate_memory_usage(raw, operation='filter')
    ica_usage = estimate_memory_usage(raw, operation='ica')
    psd_usage = estimate_memory_usage(raw, operation='psd')
    epochs_usage = estimate_memory_usage(raw, operation='epochs')
    
    # Check that the estimates are reasonable
    assert filter_usage['total'] > 0
    assert ica_usage['total'] > 0
    assert psd_usage['total'] > 0
    assert epochs_usage['total'] > 0
    
    # Check that the estimates include the expected keys
    for usage in [filter_usage, ica_usage, psd_usage, epochs_usage]:
        assert 'raw_data_mb' in usage
        assert 'operation_data_mb' in usage
        assert 'total' in usage
    
    # Test with invalid operation
    with pytest.raises(ValueError):
        estimate_memory_usage(raw, operation='invalid_operation')
    
    # Test with invalid raw object
    with pytest.raises(TypeError):
        estimate_memory_usage("not a raw object", operation='filter')


def test_check_memory_limit():
    # Test with memory usage below limit
    assert check_memory_limit(100, limit_mb=200) is True
    
    # Test with memory usage above limit
    assert check_memory_limit(300, limit_mb=200) is False
    
    # Test with automatic limit detection
    with patch('psutil.virtual_memory') as mock_vm:
        # Mock available memory to be 1000 MB
        mock_vm.return_value.available = 1000 * 1024 * 1024
        
        # Test with memory usage below 80% of available memory
        assert check_memory_limit(700) is True
        
        # Test with memory usage above 80% of available memory
        assert check_memory_limit(900) is False


def test_suggest_chunk_size():
    # Create a test Raw object
    raw = create_test_raw()
    
    # Test suggesting chunk size for different operations
    filter_chunk = suggest_chunk_size(raw, operation='filter', memory_limit_mb=100)
    ica_chunk = suggest_chunk_size(raw, operation='ica', memory_limit_mb=100)
    psd_chunk = suggest_chunk_size(raw, operation='psd', memory_limit_mb=100)
    epochs_chunk = suggest_chunk_size(raw, operation='epochs', memory_limit_mb=100)
    
    # Check that the suggestions are reasonable
    assert filter_chunk > 0
    assert ica_chunk > 0
    assert psd_chunk > 0
    assert epochs_chunk > 0
    
    # Check that more memory-intensive operations suggest smaller chunks
    assert ica_chunk <= filter_chunk
    
    # Test with automatic memory limit detection
    with patch('psutil.virtual_memory') as mock_vm:
        # Mock available memory to be 1000 MB
        mock_vm.return_value.available = 1000 * 1024 * 1024
        
        chunk_size = suggest_chunk_size(raw, operation='filter')
        assert chunk_size > 0
    
    # Test with invalid raw object
    with pytest.raises(TypeError):
        suggest_chunk_size("not a raw object", operation='filter')
