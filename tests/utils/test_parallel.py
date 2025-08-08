"""
Tests for the parallel module.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from qeeg.utils.parallel import parallel_process


def test_parallel_process():
    # Define a test function
    def square(x):
        return x ** 2
    
    # Test with a list of inputs
    inputs = [1, 2, 3, 4, 5]
    results = parallel_process(square, inputs)
    
    # Check that the results are correct
    assert results == [1, 4, 9, 16, 25]
    
    # Test with n_jobs=1 (sequential processing)
    results = parallel_process(square, inputs, n_jobs=1)
    assert results == [1, 4, 9, 16, 25]
    
    # Test with n_jobs=2 (parallel processing)
    results = parallel_process(square, inputs, n_jobs=2)
    assert results == [1, 4, 9, 16, 25]
    
    # Test with n_jobs=-1 (use all available cores)
    results = parallel_process(square, inputs, n_jobs=-1)
    assert results == [1, 4, 9, 16, 25]


def test_parallel_process_with_kwargs():
    # Define a test function with kwargs
    def power(x, exponent=2):
        return x ** exponent
    
    # Test with a list of inputs and kwargs
    inputs = [1, 2, 3, 4, 5]
    results = parallel_process(power, inputs, exponent=3)
    
    # Check that the results are correct
    assert results == [1, 8, 27, 64, 125]


def test_parallel_process_with_progress_bar():
    # Define a test function
    def square(x):
        return x ** 2
    
    # Test with progress_bar=True
    with patch('tqdm.tqdm') as mock_tqdm:
        mock_tqdm.return_value = range(5)  # Mock the tqdm iterator
        
        inputs = [1, 2, 3, 4, 5]
        results = parallel_process(square, inputs, progress_bar=True)
        
        # Check that tqdm was called
        mock_tqdm.assert_called_once()
        
        # Check that the results are correct
        assert results == [1, 4, 9, 16, 25]


def test_parallel_process_with_error():
    # Define a test function that raises an error
    def failing_func(x):
        if x == 3:
            raise ValueError("Error for x=3")
        return x ** 2
    
    # Test with a function that raises an error
    inputs = [1, 2, 3, 4, 5]
    
    # Check that the error is propagated
    with pytest.raises(ValueError):
        parallel_process(failing_func, inputs)


def test_parallel_process_with_empty_input():
    # Define a test function
    def square(x):
        return x ** 2
    
    # Test with an empty list
    results = parallel_process(square, [])
    
    # Check that the results are correct
    assert results == []


def test_parallel_process_with_numpy_array():
    # Define a test function
    def square(x):
        return x ** 2
    
    # Test with a numpy array
    inputs = np.array([1, 2, 3, 4, 5])
    results = parallel_process(square, inputs)
    
    # Check that the results are correct
    assert results == [1, 4, 9, 16, 25]


def test_parallel_process_with_joblib():
    # Define a test function
    def square(x):
        return x ** 2
    
    # Test with backend='joblib'
    with patch('joblib.Parallel') as mock_parallel:
        mock_parallel.return_value.return_value = [1, 4, 9, 16, 25]
        
        inputs = [1, 2, 3, 4, 5]
        results = parallel_process(square, inputs, backend='joblib')
        
        # Check that joblib.Parallel was called
        mock_parallel.assert_called_once()
        
        # Check that the results are correct
        assert results == [1, 4, 9, 16, 25]


def test_parallel_process_with_concurrent():
    # Define a test function
    def square(x):
        return x ** 2
    
    # Test with backend='concurrent'
    with patch('concurrent.futures.ProcessPoolExecutor') as mock_executor:
        mock_executor.return_value.__enter__.return_value.map.return_value = [1, 4, 9, 16, 25]
        
        inputs = [1, 2, 3, 4, 5]
        results = parallel_process(square, inputs, backend='concurrent')
        
        # Check that concurrent.futures.ProcessPoolExecutor was called
        mock_executor.assert_called_once()
        
        # Check that the results are correct
        assert results == [1, 4, 9, 16, 25]


def test_parallel_process_with_invalid_backend():
    # Define a test function
    def square(x):
        return x ** 2
    
    # Test with an invalid backend
    with pytest.raises(ValueError):
        parallel_process(square, [1, 2, 3], backend='invalid')
