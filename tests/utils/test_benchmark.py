"""
Tests for the benchmark module.
"""

import pytest
import time
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock

from qeeg.utils.benchmark import (
    timeit,
    benchmark_function,
    plot_benchmark_results,
    compare_functions,
    benchmark_preprocessing_pipeline,
    benchmark_memory_usage,
    create_benchmark_report
)


def test_timeit():
    # Test the timeit decorator
    @timeit
    def test_func(sleep_time):
        time.sleep(sleep_time)
        return sleep_time * 2
    
    # Test that the function returns the correct result
    result = test_func(0.01)
    assert result == 0.02
    
    # Test that the decorator preserves function metadata
    assert test_func.__name__ == 'test_func'
    assert 'sleep_time' in test_func.__code__.co_varnames


def test_benchmark_function():
    # Define a test function
    def test_func(x):
        time.sleep(0.01)  # Small sleep to ensure measurable execution time
        return x ** 2
    
    # Test benchmarking the function
    results = benchmark_function(test_func, 10, n_runs=3)
    
    # Check that the results include the expected keys
    assert 'mean' in results
    assert 'std' in results
    assert 'min' in results
    assert 'max' in results
    assert 'median' in results
    assert 'times' in results
    
    # Check that the times are reasonable
    assert results['mean'] > 0
    assert len(results['times']) == 3
    
    # Check that min <= median <= max
    assert results['min'] <= results['median'] <= results['max']


def test_plot_benchmark_results():
    # Create some benchmark results
    results = {
        'func1': {'mean': 0.1, 'std': 0.01, 'min': 0.09, 'max': 0.11, 'median': 0.1, 'times': [0.09, 0.1, 0.11]},
        'func2': {'mean': 0.2, 'std': 0.02, 'min': 0.18, 'max': 0.22, 'median': 0.2, 'times': [0.18, 0.2, 0.22]}
    }
    
    # Test plotting with default parameters
    fig = plot_benchmark_results(results, show=False)
    assert isinstance(fig, plt.Figure)
    
    # Test plotting with custom parameters
    fig = plot_benchmark_results(results, title='Custom Title', metric='min', show=False)
    assert isinstance(fig, plt.Figure)
    
    # Test plotting with save_path
    with patch('matplotlib.pyplot.savefig') as mock_savefig:
        fig = plot_benchmark_results(results, show=False, save_path='test.png')
        mock_savefig.assert_called_once_with('test.png')


def test_compare_functions():
    # Define test functions
    def func1(x):
        time.sleep(0.01)
        return x ** 2
    
    def func2(x):
        time.sleep(0.02)
        return x ** 3
    
    # Test comparing functions with default parameters
    with patch('qeeg.utils.benchmark.plot_benchmark_results') as mock_plot:
        results = compare_functions(
            {'func1': func1, 'func2': func2},
            args_list=[(10,), (10,)],
            n_runs=2,
            show=False
        )
        
        # Check that the results include both functions
        assert 'func1' in results
        assert 'func2' in results
        
        # Check that the plot function was called
        mock_plot.assert_called_once()
    
    # Test with invalid args_list length
    with pytest.raises(ValueError):
        compare_functions(
            {'func1': func1, 'func2': func2},
            args_list=[(10,)],
            n_runs=2
        )


def test_benchmark_preprocessing_pipeline():
    # Create mock Raw object and pipeline steps
    raw = MagicMock()
    
    # Define mock pipeline steps
    step1 = {'name': 'Step 1', 'func': MagicMock(return_value=raw), 'kwargs': {}}
    step2 = {'name': 'Step 2', 'func': MagicMock(return_value=raw), 'kwargs': {}}
    
    pipeline_steps = [step1, step2]
    
    # Test benchmarking the pipeline
    with patch('qeeg.utils.benchmark.benchmark_function') as mock_benchmark:
        mock_benchmark.return_value = {'mean': 0.1, 'std': 0.01, 'min': 0.09, 'max': 0.11, 'median': 0.1, 'times': [0.1]}
        
        with patch('qeeg.utils.benchmark.plot_benchmark_results') as mock_plot:
            results = benchmark_preprocessing_pipeline(raw, pipeline_steps, n_runs=2, show=False)
            
            # Check that benchmark_function was called for each step
            assert mock_benchmark.call_count == 2
            
            # Check that the results include both steps
            assert 'Step 1' in results
            assert 'Step 2' in results
            
            # Check that the plot function was called
            mock_plot.assert_called_once()


def test_benchmark_memory_usage():
    # Define a test function
    def test_func():
        # Allocate some memory
        x = np.zeros((1000, 1000))
        return x
    
    # Test with psutil available
    with patch('psutil.Process') as mock_process:
        # Mock memory_info method
        mock_memory_info = MagicMock()
        mock_memory_info.rss = 100 * 1024 * 1024  # 100 MB
        mock_process.return_value.memory_info.return_value = mock_memory_info
        
        results = benchmark_memory_usage(test_func, n_runs=2)
        
        # Check that the results include the expected keys
        assert 'peak' in results
        assert 'mean' in results
        assert 'baseline' in results
        assert 'net' in results
        
        # Check that the values are reasonable
        assert results['peak'] >= results['baseline']
        assert results['mean'] >= results['baseline']
        assert results['net'] >= 0
    
    # Test with psutil not available
    with patch('qeeg.utils.benchmark.psutil', None):
        with patch('qeeg.utils.benchmark.logger') as mock_logger:
            results = benchmark_memory_usage(test_func, n_runs=2)
            
            # Check that a warning was logged
            mock_logger.warning.assert_called_once()
            
            # Check that default values were returned
            assert results['peak'] == 0.0
            assert results['mean'] == 0.0
            assert results['baseline'] == 0.0
            assert results['net'] == 0.0


def test_create_benchmark_report():
    # Create some benchmark results
    results = {
        'func1': {'mean': 0.1, 'std': 0.01, 'min': 0.09, 'max': 0.11, 'median': 0.1, 'times': [0.09, 0.1, 0.11]},
        'func2': {'mean': 0.2, 'std': 0.02, 'min': 0.18, 'max': 0.22, 'median': 0.2, 'times': [0.18, 0.2, 0.22]}
    }
    
    # Test creating a report
    df = create_benchmark_report(results)
    
    # Check that the DataFrame has the expected shape and columns
    assert df.shape == (2, 6)  # 2 rows, 6 columns (Function, Mean, Std, Min, Max, Median)
    assert 'Function' in df.columns
    assert 'Mean (s)' in df.columns
    assert 'Std (s)' in df.columns
    assert 'Min (s)' in df.columns
    assert 'Max (s)' in df.columns
    assert 'Median (s)' in df.columns
    
    # Check that the values match the input results
    assert df.loc[df['Function'] == 'func1', 'Mean (s)'].values[0] == 0.1
    assert df.loc[df['Function'] == 'func2', 'Mean (s)'].values[0] == 0.2
    
    # Test with save_path
    with patch('pandas.DataFrame.to_csv') as mock_to_csv:
        df = create_benchmark_report(results, save_path='report.csv')
        mock_to_csv.assert_called_once_with('report.csv', index=False)
