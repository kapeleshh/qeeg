"""
Benchmarking module for EEG data processing.

This module provides functions for benchmarking the performance of
EEG data processing functions and algorithms.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Dict, List, Optional, Any, Tuple, Union
import mne
import pandas as pd
from functools import wraps

from qeeg.utils.logging import get_logger

# Get logger
logger = get_logger(__name__)


def timeit(func: Callable) -> Callable:
    """
    Decorator to measure the execution time of a function.
    
    Parameters
    ----------
    func : callable
        Function to time
    
    Returns
    -------
    callable
        Wrapped function that prints execution time
    
    Examples
    --------
    >>> from qeeg.utils.benchmark import timeit
    >>> @timeit
    ... def slow_function():
    ...     import time
    ...     time.sleep(1)
    >>> slow_function()
    Function 'slow_function' executed in 1.001 seconds
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"Function '{func.__name__}' executed in {end_time - start_time:.3f} seconds")
        return result
    return wrapper


def benchmark_function(
    func: Callable,
    *args,
    n_runs: int = 10,
    **kwargs
) -> Dict[str, Union[float, List[float]]]:
    """
    Benchmark a function by running it multiple times and measuring execution time.
    
    Parameters
    ----------
    func : callable
        Function to benchmark
    *args
        Positional arguments to pass to the function
    n_runs : int, optional
        Number of runs to perform, by default 10
    **kwargs
        Keyword arguments to pass to the function
    
    Returns
    -------
    Dict[str, Union[float, List[float]]]
        Dictionary with benchmark results
    
    Examples
    --------
    >>> from qeeg.utils.benchmark import benchmark_function
    >>> def test_func(x):
    ...     return x ** 2
    >>> results = benchmark_function(test_func, 10, n_runs=5)
    >>> print(f"Mean execution time: {results['mean']:.6f} seconds")
    """
    times = []
    
    logger.info(f"Benchmarking function '{func.__name__}' with {n_runs} runs")
    
    for i in range(n_runs):
        start_time = time.time()
        func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        times.append(execution_time)
        logger.debug(f"Run {i+1}/{n_runs}: {execution_time:.6f} seconds")
    
    results = {
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times),
        'median': np.median(times),
        'times': times
    }
    
    logger.info(f"Benchmark results for '{func.__name__}':")
    logger.info(f"  Mean: {results['mean']:.6f} seconds")
    logger.info(f"  Std: {results['std']:.6f} seconds")
    logger.info(f"  Min: {results['min']:.6f} seconds")
    logger.info(f"  Max: {results['max']:.6f} seconds")
    logger.info(f"  Median: {results['median']:.6f} seconds")
    
    return results


def plot_benchmark_results(
    results_dict: Dict[str, Dict[str, Union[float, List[float]]]],
    title: str = 'Benchmark Results',
    metric: str = 'mean',
    figsize: Tuple[int, int] = (10, 6),
    show: bool = True,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot benchmark results.
    
    Parameters
    ----------
    results_dict : Dict[str, Dict[str, Union[float, List[float]]]]
        Dictionary mapping function names to benchmark results
    title : str, optional
        Plot title, by default 'Benchmark Results'
    metric : str, optional
        Metric to plot, by default 'mean'
    figsize : Tuple[int, int], optional
        Figure size, by default (10, 6)
    show : bool, optional
        Whether to show the plot, by default True
    save_path : str or None, optional
        Path to save the plot, by default None
    
    Returns
    -------
    plt.Figure
        The matplotlib figure
    
    Examples
    --------
    >>> from qeeg.utils.benchmark import benchmark_function, plot_benchmark_results
    >>> def func1(x):
    ...     return x ** 2
    >>> def func2(x):
    ...     return x ** 3
    >>> results = {
    ...     'func1': benchmark_function(func1, 10, n_runs=5),
    ...     'func2': benchmark_function(func2, 10, n_runs=5)
    ... }
    >>> fig = plot_benchmark_results(results, title='Comparison of Functions')
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    names = list(results_dict.keys())
    values = [results_dict[name][metric] for name in names]
    
    if metric in ['mean', 'median']:
        # Add error bars for mean or median
        errors = [results_dict[name]['std'] for name in names]
        bars = ax.bar(names, values, yerr=errors, capsize=10)
    else:
        bars = ax.bar(names, values)
    
    # Add value labels on top of the bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            height,
            f'{height:.6f}s',
            ha='center',
            va='bottom',
            rotation=0
        )
    
    ax.set_ylabel(f'Execution Time ({metric})')
    ax.set_title(title)
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Benchmark plot saved to {save_path}")
    
    if show:
        plt.show()
    
    return fig


def compare_functions(
    functions: Dict[str, Callable],
    args_list: Optional[List[Tuple]] = None,
    kwargs_list: Optional[List[Dict]] = None,
    n_runs: int = 5,
    title: str = 'Function Comparison',
    show: bool = True,
    save_path: Optional[str] = None
) -> Dict[str, Dict[str, Union[float, List[float]]]]:
    """
    Compare the performance of multiple functions.
    
    Parameters
    ----------
    functions : Dict[str, Callable]
        Dictionary mapping function names to functions
    args_list : List[Tuple] or None, optional
        List of positional arguments for each function, by default None
    kwargs_list : List[Dict] or None, optional
        List of keyword arguments for each function, by default None
    n_runs : int, optional
        Number of runs to perform, by default 5
    title : str, optional
        Plot title, by default 'Function Comparison'
    show : bool, optional
        Whether to show the plot, by default True
    save_path : str or None, optional
        Path to save the plot, by default None
    
    Returns
    -------
    Dict[str, Dict[str, Union[float, List[float]]]]
        Dictionary with benchmark results
    
    Examples
    --------
    >>> from qeeg.utils.benchmark import compare_functions
    >>> def func1(x):
    ...     return x ** 2
    >>> def func2(x):
    ...     return x ** 3
    >>> functions = {'Square': func1, 'Cube': func2}
    >>> args_list = [(10,), (10,)]
    >>> results = compare_functions(functions, args_list=args_list)
    """
    if args_list is None:
        args_list = [()] * len(functions)
    
    if kwargs_list is None:
        kwargs_list = [{}] * len(functions)
    
    if len(args_list) != len(functions) or len(kwargs_list) != len(functions):
        raise ValueError("Length of args_list and kwargs_list must match length of functions")
    
    results = {}
    
    for (name, func), args, kwargs in zip(functions.items(), args_list, kwargs_list):
        logger.info(f"Benchmarking function '{name}'")
        results[name] = benchmark_function(func, *args, n_runs=n_runs, **kwargs)
    
    # Plot results
    plot_benchmark_results(results, title=title, show=show, save_path=save_path)
    
    return results


def benchmark_preprocessing_pipeline(
    raw: mne.io.Raw,
    pipeline_steps: List[Dict[str, Any]],
    n_runs: int = 3,
    show: bool = True,
    save_path: Optional[str] = None
) -> Dict[str, Dict[str, Union[float, List[float]]]]:
    """
    Benchmark a preprocessing pipeline.
    
    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data
    pipeline_steps : List[Dict[str, Any]]
        List of pipeline steps, each a dictionary with 'name', 'func', and 'kwargs'
    n_runs : int, optional
        Number of runs to perform, by default 3
    show : bool, optional
        Whether to show the plot, by default True
    save_path : str or None, optional
        Path to save the plot, by default None
    
    Returns
    -------
    Dict[str, Dict[str, Union[float, List[float]]]]
        Dictionary with benchmark results
    
    Examples
    --------
    >>> import mne
    >>> from qeeg.utils.benchmark import benchmark_preprocessing_pipeline
    >>> from qeeg.preprocessing import filtering, artifacts
    >>> raw = mne.io.read_raw_edf("sample.edf", preload=True)
    >>> pipeline_steps = [
    ...     {'name': 'Bandpass Filter', 'func': filtering.bandpass_filter, 'kwargs': {'l_freq': 1.0, 'h_freq': 40.0}},
    ...     {'name': 'Notch Filter', 'func': filtering.notch_filter, 'kwargs': {'freqs': 50.0}},
    ...     {'name': 'ICA', 'func': artifacts.remove_artifacts_ica, 'kwargs': {'n_components': 10}}
    ... ]
    >>> results = benchmark_preprocessing_pipeline(raw, pipeline_steps)
    """
    results = {}
    
    # Make a copy of the raw data for each step
    raw_copy = raw.copy()
    
    for step in pipeline_steps:
        name = step['name']
        func = step['func']
        kwargs = step.get('kwargs', {})
        
        logger.info(f"Benchmarking preprocessing step '{name}'")
        
        # Benchmark the function
        step_results = benchmark_function(func, raw_copy, n_runs=n_runs, **kwargs)
        results[name] = step_results
        
        # Apply the function once to get the output for the next step
        raw_copy = func(raw_copy, **kwargs)
    
    # Plot results
    plot_benchmark_results(
        results,
        title='Preprocessing Pipeline Benchmark',
        show=show,
        save_path=save_path
    )
    
    return results


def benchmark_memory_usage(
    func: Callable,
    *args,
    n_runs: int = 3,
    **kwargs
) -> Dict[str, float]:
    """
    Benchmark memory usage of a function.
    
    Parameters
    ----------
    func : callable
        Function to benchmark
    *args
        Positional arguments to pass to the function
    n_runs : int, optional
        Number of runs to perform, by default 3
    **kwargs
        Keyword arguments to pass to the function
    
    Returns
    -------
    Dict[str, float]
        Dictionary with memory usage statistics in MB
    
    Examples
    --------
    >>> from qeeg.utils.benchmark import benchmark_memory_usage
    >>> def create_array(size):
    ...     import numpy as np
    ...     return np.random.randn(size, size)
    >>> memory_usage = benchmark_memory_usage(create_array, 1000, n_runs=3)
    >>> print(f"Peak memory usage: {memory_usage['peak']:.2f} MB")
    """
    try:
        import psutil
        import os
        process = psutil.Process(os.getpid())
    except ImportError:
        logger.warning("psutil not installed. Cannot measure memory usage.")
        return {'peak': 0.0, 'mean': 0.0, 'baseline': 0.0, 'net': 0.0}
    
    # Get baseline memory usage
    baseline = process.memory_info().rss / (1024 * 1024)
    
    peak_memory = baseline
    memory_usage = []
    
    for _ in range(n_runs):
        # Run the function
        func(*args, **kwargs)
        
        # Measure memory usage
        current_memory = process.memory_info().rss / (1024 * 1024)
        memory_usage.append(current_memory)
        
        # Update peak memory
        peak_memory = max(peak_memory, current_memory)
    
    # Calculate statistics
    mean_memory = np.mean(memory_usage)
    net_memory = mean_memory - baseline
    
    results = {
        'peak': peak_memory,
        'mean': mean_memory,
        'baseline': baseline,
        'net': net_memory
    }
    
    logger.info(f"Memory usage for '{func.__name__}':")
    logger.info(f"  Baseline: {baseline:.2f} MB")
    logger.info(f"  Peak: {peak_memory:.2f} MB")
    logger.info(f"  Mean: {mean_memory:.2f} MB")
    logger.info(f"  Net: {net_memory:.2f} MB")
    
    return results


def create_benchmark_report(
    results: Dict[str, Dict[str, Union[float, List[float]]]],
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Create a benchmark report as a pandas DataFrame.
    
    Parameters
    ----------
    results : Dict[str, Dict[str, Union[float, List[float]]]]
        Dictionary with benchmark results
    save_path : str or None, optional
        Path to save the report as CSV, by default None
    
    Returns
    -------
    pd.DataFrame
        DataFrame with benchmark results
    
    Examples
    --------
    >>> from qeeg.utils.benchmark import benchmark_function, create_benchmark_report
    >>> def func1(x):
    ...     return x ** 2
    >>> def func2(x):
    ...     return x ** 3
    >>> results = {
    ...     'func1': benchmark_function(func1, 10, n_runs=5),
    ...     'func2': benchmark_function(func2, 10, n_runs=5)
    ... }
    >>> report = create_benchmark_report(results, save_path='benchmark_report.csv')
    >>> print(report)
    """
    # Create a list of dictionaries for the DataFrame
    data = []
    
    for name, result in results.items():
        row = {
            'Function': name,
            'Mean (s)': result['mean'],
            'Std (s)': result['std'],
            'Min (s)': result['min'],
            'Max (s)': result['max'],
            'Median (s)': result['median']
        }
        data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV if requested
    if save_path:
        df.to_csv(save_path, index=False)
        logger.info(f"Benchmark report saved to {save_path}")
    
    return df
