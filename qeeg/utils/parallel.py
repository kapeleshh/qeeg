"""
Parallel processing module for EEG data analysis.

This module provides functions for parallel processing of EEG data,
enabling faster computation for computationally intensive operations.
"""

import numpy as np
from typing import Tuple, Optional, Union, List, Dict, Any, Callable
import multiprocessing
from joblib import Parallel, delayed
import mne


def parallel_apply(
    func: Callable,
    items: List[Any],
    n_jobs: Optional[int] = None,
    backend: str = 'loky',
    verbose: int = 0,
    **kwargs
) -> List[Any]:
    """
    Apply a function to a list of items in parallel.

    Parameters
    ----------
    func : Callable
        The function to apply.
    items : List[Any]
        The list of items to apply the function to.
    n_jobs : int or None, optional
        Number of jobs to run in parallel, by default None (use all available cores)
    backend : str, optional
        Backend to use for parallelization, by default 'loky'
    verbose : int, optional
        Verbosity level, by default 0
    **kwargs
        Additional keyword arguments to pass to the function

    Returns
    -------
    List[Any]
        List of results.

    Examples
    --------
    >>> from qeeg.utils import parallel
    >>> def square(x):
    ...     return x ** 2
    >>> results = parallel.parallel_apply(square, [1, 2, 3, 4, 5])
    >>> print(results)
    [1, 4, 9, 16, 25]
    """
    # Set the number of jobs
    if n_jobs is None:
        n_jobs = multiprocessing.cpu_count()
    
    # Apply the function in parallel
    results = Parallel(n_jobs=n_jobs, backend=backend, verbose=verbose)(
        delayed(func)(item, **kwargs) for item in items
    )
    
    return results


def parallel_extract_features(
    raw_list: List[mne.io.Raw],
    feature_extractor: Callable,
    n_jobs: Optional[int] = None,
    backend: str = 'loky',
    verbose: int = 0,
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Extract features from a list of EEG data in parallel.

    Parameters
    ----------
    raw_list : List[mne.io.Raw]
        List of raw EEG data.
    feature_extractor : Callable
        Function to extract features from the EEG data.
    n_jobs : int or None, optional
        Number of jobs to run in parallel, by default None (use all available cores)
    backend : str, optional
        Backend to use for parallelization, by default 'loky'
    verbose : int, optional
        Verbosity level, by default 0
    **kwargs
        Additional keyword arguments to pass to the feature extractor

    Returns
    -------
    List[Dict[str, Any]]
        List of feature dictionaries.

    Examples
    --------
    >>> import mne
    >>> from qeeg.utils import parallel
    >>> from qeeg.ml import features
    >>> raw_list = [mne.io.read_raw_edf("sample1.edf", preload=True),
    ...             mne.io.read_raw_edf("sample2.edf", preload=True)]
    >>> feature_dicts = parallel.parallel_extract_features(raw_list, features.extract_all_features)
    """
    return parallel_apply(
        feature_extractor,
        raw_list,
        n_jobs=n_jobs,
        backend=backend,
        verbose=verbose,
        **kwargs
    )


def parallel_preprocess(
    raw_list: List[mne.io.Raw],
    preprocess_func: Callable,
    n_jobs: Optional[int] = None,
    backend: str = 'loky',
    verbose: int = 0,
    **kwargs
) -> List[mne.io.Raw]:
    """
    Preprocess a list of EEG data in parallel.

    Parameters
    ----------
    raw_list : List[mne.io.Raw]
        List of raw EEG data.
    preprocess_func : Callable
        Function to preprocess the EEG data.
    n_jobs : int or None, optional
        Number of jobs to run in parallel, by default None (use all available cores)
    backend : str, optional
        Backend to use for parallelization, by default 'loky'
    verbose : int, optional
        Verbosity level, by default 0
    **kwargs
        Additional keyword arguments to pass to the preprocessing function

    Returns
    -------
    List[mne.io.Raw]
        List of preprocessed EEG data.

    Examples
    --------
    >>> import mne
    >>> from qeeg.utils import parallel
    >>> from qeeg.preprocessing import filtering
    >>> raw_list = [mne.io.read_raw_edf("sample1.edf", preload=True),
    ...             mne.io.read_raw_edf("sample2.edf", preload=True)]
    >>> filtered_list = parallel.parallel_preprocess(raw_list, filtering.bandpass_filter,
    ...                                             l_freq=1.0, h_freq=40.0)
    """
    return parallel_apply(
        preprocess_func,
        raw_list,
        n_jobs=n_jobs,
        backend=backend,
        verbose=verbose,
        **kwargs
    )


def parallel_detect_events(
    raw_list: List[mne.io.Raw],
    detector_func: Callable,
    n_jobs: Optional[int] = None,
    backend: str = 'loky',
    verbose: int = 0,
    **kwargs
) -> List[List[Dict[str, Any]]]:
    """
    Detect events in a list of EEG data in parallel.

    Parameters
    ----------
    raw_list : List[mne.io.Raw]
        List of raw EEG data.
    detector_func : Callable
        Function to detect events in the EEG data.
    n_jobs : int or None, optional
        Number of jobs to run in parallel, by default None (use all available cores)
    backend : str, optional
        Backend to use for parallelization, by default 'loky'
    verbose : int, optional
        Verbosity level, by default 0
    **kwargs
        Additional keyword arguments to pass to the detector function

    Returns
    -------
    List[List[Dict[str, Any]]]
        List of lists of detected events.

    Examples
    --------
    >>> import mne
    >>> from qeeg.utils import parallel
    >>> from qeeg.analysis import epileptiform
    >>> raw_list = [mne.io.read_raw_edf("sample1.edf", preload=True),
    ...             mne.io.read_raw_edf("sample2.edf", preload=True)]
    >>> spikes_list = parallel.parallel_detect_events(raw_list, epileptiform.detect_spikes)
    """
    return parallel_apply(
        detector_func,
        raw_list,
        n_jobs=n_jobs,
        backend=backend,
        verbose=verbose,
        **kwargs
    )


def parallel_cross_validation(
    X: np.ndarray,
    y: np.ndarray,
    model_factory: Callable,
    cv: int = 5,
    n_jobs: Optional[int] = None,
    backend: str = 'loky',
    verbose: int = 0,
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Perform cross-validation in parallel.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target vector.
    model_factory : Callable
        Function that returns a new model instance.
    cv : int, optional
        Number of cross-validation folds, by default 5
    n_jobs : int or None, optional
        Number of jobs to run in parallel, by default None (use all available cores)
    backend : str, optional
        Backend to use for parallelization, by default 'loky'
    verbose : int, optional
        Verbosity level, by default 0
    **kwargs
        Additional keyword arguments to pass to the model factory

    Returns
    -------
    List[Dict[str, Any]]
        List of dictionaries with cross-validation results.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from qeeg.utils import parallel
    >>> X = np.random.randn(100, 10)
    >>> y = np.random.randint(0, 2, 100)
    >>> def model_factory():
    ...     return RandomForestClassifier(n_estimators=100, random_state=42)
    >>> results = parallel.parallel_cross_validation(X, y, model_factory, cv=5)
    """
    from sklearn.model_selection import KFold
    
    # Create the cross-validation object
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    
    # Define a function to train and evaluate a model on a single fold
    def train_eval_fold(fold_idx, train_idx, test_idx):
        # Split the data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Create and train the model
        model = model_factory(**kwargs)
        model.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        metrics = {
            'fold': fold_idx,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted')
        }
        
        return metrics
    
    # Create a list of fold indices and train/test indices
    fold_data = [(i, train_idx, test_idx) for i, (train_idx, test_idx) in enumerate(kf.split(X))]
    
    # Apply the function in parallel
    results = Parallel(n_jobs=n_jobs, backend=backend, verbose=verbose)(
        delayed(train_eval_fold)(fold_idx, train_idx, test_idx)
        for fold_idx, train_idx, test_idx in fold_data
    )
    
    return results


def parallel_grid_search(
    X: np.ndarray,
    y: np.ndarray,
    model_factory: Callable,
    param_grid: Dict[str, List[Any]],
    cv: int = 5,
    n_jobs: Optional[int] = None,
    backend: str = 'loky',
    verbose: int = 0
) -> List[Dict[str, Any]]:
    """
    Perform grid search in parallel.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target vector.
    model_factory : Callable
        Function that returns a new model instance.
    param_grid : Dict[str, List[Any]]
        Dictionary with parameters names (string) as keys and lists of parameter values to try.
    cv : int, optional
        Number of cross-validation folds, by default 5
    n_jobs : int or None, optional
        Number of jobs to run in parallel, by default None (use all available cores)
    backend : str, optional
        Backend to use for parallelization, by default 'loky'
    verbose : int, optional
        Verbosity level, by default 0

    Returns
    -------
    List[Dict[str, Any]]
        List of dictionaries with grid search results.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from qeeg.utils import parallel
    >>> X = np.random.randn(100, 10)
    >>> y = np.random.randint(0, 2, 100)
    >>> def model_factory(n_estimators=100, max_depth=None):
    ...     return RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    >>> param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 5, 10]}
    >>> results = parallel.parallel_grid_search(X, y, model_factory, param_grid, cv=5)
    """
    from sklearn.model_selection import ParameterGrid
    
    # Create the parameter grid
    param_combinations = list(ParameterGrid(param_grid))
    
    # Define a function to evaluate a parameter combination
    def evaluate_params(params):
        # Perform cross-validation with the given parameters
        cv_results = parallel_cross_validation(
            X, y, model_factory, cv=cv, n_jobs=1, backend=backend, verbose=0, **params
        )
        
        # Calculate mean metrics
        mean_metrics = {
            'params': params,
            'mean_accuracy': np.mean([r['accuracy'] for r in cv_results]),
            'mean_precision': np.mean([r['precision'] for r in cv_results]),
            'mean_recall': np.mean([r['recall'] for r in cv_results]),
            'mean_f1': np.mean([r['f1'] for r in cv_results]),
            'std_accuracy': np.std([r['accuracy'] for r in cv_results]),
            'cv_results': cv_results
        }
        
        return mean_metrics
    
    # Apply the function in parallel
    results = Parallel(n_jobs=n_jobs, backend=backend, verbose=verbose)(
        delayed(evaluate_params)(params) for params in param_combinations
    )
    
    # Sort the results by mean accuracy
    results.sort(key=lambda x: x['mean_accuracy'], reverse=True)
    
    return results
