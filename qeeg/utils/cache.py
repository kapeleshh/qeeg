"""
Caching module for storing and retrieving computation results.

This module provides functions for caching expensive computations
to improve performance when processing EEG data.
"""

import os
import pickle
import hashlib
import time
import functools
import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np

from qeeg.utils.logging import get_logger

# Get logger
logger = get_logger(__name__)


def get_cache_dir() -> str:
    """
    Get the cache directory path.
    
    Returns
    -------
    str
        Path to the cache directory
    
    Examples
    --------
    >>> from qeeg.utils.cache import get_cache_dir
    >>> cache_dir = get_cache_dir()
    >>> print(cache_dir)
    """
    # Default cache directory is ~/.qeeg_cache
    cache_dir = os.path.join(os.path.expanduser('~'), '.qeeg_cache')
    
    # Create the directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    
    return cache_dir


def clear_cache(older_than: Optional[float] = None) -> int:
    """
    Clear the cache directory.
    
    Parameters
    ----------
    older_than : float or None, optional
        If provided, only clear files older than this many days, by default None (clear all)
    
    Returns
    -------
    int
        Number of files removed
    
    Examples
    --------
    >>> from qeeg.utils.cache import clear_cache
    >>> # Clear all cache files
    >>> n_removed = clear_cache()
    >>> print(f"Removed {n_removed} cache files")
    >>> # Clear cache files older than 30 days
    >>> n_removed = clear_cache(older_than=30)
    >>> print(f"Removed {n_removed} cache files older than 30 days")
    """
    cache_dir = get_cache_dir()
    count = 0
    
    # Get the current time
    now = time.time()
    
    # Iterate over files in the cache directory
    for filename in os.listdir(cache_dir):
        file_path = os.path.join(cache_dir, filename)
        
        # Skip directories
        if os.path.isdir(file_path):
            continue
        
        # Check if the file is old enough to be removed
        if older_than is not None:
            file_age_days = (now - os.path.getmtime(file_path)) / (60 * 60 * 24)
            if file_age_days < older_than:
                continue
        
        # Remove the file
        try:
            os.remove(file_path)
            count += 1
        except Exception as e:
            logger.warning(f"Failed to remove cache file {file_path}: {str(e)}")
    
    logger.info(f"Removed {count} cache files")
    return count


def get_cache_size() -> Dict[str, Union[int, float]]:
    """
    Get the size of the cache directory.
    
    Returns
    -------
    Dict[str, Union[int, float]]
        Dictionary with cache size information
    
    Examples
    --------
    >>> from qeeg.utils.cache import get_cache_size
    >>> size_info = get_cache_size()
    >>> print(f"Cache size: {size_info['size_mb']:.2f} MB ({size_info['n_files']} files)")
    """
    cache_dir = get_cache_dir()
    total_size = 0
    n_files = 0
    
    # Iterate over files in the cache directory
    for filename in os.listdir(cache_dir):
        file_path = os.path.join(cache_dir, filename)
        
        # Skip directories
        if os.path.isdir(file_path):
            continue
        
        # Get file size
        try:
            total_size += os.path.getsize(file_path)
            n_files += 1
        except Exception as e:
            logger.warning(f"Failed to get size of cache file {file_path}: {str(e)}")
    
    # Convert to MB
    size_mb = total_size / (1024 * 1024)
    
    return {
        'size_bytes': total_size,
        'size_mb': size_mb,
        'n_files': n_files
    }


def cache_result(
    func: Optional[Callable] = None,
    *,
    key_prefix: Optional[str] = None,
    expire_after: Optional[float] = None
) -> Callable:
    """
    Decorator to cache function results to disk.
    
    Parameters
    ----------
    func : callable or None, optional
        Function to cache results for, by default None
    key_prefix : str or None, optional
        Prefix for cache keys, by default None (use function name)
    expire_after : float or None, optional
        Time in days after which cache entries expire, by default None (never expire)
    
    Returns
    -------
    callable
        Wrapped function with caching
    
    Examples
    --------
    >>> from qeeg.utils.cache import cache_result
    >>> @cache_result
    ... def expensive_computation(x):
    ...     import time
    ...     time.sleep(1)  # Simulate expensive computation
    ...     return x ** 2
    >>> # First call will be slow
    >>> result1 = expensive_computation(10)
    >>> # Second call will be fast (cached)
    >>> result2 = expensive_computation(10)
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get the cache directory
            cache_dir = get_cache_dir()
            
            # Create a hash of the function name, arguments, and keyword arguments
            if key_prefix is None:
                prefix = func.__name__
            else:
                prefix = key_prefix
            
            # Convert args and kwargs to a string representation
            args_str = str(args)
            kwargs_str = str(sorted(kwargs.items()))
            
            # Create a hash
            hash_str = f"{prefix}:{args_str}:{kwargs_str}"
            hash_obj = hashlib.md5(hash_str.encode())
            cache_key = hash_obj.hexdigest()
            
            # Create the cache file path
            cache_file = os.path.join(cache_dir, f"{cache_key}.pkl")
            
            # Check if the cache file exists and is not expired
            if os.path.exists(cache_file):
                if expire_after is not None:
                    # Check if the file is expired
                    file_age_days = (time.time() - os.path.getmtime(cache_file)) / (60 * 60 * 24)
                    if file_age_days > expire_after:
                        logger.debug(f"Cache entry expired for {func.__name__}")
                    else:
                        # Load the cached result
                        try:
                            with open(cache_file, 'rb') as f:
                                result = pickle.load(f)
                            logger.debug(f"Cache hit for {func.__name__}")
                            return result
                        except Exception as e:
                            logger.warning(f"Failed to load cache file {cache_file}: {str(e)}")
                else:
                    # Load the cached result
                    try:
                        with open(cache_file, 'rb') as f:
                            result = pickle.load(f)
                        logger.debug(f"Cache hit for {func.__name__}")
                        return result
                    except Exception as e:
                        logger.warning(f"Failed to load cache file {cache_file}: {str(e)}")
            
            # Call the function and cache the result
            logger.debug(f"Cache miss for {func.__name__}")
            result = func(*args, **kwargs)
            
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(result, f)
                logger.debug(f"Cached result for {func.__name__}")
            except Exception as e:
                logger.warning(f"Failed to cache result for {func.__name__}: {str(e)}")
            
            return result
        
        return wrapper
    
    if func is None:
        return decorator
    else:
        return decorator(func)


def memoize(func: Callable) -> Callable:
    """
    Decorator to memoize function results in memory.
    
    Parameters
    ----------
    func : callable
        Function to memoize
    
    Returns
    -------
    callable
        Wrapped function with memoization
    
    Examples
    --------
    >>> from qeeg.utils.cache import memoize
    >>> @memoize
    ... def fibonacci(n):
    ...     if n <= 1:
    ...         return n
    ...     return fibonacci(n-1) + fibonacci(n-2)
    >>> # This would be very slow without memoization
    >>> fibonacci(30)
    832040
    """
    cache = {}
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Create a key from the arguments
        key = str(args) + str(sorted(kwargs.items()))
        
        # Check if the result is cached
        if key in cache:
            return cache[key]
        
        # Call the function and cache the result
        result = func(*args, **kwargs)
        cache[key] = result
        
        return result
    
    # Add a method to clear the cache
    def clear_cache():
        cache.clear()
    
    wrapper.clear_cache = clear_cache
    
    return wrapper


class LRUCache:
    """
    Least Recently Used (LRU) cache implementation.
    
    Parameters
    ----------
    max_size : int, optional
        Maximum number of items to store in the cache, by default 128
    
    Examples
    --------
    >>> from qeeg.utils.cache import LRUCache
    >>> cache = LRUCache(max_size=2)
    >>> cache.set('a', 1)
    >>> cache.set('b', 2)
    >>> cache.get('a')
    1
    >>> cache.set('c', 3)  # 'b' will be evicted
    >>> cache.get('b')
    None
    """
    
    def __init__(self, max_size: int = 128):
        self.max_size = max_size
        self.cache = {}
        self.usage = []
    
    def get(self, key: Any) -> Any:
        """
        Get a value from the cache.
        
        Parameters
        ----------
        key : any
            The key to look up
        
        Returns
        -------
        any
            The cached value, or None if not found
        """
        if key in self.cache:
            # Move the key to the end of the usage list (most recently used)
            self.usage.remove(key)
            self.usage.append(key)
            return self.cache[key]
        return None
    
    def set(self, key: Any, value: Any) -> None:
        """
        Set a value in the cache.
        
        Parameters
        ----------
        key : any
            The key to store
        value : any
            The value to store
        """
        if key in self.cache:
            # Update existing key
            self.cache[key] = value
            # Move the key to the end of the usage list (most recently used)
            self.usage.remove(key)
            self.usage.append(key)
        else:
            # Add new key
            if len(self.cache) >= self.max_size:
                # Remove least recently used item
                lru_key = self.usage.pop(0)
                del self.cache[lru_key]
            
            # Add the new item
            self.cache[key] = value
            self.usage.append(key)
    
    def clear(self) -> None:
        """Clear the cache."""
        self.cache.clear()
        self.usage.clear()
    
    def __len__(self) -> int:
        """Return the number of items in the cache."""
        return len(self.cache)
    
    def __contains__(self, key: Any) -> bool:
        """Check if a key is in the cache."""
        return key in self.cache


def lru_cache(
    maxsize: int = 128,
    typed: bool = False
) -> Callable:
    """
    Decorator to cache function results with a Least Recently Used (LRU) strategy.
    
    This is a thin wrapper around functools.lru_cache for consistency with other
    caching decorators in this module.
    
    Parameters
    ----------
    maxsize : int, optional
        Maximum number of items to store in the cache, by default 128
    typed : bool, optional
        If True, arguments of different types will be cached separately, by default False
    
    Returns
    -------
    callable
        Wrapped function with LRU caching
    
    Examples
    --------
    >>> from qeeg.utils.cache import lru_cache
    >>> @lru_cache(maxsize=32)
    ... def factorial(n):
    ...     if n <= 1:
    ...         return 1
    ...     return n * factorial(n-1)
    >>> factorial(10)
    3628800
    """
    return functools.lru_cache(maxsize=maxsize, typed=typed)
