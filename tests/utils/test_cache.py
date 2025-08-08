"""
Tests for the cache module.
"""

import os
import time
import pytest
import tempfile
import shutil
from unittest.mock import patch, MagicMock

from qeeg.utils.cache import (
    get_cache_dir,
    clear_cache,
    get_cache_size,
    cache_result,
    memoize,
    LRUCache,
    lru_cache
)


def test_get_cache_dir():
    # Test that the cache directory is created
    cache_dir = get_cache_dir()
    assert os.path.exists(cache_dir)
    assert os.path.isdir(cache_dir)
    assert cache_dir.endswith('.qeeg_cache')


def test_clear_cache():
    # Create a temporary directory to use as cache
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create some test files
        for i in range(5):
            with open(os.path.join(temp_dir, f"test{i}.txt"), 'w') as f:
                f.write(f"Test file {i}")
        
        # Patch get_cache_dir to return our temporary directory
        with patch('qeeg.utils.cache.get_cache_dir', return_value=temp_dir):
            # Test clearing all files
            n_removed = clear_cache()
            assert n_removed == 5
            assert len(os.listdir(temp_dir)) == 0
            
            # Create more test files
            for i in range(5):
                with open(os.path.join(temp_dir, f"test{i}.txt"), 'w') as f:
                    f.write(f"Test file {i}")
            
            # Modify the access time of some files to make them older
            old_file = os.path.join(temp_dir, "test0.txt")
            os.utime(old_file, (time.time() - 40 * 24 * 60 * 60, time.time() - 40 * 24 * 60 * 60))
            
            # Test clearing only old files
            n_removed = clear_cache(older_than=30)
            assert n_removed == 1
            assert len(os.listdir(temp_dir)) == 4


def test_get_cache_size():
    # Create a temporary directory to use as cache
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create some test files
        for i in range(5):
            with open(os.path.join(temp_dir, f"test{i}.txt"), 'w') as f:
                f.write(f"Test file {i}" * 100)  # Make files have some size
        
        # Patch get_cache_dir to return our temporary directory
        with patch('qeeg.utils.cache.get_cache_dir', return_value=temp_dir):
            # Test getting cache size
            size_info = get_cache_size()
            assert size_info['n_files'] == 5
            assert size_info['size_bytes'] > 0
            assert size_info['size_mb'] > 0


def test_cache_result():
    # Create a temporary directory to use as cache
    with tempfile.TemporaryDirectory() as temp_dir:
        # Patch get_cache_dir to return our temporary directory
        with patch('qeeg.utils.cache.get_cache_dir', return_value=temp_dir):
            # Create a test function with a counter to track calls
            counter = {'count': 0}
            
            @cache_result
            def test_func(x):
                counter['count'] += 1
                return x ** 2
            
            # First call should compute the result
            result1 = test_func(10)
            assert result1 == 100
            assert counter['count'] == 1
            
            # Second call with same args should use cache
            result2 = test_func(10)
            assert result2 == 100
            assert counter['count'] == 1  # Counter shouldn't increment
            
            # Call with different args should compute new result
            result3 = test_func(20)
            assert result3 == 400
            assert counter['count'] == 2


def test_cache_result_with_expiry():
    # Create a temporary directory to use as cache
    with tempfile.TemporaryDirectory() as temp_dir:
        # Patch get_cache_dir to return our temporary directory
        with patch('qeeg.utils.cache.get_cache_dir', return_value=temp_dir):
            # Create a test function with a counter to track calls
            counter = {'count': 0}
            
            @cache_result(expire_after=30)  # 30 days
            def test_func(x):
                counter['count'] += 1
                return x ** 2
            
            # First call should compute the result
            result1 = test_func(10)
            assert result1 == 100
            assert counter['count'] == 1
            
            # Second call with same args should use cache
            result2 = test_func(10)
            assert result2 == 100
            assert counter['count'] == 1  # Counter shouldn't increment


def test_memoize():
    # Create a test function with a counter to track calls
    counter = {'count': 0}
    
    @memoize
    def test_func(x):
        counter['count'] += 1
        return x ** 2
    
    # First call should compute the result
    result1 = test_func(10)
    assert result1 == 100
    assert counter['count'] == 1
    
    # Second call with same args should use cache
    result2 = test_func(10)
    assert result2 == 100
    assert counter['count'] == 1  # Counter shouldn't increment
    
    # Call with different args should compute new result
    result3 = test_func(20)
    assert result3 == 400
    assert counter['count'] == 2
    
    # Test clearing the cache
    test_func.clear_cache()
    result4 = test_func(10)
    assert result4 == 100
    assert counter['count'] == 3  # Counter should increment after clearing cache


def test_lru_cache():
    # Create a test function with a counter to track calls
    counter = {'count': 0}
    
    @lru_cache(maxsize=2)
    def test_func(x):
        counter['count'] += 1
        return x ** 2
    
    # First call should compute the result
    result1 = test_func(10)
    assert result1 == 100
    assert counter['count'] == 1
    
    # Second call with same args should use cache
    result2 = test_func(10)
    assert result2 == 100
    assert counter['count'] == 1  # Counter shouldn't increment
    
    # Call with different args should compute new result
    result3 = test_func(20)
    assert result3 == 400
    assert counter['count'] == 2
    
    # Call with a third unique arg should compute new result
    result4 = test_func(30)
    assert result4 == 900
    assert counter['count'] == 3
    
    # Call with the first arg again should compute new result (LRU eviction)
    result5 = test_func(10)
    assert result5 == 100
    assert counter['count'] == 4


def test_lru_cache_class():
    # Create an LRU cache with max size 2
    cache = LRUCache(max_size=2)
    
    # Test setting and getting values
    cache.set('a', 1)
    cache.set('b', 2)
    
    assert cache.get('a') == 1
    assert cache.get('b') == 2
    
    # Test LRU eviction
    cache.set('c', 3)  # This should evict 'a'
    assert cache.get('a') is None
    assert cache.get('b') == 2
    assert cache.get('c') == 3
    
    # Test updating existing key
    cache.set('b', 20)
    assert cache.get('b') == 20
    
    # Test LRU ordering (b was accessed most recently, so c should be evicted)
    cache.set('d', 4)
    assert cache.get('b') == 20
    assert cache.get('c') is None
    assert cache.get('d') == 4
    
    # Test __len__ and __contains__
    assert len(cache) == 2
    assert 'b' in cache
    assert 'd' in cache
    assert 'a' not in cache
    assert 'c' not in cache
    
    # Test clear
    cache.clear()
    assert len(cache) == 0
    assert 'b' not in cache
    assert 'd' not in cache
