"""
Tests for the validation module.
"""

import pytest
import numpy as np
import mne
from typing import Optional, Union, List

from epilepsy_eeg.utils.validation import (
    validate_type,
    validate_range,
    validate_options,
    validate_raw,
    validate_array,
    validate_function_params,
    validate_frequency_bands,
    input_validator
)
from epilepsy_eeg.utils.exceptions import ValidationError


def test_validate_type():
    # Test valid types
    validate_type(10, int, 'param')
    validate_type(10.5, float, 'param')
    validate_type('hello', str, 'param')
    validate_type([1, 2, 3], list, 'param')
    validate_type((1, 2, 3), tuple, 'param')
    validate_type({'a': 1}, dict, 'param')
    
    # Test valid types with tuple of types
    validate_type(10, (int, float), 'param')
    validate_type(10.5, (int, float), 'param')
    
    # Test None with allow_none=True
    validate_type(None, int, 'param', allow_none=True)
    
    # Test invalid types
    with pytest.raises(ValidationError):
        validate_type('hello', int, 'param')
    
    with pytest.raises(ValidationError):
        validate_type(10, str, 'param')
    
    with pytest.raises(ValidationError):
        validate_type(10, (str, list), 'param')
    
    # Test None with allow_none=False
    with pytest.raises(ValidationError):
        validate_type(None, int, 'param', allow_none=False)


def test_validate_range():
    # Test valid ranges
    validate_range(5, 0, 10, 'param')
    validate_range(0, 0, 10, 'param')
    validate_range(10, 0, 10, 'param')
    validate_range(5, None, 10, 'param')
    validate_range(5, 0, None, 'param')
    validate_range(5, None, None, 'param')
    
    # Test exclusive bounds
    validate_range(5, 0, 10, 'param', inclusive_min=False, inclusive_max=False)
    
    # Test invalid ranges
    with pytest.raises(ValidationError):
        validate_range(-1, 0, 10, 'param')
    
    with pytest.raises(ValidationError):
        validate_range(11, 0, 10, 'param')
    
    with pytest.raises(ValidationError):
        validate_range(0, 0, 10, 'param', inclusive_min=False)
    
    with pytest.raises(ValidationError):
        validate_range(10, 0, 10, 'param', inclusive_max=False)
    
    # Test non-numeric value
    with pytest.raises(ValidationError):
        validate_range('hello', 0, 10, 'param')


def test_validate_options():
    # Test valid options
    validate_options('apple', ['apple', 'banana', 'orange'], 'param')
    validate_options(1, [1, 2, 3], 'param')
    
    # Test None with allow_none=True
    validate_options(None, ['apple', 'banana', 'orange'], 'param', allow_none=True)
    
    # Test invalid options
    with pytest.raises(ValidationError):
        validate_options('grape', ['apple', 'banana', 'orange'], 'param')
    
    with pytest.raises(ValidationError):
        validate_options(4, [1, 2, 3], 'param')
    
    # Test None with allow_none=False
    with pytest.raises(ValidationError):
        validate_options(None, ['apple', 'banana', 'orange'], 'param', allow_none=False)


def test_validate_raw():
    # Create a mock Raw object
    data = np.random.randn(2, 1000)
    info = mne.create_info(['ch1', 'ch2'], 100, 'eeg')
    raw = mne.io.RawArray(data, info)
    
    # Test valid Raw object
    validate_raw(raw, require_preload=False)
    
    # Test preloaded Raw object
    raw.load_data()
    validate_raw(raw, require_preload=True)
    
    # Test invalid type
    with pytest.raises(ValidationError):
        validate_raw('not a raw object', require_preload=False)
    
    # Create a non-preloaded Raw object
    raw = mne.io.RawArray(data, info, preload=False)
    
    # Test non-preloaded Raw object with require_preload=True
    with pytest.raises(ValidationError):
        validate_raw(raw, require_preload=True)


def test_validate_array():
    # Test valid arrays
    arr1d = np.array([1, 2, 3])
    arr2d = np.array([[1, 2], [3, 4]])
    
    validate_array(arr1d, param_name='arr1d')
    validate_array(arr2d, param_name='arr2d')
    
    # Test ndim
    validate_array(arr1d, ndim=1, param_name='arr1d')
    validate_array(arr2d, ndim=2, param_name='arr2d')
    
    # Test shape
    validate_array(arr1d, shape=(3,), param_name='arr1d')
    validate_array(arr2d, shape=(2, 2), param_name='arr2d')
    
    # Test dtype
    validate_array(arr1d, dtype=np.int64, param_name='arr1d')
    
    # Test invalid type
    with pytest.raises(ValidationError):
        validate_array('not an array', param_name='not_arr')
    
    # Test invalid ndim
    with pytest.raises(ValidationError):
        validate_array(arr1d, ndim=2, param_name='arr1d')
    
    # Test invalid shape
    with pytest.raises(ValidationError):
        validate_array(arr1d, shape=(4,), param_name='arr1d')
    
    # Test invalid dtype
    float_arr = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValidationError):
        validate_array(float_arr, dtype=np.int64, param_name='float_arr')


def test_validate_function_params():
    # Define a test function
    def test_func(a, b=1, c=2):
        return a + b + c
    
    # Test valid parameters
    params = validate_function_params(test_func, a=10, b=20)
    assert params == {'a': 10, 'b': 20}
    
    # Test invalid parameters
    with pytest.raises(ValidationError):
        validate_function_params(test_func, a=10, d=30)


def test_validate_frequency_bands():
    # Test valid frequency bands
    bands = {
        'alpha': (8, 13),
        'beta': (13, 30)
    }
    validate_frequency_bands(bands)
    
    # Test invalid type
    with pytest.raises(ValidationError):
        validate_frequency_bands('not a dict')
    
    # Test invalid band value type
    with pytest.raises(ValidationError):
        validate_frequency_bands({'alpha': 'not a tuple'})
    
    # Test invalid band value length
    with pytest.raises(ValidationError):
        validate_frequency_bands({'alpha': (8, 13, 20)})
    
    # Test negative frequency
    with pytest.raises(ValidationError):
        validate_frequency_bands({'alpha': (-1, 13)})
    
    # Test fmin >= fmax
    with pytest.raises(ValidationError):
        validate_frequency_bands({'alpha': (13, 8)})
    
    with pytest.raises(ValidationError):
        validate_frequency_bands({'alpha': (13, 13)})


def test_input_validator():
    # Define a test function with type annotations
    @input_validator
    def add_numbers(a: int, b: int) -> int:
        return a + b
    
    # Test valid inputs
    assert add_numbers(1, 2) == 3
    
    # Test invalid inputs
    with pytest.raises(ValidationError):
        add_numbers('1', 2)
    
    with pytest.raises(ValidationError):
        add_numbers(1, '2')
    
    # Define a function with Optional type
    @input_validator
    def add_with_optional(a: int, b: Optional[int] = None) -> int:
        if b is None:
            return a
        return a + b
    
    # Test with None for Optional parameter
    assert add_with_optional(1) == 1
    assert add_with_optional(1, None) == 1
    assert add_with_optional(1, 2) == 3
    
    # Test invalid input for Optional parameter
    with pytest.raises(ValidationError):
        add_with_optional(1, '2')
    
    # Define a function with Union type
    @input_validator
    def add_with_union(a: Union[int, float], b: Union[int, float]) -> Union[int, float]:
        return a + b
    
    # Test valid inputs for Union type
    assert add_with_union(1, 2) == 3
    assert add_with_union(1.0, 2) == 3.0
    assert add_with_union(1, 2.0) == 3.0
    assert add_with_union(1.0, 2.0) == 3.0
    
    # Test invalid inputs for Union type
    with pytest.raises(ValidationError):
        add_with_union('1', 2)
    
    with pytest.raises(ValidationError):
        add_with_union(1, '2')
