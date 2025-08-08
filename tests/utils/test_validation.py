"""
Tests for the validation module.
"""

import pytest
import numpy as np
import mne
from typing import Optional, Union, List

from qeeg.utils.validation import (
    validate_type,
    validate_range,
    validate_options,
    validate_raw,
    validate_epochs,
    validate_function_params,
    validate_montage,
    validate_channel_names
)
from qeeg.utils.exceptions import ValidationError, DataQualityError


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


def create_test_epochs(n_channels=5, n_samples=1000, sfreq=100, n_epochs=10):
    """Create a test MNE Epochs object with simulated EEG data."""
    # Create simulated data
    data = np.random.randn(n_epochs, n_channels, n_samples)
    
    # Create events
    events = np.column_stack([
        np.arange(n_epochs) * n_samples,
        np.zeros(n_epochs, dtype=int),
        np.ones(n_epochs, dtype=int)
    ])
    
    # Create MNE Epochs object
    ch_names = [f'CH{i}' for i in range(n_channels)]
    ch_types = ['eeg'] * n_channels
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    epochs = mne.EpochsArray(data, info, events)
    
    return epochs


def test_validate_type():
    """Test the validate_type function."""
    # Test with correct type
    validate_type(5, int)
    validate_type(5.0, float)
    validate_type("test", str)
    validate_type([1, 2, 3], list)
    validate_type((1, 2, 3), tuple)
    validate_type({"a": 1}, dict)
    
    # Test with multiple types
    validate_type(5, (int, float))
    validate_type(5.0, (int, float))
    
    # Test with incorrect type
    with pytest.raises(ValidationError):
        validate_type(5, str)
    
    with pytest.raises(ValidationError):
        validate_type("test", int)
    
    # Test with parameter name
    with pytest.raises(ValidationError) as excinfo:
        validate_type(5, str, "value")
    
    assert "value" in str(excinfo.value)
    assert "str" in str(excinfo.value)
    assert "int" in str(excinfo.value)


def test_validate_range():
    """Test the validate_range function."""
    # Test with value in range
    validate_range(5, 0, 10)
    validate_range(5.0, 0.0, 10.0)
    
    # Test with value at min
    validate_range(0, 0, 10)
    
    # Test with value at max
    validate_range(10, 0, 10)
    
    # Test with only min
    validate_range(5, 0)
    
    # Test with only max
    validate_range(5, max_value=10)
    
    # Test with value below min
    with pytest.raises(ValidationError):
        validate_range(-1, 0, 10)
    
    # Test with value above max
    with pytest.raises(ValidationError):
        validate_range(11, 0, 10)
    
    # Test with parameter name
    with pytest.raises(ValidationError) as excinfo:
        validate_range(-1, 0, 10, "value")
    
    assert "value" in str(excinfo.value)
    assert "-1" in str(excinfo.value)
    assert ">= 0" in str(excinfo.value)
    
    # Test with non-numeric value
    with pytest.raises(ValidationError):
        validate_range("test", 0, 10)


def test_validate_options():
    """Test the validate_options function."""
    # Test with value in options
    validate_options("a", ["a", "b", "c"])
    validate_options(1, [1, 2, 3])
    
    # Test with value not in options
    with pytest.raises(ValidationError):
        validate_options("d", ["a", "b", "c"])
    
    with pytest.raises(ValidationError):
        validate_options(4, [1, 2, 3])
    
    # Test with parameter name
    with pytest.raises(ValidationError) as excinfo:
        validate_options("d", ["a", "b", "c"], "value")
    
    assert "value" in str(excinfo.value)
    assert "'d'" in str(excinfo.value)
    assert "['a', 'b', 'c']" in str(excinfo.value)


def test_validate_raw():
    """Test the validate_raw function."""
    # Create a test raw object
    raw = create_test_raw()
    
    # Test with valid raw
    results = validate_raw(raw)
    
    # Check that the results are as expected
    assert results["type"] == "mne.io.Raw"
    assert results["preload"] is True
    assert results["n_channels"] == 5
    assert results["sfreq"] == 100
    assert results["duration"] > 0
    assert results["data_quality"] == "good"
    
    # Test with specific checks
    results = validate_raw(raw, checks=["type", "preload"])
    assert "type" in results
    assert "preload" in results
    assert "n_channels" not in results
    
    # Test with invalid type
    with pytest.raises(ValidationError):
        validate_raw("not a raw object")
    
    # Test with not preloaded raw
    raw_copy = raw.copy()
    raw_copy.preload = False
    with pytest.raises(ValidationError):
        validate_raw(raw_copy)
    
    # Test with no channels
    raw_copy = raw.copy()
    raw_copy.pick([])  # Pick no channels
    with pytest.raises(ValidationError):
        validate_raw(raw_copy)
    
    # Test with invalid sampling frequency
    raw_copy = raw.copy()
    raw_copy.info["sfreq"] = 0
    with pytest.raises(ValidationError):
        validate_raw(raw_copy)
    
    # Test with NaN values
    raw_copy = raw.copy()
    data = raw_copy.get_data()
    data[0, 0] = np.nan
    raw_copy._data = data
    with pytest.raises(DataQualityError):
        validate_raw(raw_copy)
    
    # Test with flat signal
    raw_copy = raw.copy()
    data = raw_copy.get_data()
    data[0, :] = 0
    raw_copy._data = data
    with pytest.raises(DataQualityError):
        validate_raw(raw_copy)


def test_validate_epochs():
    """Test the validate_epochs function."""
    # Create a test epochs object
    epochs = create_test_epochs()
    
    # Test with valid epochs
    results = validate_epochs(epochs)
    
    # Check that the results are as expected
    assert results["type"] == "mne.Epochs"
    assert results["preload"] is True
    assert results["n_channels"] == 5
    assert results["sfreq"] == 100
    assert results["n_epochs"] == 10
    assert results["data_quality"] == "good"
    
    # Test with specific checks
    results = validate_epochs(epochs, checks=["type", "preload"])
    assert "type" in results
    assert "preload" in results
    assert "n_channels" not in results
    
    # Test with invalid type
    with pytest.raises(ValidationError):
        validate_epochs("not an epochs object")
    
    # Test with not preloaded epochs
    epochs_copy = epochs.copy()
    epochs_copy.preload = False
    with pytest.raises(ValidationError):
        validate_epochs(epochs_copy)
    
    # Test with no channels
    epochs_copy = epochs.copy()
    epochs_copy.pick([])  # Pick no channels
    with pytest.raises(ValidationError):
        validate_epochs(epochs_copy)
    
    # Test with invalid sampling frequency
    epochs_copy = epochs.copy()
    epochs_copy.info["sfreq"] = 0
    with pytest.raises(ValidationError):
        validate_epochs(epochs_copy)
    
    # Test with no epochs
    epochs_copy = epochs.copy()
    epochs_copy._data = np.empty((0, 5, 1000))
    with pytest.raises(ValidationError):
        validate_epochs(epochs_copy)
    
    # Test with NaN values
    epochs_copy = epochs.copy()
    data = epochs_copy.get_data()
    data[0, 0, 0] = np.nan
    epochs_copy._data = data
    with pytest.raises(DataQualityError):
        validate_epochs(epochs_copy)
    
    # Test with flat signal
    epochs_copy = epochs.copy()
    data = epochs_copy.get_data()
    data[0, :, :] = 0
    epochs_copy._data = data
    with pytest.raises(DataQualityError):
        validate_epochs(epochs_copy)


def test_validate_function_params():
    """Test the validate_function_params function."""
    # Define a function with type annotations
    def test_func(a: int, b: float, c: str = "default", d: Optional[list] = None):
        return a, b, c, d
    
    # Test with valid parameters
    params = {"a": 1, "b": 2.0, "c": "test", "d": [1, 2, 3]}
    validated = validate_function_params(test_func, params)
    assert validated["a"] == 1
    assert validated["b"] == 2.0
    assert validated["c"] == "test"
    assert validated["d"] == [1, 2, 3]
    
    # Test with type conversion
    params = {"a": "1", "b": "2.0"}
    validated = validate_function_params(test_func, params)
    assert validated["a"] == 1
    assert validated["b"] == 2.0
    
    # Test with invalid parameter
    params = {"a": 1, "b": 2.0, "e": 3}
    with pytest.raises(ValidationError):
        validate_function_params(test_func, params)
    
    # Test with invalid type that can't be converted
    params = {"a": "not an int", "b": 2.0}
    with pytest.raises(ValidationError):
        validate_function_params(test_func, params)
    
    # Test with None for Optional parameter
    params = {"a": 1, "b": 2.0, "d": None}
    validated = validate_function_params(test_func, params)
    assert validated["d"] is None
    
    # Define a function with Union type
    def union_func(a: Union[int, str]):
        return a
    
    # Test with Union type
    params = {"a": 1}
    validated = validate_function_params(union_func, params)
    assert validated["a"] == 1
    
    params = {"a": "test"}
    validated = validate_function_params(union_func, params)
    assert validated["a"] == "test"
    
    # Define a function without type annotations
    def no_annotation_func(a, b, c="default"):
        return a, b, c
    
    # Test with function without annotations
    params = {"a": 1, "b": 2, "c": "test"}
    validated = validate_function_params(no_annotation_func, params)
    assert validated["a"] == 1
    assert validated["b"] == 2
    assert validated["c"] == "test"


def test_validate_montage():
    """Test the validate_montage function."""
    # Create a test raw object
    raw = create_test_raw()
    
    # Test with string montage name
    assert validate_montage("standard_1020")
    
    # Test with DigMontage object
    montage = mne.channels.make_standard_montage("standard_1020")
    assert validate_montage(montage)
    
    # Test with invalid montage name
    with pytest.raises(ValidationError):
        validate_montage("invalid_montage_name")
    
    # Test with invalid montage type
    with pytest.raises(ValidationError):
        validate_montage(123)
    
    # Test with raw object
    with pytest.warns(UserWarning):
        assert validate_montage(montage, raw)


def test_validate_channel_names():
    """Test the validate_channel_names function."""
    # Test with standard 10-20 channel names
    ch_names = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2']
    non_standard = validate_channel_names(ch_names, standard='10-20')
    assert len(non_standard) == 0
    
    # Test with non-standard channel names
    ch_names = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2', 'Custom1', 'Custom2']
    non_standard = validate_channel_names(ch_names, standard='10-20')
    assert len(non_standard) == 2
    assert 'Custom1' in non_standard
    assert 'Custom2' in non_standard
    
    # Test with common non-EEG channels
    ch_names = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2', 'EKG', 'ECG', 'EMG', 'EOG']
    non_standard = validate_channel_names(ch_names, standard='10-20')
    assert len(non_standard) == 0  # Common non-EEG channels should be ignored
    
    # Test with different standards
    ch_names = ['Fp1', 'Fpz', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8']
    non_standard = validate_channel_names(ch_names, standard='10-20')
    assert len(non_standard) > 0  # These are 10-10 channels, not all in 10-20
    
    non_standard = validate_channel_names(ch_names, standard='10-10')
    assert len(non_standard) == 0  # All should be valid 10-10 channels
    
    # Test with invalid standard
    with pytest.raises(ValidationError):
        validate_channel_names(ch_names, standard='invalid_standard')
