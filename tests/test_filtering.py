"""
Tests for the filtering module.
"""

import numpy as np
import pytest
import mne
from qeeg.preprocessing import filtering


def create_test_raw():
    """Create a test MNE Raw object with simulated EEG data."""
    # Create simulated data
    n_channels = 5
    n_samples = 1000
    sfreq = 100  # Hz
    data = np.random.randn(n_channels, n_samples)
    
    # Add a 10 Hz sine wave to the first channel
    t = np.arange(n_samples) / sfreq
    data[0, :] += 2 * np.sin(2 * np.pi * 10 * t)
    
    # Add a 50 Hz sine wave to the second channel (simulating line noise)
    data[1, :] += np.sin(2 * np.pi * 50 * t)
    
    # Create MNE Raw object
    ch_names = ['Fp1', 'Fp2', 'F3', 'F4', 'Cz']
    ch_types = ['eeg'] * n_channels
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(data, info)
    
    return raw


def test_bandpass_filter():
    """Test the bandpass_filter function."""
    # Create test data
    raw = create_test_raw()
    
    # Apply bandpass filter
    raw_filtered = filtering.bandpass_filter(raw, l_freq=1.0, h_freq=40.0)
    
    # Check that the output is a Raw object
    assert isinstance(raw_filtered, mne.io.Raw)
    
    # Check that the data has been modified
    assert not np.array_equal(raw.get_data(), raw_filtered.get_data())
    
    # Check that the 50 Hz component has been attenuated
    # Get the power spectrum
    psds, freqs = mne.time_frequency.psd_welch(
        raw_filtered, fmin=1, fmax=60, n_fft=256
    )
    
    # Find the index of the frequency closest to 50 Hz
    idx_50hz = np.argmin(np.abs(freqs - 50))
    
    # Check that the power at 50 Hz is lower in the filtered data
    psds_raw, _ = mne.time_frequency.psd_welch(
        raw, fmin=1, fmax=60, n_fft=256
    )
    assert psds[1, idx_50hz] < psds_raw[1, idx_50hz]


def test_notch_filter():
    """Test the notch_filter function."""
    # Create test data
    raw = create_test_raw()
    
    # Apply notch filter
    raw_filtered = filtering.notch_filter(raw, freqs=50.0)
    
    # Check that the output is a Raw object
    assert isinstance(raw_filtered, mne.io.Raw)
    
    # Check that the data has been modified
    assert not np.array_equal(raw.get_data(), raw_filtered.get_data())
    
    # Check that the 50 Hz component has been attenuated
    # Get the power spectrum
    psds, freqs = mne.time_frequency.psd_welch(
        raw_filtered, fmin=1, fmax=60, n_fft=256
    )
    
    # Find the index of the frequency closest to 50 Hz
    idx_50hz = np.argmin(np.abs(freqs - 50))
    
    # Check that the power at 50 Hz is lower in the filtered data
    psds_raw, _ = mne.time_frequency.psd_welch(
        raw, fmin=1, fmax=60, n_fft=256
    )
    assert psds[1, idx_50hz] < psds_raw[1, idx_50hz]


def test_filter_data():
    """Test the filter_data function."""
    # Create test data
    raw = create_test_raw()
    
    # Apply filter_data
    raw_filtered = filtering.filter_data(
        raw, l_freq=1.0, h_freq=40.0, notch_freq=50.0
    )
    
    # Check that the output is a Raw object
    assert isinstance(raw_filtered, mne.io.Raw)
    
    # Check that the data has been modified
    assert not np.array_equal(raw.get_data(), raw_filtered.get_data())
    
    # Check that the 50 Hz component has been attenuated
    # Get the power spectrum
    psds, freqs = mne.time_frequency.psd_welch(
        raw_filtered, fmin=1, fmax=60, n_fft=256
    )
    
    # Find the index of the frequency closest to 50 Hz
    idx_50hz = np.argmin(np.abs(freqs - 50))
    
    # Check that the power at 50 Hz is lower in the filtered data
    psds_raw, _ = mne.time_frequency.psd_welch(
        raw, fmin=1, fmax=60, n_fft=256
    )
    assert psds[1, idx_50hz] < psds_raw[1, idx_50hz]
