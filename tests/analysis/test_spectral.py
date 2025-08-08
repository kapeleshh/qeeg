import pytest
import numpy as np
import mne
from qeeg.analysis import spectral


def test_compute_psd():
    """Test the compute_psd function."""
    # Create a simple raw object with random data
    data = np.random.randn(3, 1000)
    info = mne.create_info(['Fz', 'Cz', 'Pz'], 100, 'eeg')
    raw = mne.io.RawArray(data, info)
    
    # Test compute_psd function
    psds, freqs = spectral.compute_psd(raw)
    
    # Check shapes
    assert psds.shape[0] == 3  # 3 channels
    assert len(freqs) > 0
    assert np.all(freqs >= 0)  # Frequencies should be non-negative


def test_compute_band_powers():
    """Test the compute_band_powers function."""
    # Create a simple raw object with random data
    data = np.random.randn(3, 1000)
    info = mne.create_info(['Fz', 'Cz', 'Pz'], 100, 'eeg')
    raw = mne.io.RawArray(data, info)
    
    # Define custom frequency bands
    bands = {
        "Delta": (1, 4),
        "Theta": (4, 8),
        "Alpha": (8, 13),
        "Beta": (13, 30)
    }
    
    # Test compute_band_powers function
    band_powers = spectral.compute_band_powers(raw, frequency_bands=bands)
    
    # Check that all bands are present
    assert set(band_powers.keys()) == set(bands.keys())
    
    # Check shapes
    for band, powers in band_powers.items():
        assert len(powers) == 3  # 3 channels
        assert np.all(powers >= 0)  # Powers should be non-negative


def test_frequency_bands_constant():
    """Test that the FREQUENCY_BANDS constant is properly defined."""
    # Check that FREQUENCY_BANDS is a dictionary
    assert isinstance(spectral.FREQUENCY_BANDS, dict)
    
    # Check that each band is defined as a tuple of (fmin, fmax)
    for band, (fmin, fmax) in spectral.FREQUENCY_BANDS.items():
        assert isinstance(band, str)
        assert isinstance(fmin, (int, float))
        assert isinstance(fmax, (int, float))
        assert fmin < fmax  # Lower bound should be less than upper bound
