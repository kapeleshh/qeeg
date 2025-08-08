"""
Feature extraction module for EEG data.

This module provides functions for extracting features from EEG data for machine learning.
"""

import numpy as np
from typing import Dict, List, Tuple, Union, Optional
import mne
from scipy import stats


def extract_band_power_features(
    raw: mne.io.Raw,
    bands: Optional[Dict[str, Tuple[float, float]]] = None,
    picks: str = "eeg",
    normalize: bool = True
) -> Dict[str, np.ndarray]:
    """
    Extract band power features from EEG data.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data.
    bands : dict, optional
        Dictionary mapping band names to frequency ranges.
        If None, default bands will be used.
    picks : str or list, optional
        Channels to include, by default "eeg" (all EEG channels)
    normalize : bool, optional
        Whether to normalize the band powers, by default True

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary mapping band names to band powers for each channel.

    Examples
    --------
    >>> import mne
    >>> from qeeg.ml import features
    >>> raw = mne.io.read_raw_edf("sample.edf", preload=True)
    >>> band_powers = features.extract_band_power_features(raw)
    >>> print(band_powers.keys())
    """
    from qeeg.analysis.spectral import compute_band_powers, FREQUENCY_BANDS
    
    # Use default bands if not provided
    if bands is None:
        bands = FREQUENCY_BANDS
    
    # Compute band powers
    band_powers = compute_band_powers(raw, frequency_bands=bands, picks=picks, normalize=normalize)
    
    return band_powers


def extract_statistical_features(
    raw: mne.io.Raw,
    picks: str = "eeg"
) -> Dict[str, np.ndarray]:
    """
    Extract statistical features from EEG data.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data.
    picks : str or list, optional
        Channels to include, by default "eeg" (all EEG channels)

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary mapping feature names to feature values for each channel.

    Examples
    --------
    >>> import mne
    >>> from qeeg.ml import features
    >>> raw = mne.io.read_raw_edf("sample.edf", preload=True)
    >>> stat_features = features.extract_statistical_features(raw)
    >>> print(stat_features.keys())
    """
    # Get the data
    data = raw.get_data(picks=picks)
    
    # Initialize the result dictionary
    result = {}
    
    # Calculate statistical features
    result['mean'] = np.mean(data, axis=1)
    result['std'] = np.std(data, axis=1)
    result['var'] = np.var(data, axis=1)
    result['min'] = np.min(data, axis=1)
    result['max'] = np.max(data, axis=1)
    result['range'] = result['max'] - result['min']
    result['skewness'] = stats.skew(data, axis=1)
    result['kurtosis'] = stats.kurtosis(data, axis=1)
    
    return result


def extract_connectivity_features(
    raw: mne.io.Raw,
    method: str = "coh",
    fmin: float = 0.0,
    fmax: float = 50.0,
    picks: str = "eeg"
) -> np.ndarray:
    """
    Extract connectivity features from EEG data.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data.
    method : str, optional
        Connectivity method, by default "coh" (coherence)
    fmin : float, optional
        Minimum frequency, by default 0.0
    fmax : float, optional
        Maximum frequency, by default 50.0
    picks : str or list, optional
        Channels to include, by default "eeg" (all EEG channels)

    Returns
    -------
    np.ndarray
        Connectivity matrix.

    Examples
    --------
    >>> import mne
    >>> from qeeg.ml import features
    >>> raw = mne.io.read_raw_edf("sample.edf", preload=True)
    >>> conn = features.extract_connectivity_features(raw)
    >>> print(conn.shape)
    """
    # Get the data
    data = raw.get_data(picks=picks)
    
    # Get the sampling frequency
    sfreq = raw.info['sfreq']
    
    # Calculate the connectivity
    from mne.connectivity import spectral_connectivity
    
    # Reshape the data for spectral_connectivity
    data = data.reshape(1, data.shape[0], data.shape[1])
    
    # Calculate the connectivity
    conn, freqs, times, n_epochs, n_tapers = spectral_connectivity(
        data,
        method=method,
        mode='multitaper',
        sfreq=sfreq,
        fmin=fmin,
        fmax=fmax,
        faverage=True,
        mt_adaptive=True,
        n_jobs=1
    )
    
    # Reshape the connectivity matrix
    n_channels = data.shape[1]
    conn = conn.reshape(n_channels, n_channels)
    
    return conn


def extract_features(
    raw: mne.io.Raw,
    feature_types: List[str] = ["band_power", "statistical", "connectivity"],
    **kwargs
) -> Dict[str, Union[Dict[str, np.ndarray], np.ndarray]]:
    """
    Extract multiple types of features from EEG data.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data.
    feature_types : List[str], optional
        List of feature types to extract, by default ["band_power", "statistical", "connectivity"]
    **kwargs
        Additional keyword arguments to pass to the feature extraction functions.

    Returns
    -------
    Dict[str, Union[Dict[str, np.ndarray], np.ndarray]]
        Dictionary mapping feature types to feature values.

    Examples
    --------
    >>> import mne
    >>> from qeeg.ml import features
    >>> raw = mne.io.read_raw_edf("sample.edf", preload=True)
    >>> all_features = features.extract_features(raw)
    >>> print(all_features.keys())
    """
    # Initialize the result dictionary
    result = {}
    
    # Extract band power features
    if "band_power" in feature_types:
        result["band_power"] = extract_band_power_features(raw, **kwargs)
    
    # Extract statistical features
    if "statistical" in feature_types:
        result["statistical"] = extract_statistical_features(raw, **kwargs)
    
    # Extract connectivity features
    if "connectivity" in feature_types:
        result["connectivity"] = extract_connectivity_features(raw, **kwargs)
    
    return result


def create_feature_vector(
    features: Dict[str, Union[Dict[str, np.ndarray], np.ndarray]],
    flatten: bool = True
) -> np.ndarray:
    """
    Create a feature vector from extracted features.

    Parameters
    ----------
    features : Dict[str, Union[Dict[str, np.ndarray], np.ndarray]]
        Dictionary mapping feature types to feature values.
    flatten : bool, optional
        Whether to flatten the feature vector, by default True

    Returns
    -------
    np.ndarray
        Feature vector.

    Examples
    --------
    >>> import mne
    >>> from qeeg.ml import features
    >>> raw = mne.io.read_raw_edf("sample.edf", preload=True)
    >>> all_features = features.extract_features(raw)
    >>> feature_vector = features.create_feature_vector(all_features)
    >>> print(feature_vector.shape)
    """
    # Initialize the feature vector
    feature_vector = []
    
    # Process each feature type
    for feature_type, feature_values in features.items():
        if isinstance(feature_values, dict):
            # Process dictionary of features
            for feature_name, values in feature_values.items():
                if flatten:
                    feature_vector.append(values.flatten())
                else:
                    feature_vector.append(values)
        else:
            # Process array of features
            if flatten:
                feature_vector.append(feature_values.flatten())
            else:
                feature_vector.append(feature_values)
    
    # Concatenate the feature vector
    if flatten:
        feature_vector = np.concatenate(feature_vector)
    
    return feature_vector
