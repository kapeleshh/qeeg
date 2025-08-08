"""
Feature extraction module for EEG data.

This module provides functions for extracting features from EEG data
for use in machine learning models.
"""

import numpy as np
from typing import Tuple, Optional, Union, List, Dict
import mne
from scipy.stats import skew, kurtosis
import pywt


def extract_band_power_features(
    raw: mne.io.Raw,
    frequency_bands: Optional[Dict[str, Tuple[float, float]]] = None,
    relative: bool = True,
    **kwargs
) -> Dict[str, np.ndarray]:
    """
    Extract band power features from EEG data.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data.
    frequency_bands : Dict[str, Tuple[float, float]] or None, optional
        Dictionary mapping band names to (fmin, fmax) tuples, by default None
        If None, the standard frequency bands are used.
    relative : bool, optional
        Whether to compute relative power (power in band / total power), by default True
    **kwargs
        Additional keyword arguments to pass to compute_band_powers()

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary mapping feature names to feature values.

    Examples
    --------
    >>> import mne
    >>> from epilepsy_eeg.ml import features
    >>> raw = mne.io.read_raw_edf("sample.edf", preload=True)
    >>> features_dict = features.extract_band_power_features(raw)
    """
    # Use standard frequency bands if none are provided
    if frequency_bands is None:
        from epilepsy_eeg.analysis.spectral import FREQUENCY_BANDS
        frequency_bands = FREQUENCY_BANDS
    
    # Compute the power in each frequency band
    from epilepsy_eeg.analysis.spectral import compute_band_powers
    band_powers = compute_band_powers(raw, frequency_bands=frequency_bands, relative=relative, **kwargs)
    
    # Initialize the result dictionary
    features_dict = {}
    
    # Get the channel names
    ch_names = [raw.ch_names[idx] for idx in mne.pick_types(raw.info, meg=False, eeg=True)]
    
    # Extract features for each band and channel
    for band_name, powers in band_powers.items():
        for i, ch_name in enumerate(ch_names):
            feature_name = f"{band_name}_{ch_name}"
            features_dict[feature_name] = powers[i]
    
    # Compute band power ratios
    from epilepsy_eeg.analysis.spectral import compute_band_power_ratios
    power_ratios = compute_band_power_ratios(raw, frequency_bands=frequency_bands, **kwargs)
    
    # Extract ratio features for each channel
    for ratio_name, ratios in power_ratios.items():
        for i, ch_name in enumerate(ch_names):
            feature_name = f"{ratio_name}_{ch_name}"
            features_dict[feature_name] = ratios[i]
    
    return features_dict


def extract_asymmetry_features(
    raw: mne.io.Raw,
    frequency_bands: Optional[Dict[str, Tuple[float, float]]] = None,
    electrode_pairs: Optional[List[Tuple[str, str]]] = None,
    **kwargs
) -> Dict[str, float]:
    """
    Extract asymmetry features from EEG data.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data.
    frequency_bands : Dict[str, Tuple[float, float]] or None, optional
        Dictionary mapping band names to (fmin, fmax) tuples, by default None
        If None, the standard frequency bands are used.
    electrode_pairs : List[Tuple[str, str]] or None, optional
        List of electrode pairs (left, right), by default None
        If None, standard homologous pairs are used.
    **kwargs
        Additional keyword arguments to pass to compute_band_asymmetry()

    Returns
    -------
    Dict[str, float]
        Dictionary mapping feature names to feature values.

    Examples
    --------
    >>> import mne
    >>> from epilepsy_eeg.ml import features
    >>> raw = mne.io.read_raw_edf("sample.edf", preload=True)
    >>> features_dict = features.extract_asymmetry_features(raw)
    """
    # Compute asymmetry indices
    from epilepsy_eeg.analysis.asymmetry import compute_band_asymmetry
    band_asymmetry = compute_band_asymmetry(
        raw,
        frequency_bands=frequency_bands,
        electrode_pairs=electrode_pairs,
        **kwargs
    )
    
    # Initialize the result dictionary
    features_dict = {}
    
    # Extract features for each band and pair
    for band_name, asymmetry_indices in band_asymmetry.items():
        for pair_name, asymmetry_index in asymmetry_indices.items():
            feature_name = f"asymmetry_{band_name}_{pair_name}"
            features_dict[feature_name] = asymmetry_index
    
    return features_dict


def extract_wavelet_features(
    raw: mne.io.Raw,
    wavelet: str = 'db4',
    level: int = 5,
    picks: Optional[Union[str, list]] = "eeg",
    **kwargs
) -> Dict[str, np.ndarray]:
    """
    Extract wavelet features from EEG data.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data.
    wavelet : str, optional
        Wavelet to use, by default 'db4'
    level : int, optional
        Decomposition level, by default 5
    picks : str or list, optional
        Channels to include, by default "eeg" (all EEG channels)
    **kwargs
        Additional keyword arguments

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary mapping feature names to feature values.

    Examples
    --------
    >>> import mne
    >>> from epilepsy_eeg.ml import features
    >>> raw = mne.io.read_raw_edf("sample.edf", preload=True)
    >>> features_dict = features.extract_wavelet_features(raw)
    """
    # Get the data
    data = raw.get_data(picks=picks)
    
    # Get the channel names
    ch_names = [raw.ch_names[idx] for idx in mne.pick_types(raw.info, meg=False, eeg=True, selection=picks)]
    
    # Initialize the result dictionary
    features_dict = {}
    
    # Process each channel
    for i, ch_name in enumerate(ch_names):
        # Get the channel data
        ch_data = data[i]
        
        # Perform wavelet decomposition
        coeffs = pywt.wavedec(ch_data, wavelet, level=level)
        
        # Extract features from each level
        for j, coeff in enumerate(coeffs):
            # Compute statistics
            mean = np.mean(coeff)
            std = np.std(coeff)
            skewness = skew(coeff)
            kurt = kurtosis(coeff)
            energy = np.sum(coeff**2)
            
            # Store the features
            level_name = "A" if j == 0 else f"D{level - j + 1}"
            features_dict[f"wavelet_{level_name}_{ch_name}_mean"] = mean
            features_dict[f"wavelet_{level_name}_{ch_name}_std"] = std
            features_dict[f"wavelet_{level_name}_{ch_name}_skew"] = skewness
            features_dict[f"wavelet_{level_name}_{ch_name}_kurtosis"] = kurt
            features_dict[f"wavelet_{level_name}_{ch_name}_energy"] = energy
    
    return features_dict


def extract_connectivity_features(
    raw: mne.io.Raw,
    method: str = 'coh',
    fmin: float = 0.5,
    fmax: float = 40.0,
    ch_pairs: Optional[List[Tuple[str, str]]] = None,
    **kwargs
) -> Dict[str, float]:
    """
    Extract connectivity features from EEG data.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data.
    method : str, optional
        Connectivity method, by default 'coh'
        Options: 'coh', 'plv', 'pli', 'wpli', 'ppc'
    fmin : float, optional
        Minimum frequency, by default 0.5
    fmax : float, optional
        Maximum frequency, by default 40.0
    ch_pairs : List[Tuple[str, str]] or None, optional
        List of channel pairs to compute connectivity for, by default None
        If None, uses standard homologous pairs.
    **kwargs
        Additional keyword arguments to pass to mne.connectivity.spectral_connectivity()

    Returns
    -------
    Dict[str, float]
        Dictionary mapping feature names to feature values.

    Examples
    --------
    >>> import mne
    >>> from epilepsy_eeg.ml import features
    >>> raw = mne.io.read_raw_edf("sample.edf", preload=True)
    >>> features_dict = features.extract_connectivity_features(raw)
    """
    # Define standard homologous pairs if none are provided
    if ch_pairs is None:
        ch_pairs = [
            ('F3', 'F4'),
            ('F7', 'F8'),
            ('C3', 'C4'),
            ('T3', 'T4'),
            ('T5', 'T6'),
            ('P3', 'P4'),
            ('O1', 'O2')
        ]
    
    # Filter the pairs to only include channels that exist in the data
    ch_pairs = [
        (ch1, ch2) for ch1, ch2 in ch_pairs
        if ch1 in raw.ch_names and ch2 in raw.ch_names
    ]
    
    if not ch_pairs:
        raise ValueError("No valid channel pairs found in the data.")
    
    # Create epochs from the raw data
    events = mne.make_fixed_length_events(raw, duration=2.0)
    epochs = mne.Epochs(raw, events, tmin=0, tmax=2.0, baseline=None, preload=True)
    
    # Compute the connectivity
    from mne.connectivity import spectral_connectivity
    
    # Convert channel pairs to indices
    indices = []
    for ch1, ch2 in ch_pairs:
        idx1 = raw.ch_names.index(ch1)
        idx2 = raw.ch_names.index(ch2)
        indices.append((idx1, idx2))
    
    # Compute the connectivity
    con, freqs, times, _, _ = spectral_connectivity(
        epochs,
        method=method,
        mode='multitaper',
        sfreq=raw.info['sfreq'],
        fmin=fmin,
        fmax=fmax,
        faverage=True,
        indices=indices,
        **kwargs
    )
    
    # Initialize the result dictionary
    features_dict = {}
    
    # Extract features for each pair
    for i, (ch1, ch2) in enumerate(ch_pairs):
        feature_name = f"connectivity_{method}_{ch1}_{ch2}"
        features_dict[feature_name] = con[i, 0, 0]
    
    return features_dict


def extract_epileptiform_features(
    raw: mne.io.Raw,
    **kwargs
) -> Dict[str, float]:
    """
    Extract epileptiform activity features from EEG data.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data.
    **kwargs
        Additional keyword arguments to pass to analyze_epileptiform_activity()

    Returns
    -------
    Dict[str, float]
        Dictionary mapping feature names to feature values.

    Examples
    --------
    >>> import mne
    >>> from epilepsy_eeg.ml import features
    >>> raw = mne.io.read_raw_edf("sample.edf", preload=True)
    >>> features_dict = features.extract_epileptiform_features(raw)
    """
    # Analyze epileptiform activity
    from epilepsy_eeg.analysis.epileptiform import analyze_epileptiform_activity
    summary = analyze_epileptiform_activity(raw, **kwargs)
    
    # Initialize the result dictionary
    features_dict = {}
    
    # Extract features for each activity type
    for activity_type, activity_summary in summary.items():
        # Extract count and rate
        features_dict[f"{activity_type}_count"] = activity_summary['count']
        features_dict[f"{activity_type}_rate"] = activity_summary['rate']
        
        # Extract mean duration if available
        if 'mean_duration' in activity_summary:
            features_dict[f"{activity_type}_mean_duration"] = activity_summary['mean_duration']
        
        # Extract mean amplitude if available
        if 'mean_amplitude' in activity_summary:
            features_dict[f"{activity_type}_mean_amplitude"] = activity_summary['mean_amplitude']
        
        # Extract mean power if available
        if 'mean_power' in activity_summary:
            features_dict[f"{activity_type}_mean_power"] = activity_summary['mean_power']
    
    return features_dict


def extract_all_features(
    raw: mne.io.Raw,
    include_wavelet: bool = True,
    include_connectivity: bool = True,
    **kwargs
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Extract all available features from EEG data.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data.
    include_wavelet : bool, optional
        Whether to include wavelet features, by default True
    include_connectivity : bool, optional
        Whether to include connectivity features, by default True
    **kwargs
        Additional keyword arguments to pass to feature extraction functions

    Returns
    -------
    Dict[str, Union[float, np.ndarray]]
        Dictionary mapping feature names to feature values.

    Examples
    --------
    >>> import mne
    >>> from epilepsy_eeg.ml import features
    >>> raw = mne.io.read_raw_edf("sample.edf", preload=True)
    >>> features_dict = features.extract_all_features(raw)
    """
    # Initialize the result dictionary
    features_dict = {}
    
    # Extract band power features
    band_power_features = extract_band_power_features(raw, **kwargs)
    features_dict.update(band_power_features)
    
    # Extract asymmetry features
    asymmetry_features = extract_asymmetry_features(raw, **kwargs)
    features_dict.update(asymmetry_features)
    
    # Extract epileptiform features
    epileptiform_features = extract_epileptiform_features(raw, **kwargs)
    features_dict.update(epileptiform_features)
    
    # Extract wavelet features if requested
    if include_wavelet:
        wavelet_features = extract_wavelet_features(raw, **kwargs)
        features_dict.update(wavelet_features)
    
    # Extract connectivity features if requested
    if include_connectivity:
        connectivity_features = extract_connectivity_features(raw, **kwargs)
        features_dict.update(connectivity_features)
    
    return features_dict


def create_feature_matrix(
    features_dict: Dict[str, Union[float, np.ndarray]],
    feature_names: Optional[List[str]] = None
) -> Tuple[np.ndarray, List[str]]:
    """
    Create a feature matrix from a dictionary of features.

    Parameters
    ----------
    features_dict : Dict[str, Union[float, np.ndarray]]
        Dictionary mapping feature names to feature values.
    feature_names : List[str] or None, optional
        List of feature names to include, by default None (all features)

    Returns
    -------
    Tuple[np.ndarray, List[str]]
        Feature matrix and list of feature names.

    Examples
    --------
    >>> import mne
    >>> from epilepsy_eeg.ml import features
    >>> raw = mne.io.read_raw_edf("sample.edf", preload=True)
    >>> features_dict = features.extract_all_features(raw)
    >>> X, feature_names = features.create_feature_matrix(features_dict)
    """
    # Use all feature names if none are provided
    if feature_names is None:
        feature_names = list(features_dict.keys())
    
    # Create the feature matrix
    X = np.array([features_dict[name] for name in feature_names])
    
    return X, feature_names
