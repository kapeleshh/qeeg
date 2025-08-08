"""
Artifacts module for EEG signal processing.

This module provides functions for artifact removal in EEG signals,
including ICA-based artifact removal and automatic rejection of noisy epochs.
"""

import numpy as np
from typing import Tuple, Optional, Union, List, Dict
import mne
from mne.preprocessing import ICA
from mne import Epochs, make_fixed_length_events


def create_epochs(
    raw: mne.io.Raw,
    duration: float = 2.0,
    overlap: float = 0.0,
    **kwargs
) -> mne.Epochs:
    """
    Create fixed-length epochs from continuous EEG data.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data.
    duration : float, optional
        Duration of each epoch in seconds, by default 2.0
    overlap : float, optional
        Overlap between consecutive epochs in seconds, by default 0.0
    **kwargs
        Additional keyword arguments to pass to mne.Epochs

    Returns
    -------
    mne.Epochs
        The epoched EEG data.

    Examples
    --------
    >>> import mne
    >>> from qeeg.preprocessing import artifacts
    >>> raw = mne.io.read_raw_edf("sample.edf", preload=True)
    >>> epochs = artifacts.create_epochs(raw, duration=2.0)
    """
    # Create fixed-length events
    events = make_fixed_length_events(
        raw,
        duration=duration,
        overlap=overlap
    )
    
    # Create epochs
    epochs = Epochs(
        raw,
        events,
        tmin=0,
        tmax=duration,
        baseline=None,
        preload=True,
        **kwargs
    )
    
    return epochs


def detect_bad_channels(
    raw: mne.io.Raw,
    threshold: float = 0.5,
    picks: Optional[Union[str, list]] = "eeg",
    **kwargs
) -> List[str]:
    """
    Detect bad channels in the EEG data based on signal statistics.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data.
    threshold : float, optional
        Threshold for bad channel detection, by default 0.5
    picks : str or list, optional
        Channels to include, by default "eeg" (all EEG channels)
    **kwargs
        Additional keyword arguments

    Returns
    -------
    List[str]
        List of detected bad channel names.

    Examples
    --------
    >>> import mne
    >>> from qeeg.preprocessing import artifacts
    >>> raw = mne.io.read_raw_edf("sample.edf", preload=True)
    >>> bad_channels = artifacts.detect_bad_channels(raw)
    >>> print(f"Detected bad channels: {bad_channels}")
    """
    # Get the data and channel names
    data = raw.get_data(picks=picks)
    ch_names = [raw.ch_names[idx] for idx in mne.pick_types(raw.info, meg=False, eeg=True)]
    
    # Calculate statistics for each channel
    ch_std = np.std(data, axis=1)
    ch_range = np.ptp(data, axis=1)
    
    # Normalize the statistics
    norm_std = (ch_std - np.mean(ch_std)) / np.std(ch_std)
    norm_range = (ch_range - np.mean(ch_range)) / np.std(ch_range)
    
    # Combine the statistics
    combined_score = np.abs(norm_std) + np.abs(norm_range)
    
    # Detect bad channels
    bad_indices = np.where(combined_score > threshold)[0]
    bad_channels = [ch_names[idx] for idx in bad_indices]
    
    return bad_channels


def apply_ica(
    raw: mne.io.Raw,
    n_components: Optional[int] = 15,
    random_state: int = 97,
    method: str = "fastica",
    exclude: Optional[List[int]] = None,
    **kwargs
) -> Tuple[mne.io.Raw, mne.preprocessing.ICA]:
    """
    Apply Independent Component Analysis (ICA) to remove artifacts from EEG data.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data.
    n_components : int or None, optional
        Number of ICA components, by default 15
    random_state : int, optional
        Random state for reproducibility, by default 97
    method : str, optional
        ICA method, by default "fastica"
    exclude : List[int] or None, optional
        List of ICA components to exclude, by default None
    **kwargs
        Additional keyword arguments to pass to mne.preprocessing.ICA

    Returns
    -------
    Tuple[mne.io.Raw, mne.preprocessing.ICA]
        The cleaned EEG data and the fitted ICA object.

    Examples
    --------
    >>> import mne
    >>> from qeeg.preprocessing import artifacts
    >>> raw = mne.io.read_raw_edf("sample.edf", preload=True)
    >>> cleaned_raw, ica = artifacts.apply_ica(raw)
    """
    # Create a copy of the raw data
    raw_copy = raw.copy()
    
    # Initialize ICA
    ica = ICA(
        n_components=n_components,
        random_state=random_state,
        method=method,
        **kwargs
    )
    
    # Fit ICA
    ica.fit(raw_copy)
    
    # Apply ICA with excluded components
    if exclude is not None:
        ica.exclude = exclude
    
    cleaned_raw = raw_copy.copy()
    ica.apply(cleaned_raw)
    
    return cleaned_raw, ica


def auto_detect_artifacts(
    raw: mne.io.Raw,
    n_components: Optional[int] = 15,
    threshold: float = 3.0,
    **kwargs
) -> Tuple[mne.io.Raw, mne.preprocessing.ICA, List[int]]:
    """
    Automatically detect and remove artifacts using ICA.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data.
    n_components : int or None, optional
        Number of ICA components, by default 15
    threshold : float, optional
        Threshold for artifact detection, by default 3.0
    **kwargs
        Additional keyword arguments to pass to apply_ica()

    Returns
    -------
    Tuple[mne.io.Raw, mne.preprocessing.ICA, List[int]]
        The cleaned EEG data, the fitted ICA object, and the list of excluded components.

    Examples
    --------
    >>> import mne
    >>> from qeeg.preprocessing import artifacts
    >>> raw = mne.io.read_raw_edf("sample.edf", preload=True)
    >>> cleaned_raw, ica, excluded = artifacts.auto_detect_artifacts(raw)
    >>> print(f"Excluded components: {excluded}")
    """
    # Create a copy of the raw data
    raw_copy = raw.copy()
    
    # Apply ICA
    cleaned_raw, ica = apply_ica(
        raw_copy,
        n_components=n_components,
        exclude=None,
        **kwargs
    )
    
    # Detect artifacts based on variance
    excluded = []
    for idx, component in enumerate(ica.get_components().T):
        if (component.max() - component.min()) > threshold:
            excluded.append(idx)
    
    # Apply ICA with excluded components
    ica.exclude = excluded
    cleaned_raw = raw_copy.copy()
    ica.apply(cleaned_raw)
    
    return cleaned_raw, ica, excluded


def remove_artifacts_ica(
    raw: mne.io.Raw,
    n_components: Optional[int] = 15,
    threshold: float = 3.0,
    auto_detect: bool = True,
    exclude: Optional[List[int]] = None,
    **kwargs
) -> mne.io.Raw:
    """
    Remove artifacts from EEG data using ICA.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data.
    n_components : int or None, optional
        Number of ICA components, by default 15
    threshold : float, optional
        Threshold for artifact detection, by default 3.0
    auto_detect : bool, optional
        Whether to automatically detect artifacts, by default True
    exclude : List[int] or None, optional
        List of ICA components to exclude, by default None
    **kwargs
        Additional keyword arguments to pass to apply_ica()

    Returns
    -------
    mne.io.Raw
        The cleaned EEG data.

    Examples
    --------
    >>> import mne
    >>> from qeeg.preprocessing import artifacts
    >>> raw = mne.io.read_raw_edf("sample.edf", preload=True)
    >>> cleaned_raw = artifacts.remove_artifacts_ica(raw)
    """
    # Create a copy of the raw data
    raw_copy = raw.copy()
    
    if auto_detect:
        # Automatically detect and remove artifacts
        cleaned_raw, _, _ = auto_detect_artifacts(
            raw_copy,
            n_components=n_components,
            threshold=threshold,
            **kwargs
        )
    else:
        # Apply ICA with manually specified excluded components
        cleaned_raw, _ = apply_ica(
            raw_copy,
            n_components=n_components,
            exclude=exclude,
            **kwargs
        )
    
    return cleaned_raw


def preprocess_continuous(
    raw: mne.io.Raw,
    l_freq: float = 1.0,
    h_freq: float = 40.0,
    reference: str = 'average',
    ica_components: int = 15,
    variance_threshold: float = 3.0,
    **kwargs
) -> mne.io.Raw:
    """
    Comprehensive preprocessing pipeline for continuous EEG data.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data.
    l_freq : float, optional
        Lower frequency bound for filtering, by default 1.0
    h_freq : float, optional
        Upper frequency bound for filtering, by default 40.0
    reference : str, optional
        Reference for EEG data, by default 'average'
    ica_components : int, optional
        Number of ICA components, by default 15
    variance_threshold : float, optional
        Threshold for artifact detection, by default 3.0
    **kwargs
        Additional keyword arguments

    Returns
    -------
    mne.io.Raw
        The preprocessed EEG data.

    Examples
    --------
    >>> import mne
    >>> from qeeg.preprocessing import artifacts
    >>> raw = mne.io.read_raw_edf("sample.edf", preload=True)
    >>> cleaned_raw = artifacts.preprocess_continuous(raw)
    """
    # Create a copy of the raw data
    raw_copy = raw.copy()
    
    # Step 1: Filter the data
    raw_copy.filter(l_freq=l_freq, h_freq=h_freq)
    
    # Step 2: Set reference
    if reference is not None:
        raw_copy.set_eeg_reference(reference, projection=True)
    
    # Step 3: Create fixed-length events and epochs
    events = make_fixed_length_events(raw_copy, duration=2.0)
    epochs = Epochs(raw_copy, events, tmin=0, tmax=2.0, baseline=None, preload=True)
    
    # Step 4: Apply AutoReject (if available)
    try:
        from autoreject import AutoReject
        ar = AutoReject(random_state=97)
        epochs_cleaned, _ = ar.fit_transform(epochs, return_log=True)
    except ImportError:
        # If autoreject is not available, use the original epochs
        epochs_cleaned = epochs
    
    # Step 5: Convert cleaned epochs back to raw
    raw_cleaned = mne.io.RawArray(
        epochs_cleaned.get_data().reshape((-1, raw_copy.info['nchan'])).T,
        raw_copy.info
    )
    
    # Step 6: Apply ICA
    cleaned_raw = remove_artifacts_ica(
        raw_cleaned,
        n_components=ica_components,
        threshold=variance_threshold,
        auto_detect=True
    )
    
    return cleaned_raw
