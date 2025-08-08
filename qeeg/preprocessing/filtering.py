"""
Filtering module for EEG signal processing.

This module provides functions for filtering EEG signals, including bandpass,
notch, and other filtering operations.
"""

import numpy as np
from typing import Tuple, Optional, Union
import mne


def bandpass_filter(
    raw: mne.io.Raw,
    l_freq: float = 1.0,
    h_freq: float = 40.0,
    picks: Optional[Union[str, list]] = "eeg",
    method: str = "fir",
    **kwargs
) -> mne.io.Raw:
    """
    Apply a bandpass filter to the EEG data.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data to filter.
    l_freq : float, optional
        The lower frequency bound in Hz, by default 1.0
    h_freq : float, optional
        The upper frequency bound in Hz, by default 40.0
    picks : str or list, optional
        Channels to include, by default "eeg" (all EEG channels)
    method : str, optional
        The filtering method, by default "fir"
    **kwargs
        Additional keyword arguments to pass to mne.io.Raw.filter()

    Returns
    -------
    mne.io.Raw
        The filtered EEG data.

    Examples
    --------
    >>> import mne
    >>> from qeeg.preprocessing import filtering
    >>> raw = mne.io.read_raw_edf("sample.edf", preload=True)
    >>> filtered_raw = filtering.bandpass_filter(raw, l_freq=1.0, h_freq=40.0)
    """
    # Create a copy of the raw data to avoid modifying the original
    raw_filtered = raw.copy()
    
    # Apply the bandpass filter
    raw_filtered.filter(
        l_freq=l_freq,
        h_freq=h_freq,
        picks=picks,
        method=method,
        **kwargs
    )
    
    return raw_filtered


def notch_filter(
    raw: mne.io.Raw,
    freqs: Union[float, np.ndarray] = 50.0,
    picks: Optional[Union[str, list]] = "eeg",
    method: str = "fir",
    **kwargs
) -> mne.io.Raw:
    """
    Apply a notch filter to remove power line noise from the EEG data.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data to filter.
    freqs : float or array-like, optional
        The frequencies to filter out, by default 50.0 (European power line frequency)
    picks : str or list, optional
        Channels to include, by default "eeg" (all EEG channels)
    method : str, optional
        The filtering method, by default "fir"
    **kwargs
        Additional keyword arguments to pass to mne.io.Raw.notch_filter()

    Returns
    -------
    mne.io.Raw
        The filtered EEG data.

    Examples
    --------
    >>> import mne
    >>> from qeeg.preprocessing import filtering
    >>> raw = mne.io.read_raw_edf("sample.edf", preload=True)
    >>> filtered_raw = filtering.notch_filter(raw, freqs=60.0)  # US power line frequency
    """
    # Create a copy of the raw data to avoid modifying the original
    raw_filtered = raw.copy()
    
    # Apply the notch filter
    raw_filtered.notch_filter(
        freqs=freqs,
        picks=picks,
        method=method,
        **kwargs
    )
    
    return raw_filtered


def filter_data(
    raw: mne.io.Raw,
    l_freq: Optional[float] = 1.0,
    h_freq: Optional[float] = 40.0,
    notch_freq: Optional[Union[float, np.ndarray]] = None,
    picks: Optional[Union[str, list]] = "eeg",
    **kwargs
) -> mne.io.Raw:
    """
    Apply both bandpass and notch filters to the EEG data.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data to filter.
    l_freq : float or None, optional
        The lower frequency bound in Hz, by default 1.0
    h_freq : float or None, optional
        The upper frequency bound in Hz, by default 40.0
    notch_freq : float, array-like, or None, optional
        The frequencies to filter out, by default None
    picks : str or list, optional
        Channels to include, by default "eeg" (all EEG channels)
    **kwargs
        Additional keyword arguments to pass to the filter functions

    Returns
    -------
    mne.io.Raw
        The filtered EEG data.

    Examples
    --------
    >>> import mne
    >>> from qeeg.preprocessing import filtering
    >>> raw = mne.io.read_raw_edf("sample.edf", preload=True)
    >>> filtered_raw = filtering.filter_data(raw, l_freq=1.0, h_freq=40.0, notch_freq=50.0)
    """
    # Create a copy of the raw data to avoid modifying the original
    raw_filtered = raw.copy()
    
    # Apply bandpass filter if frequencies are specified
    if l_freq is not None or h_freq is not None:
        raw_filtered = bandpass_filter(
            raw_filtered,
            l_freq=l_freq,
            h_freq=h_freq,
            picks=picks,
            **kwargs
        )
    
    # Apply notch filter if frequency is specified
    if notch_freq is not None:
        raw_filtered = notch_filter(
            raw_filtered,
            freqs=notch_freq,
            picks=picks,
            **kwargs
        )
    
    return raw_filtered
