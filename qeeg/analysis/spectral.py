"""
Spectral analysis module for EEG signal processing.

This module provides functions for spectral analysis of EEG signals,
including power spectral density calculation and frequency band analysis.
"""

import numpy as np
from typing import Tuple, Optional, Union, List, Dict
import mne
from scipy.signal import welch


# Define standard frequency bands
FREQUENCY_BANDS = {
    "Delta": (1, 4),
    "Theta": (4, 7.5),
    "Alpha": (7.5, 14),
    "Beta1": (14, 20),
    "Beta2": (20, 30),
    "Gamma": (30, 40)
}


def compute_psd(
    raw: mne.io.Raw,
    fmin: float = 0.5,
    fmax: float = 40.0,
    method: str = 'welch',
    picks: Optional[Union[str, list]] = "eeg",
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the power spectral density (PSD) of the EEG data.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data.
    fmin : float, optional
        Minimum frequency of interest, by default 0.5
    fmax : float, optional
        Maximum frequency of interest, by default 40.0
    method : str, optional
        Method to compute the PSD, by default 'welch'
    picks : str or list, optional
        Channels to include, by default "eeg" (all EEG channels)
    **kwargs
        Additional keyword arguments to pass to mne.time_frequency.psd_welch()

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        The PSD values and the corresponding frequencies.

    Examples
    --------
    >>> import mne
    >>> from qeeg.analysis import spectral
    >>> raw = mne.io.read_raw_edf("sample.edf", preload=True)
    >>> psds, freqs = spectral.compute_psd(raw)
    """
    # Compute the PSD
    psd_object = raw.compute_psd(
        method=method,
        fmin=fmin,
        fmax=fmax,
        picks=picks,
        **kwargs
    )
    
    # Extract the PSD values and frequencies
    psds, freqs = psd_object.get_data(return_freqs=True)
    
    return psds, freqs


def compute_band_powers(
    raw: mne.io.Raw,
    frequency_bands: Optional[Dict[str, Tuple[float, float]]] = None,
    relative: bool = False,
    **kwargs
) -> Dict[str, np.ndarray]:
    """
    Compute the power in each frequency band for each channel.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data.
    frequency_bands : Dict[str, Tuple[float, float]] or None, optional
        Dictionary mapping band names to (fmin, fmax) tuples, by default None
        If None, the standard frequency bands are used.
    relative : bool, optional
        Whether to compute relative power (power in band / total power), by default False
    **kwargs
        Additional keyword arguments to pass to compute_psd()

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary mapping band names to arrays of power values (one per channel).

    Examples
    --------
    >>> import mne
    >>> from qeeg.analysis import spectral
    >>> raw = mne.io.read_raw_edf("sample.edf", preload=True)
    >>> powers = spectral.compute_band_powers(raw)
    >>> print(powers['Alpha'])
    """
    # Use standard frequency bands if none are provided
    if frequency_bands is None:
        frequency_bands = FREQUENCY_BANDS
    
    # Compute the PSD
    psds, freqs = compute_psd(raw, **kwargs)
    
    # Initialize the result dictionary
    band_powers = {}
    
    # Compute the total power if relative power is requested
    if relative:
        total_power = np.sum(psds, axis=1)
    
    # Compute the power in each frequency band
    for band_name, (fmin, fmax) in frequency_bands.items():
        # Find the frequency indices corresponding to the band
        band_idx = np.logical_and(freqs >= fmin, freqs <= fmax)
        
        # Compute the power in the band
        band_power = np.sum(psds[:, band_idx], axis=1)
        
        # Compute relative power if requested
        if relative:
            band_power = band_power / total_power
        
        # Store the result
        band_powers[band_name] = band_power
    
    return band_powers


def compute_relative_band_powers(
    raw: mne.io.Raw,
    frequency_bands: Optional[Dict[str, Tuple[float, float]]] = None,
    **kwargs
) -> Dict[str, np.ndarray]:
    """
    Compute the relative power in each frequency band for each channel.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data.
    frequency_bands : Dict[str, Tuple[float, float]] or None, optional
        Dictionary mapping band names to (fmin, fmax) tuples, by default None
        If None, the standard frequency bands are used.
    **kwargs
        Additional keyword arguments to pass to compute_band_powers()

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary mapping band names to arrays of relative power values (one per channel).

    Examples
    --------
    >>> import mne
    >>> from qeeg.analysis import spectral
    >>> raw = mne.io.read_raw_edf("sample.edf", preload=True)
    >>> rel_powers = spectral.compute_relative_band_powers(raw)
    >>> print(rel_powers['Alpha'])
    """
    return compute_band_powers(raw, frequency_bands, relative=True, **kwargs)


def compute_band_power_ratios(
    raw: mne.io.Raw,
    ratios: Optional[Dict[str, Tuple[str, str]]] = None,
    frequency_bands: Optional[Dict[str, Tuple[float, float]]] = None,
    **kwargs
) -> Dict[str, np.ndarray]:
    """
    Compute power ratios between frequency bands for each channel.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data.
    ratios : Dict[str, Tuple[str, str]] or None, optional
        Dictionary mapping ratio names to (numerator_band, denominator_band) tuples, by default None
        If None, the standard ratios are used: {'Theta/Beta': ('Theta', 'Beta1')}.
    frequency_bands : Dict[str, Tuple[float, float]] or None, optional
        Dictionary mapping band names to (fmin, fmax) tuples, by default None
        If None, the standard frequency bands are used.
    **kwargs
        Additional keyword arguments to pass to compute_band_powers()

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary mapping ratio names to arrays of ratio values (one per channel).

    Examples
    --------
    >>> import mne
    >>> from qeeg.analysis import spectral
    >>> raw = mne.io.read_raw_edf("sample.edf", preload=True)
    >>> ratios = spectral.compute_band_power_ratios(raw)
    >>> print(ratios['Theta/Beta'])
    """
    # Use standard frequency bands if none are provided
    if frequency_bands is None:
        frequency_bands = FREQUENCY_BANDS
    
    # Use standard ratios if none are provided
    if ratios is None:
        ratios = {'Theta/Beta': ('Theta', 'Beta1')}
    
    # Compute the power in each frequency band
    band_powers = compute_band_powers(raw, frequency_bands, relative=False, **kwargs)
    
    # Initialize the result dictionary
    power_ratios = {}
    
    # Compute the power ratios
    for ratio_name, (numerator_band, denominator_band) in ratios.items():
        # Check if the bands exist
        if numerator_band not in band_powers or denominator_band not in band_powers:
            raise ValueError(f"Band '{numerator_band}' or '{denominator_band}' not found in frequency bands.")
        
        # Compute the ratio
        ratio = band_powers[numerator_band] / (band_powers[denominator_band] + 1e-10)  # Add small constant to avoid division by zero
        
        # Store the result
        power_ratios[ratio_name] = ratio
    
    return power_ratios


def compute_peak_frequency(
    raw: mne.io.Raw,
    fmin: float = 0.5,
    fmax: float = 40.0,
    picks: Optional[Union[str, list]] = "eeg",
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the peak frequency and corresponding power for each channel.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data.
    fmin : float, optional
        Minimum frequency of interest, by default 0.5
    fmax : float, optional
        Maximum frequency of interest, by default 40.0
    picks : str or list, optional
        Channels to include, by default "eeg" (all EEG channels)
    **kwargs
        Additional keyword arguments to pass to compute_psd()

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Arrays of peak frequencies and corresponding powers (one per channel).

    Examples
    --------
    >>> import mne
    >>> from qeeg.analysis import spectral
    >>> raw = mne.io.read_raw_edf("sample.edf", preload=True)
    >>> peak_freqs, peak_powers = spectral.compute_peak_frequency(raw)
    """
    # Compute the PSD
    psds, freqs = compute_psd(raw, fmin=fmin, fmax=fmax, picks=picks, **kwargs)
    
    # Find the peak frequency for each channel
    peak_indices = np.argmax(psds, axis=1)
    peak_freqs = freqs[peak_indices]
    peak_powers = np.array([psds[i, idx] for i, idx in enumerate(peak_indices)])
    
    return peak_freqs, peak_powers


def compute_band_peak_frequency(
    raw: mne.io.Raw,
    band_name: str = 'Alpha',
    frequency_bands: Optional[Dict[str, Tuple[float, float]]] = None,
    picks: Optional[Union[str, list]] = "eeg",
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the peak frequency and corresponding power within a specific frequency band for each channel.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data.
    band_name : str, optional
        Name of the frequency band, by default 'Alpha'
    frequency_bands : Dict[str, Tuple[float, float]] or None, optional
        Dictionary mapping band names to (fmin, fmax) tuples, by default None
        If None, the standard frequency bands are used.
    picks : str or list, optional
        Channels to include, by default "eeg" (all EEG channels)
    **kwargs
        Additional keyword arguments to pass to compute_psd()

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Arrays of peak frequencies and corresponding powers within the band (one per channel).

    Examples
    --------
    >>> import mne
    >>> from qeeg.analysis import spectral
    >>> raw = mne.io.read_raw_edf("sample.edf", preload=True)
    >>> alpha_peak_freqs, alpha_peak_powers = spectral.compute_band_peak_frequency(raw, band_name='Alpha')
    """
    # Use standard frequency bands if none are provided
    if frequency_bands is None:
        frequency_bands = FREQUENCY_BANDS
    
    # Check if the band exists
    if band_name not in frequency_bands:
        raise ValueError(f"Band '{band_name}' not found in frequency bands.")
    
    # Get the frequency range for the band
    fmin, fmax = frequency_bands[band_name]
    
    # Compute the PSD
    psds, freqs = compute_psd(raw, fmin=fmin, fmax=fmax, picks=picks, **kwargs)
    
    # Find the peak frequency for each channel within the band
    band_idx = np.logical_and(freqs >= fmin, freqs <= fmax)
    band_freqs = freqs[band_idx]
    band_psds = psds[:, band_idx]
    
    peak_indices = np.argmax(band_psds, axis=1)
    peak_freqs = band_freqs[peak_indices]
    peak_powers = np.array([band_psds[i, idx] for i, idx in enumerate(peak_indices)])
    
    return peak_freqs, peak_powers


def analyze_relative_power(
    raw: mne.io.Raw,
    frequency_bands: Optional[Dict[str, Tuple[float, float]]] = None,
    zones: Optional[Dict[str, List[str]]] = None,
    upper_percentile: float = 90,
    lower_percentile: float = 10,
    **kwargs
) -> Dict[str, Dict[str, Dict[str, List[str]]]]:
    """
    Analyze relative power across frequency bands and identify zones/channels with increased or reduced power.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data.
    frequency_bands : Dict[str, Tuple[float, float]] or None, optional
        Dictionary mapping band names to (fmin, fmax) tuples, by default None
        If None, the standard frequency bands are used.
    zones : Dict[str, List[str]] or None, optional
        Dictionary mapping zone names to lists of channel names, by default None
        If None, zones are defined based on channel names.
    upper_percentile : float, optional
        Percentile for the "increased power" threshold, by default 90
    lower_percentile : float, optional
        Percentile for the "reduced power" threshold, by default 10
    **kwargs
        Additional keyword arguments to pass to compute_relative_band_powers()

    Returns
    -------
    Dict[str, Dict[str, Dict[str, List[str]]]]
        Dictionary with results for each frequency band, with "increased" and "reduced" power for each zone.

    Examples
    --------
    >>> import mne
    >>> from qeeg.analysis import spectral
    >>> raw = mne.io.read_raw_edf("sample.edf", preload=True)
    >>> results = spectral.analyze_relative_power(raw)
    >>> print(results['Alpha']['increased']['Occipital'])
    """
    # Use standard frequency bands if none are provided
    if frequency_bands is None:
        frequency_bands = FREQUENCY_BANDS
    
    # Define zones if not provided
    if zones is None:
        from qeeg.preprocessing.montage import define_zones
        zones = define_zones(raw)
    
    # Compute relative power for each frequency band
    relative_powers = compute_relative_band_powers(raw, frequency_bands, **kwargs)
    
    # Initialize the result dictionary
    results = {band: {"increased": {}, "reduced": {}} for band in frequency_bands}
    
    # Process each frequency band
    for band_name, band_power in relative_powers.items():
        # Collect all relative powers for dynamic thresholds
        all_relative_powers = []
        for zone, channels in zones.items():
            # Get indices of channels in the zone
            picks = mne.pick_channels(raw.info["ch_names"], include=channels)
            
            # Extract powers for the selected channels
            zone_power = band_power[picks]
            
            # Collect all relative powers
            all_relative_powers.extend(zone_power)
        
        # Compute dynamic thresholds using percentiles
        increase_threshold = np.percentile(all_relative_powers, upper_percentile)
        reduce_threshold = np.percentile(all_relative_powers, lower_percentile)
        
        # Process zones with thresholds
        for zone, channels in zones.items():
            # Get indices of channels in the zone
            picks = mne.pick_channels(raw.info["ch_names"], include=channels)
            
            # Extract powers for the selected channels
            zone_power = band_power[picks]
            
            # Identify increased power
            increased_channels = [
                raw.info["ch_names"][picks[i]]
                for i in range(len(zone_power))
                if zone_power[i] > increase_threshold
            ]
            
            # Identify reduced power
            reduced_channels = [
                raw.info["ch_names"][picks[i]]
                for i in range(len(zone_power))
                if zone_power[i] < reduce_threshold
            ]
            
            # Store results
            if increased_channels:
                results[band_name]["increased"][zone] = increased_channels
            if reduced_channels:
                results[band_name]["reduced"][zone] = reduced_channels
    
    return results
