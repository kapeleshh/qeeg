"""
Autism module for EEG data analysis.

This module provides functions for analyzing EEG data in the context of autism spectrum disorder (ASD),
including detection of EEG patterns associated with ASD and calculation of relevant metrics.
"""

import numpy as np
from typing import Tuple, Optional, Union, List, Dict
import mne

from qeeg.analysis import spectral, asymmetry


def calculate_coherence(
    raw: mne.io.Raw,
    fmin: float = 0.5,
    fmax: float = 40.0,
    **kwargs
) -> Dict[Tuple[str, str], np.ndarray]:
    """
    Calculate coherence between electrode pairs.
    
    Reduced long-range coherence and increased short-range coherence
    have been observed in individuals with autism spectrum disorder.
    
    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data.
    fmin : float, optional
        Minimum frequency of interest, by default 0.5
    fmax : float, optional
        Maximum frequency of interest, by default 40.0
    **kwargs
        Additional keyword arguments
        
    Returns
    -------
    Dict[Tuple[str, str], np.ndarray]
        Dictionary mapping electrode pairs to coherence values.
        
    Examples
    --------
    >>> import mne
    >>> from qeeg.conditions import autism
    >>> raw = mne.io.read_raw_edf("sample.edf", preload=True)
    >>> coherence = autism.calculate_coherence(raw)
    >>> print(f"F3-F4 coherence: {coherence[('F3', 'F4')].mean():.2f}")
    """
    # Get the data and channel names
    data = raw.get_data()
    ch_names = raw.ch_names
    
    # Calculate the sampling frequency
    sfreq = raw.info['sfreq']
    
    # Define electrode pairs
    # Focus on long-range connections (frontal-parietal, frontal-occipital)
    # and short-range connections (frontal-frontal, parietal-parietal)
    electrode_pairs = [
        # Long-range connections
        ('F3', 'P3'),
        ('F4', 'P4'),
        ('F3', 'O1'),
        ('F4', 'O2'),
        # Short-range connections
        ('F3', 'F4'),
        ('P3', 'P4'),
        ('O1', 'O2')
    ]
    
    # Initialize the result dictionary
    coherence = {}
    
    # Calculate coherence for each pair
    for ch1, ch2 in electrode_pairs:
        # Check if both channels exist
        if ch1 in ch_names and ch2 in ch_names:
            # Get the indices of the channels
            idx1 = ch_names.index(ch1)
            idx2 = ch_names.index(ch2)
            
            # Get the data for the channels
            data1 = data[idx1]
            data2 = data[idx2]
            
            # Calculate coherence using mne.connectivity.spectral_connectivity
            from mne.connectivity import spectral_connectivity
            
            # Create epochs from continuous data
            # (required for spectral_connectivity)
            epoch_length = int(2 * sfreq)  # 2-second epochs
            n_epochs = data.shape[1] // epoch_length
            
            if n_epochs > 0:
                # Reshape data into epochs
                epochs_data = np.zeros((n_epochs, 2, epoch_length))
                for i in range(n_epochs):
                    start = i * epoch_length
                    end = start + epoch_length
                    epochs_data[i, 0, :] = data1[start:end]
                    epochs_data[i, 1, :] = data2[start:end]
                
                # Calculate coherence
                con, freqs, times, n_epochs, n_tapers = spectral_connectivity(
                    epochs_data,
                    method='coh',
                    mode='multitaper',
                    sfreq=sfreq,
                    fmin=fmin,
                    fmax=fmax,
                    faverage=True,
                    **kwargs
                )
                
                # Store the result
                coherence[(ch1, ch2)] = con[0, 1, 0]
    
    return coherence


def analyze_gamma_power(
    raw: mne.io.Raw,
    picks: Optional[Union[str, list]] = "eeg",
    **kwargs
) -> Dict[str, np.ndarray]:
    """
    Analyze gamma power, which may be altered in autism.
    
    Some studies have reported increased gamma activity in individuals with autism.
    
    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data.
    picks : str or list, optional
        Channels to include, by default "eeg" (all EEG channels)
    **kwargs
        Additional keyword arguments to pass to compute_band_powers()
        
    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary mapping frequency bands to power values.
        
    Examples
    --------
    >>> import mne
    >>> from qeeg.conditions import autism
    >>> raw = mne.io.read_raw_edf("sample.edf", preload=True)
    >>> gamma_power = autism.analyze_gamma_power(raw)
    >>> print(f"Mean gamma power: {gamma_power['Gamma'].mean():.2f}")
    """
    # Define frequency bands with focus on gamma
    frequency_bands = {
        "Gamma": (30, 80),
        "Low Gamma": (30, 50),
        "High Gamma": (50, 80)
    }
    
    # Compute band powers
    band_powers = spectral.compute_band_powers(
        raw,
        frequency_bands=frequency_bands,
        picks=picks,
        **kwargs
    )
    
    return band_powers


def detect_autism_patterns(
    raw: mne.io.Raw,
    coherence_threshold: float = 0.5,
    gamma_threshold: float = 1.5,
    **kwargs
) -> Dict[str, Union[bool, Dict[Tuple[str, str], np.ndarray], Dict[str, np.ndarray]]]:
    """
    Detect EEG patterns associated with autism spectrum disorder.
    
    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data.
    coherence_threshold : float, optional
        Threshold for reduced long-range coherence, by default 0.5
    gamma_threshold : float, optional
        Threshold for increased gamma power in standard deviations above the mean, by default 1.5
    **kwargs
        Additional keyword arguments
        
    Returns
    -------
    Dict[str, Union[bool, Dict[Tuple[str, str], np.ndarray], Dict[str, np.ndarray]]]
        Dictionary with autism-related metrics and detection results.
        
    Examples
    --------
    >>> import mne
    >>> from qeeg.conditions import autism
    >>> raw = mne.io.read_raw_edf("sample.edf", preload=True)
    >>> results = autism.detect_autism_patterns(raw)
    >>> print(f"Autism patterns detected: {results['autism_detected']}")
    """
    # Calculate coherence
    coherence = calculate_coherence(raw, **kwargs)
    
    # Analyze gamma power
    gamma_power = analyze_gamma_power(raw, **kwargs)
    
    # Calculate mean gamma power
    mean_gamma_power = np.mean(gamma_power["Gamma"])
    
    # Calculate standard deviation of gamma power
    std_gamma_power = np.std(gamma_power["Gamma"])
    
    # Check for autism patterns
    # 1. Check for reduced long-range coherence
    long_range_pairs = [('F3', 'P3'), ('F4', 'P4'), ('F3', 'O1'), ('F4', 'O2')]
    long_range_coherence = [coherence.get((ch1, ch2), 0) for ch1, ch2 in long_range_pairs if (ch1, ch2) in coherence]
    
    reduced_long_range = False
    if long_range_coherence:
        mean_long_range = np.mean(long_range_coherence)
        reduced_long_range = mean_long_range < coherence_threshold
    
    # 2. Check for increased gamma power
    elevated_gamma = mean_gamma_power > (np.mean(gamma_power["Gamma"]) + gamma_threshold * std_gamma_power)
    
    # Combine the results
    autism_detected = reduced_long_range or elevated_gamma
    
    # Return the results
    return {
        "autism_detected": autism_detected,
        "coherence": coherence,
        "gamma_power": gamma_power,
        "mean_gamma_power": mean_gamma_power,
        "reduced_long_range_coherence": reduced_long_range,
        "elevated_gamma": elevated_gamma
    }


def generate_autism_report(
    raw: mne.io.Raw,
    **kwargs
) -> str:
    """
    Generate a report on autism-related EEG patterns.
    
    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data.
    **kwargs
        Additional keyword arguments to pass to detect_autism_patterns()
        
    Returns
    -------
    str
        Report on autism-related EEG patterns.
        
    Examples
    --------
    >>> import mne
    >>> from qeeg.conditions import autism
    >>> raw = mne.io.read_raw_edf("sample.edf", preload=True)
    >>> report = autism.generate_autism_report(raw)
    >>> print(report)
    """
    # Detect autism patterns
    results = detect_autism_patterns(raw, **kwargs)
    
    # Generate the report
    report = []
    report.append("Autism Spectrum Disorder EEG Analysis Report")
    report.append("===========================================")
    report.append("")
    
    # Add the coherence results
    report.append("EEG Coherence:")
    
    # Long-range coherence
    long_range_pairs = [('F3', 'P3'), ('F4', 'P4'), ('F3', 'O1'), ('F4', 'O2')]
    report.append("  Long-range coherence:")
    for pair in long_range_pairs:
        if pair in results['coherence']:
            report.append(f"    {pair[0]}-{pair[1]}: {results['coherence'][pair]:.2f}")
    
    # Short-range coherence
    short_range_pairs = [('F3', 'F4'), ('P3', 'P4'), ('O1', 'O2')]
    report.append("  Short-range coherence:")
    for pair in short_range_pairs:
        if pair in results['coherence']:
            report.append(f"    {pair[0]}-{pair[1]}: {results['coherence'][pair]:.2f}")
    
    if results['reduced_long_range_coherence']:
        report.append("Reduced long-range coherence detected, which may be associated with autism spectrum disorder.")
    else:
        report.append("Long-range coherence is within normal limits.")
    report.append("")
    
    # Add the gamma power results
    report.append(f"Mean Gamma Power: {results['mean_gamma_power']:.2f}")
    if results['elevated_gamma']:
        report.append("Elevated gamma power detected, which may be associated with autism spectrum disorder.")
    else:
        report.append("Gamma power is within normal limits.")
    report.append("")
    
    # Add the overall result
    if results['autism_detected']:
        report.append("EEG patterns consistent with autism spectrum disorder were detected.")
    else:
        report.append("No significant autism-related EEG patterns were detected.")
    report.append("")
    
    # Add a disclaimer
    report.append("Disclaimer: This analysis is for research purposes only and should not be used for clinical diagnosis.")
    
    return "\n".join(report)
