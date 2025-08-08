"""
ADHD (Attention Deficit Hyperactivity Disorder) module for EEG data analysis.

This module provides functions for analyzing EEG data in the context of ADHD,
including detection of EEG patterns associated with ADHD and calculation of
relevant metrics.
"""

import numpy as np
from typing import Tuple, Optional, Union, List, Dict
import mne

from qeeg.analysis import spectral, asymmetry


def calculate_theta_beta_ratio(
    raw: mne.io.Raw,
    picks: Optional[Union[str, list]] = "eeg",
    **kwargs
) -> Dict[str, float]:
    """
    Calculate the theta/beta ratio for each channel.
    
    The theta/beta ratio is a common metric used in ADHD research, with higher
    ratios often associated with ADHD.
    
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
    Dict[str, float]
        Dictionary mapping channel names to theta/beta ratios.
        
    Examples
    --------
    >>> import mne
    >>> from qeeg.conditions import adhd
    >>> raw = mne.io.read_raw_edf("sample.edf", preload=True)
    >>> ratios = adhd.calculate_theta_beta_ratio(raw)
    >>> print(f"Theta/Beta ratio at Cz: {ratios['Cz']:.2f}")
    """
    # Define the frequency bands
    frequency_bands = {
        "Theta": (4, 7.5),
        "Beta": (13, 30)
    }
    
    # Compute the power in each frequency band
    band_powers = spectral.compute_band_powers(
        raw,
        frequency_bands=frequency_bands,
        picks=picks,
        **kwargs
    )
    
    # Get the channel names
    ch_names = [raw.ch_names[idx] for idx in mne.pick_types(raw.info, meg=False, eeg=True, selection=picks)]
    
    # Calculate the theta/beta ratio for each channel
    ratios = {}
    for ch_idx, ch_name in enumerate(ch_names):
        theta_power = band_powers["Theta"][ch_idx]
        beta_power = band_powers["Beta"][ch_idx]
        ratio = theta_power / (beta_power + 1e-10)  # Add small constant to avoid division by zero
        ratios[ch_name] = ratio
    
    return ratios


def analyze_frontal_asymmetry(
    raw: mne.io.Raw,
    **kwargs
) -> Dict[str, float]:
    """
    Analyze frontal asymmetry in the EEG data.
    
    Frontal asymmetry has been associated with ADHD, with some studies
    suggesting that individuals with ADHD may show different patterns of
    frontal asymmetry compared to neurotypical individuals.
    
    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data.
    **kwargs
        Additional keyword arguments to pass to compute_asymmetry_index()
        
    Returns
    -------
    Dict[str, float]
        Dictionary mapping frontal electrode pairs to asymmetry indices.
        
    Examples
    --------
    >>> import mne
    >>> from qeeg.conditions import adhd
    >>> raw = mne.io.read_raw_edf("sample.edf", preload=True)
    >>> asymmetry = adhd.analyze_frontal_asymmetry(raw)
    >>> print(f"F3-F4 asymmetry: {asymmetry['F3-F4']:.2f}")
    """
    # Define frontal electrode pairs
    frontal_pairs = [
        ('F3', 'F4'),
        ('F7', 'F8'),
        ('Fp1', 'Fp2')
    ]
    
    # Compute asymmetry indices
    asymmetry_indices = asymmetry.compute_asymmetry_index(
        raw,
        electrode_pairs=frontal_pairs,
        **kwargs
    )
    
    return asymmetry_indices


def detect_adhd_patterns(
    raw: mne.io.Raw,
    threshold: float = 1.5,
    **kwargs
) -> Dict[str, Union[bool, float, Dict[str, float]]]:
    """
    Detect EEG patterns associated with ADHD.
    
    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data.
    threshold : float, optional
        Threshold for the theta/beta ratio, by default 1.5
    **kwargs
        Additional keyword arguments
        
    Returns
    -------
    Dict[str, Union[bool, float, Dict[str, float]]]
        Dictionary with ADHD-related metrics and detection results.
        
    Examples
    --------
    >>> import mne
    >>> from qeeg.conditions import adhd
    >>> raw = mne.io.read_raw_edf("sample.edf", preload=True)
    >>> results = adhd.detect_adhd_patterns(raw)
    >>> print(f"ADHD patterns detected: {results['adhd_detected']}")
    """
    # Calculate the theta/beta ratio
    tb_ratios = calculate_theta_beta_ratio(raw, **kwargs)
    
    # Calculate the mean theta/beta ratio
    mean_tb_ratio = np.mean(list(tb_ratios.values()))
    
    # Analyze frontal asymmetry
    frontal_asymmetry = analyze_frontal_asymmetry(raw, **kwargs)
    
    # Check if the mean theta/beta ratio exceeds the threshold
    adhd_detected = mean_tb_ratio > threshold
    
    # Return the results
    return {
        "adhd_detected": adhd_detected,
        "mean_theta_beta_ratio": mean_tb_ratio,
        "theta_beta_ratios": tb_ratios,
        "frontal_asymmetry": frontal_asymmetry
    }


def generate_adhd_report(
    raw: mne.io.Raw,
    **kwargs
) -> str:
    """
    Generate a report on ADHD-related EEG patterns.
    
    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data.
    **kwargs
        Additional keyword arguments to pass to detect_adhd_patterns()
        
    Returns
    -------
    str
        Report on ADHD-related EEG patterns.
        
    Examples
    --------
    >>> import mne
    >>> from qeeg.conditions import adhd
    >>> raw = mne.io.read_raw_edf("sample.edf", preload=True)
    >>> report = adhd.generate_adhd_report(raw)
    >>> print(report)
    """
    # Detect ADHD patterns
    results = detect_adhd_patterns(raw, **kwargs)
    
    # Generate the report
    report = []
    report.append("ADHD EEG Analysis Report")
    report.append("=======================")
    report.append("")
    
    # Add the mean theta/beta ratio
    report.append(f"Mean Theta/Beta Ratio: {results['mean_theta_beta_ratio']:.2f}")
    if results['adhd_detected']:
        report.append("This ratio is elevated and may be consistent with ADHD patterns.")
    else:
        report.append("This ratio is within normal limits.")
    report.append("")
    
    # Add the theta/beta ratios for each channel
    report.append("Theta/Beta Ratios by Channel:")
    for ch_name, ratio in results['theta_beta_ratios'].items():
        report.append(f"  {ch_name}: {ratio:.2f}")
    report.append("")
    
    # Add the frontal asymmetry results
    report.append("Frontal Asymmetry:")
    for pair, asymmetry_index in results['frontal_asymmetry'].items():
        report.append(f"  {pair}: {asymmetry_index:.2f}")
    report.append("")
    
    # Add a disclaimer
    report.append("Disclaimer: This analysis is for research purposes only and should not be used for clinical diagnosis.")
    
    return "\n".join(report)
