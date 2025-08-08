"""
Anxiety module for EEG data analysis.

This module provides functions for analyzing EEG data in the context of anxiety disorders,
including detection of EEG patterns associated with anxiety and calculation of
relevant metrics.
"""

import numpy as np
from typing import Tuple, Optional, Union, List, Dict
import mne

from qeeg.analysis import spectral, asymmetry


def calculate_alpha_asymmetry(
    raw: mne.io.Raw,
    **kwargs
) -> Dict[str, float]:
    """
    Calculate frontal alpha asymmetry, which is associated with anxiety.
    
    Frontal alpha asymmetry, particularly in the F3-F4 electrode pair, has been
    associated with anxiety disorders, with greater right-sided activity (lower alpha power)
    often observed in individuals with anxiety.
    
    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data.
    **kwargs
        Additional keyword arguments to pass to compute_band_asymmetry()
        
    Returns
    -------
    Dict[str, float]
        Dictionary mapping electrode pairs to alpha asymmetry indices.
        
    Examples
    --------
    >>> import mne
    >>> from qeeg.conditions import anxiety
    >>> raw = mne.io.read_raw_edf("sample.edf", preload=True)
    >>> asymmetry = anxiety.calculate_alpha_asymmetry(raw)
    >>> print(f"F3-F4 alpha asymmetry: {asymmetry['F3-F4']:.2f}")
    """
    # Define frequency bands with focus on alpha
    frequency_bands = {
        "Alpha": (8, 13)
    }
    
    # Define electrode pairs with focus on frontal regions
    electrode_pairs = [
        ('F3', 'F4'),
        ('F7', 'F8'),
        ('Fp1', 'Fp2')
    ]
    
    # Compute band asymmetry
    band_asymmetry = asymmetry.compute_band_asymmetry(
        raw,
        frequency_bands=frequency_bands,
        electrode_pairs=electrode_pairs,
        **kwargs
    )
    
    # Extract alpha asymmetry
    alpha_asymmetry = band_asymmetry["Alpha"]
    
    return alpha_asymmetry


def analyze_beta_power(
    raw: mne.io.Raw,
    picks: Optional[Union[str, list]] = "eeg",
    **kwargs
) -> Dict[str, np.ndarray]:
    """
    Analyze beta power, which is often elevated in anxiety.
    
    Increased beta activity, particularly in frontal and central regions,
    has been associated with anxiety and stress.
    
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
    >>> from qeeg.conditions import anxiety
    >>> raw = mne.io.read_raw_edf("sample.edf", preload=True)
    >>> beta_power = anxiety.analyze_beta_power(raw)
    >>> print(f"Mean beta power: {beta_power['Beta'].mean():.2f}")
    """
    # Define frequency bands with focus on beta
    frequency_bands = {
        "Beta1": (13, 20),
        "Beta2": (20, 30),
        "Beta": (13, 30)
    }
    
    # Compute band powers
    band_powers = spectral.compute_band_powers(
        raw,
        frequency_bands=frequency_bands,
        picks=picks,
        **kwargs
    )
    
    return band_powers


def detect_anxiety_patterns(
    raw: mne.io.Raw,
    alpha_asymmetry_threshold: float = 0.2,
    beta_power_threshold: float = 1.5,
    **kwargs
) -> Dict[str, Union[bool, Dict[str, float], Dict[str, np.ndarray]]]:
    """
    Detect EEG patterns associated with anxiety.
    
    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data.
    alpha_asymmetry_threshold : float, optional
        Threshold for alpha asymmetry, by default 0.2
    beta_power_threshold : float, optional
        Threshold for beta power in standard deviations above the mean, by default 1.5
    **kwargs
        Additional keyword arguments
        
    Returns
    -------
    Dict[str, Union[bool, Dict[str, float], Dict[str, np.ndarray]]]
        Dictionary with anxiety-related metrics and detection results.
        
    Examples
    --------
    >>> import mne
    >>> from qeeg.conditions import anxiety
    >>> raw = mne.io.read_raw_edf("sample.edf", preload=True)
    >>> results = anxiety.detect_anxiety_patterns(raw)
    >>> print(f"Anxiety patterns detected: {results['anxiety_detected']}")
    """
    # Calculate alpha asymmetry
    alpha_asymmetry = calculate_alpha_asymmetry(raw, **kwargs)
    
    # Analyze beta power
    beta_power = analyze_beta_power(raw, **kwargs)
    
    # Calculate mean beta power
    mean_beta_power = np.mean(beta_power["Beta"])
    
    # Calculate standard deviation of beta power
    std_beta_power = np.std(beta_power["Beta"])
    
    # Check for anxiety patterns
    # 1. Check for right-sided frontal alpha asymmetry (negative values)
    f3_f4_asymmetry = alpha_asymmetry.get('F3-F4', 0)
    right_sided_asymmetry = f3_f4_asymmetry < -alpha_asymmetry_threshold
    
    # 2. Check for elevated beta power
    elevated_beta = mean_beta_power > (np.mean(beta_power["Beta"]) + beta_power_threshold * std_beta_power)
    
    # Combine the results
    anxiety_detected = right_sided_asymmetry or elevated_beta
    
    # Return the results
    return {
        "anxiety_detected": anxiety_detected,
        "alpha_asymmetry": alpha_asymmetry,
        "beta_power": beta_power,
        "mean_beta_power": mean_beta_power,
        "right_sided_asymmetry": right_sided_asymmetry,
        "elevated_beta": elevated_beta
    }


def generate_anxiety_report(
    raw: mne.io.Raw,
    **kwargs
) -> str:
    """
    Generate a report on anxiety-related EEG patterns.
    
    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data.
    **kwargs
        Additional keyword arguments to pass to detect_anxiety_patterns()
        
    Returns
    -------
    str
        Report on anxiety-related EEG patterns.
        
    Examples
    --------
    >>> import mne
    >>> from qeeg.conditions import anxiety
    >>> raw = mne.io.read_raw_edf("sample.edf", preload=True)
    >>> report = anxiety.generate_anxiety_report(raw)
    >>> print(report)
    """
    # Detect anxiety patterns
    results = detect_anxiety_patterns(raw, **kwargs)
    
    # Generate the report
    report = []
    report.append("Anxiety EEG Analysis Report")
    report.append("==========================")
    report.append("")
    
    # Add the alpha asymmetry results
    report.append("Frontal Alpha Asymmetry:")
    for pair, asymmetry_index in results['alpha_asymmetry'].items():
        report.append(f"  {pair}: {asymmetry_index:.2f}")
    
    if results['right_sided_asymmetry']:
        report.append("Right-sided frontal alpha asymmetry detected, which may be associated with anxiety.")
    else:
        report.append("No significant right-sided frontal alpha asymmetry detected.")
    report.append("")
    
    # Add the beta power results
    report.append(f"Mean Beta Power: {results['mean_beta_power']:.2f}")
    if results['elevated_beta']:
        report.append("Elevated beta power detected, which may be associated with anxiety or stress.")
    else:
        report.append("Beta power is within normal limits.")
    report.append("")
    
    # Add the overall result
    if results['anxiety_detected']:
        report.append("EEG patterns consistent with anxiety were detected.")
    else:
        report.append("No significant anxiety-related EEG patterns were detected.")
    report.append("")
    
    # Add a disclaimer
    report.append("Disclaimer: This analysis is for research purposes only and should not be used for clinical diagnosis.")
    
    return "\n".join(report)
