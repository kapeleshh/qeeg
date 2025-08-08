"""
Depression module for EEG data analysis.

This module provides functions for analyzing EEG data in the context of depression,
including detection of EEG patterns associated with depression and calculation of
relevant metrics.
"""

import numpy as np
from typing import Tuple, Optional, Union, List, Dict
import mne

from qeeg.analysis import spectral, asymmetry


def calculate_frontal_alpha_asymmetry(
    raw: mne.io.Raw,
    **kwargs
) -> Dict[str, float]:
    """
    Calculate frontal alpha asymmetry, which is associated with depression.
    
    Frontal alpha asymmetry, particularly in the F3-F4 electrode pair, has been
    associated with depression, with greater right-sided alpha (lower right-sided activity)
    often observed in individuals with depression.
    
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
    >>> from qeeg.conditions import depression
    >>> raw = mne.io.read_raw_edf("sample.edf", preload=True)
    >>> asymmetry = depression.calculate_frontal_alpha_asymmetry(raw)
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


def analyze_theta_activity(
    raw: mne.io.Raw,
    picks: Optional[Union[str, list]] = "eeg",
    **kwargs
) -> Dict[str, np.ndarray]:
    """
    Analyze theta activity, which may be elevated in depression.
    
    Increased theta activity, particularly in frontal regions,
    has been associated with depression.
    
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
    >>> from qeeg.conditions import depression
    >>> raw = mne.io.read_raw_edf("sample.edf", preload=True)
    >>> theta_power = depression.analyze_theta_activity(raw)
    >>> print(f"Mean theta power: {theta_power['Theta'].mean():.2f}")
    """
    # Define frequency bands with focus on theta
    frequency_bands = {
        "Theta": (4, 8),
        "Alpha": (8, 13),  # Include alpha for comparison
        "Theta/Alpha": (4, 13)  # Combined band
    }
    
    # Compute band powers
    band_powers = spectral.compute_band_powers(
        raw,
        frequency_bands=frequency_bands,
        picks=picks,
        **kwargs
    )
    
    return band_powers


def detect_depression_patterns(
    raw: mne.io.Raw,
    alpha_asymmetry_threshold: float = 0.2,
    theta_threshold: float = 1.5,
    **kwargs
) -> Dict[str, Union[bool, Dict[str, float], Dict[str, np.ndarray]]]:
    """
    Detect EEG patterns associated with depression.
    
    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data.
    alpha_asymmetry_threshold : float, optional
        Threshold for alpha asymmetry, by default 0.2
    theta_threshold : float, optional
        Threshold for elevated theta power in standard deviations above the mean, by default 1.5
    **kwargs
        Additional keyword arguments
        
    Returns
    -------
    Dict[str, Union[bool, Dict[str, float], Dict[str, np.ndarray]]]
        Dictionary with depression-related metrics and detection results.
        
    Examples
    --------
    >>> import mne
    >>> from qeeg.conditions import depression
    >>> raw = mne.io.read_raw_edf("sample.edf", preload=True)
    >>> results = depression.detect_depression_patterns(raw)
    >>> print(f"Depression patterns detected: {results['depression_detected']}")
    """
    # Calculate frontal alpha asymmetry
    alpha_asymmetry = calculate_frontal_alpha_asymmetry(raw, **kwargs)
    
    # Analyze theta activity
    theta_power = analyze_theta_activity(raw, **kwargs)
    
    # Calculate mean theta power
    mean_theta_power = np.mean(theta_power["Theta"])
    
    # Calculate standard deviation of theta power
    std_theta_power = np.std(theta_power["Theta"])
    
    # Check for depression patterns
    # 1. Check for right-sided frontal alpha asymmetry (positive values)
    f3_f4_asymmetry = alpha_asymmetry.get('F3-F4', 0)
    right_sided_asymmetry = f3_f4_asymmetry > alpha_asymmetry_threshold
    
    # 2. Check for elevated theta power
    elevated_theta = mean_theta_power > (np.mean(theta_power["Theta"]) + theta_threshold * std_theta_power)
    
    # Combine the results
    depression_detected = right_sided_asymmetry or elevated_theta
    
    # Return the results
    return {
        "depression_detected": depression_detected,
        "alpha_asymmetry": alpha_asymmetry,
        "theta_power": theta_power,
        "mean_theta_power": mean_theta_power,
        "right_sided_asymmetry": right_sided_asymmetry,
        "elevated_theta": elevated_theta
    }


def calculate_depression_severity(
    raw: mne.io.Raw,
    **kwargs
) -> Dict[str, float]:
    """
    Calculate metrics related to depression severity.
    
    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data.
    **kwargs
        Additional keyword arguments
        
    Returns
    -------
    Dict[str, float]
        Dictionary with depression severity metrics.
        
    Examples
    --------
    >>> import mne
    >>> from qeeg.conditions import depression
    >>> raw = mne.io.read_raw_edf("sample.edf", preload=True)
    >>> severity = depression.calculate_depression_severity(raw)
    >>> print(f"Depression severity score: {severity['severity_score']:.2f}")
    """
    # Detect depression patterns
    results = detect_depression_patterns(raw, **kwargs)
    
    # Calculate severity score based on alpha asymmetry and theta power
    # This is a simplified approach and should be validated with clinical data
    
    # 1. Alpha asymmetry component
    f3_f4_asymmetry = results['alpha_asymmetry'].get('F3-F4', 0)
    alpha_component = max(0, f3_f4_asymmetry)  # Only consider positive values (right-sided)
    
    # 2. Theta power component
    theta_component = results['mean_theta_power'] / 10.0  # Scale to similar range as asymmetry
    
    # Combine components into a severity score
    severity_score = alpha_component + theta_component
    
    # Normalize to 0-10 scale (approximate)
    severity_score = min(10, severity_score * 5)
    
    # Categorize severity
    if severity_score < 3:
        severity_category = "Minimal"
    elif severity_score < 5:
        severity_category = "Mild"
    elif severity_score < 7:
        severity_category = "Moderate"
    else:
        severity_category = "Severe"
    
    return {
        "severity_score": severity_score,
        "severity_category": severity_category,
        "alpha_component": alpha_component,
        "theta_component": theta_component
    }


def generate_depression_report(
    raw: mne.io.Raw,
    **kwargs
) -> str:
    """
    Generate a report on depression-related EEG patterns.
    
    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data.
    **kwargs
        Additional keyword arguments to pass to detect_depression_patterns()
        
    Returns
    -------
    str
        Report on depression-related EEG patterns.
        
    Examples
    --------
    >>> import mne
    >>> from qeeg.conditions import depression
    >>> raw = mne.io.read_raw_edf("sample.edf", preload=True)
    >>> report = depression.generate_depression_report(raw)
    >>> print(report)
    """
    # Detect depression patterns
    results = detect_depression_patterns(raw, **kwargs)
    
    # Calculate depression severity
    severity = calculate_depression_severity(raw, **kwargs)
    
    # Generate the report
    report = []
    report.append("Depression EEG Analysis Report")
    report.append("=============================")
    report.append("")
    
    # Add the alpha asymmetry results
    report.append("Frontal Alpha Asymmetry:")
    for pair, asymmetry_index in results['alpha_asymmetry'].items():
        report.append(f"  {pair}: {asymmetry_index:.2f}")
    
    if results['right_sided_asymmetry']:
        report.append("Right-sided frontal alpha asymmetry detected, which may be associated with depression.")
    else:
        report.append("No significant right-sided frontal alpha asymmetry detected.")
    report.append("")
    
    # Add the theta power results
    report.append(f"Mean Theta Power: {results['mean_theta_power']:.2f}")
    if results['elevated_theta']:
        report.append("Elevated theta power detected, which may be associated with depression.")
    else:
        report.append("Theta power is within normal limits.")
    report.append("")
    
    # Add the severity results
    report.append(f"Depression Severity Score: {severity['severity_score']:.2f}/10")
    report.append(f"Severity Category: {severity['severity_category']}")
    report.append("")
    
    # Add the overall result
    if results['depression_detected']:
        report.append("EEG patterns consistent with depression were detected.")
    else:
        report.append("No significant depression-related EEG patterns were detected.")
    report.append("")
    
    # Add a disclaimer
    report.append("Disclaimer: This analysis is for research purposes only and should not be used for clinical diagnosis.")
    
    return "\n".join(report)
