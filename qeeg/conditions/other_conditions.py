"""
Other neurological conditions module for EEG data analysis.

This module provides functions for analyzing EEG data in the context of various
neurological conditions not covered by specific modules, such as:
- Insomnia
- Traumatic Brain Injury (TBI)
- Alzheimer's Disease
- Parkinson's Disease
- Schizophrenia
"""

import numpy as np
from typing import Tuple, Optional, Union, List, Dict
import mne

from qeeg.analysis import spectral, asymmetry


def analyze_insomnia_patterns(
    raw: mne.io.Raw,
    **kwargs
) -> Dict[str, Union[bool, Dict[str, np.ndarray], float]]:
    """
    Analyze EEG patterns associated with insomnia.
    
    Insomnia is often characterized by increased beta activity and
    reduced delta activity.
    
    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data.
    **kwargs
        Additional keyword arguments
        
    Returns
    -------
    Dict[str, Union[bool, Dict[str, np.ndarray], float]]
        Dictionary with insomnia-related metrics and detection results.
        
    Examples
    --------
    >>> import mne
    >>> from qeeg.conditions import other_conditions
    >>> raw = mne.io.read_raw_edf("sample.edf", preload=True)
    >>> results = other_conditions.analyze_insomnia_patterns(raw)
    >>> print(f"Insomnia patterns detected: {results['insomnia_detected']}")
    """
    # Define frequency bands
    frequency_bands = {
        "Delta": (0.5, 4),
        "Beta": (15, 30)
    }
    
    # Compute band powers
    band_powers = spectral.compute_band_powers(
        raw,
        frequency_bands=frequency_bands,
        **kwargs
    )
    
    # Calculate mean powers
    mean_delta = np.mean(band_powers["Delta"])
    mean_beta = np.mean(band_powers["Beta"])
    
    # Calculate beta/delta ratio
    beta_delta_ratio = mean_beta / (mean_delta + 1e-10)  # Add small constant to avoid division by zero
    
    # Check for insomnia patterns (elevated beta/delta ratio)
    insomnia_detected = beta_delta_ratio > 1.5  # Threshold can be adjusted based on research
    
    # Return the results
    return {
        "insomnia_detected": insomnia_detected,
        "band_powers": band_powers,
        "mean_delta": mean_delta,
        "mean_beta": mean_beta,
        "beta_delta_ratio": beta_delta_ratio
    }


def analyze_tbi_patterns(
    raw: mne.io.Raw,
    **kwargs
) -> Dict[str, Union[bool, Dict[str, np.ndarray], float]]:
    """
    Analyze EEG patterns associated with traumatic brain injury (TBI).
    
    TBI is often characterized by increased delta and theta activity,
    particularly in the region of the injury.
    
    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data.
    **kwargs
        Additional keyword arguments
        
    Returns
    -------
    Dict[str, Union[bool, Dict[str, np.ndarray], float]]
        Dictionary with TBI-related metrics and detection results.
        
    Examples
    --------
    >>> import mne
    >>> from qeeg.conditions import other_conditions
    >>> raw = mne.io.read_raw_edf("sample.edf", preload=True)
    >>> results = other_conditions.analyze_tbi_patterns(raw)
    >>> print(f"TBI patterns detected: {results['tbi_detected']}")
    """
    # Define frequency bands
    frequency_bands = {
        "Delta": (0.5, 4),
        "Theta": (4, 8),
        "Alpha": (8, 13),
        "Beta": (13, 30)
    }
    
    # Compute band powers
    band_powers = spectral.compute_band_powers(
        raw,
        frequency_bands=frequency_bands,
        **kwargs
    )
    
    # Calculate mean powers
    mean_delta = np.mean(band_powers["Delta"])
    mean_theta = np.mean(band_powers["Theta"])
    mean_alpha = np.mean(band_powers["Alpha"])
    mean_beta = np.mean(band_powers["Beta"])
    
    # Calculate slow-to-fast ratio (delta+theta)/(alpha+beta)
    slow_fast_ratio = (mean_delta + mean_theta) / (mean_alpha + mean_beta + 1e-10)
    
    # Check for TBI patterns (elevated slow-to-fast ratio)
    tbi_detected = slow_fast_ratio > 1.5  # Threshold can be adjusted based on research
    
    # Return the results
    return {
        "tbi_detected": tbi_detected,
        "band_powers": band_powers,
        "mean_delta": mean_delta,
        "mean_theta": mean_theta,
        "mean_alpha": mean_alpha,
        "mean_beta": mean_beta,
        "slow_fast_ratio": slow_fast_ratio
    }


def analyze_alzheimers_patterns(
    raw: mne.io.Raw,
    **kwargs
) -> Dict[str, Union[bool, Dict[str, np.ndarray], float]]:
    """
    Analyze EEG patterns associated with Alzheimer's disease.
    
    Alzheimer's disease is often characterized by decreased alpha and beta activity,
    and increased delta activity, particularly in temporal and parietal regions.
    
    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data.
    **kwargs
        Additional keyword arguments
        
    Returns
    -------
    Dict[str, Union[bool, Dict[str, np.ndarray], float]]
        Dictionary with Alzheimer's-related metrics and detection results.
        
    Examples
    --------
    >>> import mne
    >>> from qeeg.conditions import other_conditions
    >>> raw = mne.io.read_raw_edf("sample.edf", preload=True)
    >>> results = other_conditions.analyze_alzheimers_patterns(raw)
    >>> print(f"Alzheimer's patterns detected: {results['alzheimers_detected']}")
    """
    # Define frequency bands
    frequency_bands = {
        "Delta": (0.5, 4),
        "Theta": (4, 8),
        "Alpha": (8, 13),
        "Beta": (13, 30)
    }
    
    # Compute band powers
    band_powers = spectral.compute_band_powers(
        raw,
        frequency_bands=frequency_bands,
        **kwargs
    )
    
    # Calculate mean powers
    mean_delta = np.mean(band_powers["Delta"])
    mean_theta = np.mean(band_powers["Theta"])
    mean_alpha = np.mean(band_powers["Alpha"])
    mean_beta = np.mean(band_powers["Beta"])
    
    # Calculate delta/alpha ratio
    delta_alpha_ratio = mean_delta / (mean_alpha + 1e-10)
    
    # Check for Alzheimer's patterns (elevated delta/alpha ratio)
    alzheimers_detected = delta_alpha_ratio > 1.5  # Threshold can be adjusted based on research
    
    # Return the results
    return {
        "alzheimers_detected": alzheimers_detected,
        "band_powers": band_powers,
        "mean_delta": mean_delta,
        "mean_theta": mean_theta,
        "mean_alpha": mean_alpha,
        "mean_beta": mean_beta,
        "delta_alpha_ratio": delta_alpha_ratio
    }


def analyze_parkinsons_patterns(
    raw: mne.io.Raw,
    **kwargs
) -> Dict[str, Union[bool, Dict[str, np.ndarray], float]]:
    """
    Analyze EEG patterns associated with Parkinson's disease.
    
    Parkinson's disease is often characterized by increased beta activity,
    particularly in frontal and central regions.
    
    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data.
    **kwargs
        Additional keyword arguments
        
    Returns
    -------
    Dict[str, Union[bool, Dict[str, np.ndarray], float]]
        Dictionary with Parkinson's-related metrics and detection results.
        
    Examples
    --------
    >>> import mne
    >>> from qeeg.conditions import other_conditions
    >>> raw = mne.io.read_raw_edf("sample.edf", preload=True)
    >>> results = other_conditions.analyze_parkinsons_patterns(raw)
    >>> print(f"Parkinson's patterns detected: {results['parkinsons_detected']}")
    """
    # Define frequency bands with focus on beta
    frequency_bands = {
        "Beta": (13, 30),
        "Beta1": (13, 20),
        "Beta2": (20, 30)
    }
    
    # Compute band powers
    band_powers = spectral.compute_band_powers(
        raw,
        frequency_bands=frequency_bands,
        **kwargs
    )
    
    # Calculate mean beta power
    mean_beta = np.mean(band_powers["Beta"])
    mean_beta1 = np.mean(band_powers["Beta1"])
    mean_beta2 = np.mean(band_powers["Beta2"])
    
    # Calculate beta2/beta1 ratio (higher in Parkinson's)
    beta_ratio = mean_beta2 / (mean_beta1 + 1e-10)
    
    # Check for Parkinson's patterns (elevated beta2/beta1 ratio)
    parkinsons_detected = beta_ratio > 1.2  # Threshold can be adjusted based on research
    
    # Return the results
    return {
        "parkinsons_detected": parkinsons_detected,
        "band_powers": band_powers,
        "mean_beta": mean_beta,
        "mean_beta1": mean_beta1,
        "mean_beta2": mean_beta2,
        "beta_ratio": beta_ratio
    }


def analyze_schizophrenia_patterns(
    raw: mne.io.Raw,
    **kwargs
) -> Dict[str, Union[bool, Dict[str, np.ndarray], float]]:
    """
    Analyze EEG patterns associated with schizophrenia.
    
    Schizophrenia is often characterized by decreased alpha activity and
    increased delta and theta activity.
    
    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data.
    **kwargs
        Additional keyword arguments
        
    Returns
    -------
    Dict[str, Union[bool, Dict[str, np.ndarray], float]]
        Dictionary with schizophrenia-related metrics and detection results.
        
    Examples
    --------
    >>> import mne
    >>> from qeeg.conditions import other_conditions
    >>> raw = mne.io.read_raw_edf("sample.edf", preload=True)
    >>> results = other_conditions.analyze_schizophrenia_patterns(raw)
    >>> print(f"Schizophrenia patterns detected: {results['schizophrenia_detected']}")
    """
    # Define frequency bands
    frequency_bands = {
        "Delta": (0.5, 4),
        "Theta": (4, 8),
        "Alpha": (8, 13),
        "Beta": (13, 30)
    }
    
    # Compute band powers
    band_powers = spectral.compute_band_powers(
        raw,
        frequency_bands=frequency_bands,
        **kwargs
    )
    
    # Calculate mean powers
    mean_delta = np.mean(band_powers["Delta"])
    mean_theta = np.mean(band_powers["Theta"])
    mean_alpha = np.mean(band_powers["Alpha"])
    mean_beta = np.mean(band_powers["Beta"])
    
    # Calculate (delta+theta)/alpha ratio
    slow_alpha_ratio = (mean_delta + mean_theta) / (mean_alpha + 1e-10)
    
    # Check for schizophrenia patterns (elevated slow/alpha ratio)
    schizophrenia_detected = slow_alpha_ratio > 2.0  # Threshold can be adjusted based on research
    
    # Return the results
    return {
        "schizophrenia_detected": schizophrenia_detected,
        "band_powers": band_powers,
        "mean_delta": mean_delta,
        "mean_theta": mean_theta,
        "mean_alpha": mean_alpha,
        "mean_beta": mean_beta,
        "slow_alpha_ratio": slow_alpha_ratio
    }


def generate_condition_report(
    raw: mne.io.Raw,
    condition: str,
    **kwargs
) -> str:
    """
    Generate a report on EEG patterns associated with a specific condition.
    
    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data.
    condition : str
        The condition to analyze. Options: 'insomnia', 'tbi', 'alzheimers', 'parkinsons', 'schizophrenia'
    **kwargs
        Additional keyword arguments
        
    Returns
    -------
    str
        Report on condition-related EEG patterns.
        
    Examples
    --------
    >>> import mne
    >>> from qeeg.conditions import other_conditions
    >>> raw = mne.io.read_raw_edf("sample.edf", preload=True)
    >>> report = other_conditions.generate_condition_report(raw, 'insomnia')
    >>> print(report)
    """
    # Check if the condition is valid
    valid_conditions = ['insomnia', 'tbi', 'alzheimers', 'parkinsons', 'schizophrenia']
    if condition not in valid_conditions:
        raise ValueError(f"Invalid condition: {condition}. Valid options are: {', '.join(valid_conditions)}")
    
    # Analyze the condition
    if condition == 'insomnia':
        results = analyze_insomnia_patterns(raw, **kwargs)
        title = "Insomnia EEG Analysis Report"
        detected_key = "insomnia_detected"
    elif condition == 'tbi':
        results = analyze_tbi_patterns(raw, **kwargs)
        title = "Traumatic Brain Injury (TBI) EEG Analysis Report"
        detected_key = "tbi_detected"
    elif condition == 'alzheimers':
        results = analyze_alzheimers_patterns(raw, **kwargs)
        title = "Alzheimer's Disease EEG Analysis Report"
        detected_key = "alzheimers_detected"
    elif condition == 'parkinsons':
        results = analyze_parkinsons_patterns(raw, **kwargs)
        title = "Parkinson's Disease EEG Analysis Report"
        detected_key = "parkinsons_detected"
    elif condition == 'schizophrenia':
        results = analyze_schizophrenia_patterns(raw, **kwargs)
        title = "Schizophrenia EEG Analysis Report"
        detected_key = "schizophrenia_detected"
    
    # Generate the report
    report = []
    report.append(title)
    report.append("=" * len(title))
    report.append("")
    
    # Add the band power results
    if 'band_powers' in results:
        report.append("Frequency Band Powers:")
        for band, powers in results['band_powers'].items():
            report.append(f"  {band}: {np.mean(powers):.2f}")
        report.append("")
    
    # Add condition-specific metrics
    if condition == 'insomnia':
        report.append(f"Beta/Delta Ratio: {results['beta_delta_ratio']:.2f}")
        if results[detected_key]:
            report.append("Elevated Beta/Delta ratio detected, which may be associated with insomnia.")
        else:
            report.append("Beta/Delta ratio is within normal limits.")
    
    elif condition == 'tbi':
        report.append(f"Slow-to-Fast Ratio: {results['slow_fast_ratio']:.2f}")
        if results[detected_key]:
            report.append("Elevated Slow-to-Fast ratio detected, which may be associated with traumatic brain injury.")
        else:
            report.append("Slow-to-Fast ratio is within normal limits.")
    
    elif condition == 'alzheimers':
        report.append(f"Delta/Alpha Ratio: {results['delta_alpha_ratio']:.2f}")
        if results[detected_key]:
            report.append("Elevated Delta/Alpha ratio detected, which may be associated with Alzheimer's disease.")
        else:
            report.append("Delta/Alpha ratio is within normal limits.")
    
    elif condition == 'parkinsons':
        report.append(f"Beta2/Beta1 Ratio: {results['beta_ratio']:.2f}")
        if results[detected_key]:
            report.append("Elevated Beta2/Beta1 ratio detected, which may be associated with Parkinson's disease.")
        else:
            report.append("Beta2/Beta1 ratio is within normal limits.")
    
    elif condition == 'schizophrenia':
        report.append(f"(Delta+Theta)/Alpha Ratio: {results['slow_alpha_ratio']:.2f}")
        if results[detected_key]:
            report.append("Elevated (Delta+Theta)/Alpha ratio detected, which may be associated with schizophrenia.")
        else:
            report.append("(Delta+Theta)/Alpha ratio is within normal limits.")
    
    report.append("")
    
    # Add the overall result
    if results[detected_key]:
        report.append(f"EEG patterns consistent with {condition} were detected.")
    else:
        report.append(f"No significant {condition}-related EEG patterns were detected.")
    report.append("")
    
    # Add a disclaimer
    report.append("Disclaimer: This analysis is for research purposes only and should not be used for clinical diagnosis.")
    
    return "\n".join(report)
