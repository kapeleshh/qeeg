"""
Asymmetry analysis module for EEG signal processing.

This module provides functions for asymmetry analysis of EEG signals,
including left-right hemisphere power asymmetry detection.
"""

import numpy as np
from typing import Tuple, Optional, Union, List, Dict
import mne
from scipy.signal import welch


def compute_asymmetry_index(
    raw: mne.io.Raw,
    electrode_pairs: Optional[List[Tuple[str, str]]] = None,
    fmin: float = 1.5,
    fmax: float = 40.0,
    **kwargs
) -> Dict[str, float]:
    """
    Compute the asymmetry index for pairs of electrodes.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data.
    electrode_pairs : List[Tuple[str, str]] or None, optional
        List of electrode pairs (left, right), by default None
        If None, standard homologous pairs are used.
    fmin : float, optional
        Minimum frequency of interest, by default 1.5
    fmax : float, optional
        Maximum frequency of interest, by default 40.0
    **kwargs
        Additional keyword arguments to pass to mne.time_frequency.psd_array_welch()

    Returns
    -------
    Dict[str, float]
        Dictionary mapping pair names to asymmetry indices.

    Examples
    --------
    >>> import mne
    >>> from qeeg.analysis import asymmetry
    >>> raw = mne.io.read_raw_edf("sample.edf", preload=True)
    >>> asymmetry_indices = asymmetry.compute_asymmetry_index(raw)
    >>> print(asymmetry_indices['F3-F4'])
    """
    # Define standard homologous pairs if none are provided
    if electrode_pairs is None:
        electrode_pairs = [
            ('F3', 'F4'),
            ('F7', 'F8'),
            ('C3', 'C4'),
            ('T3', 'T4'),
            ('T5', 'T6'),
            ('P3', 'P4'),
            ('O1', 'O2')
        ]
    
    # Get the data
    data = raw.get_data()
    
    # Compute the PSD
    psds, freqs = mne.time_frequency.psd_array_welch(
        data,
        sfreq=raw.info['sfreq'],
        fmin=fmin,
        fmax=fmax,
        **kwargs
    )
    
    # Initialize the result dictionary
    asymmetry_indices = {}
    
    # Compute the asymmetry index for each pair
    for left_electrode, right_electrode in electrode_pairs:
        # Check if the electrodes exist
        if left_electrode not in raw.ch_names or right_electrode not in raw.ch_names:
            continue
        
        # Get the indices of the electrodes
        left_idx = raw.ch_names.index(left_electrode)
        right_idx = raw.ch_names.index(right_electrode)
        
        # Compute the power in the frequency range
        range_idx = np.logical_and(freqs >= fmin, freqs <= fmax)
        left_power = np.sum(psds[left_idx, range_idx])
        right_power = np.sum(psds[right_idx, range_idx])
        
        # Compute the asymmetry index
        # AI = ln(right / left)
        # Positive values indicate more power in the right hemisphere
        # Negative values indicate more power in the left hemisphere
        asymmetry_index = np.log((right_power + 1e-10) / (left_power + 1e-10))
        
        # Store the result
        pair_name = f"{left_electrode}-{right_electrode}"
        asymmetry_indices[pair_name] = asymmetry_index
    
    return asymmetry_indices


def compute_band_asymmetry(
    raw: mne.io.Raw,
    frequency_bands: Optional[Dict[str, Tuple[float, float]]] = None,
    electrode_pairs: Optional[List[Tuple[str, str]]] = None,
    **kwargs
) -> Dict[str, Dict[str, float]]:
    """
    Compute the asymmetry index for pairs of electrodes in each frequency band.

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
        Additional keyword arguments to pass to compute_asymmetry_index()

    Returns
    -------
    Dict[str, Dict[str, float]]
        Dictionary mapping band names to dictionaries mapping pair names to asymmetry indices.

    Examples
    --------
    >>> import mne
    >>> from qeeg.analysis import asymmetry
    >>> raw = mne.io.read_raw_edf("sample.edf", preload=True)
    >>> band_asymmetry = asymmetry.compute_band_asymmetry(raw)
    >>> print(band_asymmetry['Alpha']['F3-F4'])
    """
    # Use standard frequency bands if none are provided
    if frequency_bands is None:
        from qeeg.analysis.spectral import FREQUENCY_BANDS
        frequency_bands = FREQUENCY_BANDS
    
    # Initialize the result dictionary
    band_asymmetry = {}
    
    # Compute the asymmetry index for each frequency band
    for band_name, (fmin, fmax) in frequency_bands.items():
        # Compute the asymmetry index for the band
        asymmetry_indices = compute_asymmetry_index(
            raw,
            electrode_pairs=electrode_pairs,
            fmin=fmin,
            fmax=fmax,
            **kwargs
        )
        
        # Store the result
        band_asymmetry[band_name] = asymmetry_indices
    
    return band_asymmetry


def classify_asymmetry_severity(
    asymmetry_index: float,
    mild_threshold: float = 0.2,
    moderate_threshold: float = 0.4
) -> str:
    """
    Classify the severity of asymmetry based on the asymmetry index.

    Parameters
    ----------
    asymmetry_index : float
        The asymmetry index.
    mild_threshold : float, optional
        Threshold for mild asymmetry, by default 0.2
    moderate_threshold : float, optional
        Threshold for moderate asymmetry, by default 0.4

    Returns
    -------
    str
        The severity classification: "None", "Mild", "Moderate", or "Severe".

    Examples
    --------
    >>> from qeeg.analysis import asymmetry
    >>> severity = asymmetry.classify_asymmetry_severity(0.35)
    >>> print(severity)
    """
    # Classify the severity based on the absolute value of the asymmetry index
    abs_ai = abs(asymmetry_index)
    
    if abs_ai < mild_threshold:
        return "None"
    elif abs_ai < moderate_threshold:
        return "Mild"
    elif abs_ai < 2 * moderate_threshold:
        return "Moderate"
    else:
        return "Severe"


def analyze_asymmetry(
    raw: mne.io.Raw,
    frequency_bands: Optional[Dict[str, Tuple[float, float]]] = None,
    electrode_pairs: Optional[List[Tuple[str, str]]] = None,
    asymmetry_threshold: float = 0.1,
    **kwargs
) -> Dict[str, List[Dict[str, Union[str, float, str]]]]:
    """
    Analyze asymmetry across frequency bands and identify significant asymmetries.

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
    asymmetry_threshold : float, optional
        Threshold for significant asymmetry, by default 0.1
    **kwargs
        Additional keyword arguments to pass to compute_band_asymmetry()

    Returns
    -------
    Dict[str, List[Dict[str, Union[str, float, str]]]]
        Dictionary mapping band names to lists of significant asymmetries.

    Examples
    --------
    >>> import mne
    >>> from qeeg.analysis import asymmetry
    >>> raw = mne.io.read_raw_edf("sample.edf", preload=True)
    >>> asymmetry_results = asymmetry.analyze_asymmetry(raw)
    >>> print(asymmetry_results['Alpha'])
    """
    # Compute the asymmetry index for each frequency band
    band_asymmetry = compute_band_asymmetry(
        raw,
        frequency_bands=frequency_bands,
        electrode_pairs=electrode_pairs,
        **kwargs
    )
    
    # Initialize the result dictionary
    asymmetry_results = {}
    
    # Identify significant asymmetries
    for band_name, asymmetry_indices in band_asymmetry.items():
        # Initialize the list of significant asymmetries for the band
        significant_asymmetries = []
        
        # Check each pair for significant asymmetry
        for pair_name, asymmetry_index in asymmetry_indices.items():
            # Check if the asymmetry is significant
            if abs(asymmetry_index) > asymmetry_threshold:
                # Classify the severity
                severity = classify_asymmetry_severity(asymmetry_index)
                
                # Determine the direction of asymmetry
                direction = "right > left" if asymmetry_index > 0 else "left > right"
                
                # Store the result
                significant_asymmetries.append({
                    "pair": pair_name,
                    "asymmetry_index": asymmetry_index,
                    "severity": severity,
                    "direction": direction
                })
        
        # Store the result for the band
        asymmetry_results[band_name] = significant_asymmetries
    
    return asymmetry_results


def get_zone_from_electrode(electrode: str) -> str:
    """
    Get the zone name for an electrode based on its name.

    Parameters
    ----------
    electrode : str
        The electrode name.

    Returns
    -------
    str
        The zone name: "Frontal", "Central", "Temporal", "Parietal", or "Occipital".

    Examples
    --------
    >>> from qeeg.analysis import asymmetry
    >>> zone = asymmetry.get_zone_from_electrode('F3')
    >>> print(zone)
    """
    # Map electrode prefixes to zones
    prefix_to_zone = {
        'F': 'Frontal',
        'C': 'Central',
        'T': 'Temporal',
        'P': 'Parietal',
        'O': 'Occipital'
    }
    
    # Handle special cases
    if electrode.startswith('Fp'):
        return 'Frontal'
    
    # Get the zone based on the first character
    first_char = electrode[0]
    return prefix_to_zone.get(first_char, 'Unknown')


def format_asymmetry_results(
    asymmetry_results: Dict[str, List[Dict[str, Union[str, float, str]]]],
    **kwargs
) -> Dict[str, List[str]]:
    """
    Format asymmetry results into human-readable strings.

    Parameters
    ----------
    asymmetry_results : Dict[str, List[Dict[str, Union[str, float, str]]]]
        The asymmetry results from analyze_asymmetry().
    **kwargs
        Additional keyword arguments

    Returns
    -------
    Dict[str, List[str]]
        Dictionary mapping band names to lists of formatted asymmetry strings.

    Examples
    --------
    >>> import mne
    >>> from qeeg.analysis import asymmetry
    >>> raw = mne.io.read_raw_edf("sample.edf", preload=True)
    >>> asymmetry_results = asymmetry.analyze_asymmetry(raw)
    >>> formatted_results = asymmetry.format_asymmetry_results(asymmetry_results)
    >>> print(formatted_results['Alpha'])
    """
    # Initialize the result dictionary
    formatted_results = {}
    
    # Format the results for each band
    for band_name, asymmetries in asymmetry_results.items():
        # Initialize the list of formatted asymmetries for the band
        formatted_asymmetries = []
        
        # Group asymmetries by zone
        zone_asymmetries = {}
        for asymmetry in asymmetries:
            # Get the pair name
            pair_name = asymmetry['pair']
            
            # Extract the electrodes
            left_electrode, right_electrode = pair_name.split('-')
            
            # Get the zone
            zone = get_zone_from_electrode(left_electrode)
            
            # Add the asymmetry to the zone
            if zone not in zone_asymmetries:
                zone_asymmetries[zone] = []
            zone_asymmetries[zone].append(asymmetry)
        
        # Format the asymmetries for each zone
        for zone, zone_asyms in zone_asymmetries.items():
            # Format the asymmetries
            asymmetry_strings = []
            for asymmetry in zone_asyms:
                pair_name = asymmetry['pair']
                severity = asymmetry['severity']
                ai = asymmetry['asymmetry_index']
                asymmetry_strings.append(f"{pair_name} ({severity} AI: {ai:.3f})")
            
            # Join the asymmetry strings
            asymmetry_string = ", ".join(asymmetry_strings)
            
            # Format the zone asymmetry
            formatted_asymmetry = f"{zone}: {asymmetry_string}"
            formatted_asymmetries.append(formatted_asymmetry)
        
        # Store the result for the band
        formatted_results[band_name] = formatted_asymmetries
    
    return formatted_results
