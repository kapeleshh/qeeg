"""
Brodmann area analysis module for EEG signal processing.

This module provides functions for Brodmann area analysis, mapping EEG channels
to Brodmann areas, and analyzing activity in different Brodmann areas.
"""

import numpy as np
from typing import Tuple, Optional, Union, List, Dict
import mne


# Define mapping from 10-20 system electrodes to Brodmann areas
# This is a simplified mapping and should be used with caution
ELECTRODE_TO_BRODMANN = {
    'Fp1': [10],  # Frontopolar area
    'Fp2': [10],  # Frontopolar area
    'F7': [45, 47],  # Inferior frontal gyrus
    'F3': [8, 9],  # Dorsolateral prefrontal cortex
    'Fz': [8],  # Supplementary motor area
    'F4': [8, 9],  # Dorsolateral prefrontal cortex
    'F8': [45, 47],  # Inferior frontal gyrus
    'T3': [21, 22],  # Middle temporal gyrus
    'C3': [4, 6],  # Primary motor cortex, premotor cortex
    'Cz': [4, 6],  # Primary motor cortex, premotor cortex
    'C4': [4, 6],  # Primary motor cortex, premotor cortex
    'T4': [21, 22],  # Middle temporal gyrus
    'T5': [37, 39],  # Fusiform gyrus, angular gyrus
    'P3': [7, 40],  # Somatosensory association cortex
    'Pz': [7],  # Somatosensory association cortex
    'P4': [7, 40],  # Somatosensory association cortex
    'T6': [37, 39],  # Fusiform gyrus, angular gyrus
    'O1': [17, 18, 19],  # Primary visual cortex, visual association areas
    'Oz': [17, 18],  # Primary visual cortex, visual association areas
    'O2': [17, 18, 19]  # Primary visual cortex, visual association areas
}

# Define Brodmann area functions
BRODMANN_FUNCTIONS = {
    1: 'Primary somatosensory cortex',
    2: 'Primary somatosensory cortex',
    3: 'Primary somatosensory cortex',
    4: 'Primary motor cortex',
    5: 'Somatosensory association cortex',
    6: 'Premotor cortex and supplementary motor cortex',
    7: 'Somatosensory association cortex',
    8: 'Frontal eye fields',
    9: 'Dorsolateral prefrontal cortex',
    10: 'Frontopolar area',
    11: 'Orbitofrontal area',
    12: 'Orbitofrontal area',
    17: 'Primary visual cortex',
    18: 'Visual association cortex',
    19: 'Visual association cortex',
    20: 'Inferior temporal gyrus',
    21: 'Middle temporal gyrus',
    22: 'Superior temporal gyrus',
    23: 'Ventral posterior cingulate cortex',
    24: 'Ventral anterior cingulate cortex',
    25: 'Subgenual cortex',
    26: 'Ectosplenial area',
    27: 'Piriform cortex',
    28: 'Ventral entorhinal cortex',
    29: 'Retrosplenial cingulate cortex',
    30: 'Part of cingulate cortex',
    31: 'Dorsal posterior cingulate cortex',
    32: 'Dorsal anterior cingulate cortex',
    33: 'Part of anterior cingulate cortex',
    34: 'Dorsal entorhinal cortex',
    35: 'Perirhinal cortex',
    36: 'Parahippocampal cortex',
    37: 'Fusiform gyrus',
    38: 'Temporopolar area',
    39: 'Angular gyrus',
    40: 'Supramarginal gyrus',
    41: 'Primary auditory cortex',
    42: 'Auditory association cortex',
    43: 'Subcentral area',
    44: 'Pars opercularis (part of Broca\'s area)',
    45: 'Pars triangularis (part of Broca\'s area)',
    46: 'Dorsolateral prefrontal cortex',
    47: 'Inferior prefrontal gyrus',
    48: 'Retrosubicular area',
    52: 'Parainsular area'
}


def get_brodmann_areas_for_electrode(
    electrode: str
) -> List[int]:
    """
    Get the Brodmann areas associated with an electrode.

    Parameters
    ----------
    electrode : str
        The electrode name.

    Returns
    -------
    List[int]
        List of Brodmann area numbers.

    Examples
    --------
    >>> from epilepsy_eeg.analysis import brodmann
    >>> areas = brodmann.get_brodmann_areas_for_electrode('F3')
    >>> print(areas)
    """
    # Standardize electrode name
    electrode = electrode.upper()
    
    # Check if the electrode is in the mapping
    if electrode in ELECTRODE_TO_BRODMANN:
        return ELECTRODE_TO_BRODMANN[electrode]
    else:
        return []


def get_brodmann_function(
    area: int
) -> str:
    """
    Get the function of a Brodmann area.

    Parameters
    ----------
    area : int
        The Brodmann area number.

    Returns
    -------
    str
        The function of the Brodmann area.

    Examples
    --------
    >>> from epilepsy_eeg.analysis import brodmann
    >>> function = brodmann.get_brodmann_function(17)
    >>> print(function)
    """
    # Check if the area is in the mapping
    if area in BRODMANN_FUNCTIONS:
        return BRODMANN_FUNCTIONS[area]
    else:
        return "Unknown function"


def map_channels_to_brodmann(
    raw: mne.io.Raw,
    picks: Optional[Union[str, list]] = "eeg"
) -> Dict[str, List[int]]:
    """
    Map EEG channels to Brodmann areas.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data.
    picks : str or list, optional
        Channels to include, by default "eeg" (all EEG channels)

    Returns
    -------
    Dict[str, List[int]]
        Dictionary mapping channel names to lists of Brodmann area numbers.

    Examples
    --------
    >>> import mne
    >>> from epilepsy_eeg.analysis import brodmann
    >>> raw = mne.io.read_raw_edf("sample.edf", preload=True)
    >>> channel_to_areas = brodmann.map_channels_to_brodmann(raw)
    >>> print(channel_to_areas)
    """
    # Get the channel names
    ch_names = [raw.ch_names[idx] for idx in mne.pick_types(raw.info, meg=False, eeg=True, selection=picks)]
    
    # Initialize the result dictionary
    channel_to_areas = {}
    
    # Map each channel to Brodmann areas
    for ch_name in ch_names:
        # Standardize channel name
        std_name = ch_name.upper()
        
        # Extract the electrode name (remove prefixes like "EEG ")
        if " " in std_name:
            electrode = std_name.split(" ")[-1]
        else:
            electrode = std_name
        
        # Get the Brodmann areas for the electrode
        areas = get_brodmann_areas_for_electrode(electrode)
        
        # Store the result
        channel_to_areas[ch_name] = areas
    
    return channel_to_areas


def map_brodmann_to_channels(
    raw: mne.io.Raw,
    picks: Optional[Union[str, list]] = "eeg"
) -> Dict[int, List[str]]:
    """
    Map Brodmann areas to EEG channels.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data.
    picks : str or list, optional
        Channels to include, by default "eeg" (all EEG channels)

    Returns
    -------
    Dict[int, List[str]]
        Dictionary mapping Brodmann area numbers to lists of channel names.

    Examples
    --------
    >>> import mne
    >>> from epilepsy_eeg.analysis import brodmann
    >>> raw = mne.io.read_raw_edf("sample.edf", preload=True)
    >>> area_to_channels = brodmann.map_brodmann_to_channels(raw)
    >>> print(area_to_channels)
    """
    # Map channels to Brodmann areas
    channel_to_areas = map_channels_to_brodmann(raw, picks=picks)
    
    # Initialize the result dictionary
    area_to_channels = {}
    
    # Invert the mapping
    for ch_name, areas in channel_to_areas.items():
        for area in areas:
            if area not in area_to_channels:
                area_to_channels[area] = []
            area_to_channels[area].append(ch_name)
    
    return area_to_channels


def compute_brodmann_band_powers(
    raw: mne.io.Raw,
    frequency_bands: Optional[Dict[str, Tuple[float, float]]] = None,
    picks: Optional[Union[str, list]] = "eeg",
    **kwargs
) -> Dict[int, Dict[str, float]]:
    """
    Compute the power in each frequency band for each Brodmann area.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data.
    frequency_bands : Dict[str, Tuple[float, float]] or None, optional
        Dictionary mapping band names to (fmin, fmax) tuples, by default None
        If None, the standard frequency bands are used.
    picks : str or list, optional
        Channels to include, by default "eeg" (all EEG channels)
    **kwargs
        Additional keyword arguments to pass to compute_band_powers()

    Returns
    -------
    Dict[int, Dict[str, float]]
        Dictionary mapping Brodmann area numbers to dictionaries mapping band names to power values.

    Examples
    --------
    >>> import mne
    >>> from epilepsy_eeg.analysis import brodmann
    >>> raw = mne.io.read_raw_edf("sample.edf", preload=True)
    >>> powers = brodmann.compute_brodmann_band_powers(raw)
    >>> print(powers[17]['Alpha'])
    """
    # Use standard frequency bands if none are provided
    if frequency_bands is None:
        from epilepsy_eeg.analysis.spectral import FREQUENCY_BANDS
        frequency_bands = FREQUENCY_BANDS
    
    # Compute the power in each frequency band for each channel
    from epilepsy_eeg.analysis.spectral import compute_band_powers
    band_powers = compute_band_powers(raw, frequency_bands=frequency_bands, picks=picks, **kwargs)
    
    # Map channels to Brodmann areas
    channel_to_areas = map_channels_to_brodmann(raw, picks=picks)
    
    # Initialize the result dictionary
    brodmann_powers = {}
    
    # Process each channel
    ch_names = [raw.ch_names[idx] for idx in mne.pick_types(raw.info, meg=False, eeg=True, selection=picks)]
    for ch_idx, ch_name in enumerate(ch_names):
        # Get the Brodmann areas for the channel
        areas = channel_to_areas.get(ch_name, [])
        
        # Process each area
        for area in areas:
            # Initialize the area if it doesn't exist
            if area not in brodmann_powers:
                brodmann_powers[area] = {band: 0.0 for band in band_powers}
                brodmann_powers[area]['count'] = 0
            
            # Add the power for each band
            for band, powers in band_powers.items():
                brodmann_powers[area][band] += powers[ch_idx]
            
            # Increment the count
            brodmann_powers[area]['count'] += 1
    
    # Average the powers
    for area, powers in brodmann_powers.items():
        count = powers.pop('count')
        for band in powers:
            powers[band] /= count
    
    return brodmann_powers


def analyze_brodmann_asymmetry(
    raw: mne.io.Raw,
    frequency_bands: Optional[Dict[str, Tuple[float, float]]] = None,
    **kwargs
) -> Dict[str, Dict[int, float]]:
    """
    Analyze asymmetry in Brodmann areas.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data.
    frequency_bands : Dict[str, Tuple[float, float]] or None, optional
        Dictionary mapping band names to (fmin, fmax) tuples, by default None
        If None, the standard frequency bands are used.
    **kwargs
        Additional keyword arguments to pass to compute_brodmann_band_powers()

    Returns
    -------
    Dict[str, Dict[int, float]]
        Dictionary mapping band names to dictionaries mapping Brodmann area numbers to asymmetry indices.

    Examples
    --------
    >>> import mne
    >>> from epilepsy_eeg.analysis import brodmann
    >>> raw = mne.io.read_raw_edf("sample.edf", preload=True)
    >>> asymmetry = brodmann.analyze_brodmann_asymmetry(raw)
    >>> print(asymmetry['Alpha'][17])
    """
    # Use standard frequency bands if none are provided
    if frequency_bands is None:
        from epilepsy_eeg.analysis.spectral import FREQUENCY_BANDS
        frequency_bands = FREQUENCY_BANDS
    
    # Compute the power in each frequency band for each Brodmann area
    brodmann_powers = compute_brodmann_band_powers(raw, frequency_bands=frequency_bands, **kwargs)
    
    # Initialize the result dictionary
    asymmetry = {band: {} for band in frequency_bands}
    
    # Define pairs of homologous Brodmann areas
    homologous_pairs = [
        (1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12),
        (17, 18), (19, 20), (21, 22), (37, 39), (40, 41), (42, 43),
        (44, 45), (46, 47)
    ]
    
    # Process each band
    for band in frequency_bands:
        # Process each pair of homologous areas
        for left_area, right_area in homologous_pairs:
            # Check if both areas exist
            if left_area in brodmann_powers and right_area in brodmann_powers:
                # Get the powers
                left_power = brodmann_powers[left_area][band]
                right_power = brodmann_powers[right_area][band]
                
                # Compute the asymmetry index
                # AI = ln(right / left)
                # Positive values indicate more power in the right hemisphere
                # Negative values indicate more power in the left hemisphere
                asymmetry_index = np.log((right_power + 1e-10) / (left_power + 1e-10))
                
                # Store the result
                asymmetry[band][left_area] = asymmetry_index
                asymmetry[band][right_area] = -asymmetry_index
    
    return asymmetry


def analyze_brodmann_activity(
    raw: mne.io.Raw,
    frequency_bands: Optional[Dict[str, Tuple[float, float]]] = None,
    threshold: float = 1.5,
    **kwargs
) -> Dict[str, Dict[str, List[int]]]:
    """
    Analyze activity in Brodmann areas and identify areas with increased or reduced activity.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data.
    frequency_bands : Dict[str, Tuple[float, float]] or None, optional
        Dictionary mapping band names to (fmin, fmax) tuples, by default None
        If None, the standard frequency bands are used.
    threshold : float, optional
        Threshold for increased or reduced activity in standard deviations, by default 1.5
    **kwargs
        Additional keyword arguments to pass to compute_brodmann_band_powers()

    Returns
    -------
    Dict[str, Dict[str, List[int]]]
        Dictionary mapping band names to dictionaries with "increased" and "reduced" activity for each Brodmann area.

    Examples
    --------
    >>> import mne
    >>> from epilepsy_eeg.analysis import brodmann
    >>> raw = mne.io.read_raw_edf("sample.edf", preload=True)
    >>> activity = brodmann.analyze_brodmann_activity(raw)
    >>> print(activity['Alpha']['increased'])
    """
    # Use standard frequency bands if none are provided
    if frequency_bands is None:
        from epilepsy_eeg.analysis.spectral import FREQUENCY_BANDS
        frequency_bands = FREQUENCY_BANDS
    
    # Compute the power in each frequency band for each Brodmann area
    brodmann_powers = compute_brodmann_band_powers(raw, frequency_bands=frequency_bands, **kwargs)
    
    # Initialize the result dictionary
    activity = {band: {"increased": [], "reduced": []} for band in frequency_bands}
    
    # Process each band
    for band in frequency_bands:
        # Get the powers for the band
        powers = [brodmann_powers[area][band] for area in brodmann_powers]
        
        # Calculate the mean and standard deviation
        mean_power = np.mean(powers)
        std_power = np.std(powers)
        
        # Process each area
        for area, area_powers in brodmann_powers.items():
            # Get the power for the band
            power = area_powers[band]
            
            # Check if the power is increased or reduced
            if power > mean_power + threshold * std_power:
                activity[band]["increased"].append(area)
            elif power < mean_power - threshold * std_power:
                activity[band]["reduced"].append(area)
    
    return activity


def format_brodmann_results(
    activity: Dict[str, Dict[str, List[int]]],
    **kwargs
) -> Dict[str, Dict[str, List[str]]]:
    """
    Format Brodmann area activity results into human-readable strings.

    Parameters
    ----------
    activity : Dict[str, Dict[str, List[int]]]
        The activity results from analyze_brodmann_activity().
    **kwargs
        Additional keyword arguments

    Returns
    -------
    Dict[str, Dict[str, List[str]]]
        Dictionary mapping band names to dictionaries with "increased" and "reduced" activity as formatted strings.

    Examples
    --------
    >>> import mne
    >>> from epilepsy_eeg.analysis import brodmann
    >>> raw = mne.io.read_raw_edf("sample.edf", preload=True)
    >>> activity = brodmann.analyze_brodmann_activity(raw)
    >>> formatted = brodmann.format_brodmann_results(activity)
    >>> print(formatted['Alpha']['increased'])
    """
    # Initialize the result dictionary
    formatted = {band: {"increased": [], "reduced": []} for band in activity}
    
    # Process each band
    for band, band_activity in activity.items():
        # Process increased activity
        for area in band_activity["increased"]:
            # Get the function of the area
            function = get_brodmann_function(area)
            
            # Format the result
            formatted_result = f"Area {area} ({function})"
            formatted[band]["increased"].append(formatted_result)
        
        # Process reduced activity
        for area in band_activity["reduced"]:
            # Get the function of the area
            function = get_brodmann_function(area)
            
            # Format the result
            formatted_result = f"Area {area} ({function})"
            formatted[band]["reduced"].append(formatted_result)
    
    return formatted
