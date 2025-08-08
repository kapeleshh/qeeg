"""
Montage module for EEG signal processing.

This module provides functions for EEG montage setup, channel selection,
and reference setting.
"""

import numpy as np
from typing import Tuple, Optional, Union, List, Dict
import mne


def set_standard_montage(
    raw: mne.io.Raw,
    montage_name: str = 'standard_1020',
    **kwargs
) -> mne.io.Raw:
    """
    Set a standard montage for the EEG data.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data.
    montage_name : str, optional
        Name of the standard montage, by default 'standard_1020'
    **kwargs
        Additional keyword arguments to pass to mne.channels.make_standard_montage()

    Returns
    -------
    mne.io.Raw
        The raw EEG data with the montage set.

    Examples
    --------
    >>> import mne
    >>> from qeeg.preprocessing import montage
    >>> raw = mne.io.read_raw_edf("sample.edf", preload=True)
    >>> raw = montage.set_standard_montage(raw)
    """
    # Create a copy of the raw data
    raw_copy = raw.copy()
    
    # Create the standard montage
    standard_montage = mne.channels.make_standard_montage(montage_name, **kwargs)
    
    # Set the montage
    raw_copy.set_montage(standard_montage)
    
    return raw_copy


def select_channels(
    raw: mne.io.Raw,
    channels: List[str],
    **kwargs
) -> mne.io.Raw:
    """
    Select specific channels from the EEG data.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data.
    channels : List[str]
        List of channel names to select.
    **kwargs
        Additional keyword arguments to pass to mne.io.Raw.pick_channels()

    Returns
    -------
    mne.io.Raw
        The raw EEG data with only the selected channels.

    Examples
    --------
    >>> import mne
    >>> from qeeg.preprocessing import montage
    >>> raw = mne.io.read_raw_edf("sample.edf", preload=True)
    >>> channels = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8']
    >>> raw = montage.select_channels(raw, channels)
    """
    # Create a copy of the raw data
    raw_copy = raw.copy()
    
    # Select the channels
    raw_copy.pick_channels(channels, **kwargs)
    
    return raw_copy


def set_channel_types(
    raw: mne.io.Raw,
    channel_types: Dict[str, str],
    **kwargs
) -> mne.io.Raw:
    """
    Set the types of specific channels in the EEG data.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data.
    channel_types : Dict[str, str]
        Dictionary mapping channel names to their types.
    **kwargs
        Additional keyword arguments to pass to mne.io.Raw.set_channel_types()

    Returns
    -------
    mne.io.Raw
        The raw EEG data with updated channel types.

    Examples
    --------
    >>> import mne
    >>> from qeeg.preprocessing import montage
    >>> raw = mne.io.read_raw_edf("sample.edf", preload=True)
    >>> channel_types = {'ECG': 'ecg', 'EOG': 'eog'}
    >>> raw = montage.set_channel_types(raw, channel_types)
    """
    # Create a copy of the raw data
    raw_copy = raw.copy()
    
    # Set the channel types
    raw_copy.set_channel_types(channel_types, **kwargs)
    
    return raw_copy


def set_reference(
    raw: mne.io.Raw,
    ref_channels: Union[str, List[str]] = 'average',
    projection: bool = True,
    **kwargs
) -> mne.io.Raw:
    """
    Set the reference for the EEG data.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data.
    ref_channels : str or List[str], optional
        Reference channel(s), by default 'average'
    projection : bool, optional
        Whether to use projection, by default True
    **kwargs
        Additional keyword arguments to pass to mne.io.Raw.set_eeg_reference()

    Returns
    -------
    mne.io.Raw
        The raw EEG data with the reference set.

    Examples
    --------
    >>> import mne
    >>> from qeeg.preprocessing import montage
    >>> raw = mne.io.read_raw_edf("sample.edf", preload=True)
    >>> raw = montage.set_reference(raw, ref_channels='average')
    """
    # Create a copy of the raw data
    raw_copy = raw.copy()
    
    # Set the reference
    raw_copy.set_eeg_reference(ref_channels, projection=projection, **kwargs)
    
    return raw_copy


def rename_channels(
    raw: mne.io.Raw,
    mapping: Dict[str, str],
    **kwargs
) -> mne.io.Raw:
    """
    Rename channels in the EEG data.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data.
    mapping : Dict[str, str]
        Dictionary mapping old channel names to new channel names.
    **kwargs
        Additional keyword arguments to pass to mne.io.Raw.rename_channels()

    Returns
    -------
    mne.io.Raw
        The raw EEG data with renamed channels.

    Examples
    --------
    >>> import mne
    >>> from qeeg.preprocessing import montage
    >>> raw = mne.io.read_raw_edf("sample.edf", preload=True)
    >>> mapping = {'EEG Fp1': 'Fp1', 'EEG Fp2': 'Fp2'}
    >>> raw = montage.rename_channels(raw, mapping)
    """
    # Create a copy of the raw data
    raw_copy = raw.copy()
    
    # Rename the channels
    raw_copy.rename_channels(mapping, **kwargs)
    
    return raw_copy


def setup_standard_montage(
    raw: mne.io.Raw,
    montage_name: str = 'standard_1020',
    eeg_prefix: Optional[str] = 'EEG ',
    non_eeg_channels: Optional[List[str]] = None,
    desired_channels: Optional[List[str]] = None,
    **kwargs
) -> mne.io.Raw:
    """
    Set up a standard EEG montage with common preprocessing steps.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data.
    montage_name : str, optional
        Name of the standard montage, by default 'standard_1020'
    eeg_prefix : str or None, optional
        Prefix to remove from EEG channel names, by default 'EEG '
    non_eeg_channels : List[str] or None, optional
        List of non-EEG channels to set as misc, by default None
    desired_channels : List[str] or None, optional
        List of channels to select, by default None
    **kwargs
        Additional keyword arguments

    Returns
    -------
    mne.io.Raw
        The raw EEG data with the standard montage set up.

    Examples
    --------
    >>> import mne
    >>> from qeeg.preprocessing import montage
    >>> raw = mne.io.read_raw_edf("sample.edf", preload=True)
    >>> desired_channels = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8']
    >>> raw = montage.setup_standard_montage(raw, desired_channels=desired_channels)
    """
    # Create a copy of the raw data
    raw_copy = raw.copy()
    
    # Step 1: Rename EEG channels if prefix is specified
    if eeg_prefix is not None:
        mapping = {ch: ch.replace(eeg_prefix, '') for ch in raw_copy.ch_names if eeg_prefix in ch}
        if mapping:
            raw_copy = rename_channels(raw_copy, mapping)
    
    # Step 2: Set non-EEG channels as misc
    if non_eeg_channels is not None:
        channel_types = {ch: 'misc' for ch in non_eeg_channels if ch in raw_copy.ch_names}
        if channel_types:
            raw_copy = set_channel_types(raw_copy, channel_types)
    
    # Step 3: Select desired channels
    if desired_channels is not None:
        # Filter out channels that don't exist in the raw data
        available_channels = [ch for ch in desired_channels if ch in raw_copy.ch_names]
        if available_channels:
            raw_copy = select_channels(raw_copy, available_channels)
    
    # Step 4: Set the standard montage
    raw_copy = set_standard_montage(raw_copy, montage_name)
    
    return raw_copy


def get_channel_positions(
    raw: mne.io.Raw,
    **kwargs
) -> Dict[str, np.ndarray]:
    """
    Get the 3D positions of EEG channels.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data with a montage set.
    **kwargs
        Additional keyword arguments

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary mapping channel names to their 3D positions.

    Examples
    --------
    >>> import mne
    >>> from qeeg.preprocessing import montage
    >>> raw = mne.io.read_raw_edf("sample.edf", preload=True)
    >>> raw = montage.set_standard_montage(raw)
    >>> positions = montage.get_channel_positions(raw)
    """
    # Get the montage
    montage = raw.get_montage()
    
    if montage is None:
        raise ValueError("No montage set for the raw data. Use set_standard_montage() first.")
    
    # Get the positions
    positions = montage.get_positions()
    
    if positions is None or 'ch_pos' not in positions:
        raise ValueError("No channel positions found in the montage.")
    
    # Extract channel positions
    channel_positions = {ch: positions['ch_pos'][ch] for ch in raw.ch_names if ch in positions['ch_pos']}
    
    return channel_positions


def define_zones(
    raw: mne.io.Raw,
    **kwargs
) -> Dict[str, List[str]]:
    """
    Define standard zones based on channel names.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data.
    **kwargs
        Additional keyword arguments

    Returns
    -------
    Dict[str, List[str]]
        Dictionary mapping zone names to lists of channel names.

    Examples
    --------
    >>> import mne
    >>> from qeeg.preprocessing import montage
    >>> raw = mne.io.read_raw_edf("sample.edf", preload=True)
    >>> zones = montage.define_zones(raw)
    >>> print(zones)
    """
    # Define standard zones
    zones = {
        'Frontal': [],
        'Temporal': [],
        'Central': [],
        'Parietal': [],
        'Occipital': []
    }
    
    # Map channels to zones based on their names
    for ch in raw.ch_names:
        if ch.startswith('Fp') or ch.startswith('F') or ch == 'Fz':
            zones['Frontal'].append(ch)
        elif ch.startswith('T'):
            zones['Temporal'].append(ch)
        elif ch.startswith('C') or ch == 'Cz':
            zones['Central'].append(ch)
        elif ch.startswith('P') or ch == 'Pz':
            zones['Parietal'].append(ch)
        elif ch.startswith('O') or ch == 'Oz':
            zones['Occipital'].append(ch)
    
    # Remove empty zones
    zones = {zone: channels for zone, channels in zones.items() if channels}
    
    return zones
