"""
Epileptiform activity detection module for EEG signal processing.

This module provides functions for detecting epileptiform activity in EEG signals,
such as spikes, OIRDA (Occipital Intermittent Rhythmic Delta Activity),
and FIRDA (Frontal Intermittent Rhythmic Delta Activity).
"""

import numpy as np
from typing import Tuple, Optional, Union, List, Dict
import mne
from scipy.signal import find_peaks
import pywt


def detect_spikes(
    raw: mne.io.Raw,
    threshold: float = 3.0,
    min_duration: float = 0.02,
    max_duration: float = 0.2,
    picks: Optional[Union[str, list]] = "eeg",
    **kwargs
) -> List[Dict[str, Union[str, float, int]]]:
    """
    Detect spikes in the EEG data.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data.
    threshold : float, optional
        Threshold for spike detection in standard deviations, by default 3.0
    min_duration : float, optional
        Minimum duration of a spike in seconds, by default 0.02
    max_duration : float, optional
        Maximum duration of a spike in seconds, by default 0.2
    picks : str or list, optional
        Channels to include, by default "eeg" (all EEG channels)
    **kwargs
        Additional keyword arguments

    Returns
    -------
    List[Dict[str, Union[str, float, int]]]
        List of detected spikes with channel, onset, duration, and amplitude.

    Examples
    --------
    >>> import mne
    >>> from epilepsy_eeg.analysis import epileptiform
    >>> raw = mne.io.read_raw_edf("sample.edf", preload=True)
    >>> spikes = epileptiform.detect_spikes(raw)
    >>> print(f"Detected {len(spikes)} spikes.")
    """
    # Get the data and channel names
    data = raw.get_data(picks=picks)
    ch_names = [raw.ch_names[idx] for idx in mne.pick_types(raw.info, meg=False, eeg=True)]
    
    # Calculate the sampling frequency
    sfreq = raw.info['sfreq']
    
    # Calculate the minimum and maximum number of samples for a spike
    min_samples = int(min_duration * sfreq)
    max_samples = int(max_duration * sfreq)
    
    # Initialize the list of detected spikes
    spikes = []
    
    # Process each channel
    for ch_idx, ch_name in enumerate(ch_names):
        # Get the channel data
        ch_data = data[ch_idx]
        
        # Calculate the mean and standard deviation
        mean = np.mean(ch_data)
        std = np.std(ch_data)
        
        # Find peaks above the threshold
        peaks, properties = find_peaks(
            np.abs(ch_data - mean),
            height=threshold * std,
            width=(min_samples, max_samples),
            **kwargs
        )
        
        # Process each peak
        for peak_idx, peak in enumerate(peaks):
            # Calculate the onset and duration
            onset = peak / sfreq
            width = properties['widths'][peak_idx] / sfreq
            
            # Calculate the amplitude
            amplitude = ch_data[peak] - mean
            
            # Store the spike
            spikes.append({
                'channel': ch_name,
                'onset': onset,
                'duration': width,
                'amplitude': amplitude
            })
    
    return spikes


def detect_rhythmic_delta(
    raw: mne.io.Raw,
    zone: str = 'Occipital',
    threshold: float = 2.0,
    min_duration: float = 0.5,
    **kwargs
) -> List[Dict[str, Union[str, float, int]]]:
    """
    Detect rhythmic delta activity in the EEG data.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data.
    zone : str, optional
        The zone to analyze, by default 'Occipital'
        Use 'Occipital' for OIRDA or 'Frontal' for FIRDA.
    threshold : float, optional
        Threshold for delta activity detection in standard deviations, by default 2.0
    min_duration : float, optional
        Minimum duration of rhythmic delta activity in seconds, by default 0.5
    **kwargs
        Additional keyword arguments

    Returns
    -------
    List[Dict[str, Union[str, float, int]]]
        List of detected rhythmic delta activities with channel, onset, duration, and power.

    Examples
    --------
    >>> import mne
    >>> from epilepsy_eeg.analysis import epileptiform
    >>> raw = mne.io.read_raw_edf("sample.edf", preload=True)
    >>> oirda = epileptiform.detect_rhythmic_delta(raw, zone='Occipital')
    >>> print(f"Detected {len(oirda)} OIRDA events.")
    """
    # Define zones
    from epilepsy_eeg.preprocessing.montage import define_zones
    zones = define_zones(raw)
    
    # Check if the zone exists
    if zone not in zones:
        raise ValueError(f"Zone '{zone}' not found in the EEG data.")
    
    # Get the channels in the zone
    zone_channels = zones[zone]
    
    # Get the data for the zone channels
    picks = mne.pick_channels(raw.ch_names, include=zone_channels)
    data = raw.get_data(picks=picks)
    
    # Calculate the sampling frequency
    sfreq = raw.info['sfreq']
    
    # Filter the data to the delta band (1-4 Hz)
    filtered_data = mne.filter.filter_data(
        data,
        sfreq=sfreq,
        l_freq=1.0,
        h_freq=4.0,
        **kwargs
    )
    
    # Calculate the delta power
    delta_power = np.mean(filtered_data ** 2, axis=0)
    
    # Calculate the mean and standard deviation of the delta power
    mean_power = np.mean(delta_power)
    std_power = np.std(delta_power)
    
    # Find segments with high delta power
    high_delta = delta_power > (mean_power + threshold * std_power)
    
    # Find the onsets and offsets of high delta segments
    onsets = np.where(np.diff(high_delta.astype(int)) == 1)[0] + 1
    offsets = np.where(np.diff(high_delta.astype(int)) == -1)[0] + 1
    
    # Ensure the same number of onsets and offsets
    if len(onsets) > len(offsets):
        onsets = onsets[:len(offsets)]
    elif len(offsets) > len(onsets):
        offsets = offsets[:len(onsets)]
    
    # Calculate the minimum number of samples for rhythmic delta activity
    min_samples = int(min_duration * sfreq)
    
    # Initialize the list of detected rhythmic delta activities
    rhythmic_delta = []
    
    # Process each segment
    for onset, offset in zip(onsets, offsets):
        # Check if the segment is long enough
        if offset - onset >= min_samples:
            # Calculate the onset and duration
            onset_time = onset / sfreq
            duration = (offset - onset) / sfreq
            
            # Calculate the power
            power = np.mean(delta_power[onset:offset])
            
            # Store the rhythmic delta activity
            rhythmic_delta.append({
                'zone': zone,
                'onset': onset_time,
                'duration': duration,
                'power': power
            })
    
    return rhythmic_delta


def detect_oirda(
    raw: mne.io.Raw,
    **kwargs
) -> List[Dict[str, Union[str, float, int]]]:
    """
    Detect Occipital Intermittent Rhythmic Delta Activity (OIRDA) in the EEG data.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data.
    **kwargs
        Additional keyword arguments to pass to detect_rhythmic_delta()

    Returns
    -------
    List[Dict[str, Union[str, float, int]]]
        List of detected OIRDA events with zone, onset, duration, and power.

    Examples
    --------
    >>> import mne
    >>> from epilepsy_eeg.analysis import epileptiform
    >>> raw = mne.io.read_raw_edf("sample.edf", preload=True)
    >>> oirda = epileptiform.detect_oirda(raw)
    >>> print(f"Detected {len(oirda)} OIRDA events.")
    """
    return detect_rhythmic_delta(raw, zone='Occipital', **kwargs)


def detect_firda(
    raw: mne.io.Raw,
    **kwargs
) -> List[Dict[str, Union[str, float, int]]]:
    """
    Detect Frontal Intermittent Rhythmic Delta Activity (FIRDA) in the EEG data.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data.
    **kwargs
        Additional keyword arguments to pass to detect_rhythmic_delta()

    Returns
    -------
    List[Dict[str, Union[str, float, int]]]
        List of detected FIRDA events with zone, onset, duration, and power.

    Examples
    --------
    >>> import mne
    >>> from epilepsy_eeg.analysis import epileptiform
    >>> raw = mne.io.read_raw_edf("sample.edf", preload=True)
    >>> firda = epileptiform.detect_firda(raw)
    >>> print(f"Detected {len(firda)} FIRDA events.")
    """
    return detect_rhythmic_delta(raw, zone='Frontal', **kwargs)


def detect_sharp_waves(
    raw: mne.io.Raw,
    threshold: float = 2.5,
    min_duration: float = 0.07,
    max_duration: float = 0.2,
    picks: Optional[Union[str, list]] = "eeg",
    **kwargs
) -> List[Dict[str, Union[str, float, int]]]:
    """
    Detect sharp waves in the EEG data.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data.
    threshold : float, optional
        Threshold for sharp wave detection in standard deviations, by default 2.5
    min_duration : float, optional
        Minimum duration of a sharp wave in seconds, by default 0.07
    max_duration : float, optional
        Maximum duration of a sharp wave in seconds, by default 0.2
    picks : str or list, optional
        Channels to include, by default "eeg" (all EEG channels)
    **kwargs
        Additional keyword arguments

    Returns
    -------
    List[Dict[str, Union[str, float, int]]]
        List of detected sharp waves with channel, onset, duration, and amplitude.

    Examples
    --------
    >>> import mne
    >>> from epilepsy_eeg.analysis import epileptiform
    >>> raw = mne.io.read_raw_edf("sample.edf", preload=True)
    >>> sharp_waves = epileptiform.detect_sharp_waves(raw)
    >>> print(f"Detected {len(sharp_waves)} sharp waves.")
    """
    # Get the data and channel names
    data = raw.get_data(picks=picks)
    ch_names = [raw.ch_names[idx] for idx in mne.pick_types(raw.info, meg=False, eeg=True)]
    
    # Calculate the sampling frequency
    sfreq = raw.info['sfreq']
    
    # Calculate the minimum and maximum number of samples for a sharp wave
    min_samples = int(min_duration * sfreq)
    max_samples = int(max_duration * sfreq)
    
    # Initialize the list of detected sharp waves
    sharp_waves = []
    
    # Process each channel
    for ch_idx, ch_name in enumerate(ch_names):
        # Get the channel data
        ch_data = data[ch_idx]
        
        # Calculate the mean and standard deviation
        mean = np.mean(ch_data)
        std = np.std(ch_data)
        
        # Find peaks above the threshold
        peaks, properties = find_peaks(
            np.abs(ch_data - mean),
            height=threshold * std,
            width=(min_samples, max_samples),
            prominence=threshold * std,
            **kwargs
        )
        
        # Process each peak
        for peak_idx, peak in enumerate(peaks):
            # Calculate the onset and duration
            onset = peak / sfreq
            width = properties['widths'][peak_idx] / sfreq
            
            # Calculate the amplitude
            amplitude = ch_data[peak] - mean
            
            # Store the sharp wave
            sharp_waves.append({
                'channel': ch_name,
                'onset': onset,
                'duration': width,
                'amplitude': amplitude
            })
    
    return sharp_waves


def detect_spike_wave_complexes(
    raw: mne.io.Raw,
    threshold: float = 3.0,
    min_duration: float = 0.2,
    max_duration: float = 0.5,
    picks: Optional[Union[str, list]] = "eeg",
    **kwargs
) -> List[Dict[str, Union[str, float, int]]]:
    """
    Detect spike-wave complexes in the EEG data.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data.
    threshold : float, optional
        Threshold for spike-wave complex detection in standard deviations, by default 3.0
    min_duration : float, optional
        Minimum duration of a spike-wave complex in seconds, by default 0.2
    max_duration : float, optional
        Maximum duration of a spike-wave complex in seconds, by default 0.5
    picks : str or list, optional
        Channels to include, by default "eeg" (all EEG channels)
    **kwargs
        Additional keyword arguments

    Returns
    -------
    List[Dict[str, Union[str, float, int]]]
        List of detected spike-wave complexes with channel, onset, duration, and amplitude.

    Examples
    --------
    >>> import mne
    >>> from epilepsy_eeg.analysis import epileptiform
    >>> raw = mne.io.read_raw_edf("sample.edf", preload=True)
    >>> spike_waves = epileptiform.detect_spike_wave_complexes(raw)
    >>> print(f"Detected {len(spike_waves)} spike-wave complexes.")
    """
    # Get the data and channel names
    data = raw.get_data(picks=picks)
    ch_names = [raw.ch_names[idx] for idx in mne.pick_types(raw.info, meg=False, eeg=True)]
    
    # Calculate the sampling frequency
    sfreq = raw.info['sfreq']
    
    # Calculate the minimum and maximum number of samples for a spike-wave complex
    min_samples = int(min_duration * sfreq)
    max_samples = int(max_duration * sfreq)
    
    # Initialize the list of detected spike-wave complexes
    spike_waves = []
    
    # Process each channel
    for ch_idx, ch_name in enumerate(ch_names):
        # Get the channel data
        ch_data = data[ch_idx]
        
        # Calculate the mean and standard deviation
        mean = np.mean(ch_data)
        std = np.std(ch_data)
        
        # Find peaks above the threshold
        peaks, properties = find_peaks(
            np.abs(ch_data - mean),
            height=threshold * std,
            width=(min_samples, max_samples),
            **kwargs
        )
        
        # Process each peak
        for peak_idx, peak in enumerate(peaks):
            # Calculate the onset and duration
            onset = peak / sfreq
            width = properties['widths'][peak_idx] / sfreq
            
            # Calculate the amplitude
            amplitude = ch_data[peak] - mean
            
            # Check for a slow wave following the spike
            if peak + int(0.1 * sfreq) < len(ch_data):
                # Check if there's a negative peak after the spike
                post_spike = ch_data[peak:peak + int(0.1 * sfreq)]
                if np.min(post_spike) < mean - threshold * std:
                    # Store the spike-wave complex
                    spike_waves.append({
                        'channel': ch_name,
                        'onset': onset,
                        'duration': width,
                        'amplitude': amplitude
                    })
    
    return spike_waves


def detect_epileptiform_activity(
    raw: mne.io.Raw,
    **kwargs
) -> Dict[str, List[Dict[str, Union[str, float, int]]]]:
    """
    Detect various types of epileptiform activity in the EEG data.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data.
    **kwargs
        Additional keyword arguments to pass to the detection functions

    Returns
    -------
    Dict[str, List[Dict[str, Union[str, float, int]]]]
        Dictionary mapping activity types to lists of detected events.

    Examples
    --------
    >>> import mne
    >>> from epilepsy_eeg.analysis import epileptiform
    >>> raw = mne.io.read_raw_edf("sample.edf", preload=True)
    >>> activities = epileptiform.detect_epileptiform_activity(raw)
    >>> for activity_type, events in activities.items():
    ...     print(f"Detected {len(events)} {activity_type} events.")
    """
    # Initialize the result dictionary
    activities = {}
    
    # Detect spikes
    activities['spikes'] = detect_spikes(raw, **kwargs)
    
    # Detect sharp waves
    activities['sharp_waves'] = detect_sharp_waves(raw, **kwargs)
    
    # Detect spike-wave complexes
    activities['spike_waves'] = detect_spike_wave_complexes(raw, **kwargs)
    
    # Detect OIRDA
    activities['oirda'] = detect_oirda(raw, **kwargs)
    
    # Detect FIRDA
    activities['firda'] = detect_firda(raw, **kwargs)
    
    return activities


def analyze_epileptiform_activity(
    raw: mne.io.Raw,
    **kwargs
) -> Dict[str, Dict[str, Union[int, float, List[str]]]]:
    """
    Analyze epileptiform activity in the EEG data and provide a summary.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data.
    **kwargs
        Additional keyword arguments to pass to detect_epileptiform_activity()

    Returns
    -------
    Dict[str, Dict[str, Union[int, float, List[str]]]]
        Dictionary with summary statistics for each activity type.

    Examples
    --------
    >>> import mne
    >>> from epilepsy_eeg.analysis import epileptiform
    >>> raw = mne.io.read_raw_edf("sample.edf", preload=True)
    >>> summary = epileptiform.analyze_epileptiform_activity(raw)
    >>> print(summary['spikes']['count'])
    """
    # Detect epileptiform activity
    activities = detect_epileptiform_activity(raw, **kwargs)
    
    # Initialize the result dictionary
    summary = {}
    
    # Process each activity type
    for activity_type, events in activities.items():
        # Initialize the summary for the activity type
        activity_summary = {
            'count': len(events),
            'rate': len(events) / (raw.times[-1] / 60),  # Events per minute
            'channels': []
        }
        
        # Process each event
        if events:
            # Collect channels with events
            if 'channel' in events[0]:
                channels = [event['channel'] for event in events]
                activity_summary['channels'] = list(set(channels))
            
            # Calculate the mean duration
            if 'duration' in events[0]:
                durations = [event['duration'] for event in events]
                activity_summary['mean_duration'] = np.mean(durations)
                activity_summary['max_duration'] = np.max(durations)
            
            # Calculate the mean amplitude
            if 'amplitude' in events[0]:
                amplitudes = [event['amplitude'] for event in events]
                activity_summary['mean_amplitude'] = np.mean(amplitudes)
                activity_summary['max_amplitude'] = np.max(amplitudes)
            
            # Calculate the mean power
            if 'power' in events[0]:
                powers = [event['power'] for event in events]
                activity_summary['mean_power'] = np.mean(powers)
                activity_summary['max_power'] = np.max(powers)
        
        # Store the summary
        summary[activity_type] = activity_summary
    
    return summary
