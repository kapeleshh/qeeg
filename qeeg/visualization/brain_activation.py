"""
Brain activation visualization module for EEG data.

This module provides functions for visualizing brain activation patterns from EEG data,
including 3D brain models, source localization, and connectivity visualization.
"""

import numpy as np
from typing import Tuple, Optional, Union, List, Dict
import matplotlib.pyplot as plt
import mne
from mne.viz import plot_topomap
import nibabel as nib
from nilearn import plotting


def plot_brain_activation(
    raw: mne.io.Raw,
    method: str = 'dics',
    band: str = 'alpha',
    subjects_dir: Optional[str] = None,
    subject: Optional[str] = None,
    show: bool = True,
    **kwargs
) -> plt.Figure:
    """
    Plot brain activation from EEG data using source localization.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data.
    method : str, optional
        Source localization method, by default 'dics'
        Options: 'dics', 'lcmv', 'mne', 'sloreta'
    band : str, optional
        Frequency band to analyze, by default 'alpha'
        Options: 'delta', 'theta', 'alpha', 'beta', 'gamma'
    subjects_dir : str or None, optional
        Directory containing FreeSurfer subjects, by default None
    subject : str or None, optional
        Subject name, by default None
    show : bool, optional
        Whether to show the figure, by default True
    **kwargs
        Additional keyword arguments to pass to source localization functions

    Returns
    -------
    plt.Figure
        The matplotlib figure containing the brain activation plot.

    Examples
    --------
    >>> import mne
    >>> from qeeg.visualization import brain_activation
    >>> raw = mne.io.read_raw_edf("sample.edf", preload=True)
    >>> fig = brain_activation.plot_brain_activation(raw, band='alpha')
    """
    # Define frequency bands
    bands = {
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 80)
    }
    
    # Check if the band is valid
    if band not in bands:
        raise ValueError(f"Invalid band: {band}. Valid options are: {', '.join(bands.keys())}")
    
    # Get the frequency range for the band
    fmin, fmax = bands[band]
    
    # Create epochs from the raw data
    events = mne.make_fixed_length_events(raw, duration=2.0)
    epochs = mne.Epochs(raw, events, tmin=0, tmax=2.0, baseline=None, preload=True)
    
    # Compute the forward solution
    # This requires a head model and source space, which might not be available
    # For demonstration purposes, we'll create a figure with a placeholder
    
    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Add a placeholder image
    ax.text(0.5, 0.5, f"Brain Activation ({band.capitalize()} band, {fmin}-{fmax} Hz)\n"
            f"Method: {method.upper()}\n\n"
            "Note: Actual source localization requires additional data\n"
            "(head model, source space, etc.)",
            ha='center', va='center', fontsize=14)
    
    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Set the title
    ax.set_title(f"Brain Activation - {band.capitalize()} Band ({fmin}-{fmax} Hz)")
    
    # Adjust the layout
    fig.tight_layout()
    
    # Show the figure if requested
    if show:
        plt.show()
    
    return fig


def plot_connectivity_graph(
    raw: mne.io.Raw,
    band: str = 'alpha',
    method: str = 'wpli',
    threshold: float = 0.7,
    show: bool = True,
    **kwargs
) -> plt.Figure:
    """
    Plot a brain connectivity graph from EEG data.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data.
    band : str, optional
        Frequency band to analyze, by default 'alpha'
        Options: 'delta', 'theta', 'alpha', 'beta', 'gamma'
    method : str, optional
        Connectivity method, by default 'wpli'
        Options: 'coh', 'plv', 'pli', 'wpli', 'imcoh'
    threshold : float, optional
        Threshold for connectivity values, by default 0.7
    show : bool, optional
        Whether to show the figure, by default True
    **kwargs
        Additional keyword arguments to pass to connectivity functions

    Returns
    -------
    plt.Figure
        The matplotlib figure containing the connectivity graph.

    Examples
    --------
    >>> import mne
    >>> from qeeg.visualization import brain_activation
    >>> raw = mne.io.read_raw_edf("sample.edf", preload=True)
    >>> fig = brain_activation.plot_connectivity_graph(raw, band='alpha')
    """
    # Define frequency bands
    bands = {
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 80)
    }
    
    # Check if the band is valid
    if band not in bands:
        raise ValueError(f"Invalid band: {band}. Valid options are: {', '.join(bands.keys())}")
    
    # Get the frequency range for the band
    fmin, fmax = bands[band]
    
    # Create epochs from the raw data
    events = mne.make_fixed_length_events(raw, duration=2.0)
    epochs = mne.Epochs(raw, events, tmin=0, tmax=2.0, baseline=None, preload=True)
    
    # Compute connectivity
    from mne.connectivity import spectral_connectivity
    
    # Get the channel names and positions
    picks = mne.pick_types(raw.info, meg=False, eeg=True)
    ch_names = [raw.ch_names[i] for i in picks]
    
    # Compute connectivity
    con, freqs, times, n_epochs, n_tapers = spectral_connectivity(
        epochs,
        method=method,
        mode='multitaper',
        sfreq=raw.info['sfreq'],
        fmin=fmin,
        fmax=fmax,
        faverage=True,
        **kwargs
    )
    
    # Get the connectivity matrix
    n_channels = len(picks)
    con_matrix = np.zeros((n_channels, n_channels))
    
    # Fill the connectivity matrix
    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            idx = (i * n_channels + j) - (i * (i + 1)) // 2 - i - 1
            con_matrix[i, j] = con[idx, 0, 0]
            con_matrix[j, i] = con[idx, 0, 0]  # Symmetrical
    
    # Apply threshold
    con_matrix[con_matrix < threshold] = 0
    
    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot the connectivity matrix as a graph
    try:
        # Try to use MNE's plot_connectivity_circle
        from mne.viz import plot_connectivity_circle
        
        # Get the channel positions
        pos = mne.channels.layout._find_topomap_coords(raw.info, picks=picks)
        
        # Plot the connectivity circle
        plot_connectivity_circle(
            con_matrix,
            ch_names,
            ax=ax,
            title=f"{band.capitalize()} Band Connectivity ({method.upper()})",
            show=False
        )
    except Exception:
        # Fallback to a simple heatmap
        im = ax.imshow(con_matrix, cmap='viridis', interpolation='nearest')
        plt.colorbar(im, ax=ax)
        ax.set_xticks(np.arange(len(ch_names)))
        ax.set_yticks(np.arange(len(ch_names)))
        ax.set_xticklabels(ch_names, rotation=90)
        ax.set_yticklabels(ch_names)
        ax.set_title(f"{band.capitalize()} Band Connectivity ({method.upper()})")
    
    # Adjust the layout
    fig.tight_layout()
    
    # Show the figure if requested
    if show:
        plt.show()
    
    return fig


def plot_brodmann_areas(
    raw: mne.io.Raw,
    band: str = 'alpha',
    show: bool = True,
    **kwargs
) -> plt.Figure:
    """
    Plot activation in Brodmann areas from EEG data.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data.
    band : str, optional
        Frequency band to analyze, by default 'alpha'
        Options: 'delta', 'theta', 'alpha', 'beta', 'gamma'
    show : bool, optional
        Whether to show the figure, by default True
    **kwargs
        Additional keyword arguments to pass to analysis functions

    Returns
    -------
    plt.Figure
        The matplotlib figure containing the Brodmann areas plot.

    Examples
    --------
    >>> import mne
    >>> from qeeg.visualization import brain_activation
    >>> raw = mne.io.read_raw_edf("sample.edf", preload=True)
    >>> fig = brain_activation.plot_brodmann_areas(raw, band='alpha')
    """
    # Analyze Brodmann area activity
    from qeeg.analysis.brodmann import analyze_brodmann_activity
    activity = analyze_brodmann_activity(raw, **kwargs)
    
    # Check if the band is valid
    if band not in activity:
        raise ValueError(f"Invalid band: {band}. Valid options are: {', '.join(activity.keys())}")
    
    # Get the activity for the band
    band_activity = activity[band]
    
    # Create a figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create a dictionary to store the activity values for each Brodmann area
    area_values = {}
    
    # Process increased activity
    for area in band_activity["increased"]:
        area_values[area] = 1  # Increased activity
    
    # Process reduced activity
    for area in band_activity["reduced"]:
        area_values[area] = -1  # Reduced activity
    
    # Create a list of areas and values
    areas = list(area_values.keys())
    values = [area_values[area] for area in areas]
    
    # Create a bar plot
    bars = ax.bar(areas, values, color=['r' if v > 0 else 'b' for v in values])
    
    # Add labels and title
    ax.set_xlabel('Brodmann Area')
    ax.set_ylabel('Activity (1: Increased, -1: Reduced)')
    ax.set_title(f'Brodmann Area Activity - {band.capitalize()} Band')
    
    # Add a grid
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Add area labels
    from qeeg.analysis.brodmann import get_brodmann_function
    for i, area in enumerate(areas):
        function = get_brodmann_function(area)
        ax.text(i, values[i] * 0.9, f"Area {area}\n{function}", ha='center', va='center', fontsize=8, rotation=90)
    
    # Adjust the layout
    fig.tight_layout()
    
    # Show the figure if requested
    if show:
        plt.show()
    
    return fig


def plot_3d_brain(
    raw: mne.io.Raw,
    band: str = 'alpha',
    view: str = 'lateral',
    hemisphere: str = 'both',
    show: bool = True,
    **kwargs
) -> plt.Figure:
    """
    Plot a 3D brain model with EEG data.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data.
    band : str, optional
        Frequency band to analyze, by default 'alpha'
        Options: 'delta', 'theta', 'alpha', 'beta', 'gamma'
    view : str, optional
        View angle, by default 'lateral'
        Options: 'lateral', 'medial', 'dorsal', 'ventral', 'anterior', 'posterior'
    hemisphere : str, optional
        Hemisphere to show, by default 'both'
        Options: 'left', 'right', 'both'
    show : bool, optional
        Whether to show the figure, by default True
    **kwargs
        Additional keyword arguments to pass to plotting functions

    Returns
    -------
    plt.Figure
        The matplotlib figure containing the 3D brain plot.

    Examples
    --------
    >>> import mne
    >>> from qeeg.visualization import brain_activation
    >>> raw = mne.io.read_raw_edf("sample.edf", preload=True)
    >>> fig = brain_activation.plot_3d_brain(raw, band='alpha')
    """
    # Define frequency bands
    bands = {
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 80)
    }
    
    # Check if the band is valid
    if band not in bands:
        raise ValueError(f"Invalid band: {band}. Valid options are: {', '.join(bands.keys())}")
    
    # Get the frequency range for the band
    fmin, fmax = bands[band]
    
    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Add a placeholder image
    ax.text(0.5, 0.5, f"3D Brain Visualization ({band.capitalize()} band, {fmin}-{fmax} Hz)\n"
            f"View: {view}, Hemisphere: {hemisphere}\n\n"
            "Note: Actual 3D visualization requires additional data\n"
            "(MRI data, source space, etc.)",
            ha='center', va='center', fontsize=14)
    
    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Set the title
    ax.set_title(f"3D Brain - {band.capitalize()} Band ({fmin}-{fmax} Hz)")
    
    # Adjust the layout
    fig.tight_layout()
    
    # Show the figure if requested
    if show:
        plt.show()
    
    return fig
