"""
Topographic mapping module for EEG visualization.

This module provides functions for creating topographic maps of EEG data,
including power maps, asymmetry maps, and statistical maps.
"""

import numpy as np
from typing import Tuple, Optional, Union, List, Dict
import matplotlib.pyplot as plt
import mne
from mne.viz import plot_topomap


def plot_band_topomaps(
    raw: mne.io.Raw,
    bands: Optional[Dict[str, Tuple[float, float]]] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = 'viridis',
    show: bool = True,
    **kwargs
) -> plt.Figure:
    """
    Plot topographic maps of power in different frequency bands.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data.
    bands : Dict[str, Tuple[float, float]] or None, optional
        Dictionary mapping band names to (fmin, fmax) tuples, by default None
        If None, the standard frequency bands are used.
    vmin : float or None, optional
        Minimum value for color scaling, by default None (auto-scaling)
    vmax : float or None, optional
        Maximum value for color scaling, by default None (auto-scaling)
    cmap : str, optional
        Colormap name, by default 'viridis'
    show : bool, optional
        Whether to show the figure, by default True
    **kwargs
        Additional keyword arguments to pass to mne.viz.plot_topomap()

    Returns
    -------
    plt.Figure
        The matplotlib figure containing the topomaps.

    Examples
    --------
    >>> import mne
    >>> from epilepsy_eeg.visualization import topomaps
    >>> raw = mne.io.read_raw_edf("sample.edf", preload=True)
    >>> fig = topomaps.plot_band_topomaps(raw)
    """
    # Use standard frequency bands if none are provided
    if bands is None:
        from epilepsy_eeg.analysis.spectral import FREQUENCY_BANDS
        bands = FREQUENCY_BANDS
    
    # Compute the power in each frequency band
    from epilepsy_eeg.analysis.spectral import compute_band_powers
    band_powers = compute_band_powers(raw, frequency_bands=bands)
    
    # Create a figure with subplots for each band
    n_bands = len(bands)
    n_rows = int(np.ceil(n_bands / 3))
    n_cols = min(n_bands, 3)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_bands == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Get the channel positions
    pos = mne.channels.layout._find_topomap_coords(raw.info, picks='eeg')
    
    # Plot each band
    for i, (band_name, powers) in enumerate(band_powers.items()):
        if i < len(axes):
            # Plot the topomap
            im, _ = plot_topomap(
                powers,
                pos,
                axes=axes[i],
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
                show=False,
                **kwargs
            )
            
            # Add a colorbar
            cbar = plt.colorbar(im, ax=axes[i], shrink=0.8)
            cbar.set_label('Power (µV²)')
            
            # Set the title
            axes[i].set_title(f"{band_name} ({bands[band_name][0]}-{bands[band_name][1]} Hz)")
    
    # Hide any unused subplots
    for i in range(n_bands, len(axes)):
        axes[i].set_visible(False)
    
    # Add a title to the figure
    fig.suptitle('Power Topographic Maps by Frequency Band', fontsize=16)
    
    # Adjust the layout
    fig.tight_layout()
    
    # Show the figure if requested
    if show:
        plt.show()
    
    return fig


def plot_asymmetry_topomap(
    raw: mne.io.Raw,
    band: str = 'Alpha',
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = 'RdBu_r',
    show: bool = True,
    **kwargs
) -> plt.Figure:
    """
    Plot a topographic map of asymmetry in a frequency band.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data.
    band : str, optional
        The frequency band to plot, by default 'Alpha'
    vmin : float or None, optional
        Minimum value for color scaling, by default None (auto-scaling)
    vmax : float or None, optional
        Maximum value for color scaling, by default None (auto-scaling)
    cmap : str, optional
        Colormap name, by default 'RdBu_r'
    show : bool, optional
        Whether to show the figure, by default True
    **kwargs
        Additional keyword arguments to pass to mne.viz.plot_topomap()

    Returns
    -------
    plt.Figure
        The matplotlib figure containing the topomap.

    Examples
    --------
    >>> import mne
    >>> from epilepsy_eeg.visualization import topomaps
    >>> raw = mne.io.read_raw_edf("sample.edf", preload=True)
    >>> fig = topomaps.plot_asymmetry_topomap(raw, band='Alpha')
    """
    # Compute asymmetry indices
    from epilepsy_eeg.analysis.asymmetry import compute_band_asymmetry
    asymmetry = compute_band_asymmetry(raw)
    
    # Check if the band exists
    if band not in asymmetry:
        raise ValueError(f"Band '{band}' not found in asymmetry results.")
    
    # Get the asymmetry indices for the band
    band_asymmetry = asymmetry[band]
    
    # Create a figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Get the channel positions
    pos = mne.channels.layout._find_topomap_coords(raw.info, picks='eeg')
    
    # Create a data array for the topomap
    data = np.zeros(len(raw.ch_names))
    
    # Fill in the asymmetry values
    for pair, index in band_asymmetry.items():
        # Extract the electrodes
        left_electrode, right_electrode = pair.split('-')
        
        # Find the indices of the electrodes
        try:
            left_idx = raw.ch_names.index(left_electrode)
            right_idx = raw.ch_names.index(right_electrode)
            
            # Set the asymmetry values
            data[left_idx] = -index  # Negative for left hemisphere
            data[right_idx] = index  # Positive for right hemisphere
        except ValueError:
            # Skip if the electrodes are not found
            continue
    
    # Plot the topomap
    im, _ = plot_topomap(
        data,
        pos,
        axes=ax,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        show=False,
        **kwargs
    )
    
    # Add a colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Asymmetry Index')
    
    # Set the title
    ax.set_title(f"{band} Band Asymmetry")
    
    # Adjust the layout
    fig.tight_layout()
    
    # Show the figure if requested
    if show:
        plt.show()
    
    return fig


def plot_epileptiform_activity(
    raw: mne.io.Raw,
    spikes: List[Dict[str, Union[str, float, int]]],
    show: bool = True,
    **kwargs
) -> plt.Figure:
    """
    Plot epileptiform activity on a topographic map.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data.
    spikes : List[Dict[str, Union[str, float, int]]]
        List of detected spikes with channel, onset, duration, and amplitude.
    show : bool, optional
        Whether to show the figure, by default True
    **kwargs
        Additional keyword arguments to pass to mne.viz.plot_topomap()

    Returns
    -------
    plt.Figure
        The matplotlib figure containing the topomap.

    Examples
    --------
    >>> import mne
    >>> from epilepsy_eeg.visualization import topomaps
    >>> from epilepsy_eeg.analysis import epileptiform
    >>> raw = mne.io.read_raw_edf("sample.edf", preload=True)
    >>> spikes = epileptiform.detect_spikes(raw)
    >>> fig = topomaps.plot_epileptiform_activity(raw, spikes)
    """
    # Create a figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Get the channel positions
    pos = mne.channels.layout._find_topomap_coords(raw.info, picks='eeg')
    
    # Create a data array for the topomap
    data = np.zeros(len(raw.ch_names))
    
    # Count spikes per channel
    for spike in spikes:
        channel = spike['channel']
        try:
            idx = raw.ch_names.index(channel)
            data[idx] += 1
        except ValueError:
            # Skip if the channel is not found
            continue
    
    # Plot the topomap
    im, _ = plot_topomap(
        data,
        pos,
        axes=ax,
        cmap='hot',
        show=False,
        **kwargs
    )
    
    # Add a colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Number of Spikes')
    
    # Set the title
    ax.set_title('Epileptiform Activity')
    
    # Adjust the layout
    fig.tight_layout()
    
    # Show the figure if requested
    if show:
        plt.show()
    
    return fig


def plot_brodmann_activity(
    raw: mne.io.Raw,
    band: str = 'Alpha',
    threshold: float = 1.5,
    show: bool = True,
    **kwargs
) -> plt.Figure:
    """
    Plot Brodmann area activity on a topographic map.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data.
    band : str, optional
        The frequency band to plot, by default 'Alpha'
    threshold : float, optional
        Threshold for increased or reduced activity in standard deviations, by default 1.5
    show : bool, optional
        Whether to show the figure, by default True
    **kwargs
        Additional keyword arguments to pass to mne.viz.plot_topomap()

    Returns
    -------
    plt.Figure
        The matplotlib figure containing the topomap.

    Examples
    --------
    >>> import mne
    >>> from epilepsy_eeg.visualization import topomaps
    >>> raw = mne.io.read_raw_edf("sample.edf", preload=True)
    >>> fig = topomaps.plot_brodmann_activity(raw, band='Alpha')
    """
    # Analyze Brodmann area activity
    from epilepsy_eeg.analysis.brodmann import analyze_brodmann_activity
    activity = analyze_brodmann_activity(raw, threshold=threshold)
    
    # Check if the band exists
    if band not in activity:
        raise ValueError(f"Band '{band}' not found in activity results.")
    
    # Get the activity for the band
    band_activity = activity[band]
    
    # Create a figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Get the channel positions
    pos = mne.channels.layout._find_topomap_coords(raw.info, picks='eeg')
    
    # Create a data array for the topomap
    data = np.zeros(len(raw.ch_names))
    
    # Map channels to Brodmann areas
    from epilepsy_eeg.analysis.brodmann import map_channels_to_brodmann
    channel_to_areas = map_channels_to_brodmann(raw)
    
    # Fill in the activity values
    for ch_idx, ch_name in enumerate(raw.ch_names):
        # Get the Brodmann areas for the channel
        areas = channel_to_areas.get(ch_name, [])
        
        # Check if any of the areas have increased or reduced activity
        for area in areas:
            if area in band_activity['increased']:
                data[ch_idx] = 1  # Increased activity
                break
            elif area in band_activity['reduced']:
                data[ch_idx] = -1  # Reduced activity
                break
    
    # Plot the topomap
    im, _ = plot_topomap(
        data,
        pos,
        axes=ax,
        vmin=-1,
        vmax=1,
        cmap='RdBu_r',
        show=False,
        **kwargs
    )
    
    # Add a colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Activity')
    cbar.set_ticks([-1, 0, 1])
    cbar.set_ticklabels(['Reduced', 'Normal', 'Increased'])
    
    # Set the title
    ax.set_title(f"{band} Band Brodmann Area Activity")
    
    # Adjust the layout
    fig.tight_layout()
    
    # Show the figure if requested
    if show:
        plt.show()
    
    return fig
