"""
Spectral visualization module for EEG data.

This module provides functions for visualizing spectral properties of EEG data,
including power spectra, time-frequency plots, and coherence plots.
"""

import numpy as np
from typing import Tuple, Optional, Union, List, Dict
import matplotlib.pyplot as plt
import mne
from mne.time_frequency import tfr_morlet


def plot_power_spectrum(
    raw: mne.io.Raw,
    fmin: float = 0.5,
    fmax: float = 40.0,
    picks: Optional[Union[str, list]] = "eeg",
    n_channels: int = 5,
    log_scale: bool = True,
    show: bool = True,
    **kwargs
) -> plt.Figure:
    """
    Plot the power spectrum of EEG data.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data.
    fmin : float, optional
        Minimum frequency to plot, by default 0.5
    fmax : float, optional
        Maximum frequency to plot, by default 40.0
    picks : str or list, optional
        Channels to include, by default "eeg" (all EEG channels)
    n_channels : int, optional
        Number of channels to plot, by default 5
    log_scale : bool, optional
        Whether to use a logarithmic scale for the y-axis, by default True
    show : bool, optional
        Whether to show the figure, by default True
    **kwargs
        Additional keyword arguments to pass to mne.time_frequency.psd_welch()

    Returns
    -------
    plt.Figure
        The matplotlib figure containing the power spectrum.

    Examples
    --------
    >>> import mne
    >>> from epilepsy_eeg.visualization import spectra
    >>> raw = mne.io.read_raw_edf("sample.edf", preload=True)
    >>> fig = spectra.plot_power_spectrum(raw)
    """
    # Compute the PSD
    from epilepsy_eeg.analysis.spectral import compute_psd
    psds, freqs = compute_psd(raw, fmin=fmin, fmax=fmax, picks=picks, **kwargs)
    
    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get the channel names
    ch_names = [raw.ch_names[idx] for idx in mne.pick_types(raw.info, meg=False, eeg=True, selection=picks)]
    
    # Limit the number of channels to plot
    n_channels = min(n_channels, len(ch_names))
    
    # Plot the PSD for each channel
    for i in range(n_channels):
        if log_scale:
            ax.semilogy(freqs, psds[i], label=ch_names[i])
        else:
            ax.plot(freqs, psds[i], label=ch_names[i])
    
    # Add frequency band shading
    from epilepsy_eeg.analysis.spectral import FREQUENCY_BANDS
    colors = plt.cm.tab10(np.linspace(0, 1, len(FREQUENCY_BANDS)))
    
    for (band_name, (fmin_band, fmax_band)), color in zip(FREQUENCY_BANDS.items(), colors):
        # Skip bands outside the plotted frequency range
        if fmax_band < fmin or fmin_band > fmax:
            continue
        
        # Adjust band limits to the plotted frequency range
        fmin_plot = max(fmin_band, fmin)
        fmax_plot = min(fmax_band, fmax)
        
        # Add a shaded area for the band
        ax.axvspan(fmin_plot, fmax_plot, alpha=0.2, color=color, label=f"{band_name} ({fmin_band}-{fmax_band} Hz)")
    
    # Set the axis labels and title
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power Spectral Density (µV²/Hz)')
    ax.set_title('Power Spectrum')
    
    # Set the x-axis limits
    ax.set_xlim(fmin, fmax)
    
    # Add a legend
    ax.legend(loc='upper right')
    
    # Add a grid
    ax.grid(True)
    
    # Adjust the layout
    fig.tight_layout()
    
    # Show the figure if requested
    if show:
        plt.show()
    
    return fig


def plot_time_frequency(
    raw: mne.io.Raw,
    channel: Optional[str] = None,
    fmin: float = 1.0,
    fmax: float = 40.0,
    tmin: Optional[float] = None,
    tmax: Optional[float] = None,
    n_cycles: Union[int, List[int], np.ndarray] = 7,
    show: bool = True,
    **kwargs
) -> plt.Figure:
    """
    Plot a time-frequency representation of EEG data.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data.
    channel : str or None, optional
        Channel to plot, by default None (uses the first EEG channel)
    fmin : float, optional
        Minimum frequency to plot, by default 1.0
    fmax : float, optional
        Maximum frequency to plot, by default 40.0
    tmin : float or None, optional
        Start time in seconds, by default None (start of the data)
    tmax : float or None, optional
        End time in seconds, by default None (end of the data)
    n_cycles : int or list or np.ndarray, optional
        Number of cycles in the Morlet wavelet, by default 7
    show : bool, optional
        Whether to show the figure, by default True
    **kwargs
        Additional keyword arguments to pass to mne.time_frequency.tfr_morlet()

    Returns
    -------
    plt.Figure
        The matplotlib figure containing the time-frequency plot.

    Examples
    --------
    >>> import mne
    >>> from epilepsy_eeg.visualization import spectra
    >>> raw = mne.io.read_raw_edf("sample.edf", preload=True)
    >>> fig = spectra.plot_time_frequency(raw, channel='Fz')
    """
    # Select the channel
    if channel is None:
        picks = mne.pick_types(raw.info, meg=False, eeg=True)
        if len(picks) == 0:
            raise ValueError("No EEG channels found in the data.")
        channel = raw.ch_names[picks[0]]
    
    # Create epochs from the raw data
    events = mne.make_fixed_length_events(raw, duration=1.0)
    epochs = mne.Epochs(raw, events, tmin=0, tmax=1.0, baseline=None, preload=True)
    
    # Set the frequencies
    freqs = np.arange(fmin, fmax + 1, 1.0)
    
    # Compute the time-frequency representation
    power = tfr_morlet(
        epochs,
        freqs=freqs,
        n_cycles=n_cycles,
        picks=channel,
        return_itc=False,
        **kwargs
    )
    
    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the time-frequency representation
    power.plot(
        picks=channel,
        baseline=None,
        mode='logratio',
        title=f'Time-Frequency Representation - {channel}',
        axes=ax,
        show=False
    )
    
    # Adjust the layout
    fig.tight_layout()
    
    # Show the figure if requested
    if show:
        plt.show()
    
    return fig


def plot_band_power_comparison(
    raw: mne.io.Raw,
    bands: Optional[Dict[str, Tuple[float, float]]] = None,
    show: bool = True,
    **kwargs
) -> plt.Figure:
    """
    Plot a comparison of power in different frequency bands.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data.
    bands : Dict[str, Tuple[float, float]] or None, optional
        Dictionary mapping band names to (fmin, fmax) tuples, by default None
        If None, the standard frequency bands are used.
    show : bool, optional
        Whether to show the figure, by default True
    **kwargs
        Additional keyword arguments to pass to compute_band_powers()

    Returns
    -------
    plt.Figure
        The matplotlib figure containing the band power comparison.

    Examples
    --------
    >>> import mne
    >>> from epilepsy_eeg.visualization import spectra
    >>> raw = mne.io.read_raw_edf("sample.edf", preload=True)
    >>> fig = spectra.plot_band_power_comparison(raw)
    """
    # Use standard frequency bands if none are provided
    if bands is None:
        from epilepsy_eeg.analysis.spectral import FREQUENCY_BANDS
        bands = FREQUENCY_BANDS
    
    # Compute the power in each frequency band
    from epilepsy_eeg.analysis.spectral import compute_band_powers
    band_powers = compute_band_powers(raw, frequency_bands=bands, **kwargs)
    
    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get the channel names
    ch_names = [raw.ch_names[idx] for idx in mne.pick_types(raw.info, meg=False, eeg=True)]
    
    # Compute the average power for each band
    avg_powers = {band: np.mean(powers) for band, powers in band_powers.items()}
    
    # Plot the average power for each band
    bands_list = list(bands.keys())
    powers_list = [avg_powers[band] for band in bands_list]
    
    # Create a bar plot
    bars = ax.bar(bands_list, powers_list)
    
    # Add labels and title
    ax.set_xlabel('Frequency Band')
    ax.set_ylabel('Average Power (µV²)')
    ax.set_title('Average Power by Frequency Band')
    
    # Add a grid
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on top of the bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            height,
            f'{height:.2f}',
            ha='center',
            va='bottom',
            rotation=0
        )
    
    # Adjust the layout
    fig.tight_layout()
    
    # Show the figure if requested
    if show:
        plt.show()
    
    return fig


def plot_coherence(
    raw: mne.io.Raw,
    ch_pairs: Optional[List[Tuple[str, str]]] = None,
    fmin: float = 0.5,
    fmax: float = 40.0,
    show: bool = True,
    **kwargs
) -> plt.Figure:
    """
    Plot coherence between pairs of EEG channels.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data.
    ch_pairs : List[Tuple[str, str]] or None, optional
        List of channel pairs to compute coherence for, by default None
        If None, uses standard homologous pairs.
    fmin : float, optional
        Minimum frequency to plot, by default 0.5
    fmax : float, optional
        Maximum frequency to plot, by default 40.0
    show : bool, optional
        Whether to show the figure, by default True
    **kwargs
        Additional keyword arguments to pass to mne.connectivity.spectral_connectivity()

    Returns
    -------
    plt.Figure
        The matplotlib figure containing the coherence plot.

    Examples
    --------
    >>> import mne
    >>> from epilepsy_eeg.visualization import spectra
    >>> raw = mne.io.read_raw_edf("sample.edf", preload=True)
    >>> fig = spectra.plot_coherence(raw)
    """
    # Define standard homologous pairs if none are provided
    if ch_pairs is None:
        ch_pairs = [
            ('F3', 'F4'),
            ('F7', 'F8'),
            ('C3', 'C4'),
            ('T3', 'T4'),
            ('T5', 'T6'),
            ('P3', 'P4'),
            ('O1', 'O2')
        ]
    
    # Filter the pairs to only include channels that exist in the data
    ch_pairs = [
        (ch1, ch2) for ch1, ch2 in ch_pairs
        if ch1 in raw.ch_names and ch2 in raw.ch_names
    ]
    
    if not ch_pairs:
        raise ValueError("No valid channel pairs found in the data.")
    
    # Create epochs from the raw data
    events = mne.make_fixed_length_events(raw, duration=2.0)
    epochs = mne.Epochs(raw, events, tmin=0, tmax=2.0, baseline=None, preload=True)
    
    # Compute the coherence
    from mne.connectivity import spectral_connectivity
    
    # Convert channel pairs to indices
    indices = []
    for ch1, ch2 in ch_pairs:
        idx1 = raw.ch_names.index(ch1)
        idx2 = raw.ch_names.index(ch2)
        indices.append((idx1, idx2))
    
    # Compute the coherence
    con, freqs, times, _, _ = spectral_connectivity(
        epochs,
        method='coh',
        mode='multitaper',
        sfreq=raw.info['sfreq'],
        fmin=fmin,
        fmax=fmax,
        faverage=False,
        indices=indices,
        **kwargs
    )
    
    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the coherence for each channel pair
    for i, (ch1, ch2) in enumerate(ch_pairs):
        ax.plot(freqs, con[i, :, 0], label=f'{ch1}-{ch2}')
    
    # Add frequency band shading
    from epilepsy_eeg.analysis.spectral import FREQUENCY_BANDS
    colors = plt.cm.tab10(np.linspace(0, 1, len(FREQUENCY_BANDS)))
    
    for (band_name, (fmin_band, fmax_band)), color in zip(FREQUENCY_BANDS.items(), colors):
        # Skip bands outside the plotted frequency range
        if fmax_band < fmin or fmin_band > fmax:
            continue
        
        # Adjust band limits to the plotted frequency range
        fmin_plot = max(fmin_band, fmin)
        fmax_plot = min(fmax_band, fmax)
        
        # Add a shaded area for the band
        ax.axvspan(fmin_plot, fmax_plot, alpha=0.2, color=color, label=f"{band_name} ({fmin_band}-{fmax_band} Hz)")
    
    # Set the axis labels and title
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Coherence')
    ax.set_title('Coherence between Channel Pairs')
    
    # Set the x-axis limits
    ax.set_xlim(fmin, fmax)
    
    # Set the y-axis limits
    ax.set_ylim(0, 1)
    
    # Add a legend
    ax.legend(loc='upper right')
    
    # Add a grid
    ax.grid(True)
    
    # Adjust the layout
    fig.tight_layout()
    
    # Show the figure if requested
    if show:
        plt.show()
    
    return fig
