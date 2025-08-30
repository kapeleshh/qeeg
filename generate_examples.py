"""
Comprehensive example generator for the QEEG package.

This script generates various examples using real EEG data and saves
the visualizations to the examples/output directory.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import mne
from pathlib import Path

# Import QEEG modules
import qeeg
from qeeg.analysis import spectral, asymmetry, epileptiform, brodmann
from qeeg.visualization import topomaps, spectra, reports, brain_activation
from qeeg.preprocessing import filtering, artifacts, montage
from qeeg.ml import features, classification
from qeeg.conditions import adhd, anxiety, autism, depression, other_conditions

# Set up output directory for visualizations
OUTPUT_DIR = Path("examples/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Configure matplotlib for non-interactive backend
plt.switch_backend('agg')

def check_data_file():
    """Check if the data file exists."""
    data_path = Path("data/ECEDNEW0000.edf")
    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        print("Please make sure the data file is in the correct location.")
        sys.exit(1)
    return data_path

def load_and_preprocess_data(data_path):
    """Load and preprocess the EEG data."""
    print(f"Loading EEG data from {data_path}...")
    # Use latin1 encoding as suggested by the error message
    raw = mne.io.read_raw_edf(data_path, preload=True, encoding='latin1')
    
    # Print information about the data
    print("\nEEG Data Information:")
    print(f"Number of channels: {len(raw.ch_names)}")
    print(f"Channel names: {raw.ch_names}")
    print(f"Sampling frequency: {raw.info['sfreq']} Hz")
    print(f"Duration: {raw.times[-1]:.2f} seconds")
    
    # Apply bandpass filter
    print("\nApplying bandpass filter...")
    raw_filtered = filtering.bandpass_filter(
        raw, l_freq=1.0, h_freq=40.0
    )
    
    # Apply notch filter to remove line noise
    print("Applying notch filter...")
    raw_filtered = filtering.notch_filter(
        raw_filtered, freqs=50.0
    )
    
    # Remove artifacts using ICA
    print("Removing artifacts using ICA...")
    raw_cleaned = artifacts.remove_artifacts_ica(
        raw_filtered, n_components=15
    )
    
    return raw, raw_filtered, raw_cleaned

def generate_preprocessing_examples(raw, raw_filtered, raw_cleaned):
    """Generate preprocessing examples."""
    print("\nGenerating preprocessing examples...")
    
    # Use MNE's built-in plotting functions for better visualizations
    
    # Plot raw data
    print("Generating raw EEG plot...")
    fig_raw = raw.plot(duration=10, n_channels=10, scalings='auto', show=False)
    fig_raw.suptitle('Raw EEG Data', fontsize=16)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'raw_eeg.png', dpi=300)
    plt.close(fig_raw)
    
    # Plot filtered data
    print("Generating filtered EEG plot...")
    fig_filtered = raw_filtered.plot(duration=10, n_channels=10, scalings='auto', show=False)
    fig_filtered.suptitle('Filtered EEG Data (1-40 Hz)', fontsize=16)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'filtered_eeg.png', dpi=300)
    plt.close(fig_filtered)
    
    # Plot cleaned data
    print("Generating cleaned EEG plot...")
    fig_cleaned = raw_cleaned.plot(duration=10, n_channels=10, scalings='auto', show=False)
    fig_cleaned.suptitle('Cleaned EEG Data (After ICA)', fontsize=16)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'cleaned_eeg.png', dpi=300)
    plt.close(fig_cleaned)
    
    # Plot PSD comparison
    print("Generating PSD comparison plot...")
    fig_psd, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True, sharey=True)
    
    raw.plot_psd(fmax=50, ax=axes[0], show=False)
    axes[0].set_title('Raw EEG - Power Spectral Density')
    
    raw_filtered.plot_psd(fmax=50, ax=axes[1], show=False)
    axes[1].set_title('Filtered EEG - Power Spectral Density')
    
    raw_cleaned.plot_psd(fmax=50, ax=axes[2], show=False)
    axes[2].set_title('Cleaned EEG - Power Spectral Density')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'psd_comparison.png', dpi=300)
    plt.close(fig_psd)
    
    print("Saved EEG preprocessing visualizations")

def generate_spectral_analysis_examples(raw_cleaned):
    """Generate spectral analysis examples."""
    print("\nGenerating spectral analysis examples...")
    
    # Compute power spectral density
    psds, freqs = spectral.compute_psd(raw_cleaned)
    
    # Compute power in frequency bands
    band_powers = spectral.compute_band_powers(raw_cleaned)
    
    # Print average power in each band
    print("\nAverage power in each frequency band:")
    for band, powers in band_powers.items():
        print(f"{band}: {np.mean(powers):.6f}")
    
    # Plot the power spectrum
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, ch_name in enumerate(raw_cleaned.ch_names[:5]):
        ax.semilogy(freqs, psds[i], label=ch_name)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power Spectral Density (µV²/Hz)')
    ax.set_title('Power Spectrum')
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'power_spectrum.png')
    plt.close(fig)
    
    print("Saved power spectrum to power_spectrum.png")
    
    # Plot relative band powers
    rel_powers = spectral.compute_relative_band_powers(raw_cleaned)
    fig, ax = plt.subplots(figsize=(10, 6))
    bands = list(rel_powers.keys())
    x = np.arange(len(bands))
    width = 0.8 / len(raw_cleaned.ch_names[:5])
    
    for i, ch_name in enumerate(raw_cleaned.ch_names[:5]):
        values = [rel_powers[band][i] for band in bands]
        ax.bar(x + i * width, values, width, label=ch_name)
    
    ax.set_xlabel('Frequency Band')
    ax.set_ylabel('Relative Power')
    ax.set_title('Relative Band Powers')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(bands)
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'relative_band_powers.png')
    plt.close(fig)
    
    print("Saved relative band powers to relative_band_powers.png")

def generate_topographic_maps(raw_cleaned):
    """Generate topographic maps."""
    print("\nGenerating topographic maps...")
    
    # Plot topographic maps of band powers
    fig = topomaps.plot_band_topomaps(raw_cleaned, show=False)
    plt.savefig(OUTPUT_DIR / 'band_topomaps.png')
    plt.close(fig)
    
    print("Saved band topomaps to band_topomaps.png")
    
    # Plot asymmetry topomap
    try:
        fig = topomaps.plot_asymmetry_topomap(raw_cleaned, band='Alpha', show=False)
        plt.savefig(OUTPUT_DIR / 'alpha_asymmetry_topomap.png')
        plt.close(fig)
        print("Saved alpha asymmetry topomap to alpha_asymmetry_topomap.png")
    except Exception as e:
        print(f"Could not generate asymmetry topomap: {e}")

def generate_condition_specific_examples(raw_cleaned):
    """Generate condition-specific examples."""
    print("\nGenerating condition-specific examples...")
    
    # ADHD analysis
    print("\nPerforming ADHD analysis...")
    tb_ratios = adhd.calculate_theta_beta_ratio(raw_cleaned)
    print("\nTheta/Beta ratios (relevant for ADHD):")
    for ch_name, ratio in list(tb_ratios.items())[:5]:  # Print first 5 channels
        print(f"{ch_name}: {ratio:.4f}")
    
    # Create a bar chart of theta/beta ratios
    fig, ax = plt.subplots(figsize=(10, 6))
    channels = list(tb_ratios.keys())[:10]  # First 10 channels
    values = [tb_ratios[ch] for ch in channels]
    ax.bar(channels, values)
    ax.set_xlabel('Channel')
    ax.set_ylabel('Theta/Beta Ratio')
    ax.set_title('Theta/Beta Ratios (ADHD Marker)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'adhd_theta_beta_ratios.png')
    plt.close(fig)
    
    print("Saved ADHD theta/beta ratios to adhd_theta_beta_ratios.png")
    
    # Anxiety analysis
    print("\nPerforming anxiety analysis...")
    try:
        alpha_asymmetry = anxiety.calculate_frontal_alpha_asymmetry(raw_cleaned)
        print("\nFrontal Alpha Asymmetry (relevant for anxiety):")
        for pair, asymmetry_value in list(alpha_asymmetry.items())[:3]:  # Print first 3 pairs
            print(f"{pair}: {asymmetry_value:.4f}")
        
        # Create a bar chart of alpha asymmetry
        fig, ax = plt.subplots(figsize=(10, 6))
        pairs = list(alpha_asymmetry.keys())
        values = [alpha_asymmetry[pair] for pair in pairs]
        ax.bar(pairs, values)
        ax.set_xlabel('Electrode Pair')
        ax.set_ylabel('Alpha Asymmetry Index')
        ax.set_title('Frontal Alpha Asymmetry (Anxiety Marker)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'anxiety_alpha_asymmetry.png')
        plt.close(fig)
        
        print("Saved anxiety alpha asymmetry to anxiety_alpha_asymmetry.png")
    except Exception as e:
        print(f"Could not generate anxiety analysis: {e}")
    
    # Depression analysis
    print("\nPerforming depression analysis...")
    try:
        depression_metrics = depression.analyze_depression_markers(raw_cleaned)
        print("\nDepression markers:")
        if isinstance(depression_metrics, dict) and 'alpha_asymmetry' in depression_metrics:
            for pair, value in list(depression_metrics['alpha_asymmetry'].items())[:3]:
                print(f"Alpha asymmetry {pair}: {value:.4f}")
    except Exception as e:
        print(f"Could not generate depression analysis: {e}")

def generate_ml_examples(raw_cleaned):
    """Generate machine learning examples."""
    print("\nGenerating machine learning examples...")
    
    # Extract features
    feature_dict = features.extract_features(raw_cleaned)
    
    # Create feature vector
    feature_vector = features.create_feature_vector(feature_dict)
    
    print(f"Extracted {len(feature_vector)} features")
    
    # Create a visualization of feature importance
    # For this example, we'll just create a bar chart of the first 20 features
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(min(20, len(feature_vector)))
    ax.bar(x, feature_vector[:20])
    ax.set_xlabel('Feature Index')
    ax.set_ylabel('Feature Value')
    ax.set_title('First 20 EEG Features')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'ml_features.png')
    plt.close(fig)
    
    print("Saved ML features visualization to ml_features.png")

def generate_time_frequency_analysis(raw_cleaned):
    """Generate time-frequency analysis visualizations."""
    print("\nGenerating time-frequency analysis visualizations...")
    
    # Create a shorter segment for time-frequency analysis
    segment = raw_cleaned.copy().crop(tmin=0, tmax=60)  # 1-minute segment
    
    # Generate spectrogram for a single channel
    print("Generating spectrogram...")
    ch_idx = raw_cleaned.ch_names.index('EEG Cz')  # Use Cz channel
    
    # Calculate spectrogram using MNE's built-in function
    from mne.time_frequency import tfr_array_morlet
    
    # Get data from the channel
    data = segment.get_data()[ch_idx:ch_idx+1]
    sfreq = segment.info['sfreq']
    
    # Define frequencies of interest
    freqs = np.arange(1, 41, 1)  # 1 to 40 Hz
    
    # Calculate time-frequency representation
    tfr = tfr_array_morlet(data, sfreq=sfreq, freqs=freqs, 
                          n_cycles=freqs/2, output='power')
    
    # Plot spectrogram
    fig, ax = plt.subplots(figsize=(12, 6))
    times = segment.times
    extent = [times[0], times[-1], freqs[0], freqs[-1]]
    im = ax.imshow(np.log10(tfr[0]), aspect='auto', origin='lower', extent=extent,
                  cmap='viridis')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title(f'Spectrogram - Channel {raw_cleaned.ch_names[ch_idx]}')
    plt.colorbar(im, ax=ax, label='Log Power')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'spectrogram.png', dpi=300)
    plt.close(fig)
    
    print("Saved spectrogram to spectrogram.png")
    
    # Generate time-frequency topomaps
    print("Generating time-frequency topomaps...")
    
    # We'll create topomaps for different frequency bands at different time points
    bands = {
        'Delta': (1, 4),
        'Theta': (4, 8),
        'Alpha': (8, 13),
        'Beta': (13, 30)
    }
    
    # Select time points (every 10 seconds)
    time_points = np.arange(0, 60, 10)
    
    # Create a figure for the topomaps
    fig, axes = plt.subplots(len(bands), len(time_points), 
                            figsize=(3*len(time_points), 3*len(bands)))
    
    # Get the data and positions
    data = segment.get_data()
    pos = mne.channels.layout._find_topomap_coords(segment.info, picks='eeg')
    
    # Loop through bands and time points
    for i, (band_name, (fmin, fmax)) in enumerate(bands.items()):
        # Calculate band power for the entire segment
        from mne.time_frequency import psd_welch
        psds, freqs = psd_welch(segment, fmin=fmin, fmax=fmax, n_fft=1024)
        band_power = np.mean(psds, axis=1)
        
        for j, t in enumerate(time_points):
            # Convert time to sample index
            t_idx = int(t * sfreq)
            if t_idx >= len(segment.times):
                t_idx = len(segment.times) - 1
            
            # Plot the topomap
            ax = axes[i, j]
            im, _ = mne.viz.plot_topomap(band_power, pos, axes=ax, show=False,
                                        cmap='viridis', contours=0)
            
            # Add titles
            if i == 0:
                ax.set_title(f't = {t}s')
            if j == 0:
                ax.set_ylabel(band_name)
    
    # Add a colorbar
    cbar_ax = fig.add_axes([0.92, 0.1, 0.02, 0.8])
    fig.colorbar(im, cax=cbar_ax, label='Power')
    
    plt.suptitle('Time-Frequency Topomaps', fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    plt.savefig(OUTPUT_DIR / 'time_frequency_topomaps.png', dpi=300)
    plt.close(fig)
    
    print("Saved time-frequency topomaps to time_frequency_topomaps.png")


def generate_connectivity_analysis(raw_cleaned):
    """Generate connectivity analysis visualizations."""
    print("\nGenerating connectivity analysis visualizations...")
    
    # Create a shorter segment for connectivity analysis
    segment = raw_cleaned.copy().crop(tmin=0, tmax=60)  # 1-minute segment
    
    # Select only EEG channels (exclude non-EEG channels)
    picks = mne.pick_types(segment.info, eeg=True, exclude=[])
    segment = segment.pick(picks)
    
    # Calculate connectivity matrix using spectral connectivity
    print("Calculating connectivity...")
    from mne.connectivity import spectral_connectivity
    
    # Reshape data for spectral_connectivity
    data = segment.get_data().reshape(1, len(segment.ch_names), -1)
    
    # Calculate connectivity (coherence)
    con, freqs, times, n_epochs, n_tapers = spectral_connectivity(
        data, method='coh', mode='multitaper', sfreq=segment.info['sfreq'],
        fmin=8, fmax=13, faverage=True, mt_adaptive=True, n_jobs=1
    )
    
    # Reshape connectivity matrix
    n_channels = len(segment.ch_names)
    con_matrix = con.reshape(n_channels, n_channels)
    
    # Plot connectivity matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(con_matrix, cmap='viridis', interpolation='nearest')
    ax.set_title('Alpha Band (8-13 Hz) Coherence Matrix')
    
    # Add channel names as ticks
    ax.set_xticks(np.arange(n_channels))
    ax.set_yticks(np.arange(n_channels))
    ax.set_xticklabels(segment.ch_names, rotation=90, fontsize=8)
    ax.set_yticklabels(segment.ch_names, fontsize=8)
    
    # Add colorbar
    plt.colorbar(im, ax=ax, label='Coherence')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'connectivity_matrix.png', dpi=300)
    plt.close(fig)
    
    print("Saved connectivity matrix to connectivity_matrix.png")
    
    # Generate circular connectivity plot
    print("Generating circular connectivity plot...")
    
    # Create a threshold to show only strong connections
    threshold = np.percentile(con_matrix, 90)  # Show top 10% of connections
    
    # Create a circular plot
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # Calculate node positions
    n_nodes = n_channels
    node_angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
    
    # Plot nodes
    ax.scatter(node_angles, np.ones(n_nodes), s=100, c='blue', zorder=3)
    
    # Plot connections
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            if con_matrix[i, j] > threshold:
                # Calculate arc
                theta = np.linspace(node_angles[i], node_angles[j], 100)
                radius = 1 - 0.1 * np.sin(np.linspace(0, np.pi, 100))
                
                # Plot the arc with color based on strength
                ax.plot(theta, radius, color='red', alpha=con_matrix[i, j], linewidth=con_matrix[i, j] * 3)
    
    # Add node labels
    for i, (angle, name) in enumerate(zip(node_angles, segment.ch_names)):
        ha = 'left' if np.pi/2 <= angle < 3*np.pi/2 else 'right'
        ax.text(angle, 1.1, name, ha=ha, va='center', rotation=np.degrees(angle), fontsize=8)
    
    ax.set_title('Alpha Band (8-13 Hz) Connectivity (Top 10%)')
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_frame_on(False)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'circular_connectivity.png', dpi=300)
    plt.close(fig)
    
    print("Saved circular connectivity plot to circular_connectivity.png")


def generate_source_localization(raw_cleaned):
    """Generate source localization visualizations."""
    print("\nGenerating source localization visualizations...")
    
    # This is a simplified version as full source localization requires additional data
    # like head model, MRI, etc. We'll create a simulated version for demonstration
    
    print("Generating simulated Brodmann area activity map...")
    
    # Create a figure showing Brodmann areas with simulated activity
    from nilearn import plotting
    import nibabel as nib
    from nilearn import datasets
    
    # Get a template MNI brain
    fsaverage = datasets.fetch_surf_fsaverage()
    
    # Create simulated activity data for Brodmann areas
    # This is just for visualization purposes
    n_vertices = 20484  # Number of vertices in fsaverage
    activity = np.zeros(n_vertices)
    
    # Simulate activity in some regions (this is just for demonstration)
    # In a real scenario, this would be calculated from the EEG data
    import random
    random.seed(42)  # For reproducibility
    
    # Simulate some active regions
    for i in range(2000):
        idx = random.randint(0, n_vertices-1)
        activity[idx] = random.uniform(0, 1)
    
    # Plot the activity on the brain surface
    fig = plt.figure(figsize=(12, 8))
    
    # Left hemisphere
    ax1 = fig.add_subplot(121, projection='3d')
    plotting.plot_surf_stat_map(
        fsaverage['infl_left'], activity[:n_vertices//2],
        hemi='left', view='lateral', colorbar=True,
        title='Simulated Brain Activity (Left)',
        axes=ax1
    )
    
    # Right hemisphere
    ax2 = fig.add_subplot(122, projection='3d')
    plotting.plot_surf_stat_map(
        fsaverage['infl_right'], activity[n_vertices//2:],
        hemi='right', view='lateral', colorbar=True,
        title='Simulated Brain Activity (Right)',
        axes=ax2
    )
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'source_localization.png', dpi=300)
    plt.close(fig)
    
    print("Saved simulated source localization to source_localization.png")


def generate_advanced_statistical_visualizations(raw_cleaned):
    """Generate advanced statistical visualizations."""
    print("\nGenerating advanced statistical visualizations...")
    
    # Extract features for statistical analysis
    from qeeg.ml import features
    
    print("Extracting features for statistical analysis...")
    feature_dict = features.extract_features(raw_cleaned)
    feature_vector = features.create_feature_vector(feature_dict)
    
    # Create a radar chart of normalized feature values
    print("Generating feature radar chart...")
    
    # Select a subset of features for the radar chart
    n_features = 8
    feature_indices = np.linspace(0, len(feature_vector)-1, n_features, dtype=int)
    selected_features = feature_vector[feature_indices]
    
    # Normalize the selected features
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    normalized_features = scaler.fit_transform(selected_features.reshape(-1, 1)).flatten()
    
    # Create feature names
    feature_names = [f"Feature {i+1}" for i in range(n_features)]
    
    # Create the radar chart
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, polar=True)
    
    # Set the angles for each feature
    angles = np.linspace(0, 2*np.pi, n_features, endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    # Add the feature values
    values = normalized_features.tolist()
    values += values[:1]  # Close the loop
    
    # Plot the values
    ax.plot(angles, values, 'o-', linewidth=2)
    ax.fill(angles, values, alpha=0.25)
    
    # Set the labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feature_names)
    
    # Set y-ticks
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
    
    ax.set_title('EEG Feature Radar Chart', size=15)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'feature_radar_chart.png', dpi=300)
    plt.close(fig)
    
    print("Saved feature radar chart to feature_radar_chart.png")
    
    # Create a heatmap of feature correlations
    print("Generating feature correlation heatmap...")
    
    # Reshape feature vector for correlation analysis
    # We'll use the band power features
    if 'band_power' in feature_dict:
        band_powers = feature_dict['band_power']
        
        # Get band names and channel names
        bands = list(band_powers.keys())
        channels = raw_cleaned.ch_names
        
        # Create a matrix of band powers (channels x bands)
        band_power_matrix = np.zeros((len(channels), len(bands)))
        for i, band in enumerate(bands):
            band_power_matrix[:, i] = band_powers[band]
        
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(band_power_matrix.T)
        
        # Plot correlation matrix
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        
        # Add band names as ticks
        ax.set_xticks(np.arange(len(bands)))
        ax.set_yticks(np.arange(len(bands)))
        ax.set_xticklabels(bands)
        ax.set_yticklabels(bands)
        
        # Rotate x tick labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Correlation Coefficient')
        
        # Add title
        ax.set_title('Frequency Band Correlation Matrix')
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'band_correlation_matrix.png', dpi=300)
        plt.close(fig)
        
        print("Saved band correlation matrix to band_correlation_matrix.png")


def main():
    """Main function to generate all examples."""
    print("QEEG Example Generator")
    print("=====================\n")
    
    # Check if data file exists
    data_path = check_data_file()
    
    # Load and preprocess data
    raw, raw_filtered, raw_cleaned = load_and_preprocess_data(data_path)
    
    # Generate basic examples
    generate_preprocessing_examples(raw, raw_filtered, raw_cleaned)
    generate_spectral_analysis_examples(raw_cleaned)
    generate_topographic_maps(raw_cleaned)
    generate_condition_specific_examples(raw_cleaned)
    generate_ml_examples(raw_cleaned)
    
    # Generate advanced visualizations
    generate_time_frequency_analysis(raw_cleaned)
    generate_connectivity_analysis(raw_cleaned)
    generate_source_localization(raw_cleaned)
    generate_advanced_statistical_visualizations(raw_cleaned)
    
    print("\nAll examples generated successfully!")
    print(f"Output files are in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
