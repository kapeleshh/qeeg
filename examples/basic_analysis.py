"""
Basic EEG analysis example using the epilepsy_eeg package.

This example demonstrates how to:
1. Load EEG data
2. Preprocess the data (filtering, artifact removal)
3. Perform spectral analysis
4. Detect epileptiform activity
5. Analyze asymmetry
6. Analyze Brodmann areas
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import mne
import epilepsy_eeg as eeg

# Set random seed for reproducibility
np.random.seed(42)

# Create a simulated EEG dataset if no real data is available
def create_simulated_eeg(duration=60, sfreq=256, n_channels=19):
    """Create a simulated EEG dataset."""
    # Define channel names based on the 10-20 system
    ch_names = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 
                'T3', 'C3', 'Cz', 'C4', 'T4', 
                'T5', 'P3', 'Pz', 'P4', 'T6', 
                'O1', 'O2'][:n_channels]
    
    # Create simulated data
    n_samples = int(duration * sfreq)
    data = np.random.randn(n_channels, n_samples)
    
    # Add alpha oscillations (8-12 Hz) to occipital channels
    t = np.arange(n_samples) / sfreq
    alpha_oscillation = np.sin(2 * np.pi * 10 * t)  # 10 Hz oscillation
    for ch_name in ['O1', 'O2', 'P3', 'P4']:
        if ch_name in ch_names:
            ch_idx = ch_names.index(ch_name)
            data[ch_idx, :] += 2 * alpha_oscillation
    
    # Add some spikes to simulate epileptiform activity
    for _ in range(5):
        ch_idx = np.random.randint(0, n_channels)
        sample_idx = np.random.randint(0, n_samples - 100)
        data[ch_idx, sample_idx:sample_idx+50] += 5 * np.sin(np.pi * np.arange(50) / 50)
    
    # Create MNE Raw object
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    raw = mne.io.RawArray(data, info)
    
    # Set montage
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage)
    
    return raw

def main():
    # Check if a sample EEG file exists, otherwise create simulated data
    sample_file = os.path.join('examples', 'data', 'sample.edf')
    if os.path.exists(sample_file):
        print(f"Loading EEG data from {sample_file}")
        raw = mne.io.read_raw_edf(sample_file, preload=True)
    else:
        print("No sample EEG file found. Creating simulated data.")
        raw = create_simulated_eeg()
    
    # Print information about the data
    print("\nEEG Data Information:")
    print(f"Number of channels: {len(raw.ch_names)}")
    print(f"Channel names: {raw.ch_names}")
    print(f"Sampling frequency: {raw.info['sfreq']} Hz")
    print(f"Duration: {raw.times[-1]:.2f} seconds")
    
    # Step 1: Preprocess the data
    print("\nStep 1: Preprocessing the data...")
    
    # Apply bandpass filter
    raw_filtered = eeg.preprocessing.filtering.bandpass_filter(
        raw, l_freq=1.0, h_freq=40.0
    )
    
    # Remove artifacts using ICA
    raw_cleaned = eeg.preprocessing.artifacts.remove_artifacts_ica(
        raw_filtered, n_components=10
    )
    
    # Step 2: Perform spectral analysis
    print("\nStep 2: Performing spectral analysis...")
    
    # Compute power in frequency bands
    band_powers = eeg.analysis.spectral.compute_band_powers(raw_cleaned)
    
    # Print average power in each band
    print("\nAverage power in each frequency band:")
    for band, powers in band_powers.items():
        print(f"{band}: {np.mean(powers):.6f}")
    
    # Step 3: Detect epileptiform activity
    print("\nStep 3: Detecting epileptiform activity...")
    
    # Detect spikes
    spikes = eeg.analysis.epileptiform.detect_spikes(raw_cleaned)
    print(f"Detected {len(spikes)} spikes.")
    
    # Detect OIRDA and FIRDA
    oirda = eeg.analysis.epileptiform.detect_oirda(raw_cleaned)
    firda = eeg.analysis.epileptiform.detect_firda(raw_cleaned)
    print(f"Detected {len(oirda)} OIRDA events and {len(firda)} FIRDA events.")
    
    # Step 4: Analyze asymmetry
    print("\nStep 4: Analyzing asymmetry...")
    
    # Compute asymmetry indices
    asymmetry_indices = eeg.analysis.asymmetry.compute_asymmetry_index(raw_cleaned)
    
    # Print asymmetry indices
    print("\nAsymmetry indices:")
    for pair, index in asymmetry_indices.items():
        severity = eeg.analysis.asymmetry.classify_asymmetry_severity(index)
        print(f"{pair}: {index:.4f} ({severity})")
    
    # Step 5: Analyze Brodmann areas
    print("\nStep 5: Analyzing Brodmann areas...")
    
    # Map channels to Brodmann areas
    channel_to_areas = eeg.analysis.brodmann.map_channels_to_brodmann(raw_cleaned)
    
    # Print mapping for a few channels
    print("\nBrodmann areas for selected channels:")
    for ch_name in raw_cleaned.ch_names[:5]:
        areas = channel_to_areas.get(ch_name, [])
        if areas:
            area_functions = [f"Area {area} ({eeg.analysis.brodmann.get_brodmann_function(area)})" for area in areas]
            print(f"{ch_name}: {', '.join(area_functions)}")
        else:
            print(f"{ch_name}: No Brodmann areas mapped")
    
    # Step 6: Plot the results
    print("\nStep 6: Plotting the results...")
    
    # Plot the raw and cleaned data
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    raw.plot(n_channels=5, duration=10, show=False, scalings='auto', ax=axes[0])
    axes[0].set_title('Raw EEG')
    raw_cleaned.plot(n_channels=5, duration=10, show=False, scalings='auto', ax=axes[1])
    axes[1].set_title('Cleaned EEG')
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    os.makedirs('examples/output', exist_ok=True)
    
    # Save the figure
    plt.savefig('examples/output/eeg_comparison.png')
    print("Saved EEG comparison plot to examples/output/eeg_comparison.png")
    
    # Plot the power spectrum
    fig, ax = plt.subplots(figsize=(10, 6))
    psds, freqs = eeg.analysis.spectral.compute_psd(raw_cleaned)
    for i, ch_name in enumerate(raw_cleaned.ch_names[:5]):
        ax.semilogy(freqs, psds[i], label=ch_name)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power Spectral Density (µV²/Hz)')
    ax.set_title('Power Spectrum')
    ax.legend()
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('examples/output/power_spectrum.png')
    print("Saved power spectrum plot to examples/output/power_spectrum.png")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
