# Epilepsy-EEG

A Python package for EEG analysis focused on epilepsy detection and neurological condition assessment.

## Features

- EEG data preprocessing (filtering, artifact removal)
- Spectral analysis (power spectral density, frequency bands)
- Asymmetry analysis
- Brodmann area analysis
- Epileptiform activity detection (OIRDA, FIRDA, spikes)
- Neurological condition assessment (ADHD, depression, anxiety, etc.)
- EEG visualization tools

## Installation

```bash
# Install from PyPI (not yet available)
pip install epilepsy-eeg

# Install from source
git clone https://github.com/kapeleshh/epilepsy-eeg.git
cd epilepsy-eeg
pip install -e .
```

## Quick Start

```python
import mne
import epilepsy_eeg as eeg

# Load EEG data
raw = mne.io.read_raw_edf('your_eeg_file.edf', preload=True)

# Preprocess the data
raw_filtered = eeg.preprocessing.filtering.bandpass_filter(raw, l_freq=1.0, h_freq=40.0)
raw_cleaned = eeg.preprocessing.artifacts.remove_artifacts_ica(raw_filtered)

# Perform spectral analysis
bands = eeg.analysis.spectral.compute_band_powers(raw_cleaned)

# Detect epileptiform activity
spikes = eeg.analysis.epileptiform.detect_spikes(raw_cleaned)

# Visualize results
eeg.visualization.topomaps.plot_band_topomaps(bands)
```

## Documentation

For detailed documentation, see [docs/](docs/).

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
