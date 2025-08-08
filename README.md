# Qeeg

A Python package for quantitative EEG analysis and neurological condition assessment.

## Features

- EEG data preprocessing (filtering, artifact removal)
- Spectral analysis (power spectral density, frequency bands)
- Asymmetry analysis
- Brodmann area analysis
- Epileptiform activity detection (OIRDA, FIRDA, spikes)
- Neurological condition assessment (epilepsy, ADHD, depression, anxiety, etc.)
- EEG visualization tools
- Memory-efficient processing for large datasets
- Performance benchmarking and optimization
- Comprehensive validation and error handling

## Installation

```bash
# Install from PyPI (not yet available)
pip install qeeg

# Install from source
git clone https://github.com/kapeleshh/qeeg.git
cd qeeg
pip install -e .
```

## Quick Start

```python
import mne
import qeeg

# Load EEG data
raw = mne.io.read_raw_edf('your_eeg_file.edf', preload=True)

# Preprocess the data
raw_filtered = qeeg.preprocessing.filtering.bandpass_filter(raw, l_freq=1.0, h_freq=40.0)
raw_cleaned = qeeg.preprocessing.artifacts.remove_artifacts_ica(raw_filtered)

# Perform spectral analysis
bands = qeeg.analysis.spectral.compute_band_powers(raw_cleaned)

# Detect epileptiform activity
spikes = qeeg.analysis.epileptiform.detect_spikes(raw_cleaned)

# Visualize results
qeeg.visualization.topomaps.plot_band_topomaps(bands)
```

## Documentation

For detailed documentation, see [docs/](docs/).

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
