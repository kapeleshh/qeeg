# QEEG: Quantitative EEG Analysis Toolkit

A Python package for analyzing EEG data with focus on neurological conditions.

## ⚠️ Research Software Notice

This package is primarily designed for research purposes. The API may change between minor versions during the 0.x development phase. For production use, please pin to a specific version.

## Installation

```bash
pip install qeeg
```

## Quick Start

```python
import mne
from qeeg.analysis import spectral
from qeeg.visualization import topomaps

# Load EEG data
raw = mne.io.read_raw_edf("sample.edf", preload=True)

# Compute band powers
band_powers = spectral.compute_band_powers(raw)

# Visualize topographic maps
fig = topomaps.plot_band_topomaps(raw)
```

## Features

- **Analysis Modules**:
  - Spectral analysis (PSD, band powers)
  - Asymmetry analysis
  - Epileptiform activity detection
  - Brodmann area mapping and analysis

- **Condition-Specific Analysis**:
  - ADHD (theta/beta ratio, frontal asymmetry)
  - Anxiety (alpha asymmetry, beta power)
  - Autism (coherence, gamma power)
  - Depression (frontal alpha asymmetry, theta activity)
  - Other conditions (insomnia, TBI, Alzheimer's, Parkinson's, schizophrenia)

- **Visualization Tools**:
  - Topographic maps
  - Power spectra and time-frequency plots
  - Brain activation visualization
  - Comprehensive HTML reports

- **Command-Line Interface**:
  - Data preprocessing
  - Analysis execution
  - Visualization generation

## Documentation

Full documentation is available at [https://qeeg.readthedocs.io](https://qeeg.readthedocs.io)

## Development

### Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/yourusername/qeeg.git
cd qeeg

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"
```

### Run Tests

```bash
pytest
```

## Citation

If you use this software in your research, please cite:

```
Your Name et al. (2025). QEEG: Quantitative EEG Analysis Toolkit. GitHub. https://github.com/yourusername/qeeg
```

## License

MIT License
