# QEEG: Quantitative EEG Analysis Toolkit

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Documentation Status](https://img.shields.io/badge/docs-in%20progress-orange.svg)](https://github.com/kapeleshh/qeeg/tree/main/docs)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A comprehensive Python package for quantitative EEG (QEEG) analysis and neurological condition assessment. QEEG provides tools for preprocessing, analyzing, and visualizing EEG data, with a focus on clinical applications and research in neurology and neuropsychiatry.

<p align="center">
  <img src="https://via.placeholder.com/800x400?text=QEEG+Visualization+Example" alt="QEEG Visualization Example" width="600"/>
</p>

## 🌟 Features

### Data Preprocessing
- **Filtering**: Bandpass, notch, and custom filters for noise removal
- **Artifact Removal**: ICA-based and automated artifact detection and removal
- **Montage Management**: Support for various EEG montages and reference schemes

### Analysis
- **Spectral Analysis**: Power spectral density, frequency bands, and spectral indices
- **Asymmetry Analysis**: Interhemispheric asymmetry metrics and visualization
- **Brodmann Area Analysis**: Mapping EEG activity to Brodmann areas
- **Epileptiform Activity Detection**: Automated detection of OIRDA, FIRDA, spikes, and other epileptiform patterns

### Clinical Applications
- **Neurological Condition Assessment**: Tools for analyzing EEG patterns associated with:
  - Epilepsy
  - ADHD
  - Depression
  - Anxiety
  - Autism Spectrum Disorder
  - Other neurological conditions

### Visualization
- **Topographic Maps**: Brain activity visualization with customizable parameters
- **Spectral Plots**: PSD, time-frequency plots, and coherence visualizations
- **Report Generation**: Automated clinical report generation

### Advanced Utilities
- **Memory Management**: Efficient processing of large EEG datasets
- **Validation**: Comprehensive input validation and data quality checks
- **Error Handling**: Detailed error messages with troubleshooting guidance
- **Parallel Processing**: Multi-core processing for computationally intensive operations

## 📦 Installation

### From PyPI (coming soon)
```bash
pip install qeeg
```

### From Source
```bash
# Clone the repository
git clone https://github.com/kapeleshh/qeeg.git
cd qeeg

# Install in development mode
pip install -e .

# Install with development dependencies
pip install -e ".[dev]"
```

### Dependencies
QEEG requires the following main dependencies:
- numpy (≥1.20.0)
- scipy (≥1.6.0)
- matplotlib (≥3.3.0)
- mne (≥1.0.0)
- pandas (≥1.2.0)
- scikit-learn (≥0.24.0)
- networkx (≥2.5.0)

## 🚀 Quick Start

### Basic EEG Analysis

```python
import mne
import qeeg
import matplotlib.pyplot as plt

# Load EEG data
raw = mne.io.read_raw_edf('your_eeg_file.edf', preload=True)

# Preprocess the data
raw_filtered = qeeg.preprocessing.filtering.bandpass_filter(raw, l_freq=1.0, h_freq=40.0)
raw_cleaned = qeeg.preprocessing.artifacts.remove_artifacts_ica(raw_filtered)

# Perform spectral analysis
bands = qeeg.analysis.spectral.compute_band_powers(raw_cleaned)

# Visualize results
fig = qeeg.visualization.topomaps.plot_band_topomaps(bands)
plt.show()
```

### Memory-Efficient Processing for Large Datasets

```python
import mne
from qeeg.utils.memory import process_large_eeg, MemoryMonitor
from qeeg.preprocessing.filtering import bandpass_filter

# Load a large EEG file
raw = mne.io.read_raw_edf("large_eeg_file.edf", preload=True)

# Process in chunks to save memory
with MemoryMonitor(warning_threshold=70.0, error_threshold=90.0):
    filtered_raw = process_large_eeg(
        raw,
        bandpass_filter,
        chunk_duration=30.0,  # Process 30 seconds at a time
        overlap=2.0,          # 2 seconds overlap between chunks
        l_freq=1.0,
        h_freq=40.0
    )
```

### Epileptiform Activity Detection

```python
import mne
import qeeg
import matplotlib.pyplot as plt

# Load EEG data
raw = mne.io.read_raw_edf('epilepsy_sample.edf', preload=True)

# Preprocess
raw_filtered = qeeg.preprocessing.filtering.bandpass_filter(raw, l_freq=1.0, h_freq=40.0)

# Detect spikes
spikes = qeeg.analysis.epileptiform.detect_spikes(raw_filtered)

# Print results
print(f"Detected {len(spikes)} spikes")

# Plot the first spike
if spikes:
    start_time = spikes[0]['time'] - 0.5
    raw_filtered.plot(start=start_time, duration=2.0, 
                     highlight_spans=[(spikes[0]['time'], spikes[0]['time'] + 0.1)])
```

### Machine Learning for EEG Classification

```python
import qeeg
from qeeg.ml.features import extract_features
from qeeg.ml.classification import train_classifier, evaluate_classifier

# Extract features from preprocessed EEG data
features = extract_features(raw_cleaned)

# Train a classifier (e.g., for epilepsy detection)
X_train, X_test, y_train, y_test = train_test_split(features, labels)
classifier = train_classifier(X_train, y_train, model_type='random_forest')

# Evaluate the classifier
accuracy, precision, recall, f1 = evaluate_classifier(classifier, X_test, y_test)
print(f"Accuracy: {accuracy:.2f}, F1 Score: {f1:.2f}")
```

## 📚 Documentation

### API Reference
For detailed API documentation, see [docs/](docs/).

### Tutorials
- [Basic EEG Preprocessing](docs/source/tutorials/preprocessing.md)
- [Spectral Analysis](docs/source/tutorials/spectral_analysis.md)
- [Epileptiform Activity Detection](docs/source/tutorials/epileptiform_detection.md)
- [Working with Large Datasets](docs/source/tutorials/large_datasets.md)
- [Machine Learning with EEG](docs/source/tutorials/machine_learning.md)

### Utility Modules

QEEG includes several utility modules to enhance reliability and usability:

#### Exception Handling
```python
from qeeg.utils.exceptions import ValidationError, DataQualityError

try:
    # Your code here
except DataQualityError as e:
    print(f"Data quality issue: {e}")
    print(qeeg.utils.exceptions.get_troubleshooting_info(e))
```

#### Input Validation
```python
from qeeg.utils.validation import validate_raw

# Validate raw EEG data before processing
validation_results = validate_raw(raw)
print(f"Data duration: {validation_results['duration']:.2f} seconds")
print(f"Number of channels: {validation_results['n_channels']}")
```

#### Memory Management
```python
from qeeg.utils.memory import memory_usage

@memory_usage
def process_data(raw):
    # Process data
    return processed_data

# Call the function with memory usage tracking
result = process_data(raw)
```

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup
```bash
# Clone the repository
git clone https://github.com/kapeleshh/qeeg.git
cd qeeg

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Check code style
flake8 qeeg
black qeeg
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📝 Citation

If you use QEEG in your research, please cite it as follows:

```bibtex
@software{qeeg2025,
  author = {EEG Analysis Team},
  title = {QEEG: Quantitative EEG Analysis Toolkit},
  url = {https://github.com/kapeleshh/qeeg},
  version = {0.1.0},
  year = {2025},
}
```

## 🙏 Acknowledgments

- [MNE-Python](https://mne.tools/stable/index.html) for providing the foundation for EEG data handling
- The neuroscience community for their research and insights into EEG analysis
- All contributors who have helped improve this package
