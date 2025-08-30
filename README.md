# QEEG: Quantitative EEG Analysis Toolkit

A Python package for analyzing EEG data with focus on neurological conditions.

## ⚠️ Research Software Notice

This package is primarily designed for research purposes. The API may change between minor versions during the 0.x development phase. For production use, please pin to a specific version.

## Installation

### Using pip

```bash
pip install qeeg
```

### Using Docker

We provide Docker support for easy setup and consistent environment:

```bash
# Clone the repository
git clone https://github.com/kapeleshh/qeeg.git
cd qeeg

# Build and run the Docker container
docker-compose build
docker-compose run --rm qeeg
```

#### Generating Examples with Docker

To generate comprehensive examples using real EEG data:

```bash
# Run the example generator script
./run_example.sh
```

This will:
1. Build the Docker image if needed
2. Create the output directory
3. Run the example generator inside the Docker container
4. Generate visualizations in the examples/output directory

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

## Advanced Examples

### Spectral Analysis

```python
from qeeg.analysis import spectral

# Compute power spectral density
psds, freqs = spectral.compute_psd(raw)

# Compute band powers
band_powers = spectral.compute_band_powers(raw)

# Compute relative band powers
rel_powers = spectral.compute_relative_band_powers(raw)

# Compute band power ratios (e.g., theta/beta ratio for ADHD)
ratios = spectral.compute_band_power_ratios(raw)

# Compute peak frequency
peak_freqs, peak_powers = spectral.compute_peak_frequency(raw)

# Analyze relative power across the scalp
results = spectral.analyze_relative_power(raw)
```

### Time-Frequency Analysis

```python
import numpy as np
import matplotlib.pyplot as plt
from mne.time_frequency import tfr_array_morlet

# Get data from a specific channel
ch_idx = raw.ch_names.index('Cz')
data = raw.get_data()[ch_idx:ch_idx+1]
sfreq = raw.info['sfreq']

# Define frequencies of interest
freqs = np.arange(1, 41, 1)  # 1 to 40 Hz

# Calculate time-frequency representation
tfr = tfr_array_morlet(data, sfreq=sfreq, freqs=freqs, 
                      n_cycles=freqs/2, output='power')

# Plot spectrogram
plt.figure(figsize=(12, 6))
plt.imshow(np.log10(tfr[0]), aspect='auto', origin='lower',
          extent=[raw.times[0], raw.times[-1], freqs[0], freqs[-1]],
          cmap='viridis')
plt.colorbar(label='Log Power')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title('Spectrogram')
plt.tight_layout()
```

### Connectivity Analysis

```python
from mne.connectivity import spectral_connectivity

# Reshape data for spectral_connectivity
data = raw.get_data().reshape(1, len(raw.ch_names), -1)

# Calculate connectivity (coherence)
con, freqs, times, n_epochs, n_tapers = spectral_connectivity(
    data, method='coh', mode='multitaper', sfreq=raw.info['sfreq'],
    fmin=8, fmax=13, faverage=True, mt_adaptive=True, n_jobs=1
)

# Reshape connectivity matrix
n_channels = len(raw.ch_names)
con_matrix = con.reshape(n_channels, n_channels)

# Plot connectivity matrix
plt.figure(figsize=(10, 8))
plt.imshow(con_matrix, cmap='viridis')
plt.colorbar(label='Coherence')
plt.title('Alpha Band (8-13 Hz) Coherence Matrix')
plt.tight_layout()
```

### Condition-Specific Analysis

```python
from qeeg.conditions import adhd, anxiety, depression

# ADHD analysis
tb_ratios = adhd.calculate_theta_beta_ratio(raw)
frontal_asymmetry = adhd.analyze_frontal_asymmetry(raw)
adhd_results = adhd.detect_adhd_patterns(raw)
adhd_report = adhd.generate_adhd_report(raw)

# Anxiety analysis
alpha_asymmetry = anxiety.calculate_frontal_alpha_asymmetry(raw)
anxiety_results = anxiety.detect_anxiety_patterns(raw)

# Depression analysis
depression_metrics = depression.analyze_depression_markers(raw)
depression_report = depression.generate_depression_report(raw)
```

### Machine Learning

```python
from qeeg.ml import features, classification
import numpy as np

# Extract features
feature_dict = features.extract_features(raw)
feature_vector = features.create_feature_vector(feature_dict)

# Create labels (example)
labels = np.array([0, 1, 0, 1, 0])  # Binary classification example

# Train a classifier
pipeline, metrics = classification.train_classifier(feature_vector, labels, classifier="svm")

# Cross-validate
cv_results = classification.cross_validate(feature_vector, labels, cv=5)

# Make predictions on new data
new_predictions = classification.predict(new_raw, pipeline)
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
<!-- 
## Documentation

Full documentation is available at [https://qeeg.readthedocs.io](https://qeeg.readthedocs.io)
-->
## Development

### Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/kapeleshh/qeeg.git
cd qeeg

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"
```

### Using Docker for Development

Docker provides a consistent development environment across different platforms:

```bash
# Clone the repository
git clone https://github.com/kapeleshh/qeeg.git
cd qeeg

# Build and start the Docker container
docker-compose build
docker-compose run --rm qeeg

# Inside the container, you can run tests and examples
python -m pytest
python generate_examples.py
```

#### Interactive Development with Docker

For interactive development inside the Docker container, you can use the provided script:

```bash
# Run an interactive bash shell inside the Docker container
./docker_bash.sh
```

This will give you a bash shell inside the container where you can:
- Run Python scripts interactively
- Install additional packages
- Debug issues
- Explore the file system

All changes to files in the mounted volumes (the project directory) will persist outside the container.

### Run Tests

```bash
pytest
```

## Citation

If you use this software in your research, please cite:

```
Kapeleshh KS et al. (2025). QEEG: Quantitative EEG Analysis Toolkit. GitHub. https://github.com/kapeleshh/qeeg
```
