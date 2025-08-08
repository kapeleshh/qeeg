Usage
=====

This page provides examples of how to use the QEEG package for EEG data analysis.

Basic Usage
----------

Import the package:

.. code-block:: python

    import qeeg

Loading EEG Data
---------------

QEEG uses MNE-Python for EEG data handling:

.. code-block:: python

    import mne
    from qeeg.preprocessing import filtering

    # Load EEG data
    raw = mne.io.read_raw_edf("sample.edf", preload=True)
    
    # Apply a bandpass filter
    filtered_raw = filtering.bandpass_filter(raw, l_freq=1.0, h_freq=40.0)

Artifact Removal
---------------

Remove artifacts from EEG data:

.. code-block:: python

    from qeeg.preprocessing import artifacts
    
    # Remove eye blink artifacts using ICA
    cleaned_raw = artifacts.remove_eye_artifacts(filtered_raw, n_components=20)

Spectral Analysis
----------------

Perform spectral analysis on EEG data:

.. code-block:: python

    from qeeg.analysis import spectral
    
    # Calculate power spectral density
    psd, freqs = spectral.compute_psd(cleaned_raw, fmin=1.0, fmax=40.0)
    
    # Calculate band powers
    band_powers = spectral.compute_band_powers(
        cleaned_raw, 
        bands={
            'delta': (1, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 45)
        }
    )

Visualization
------------

Visualize EEG data and analysis results:

.. code-block:: python

    from qeeg.visualization import spectra, topomaps
    
    # Plot power spectrum
    spectra.plot_psd(cleaned_raw, fmin=1.0, fmax=40.0)
    
    # Plot topographic maps of band powers
    topomaps.plot_band_topomaps(band_powers, cleaned_raw.info)

Machine Learning
--------------

Use machine learning for EEG analysis:

.. code-block:: python

    from qeeg.ml import features, classification
    
    # Extract features from EEG data
    feature_dict = features.extract_all_features(cleaned_raw)
    
    # Use features for classification
    X = feature_dict['features']
    y = [0, 1, 0, 1]  # Example labels
    
    # Train a classifier
    clf = classification.train_classifier(X, y, classifier='svm')
