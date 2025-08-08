"""
Parallel processing example using the qeeg package.

This example demonstrates how to:
1. Use parallel processing for feature extraction
2. Use parallel processing for preprocessing
3. Use parallel processing for cross-validation
4. Use parallel processing for grid search
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import mne
import qeeg

# Set random seed for reproducibility
np.random.seed(42)

# Create a simulated EEG dataset if no real data is available
def create_simulated_eeg_dataset(n_samples=20, duration=60, sfreq=256, n_channels=19):
    """Create a simulated EEG dataset with labels."""
    # Define channel names based on the 10-20 system
    ch_names = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 
                'T3', 'C3', 'Cz', 'C4', 'T4', 
                'T5', 'P3', 'Pz', 'P4', 'T6', 
                'O1', 'O2'][:n_channels]
    
    # Initialize lists to store data and labels
    raw_list = []
    labels = []
    
    # Create normal EEG samples
    for i in range(n_samples // 2):
        # Create simulated data
        n_samples_eeg = int(duration * sfreq)
        data = np.random.randn(n_channels, n_samples_eeg) * 0.5
        
        # Add alpha oscillations (8-12 Hz) to occipital channels
        t = np.arange(n_samples_eeg) / sfreq
        alpha_oscillation = np.sin(2 * np.pi * 10 * t)  # 10 Hz oscillation
        for ch_name in ['O1', 'O2', 'P3', 'P4']:
            if ch_name in ch_names:
                ch_idx = ch_names.index(ch_name)
                data[ch_idx, :] += 2 * alpha_oscillation
        
        # Create MNE Raw object
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
        raw = mne.io.RawArray(data, info)
        
        # Set montage
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage)
        
        # Add to list
        raw_list.append(raw)
        labels.append(0)  # Normal
    
    # Create abnormal EEG samples (with simulated epileptiform activity)
    for i in range(n_samples // 2):
        # Create simulated data
        n_samples_eeg = int(duration * sfreq)
        data = np.random.randn(n_channels, n_samples_eeg) * 0.5
        
        # Add alpha oscillations (8-12 Hz) to occipital channels
        t = np.arange(n_samples_eeg) / sfreq
        alpha_oscillation = np.sin(2 * np.pi * 10 * t)  # 10 Hz oscillation
        for ch_name in ['O1', 'O2', 'P3', 'P4']:
            if ch_name in ch_names:
                ch_idx = ch_names.index(ch_name)
                data[ch_idx, :] += 2 * alpha_oscillation
        
        # Add spikes to simulate epileptiform activity
        for _ in range(10):
            ch_idx = np.random.randint(0, n_channels)
            sample_idx = np.random.randint(0, n_samples_eeg - 100)
            data[ch_idx, sample_idx:sample_idx+50] += 5 * np.sin(np.pi * np.arange(50) / 50)
        
        # Add asymmetry
        if 'F3' in ch_names and 'F4' in ch_names:
            f3_idx = ch_names.index('F3')
            f4_idx = ch_names.index('F4')
            data[f3_idx, :] *= 1.5  # Increase power in left frontal region
        
        # Create MNE Raw object
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
        raw = mne.io.RawArray(data, info)
        
        # Set montage
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage)
        
        # Add to list
        raw_list.append(raw)
        labels.append(1)  # Abnormal
    
    return raw_list, np.array(labels)

def main():
    print("Parallel Processing Example")
    print("==========================")
    
    # Step 1: Load or create EEG data
    print("\nStep 1: Loading/creating EEG data...")
    raw_list, labels = create_simulated_eeg_dataset(n_samples=20)
    print(f"Created dataset with {len(raw_list)} samples ({np.sum(labels == 0)} normal, {np.sum(labels == 1)} abnormal)")
    
    # Step 2: Parallel preprocessing
    print("\nStep 2: Parallel preprocessing...")
    
    # Sequential preprocessing (for comparison)
    start_time = time.time()
    filtered_list_seq = []
    for raw in raw_list:
        filtered = qeeg.preprocessing.filtering.bandpass_filter(raw, l_freq=1.0, h_freq=40.0)
        filtered_list_seq.append(filtered)
    seq_time = time.time() - start_time
    print(f"Sequential preprocessing time: {seq_time:.2f} seconds")
    
    # Parallel preprocessing
    start_time = time.time()
    filtered_list_par = qeeg.utils.parallel.parallel_preprocess(
        raw_list,
        qeeg.preprocessing.filtering.bandpass_filter,
        l_freq=1.0,
        h_freq=40.0
    )
    par_time = time.time() - start_time
    print(f"Parallel preprocessing time: {par_time:.2f} seconds")
    print(f"Speedup: {seq_time / par_time:.2f}x")
    
    # Step 3: Parallel feature extraction
    print("\nStep 3: Parallel feature extraction...")
    
    # Sequential feature extraction (for comparison)
    start_time = time.time()
    features_list_seq = []
    for raw in filtered_list_par:
        features = qeeg.ml.features.extract_all_features(
            raw,
            include_wavelet=False,  # Exclude wavelet features for speed
            include_connectivity=True
        )
        features_list_seq.append(features)
    seq_time = time.time() - start_time
    print(f"Sequential feature extraction time: {seq_time:.2f} seconds")
    
    # Parallel feature extraction
    start_time = time.time()
    features_list_par = qeeg.utils.parallel.parallel_extract_features(
        filtered_list_par,
        qeeg.ml.features.extract_all_features,
        include_wavelet=False,  # Exclude wavelet features for speed
        include_connectivity=True
    )
    par_time = time.time() - start_time
    print(f"Parallel feature extraction time: {par_time:.2f} seconds")
    print(f"Speedup: {seq_time / par_time:.2f}x")
    
    # Step 4: Create feature matrices
    print("\nStep 4: Creating feature matrices...")
    # Get all feature names from the first sample
    all_feature_names = list(features_list_par[0].keys())
    print(f"Total number of features: {len(all_feature_names)}")
    
    # Create X matrix
    X = np.zeros((len(features_list_par), len(all_feature_names)))
    for i, features_dict in enumerate(features_list_par):
        for j, feature_name in enumerate(all_feature_names):
            X[i, j] = features_dict.get(feature_name, 0)
    
    # Step 5: Parallel cross-validation
    print("\nStep 5: Parallel cross-validation...")
    
    # Define model factory
    def rf_factory(n_estimators=100, max_depth=None):
        return RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
    
    # Sequential cross-validation (for comparison)
    start_time = time.time()
    cv_results_seq = []
    for fold in range(5):
        # Create train/test split
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        train_idx, test_idx = list(kf.split(X))[fold]
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]
        
        # Train and evaluate model
        model = rf_factory()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(y_test, y_pred)
        cv_results_seq.append({'fold': fold, 'accuracy': accuracy})
    seq_time = time.time() - start_time
    print(f"Sequential cross-validation time: {seq_time:.2f} seconds")
    
    # Parallel cross-validation
    start_time = time.time()
    cv_results_par = qeeg.utils.parallel.parallel_cross_validation(
        X, labels, rf_factory, cv=5
    )
    par_time = time.time() - start_time
    print(f"Parallel cross-validation time: {par_time:.2f} seconds")
    print(f"Speedup: {seq_time / par_time:.2f}x")
    
    # Print cross-validation results
    print("\nCross-validation results:")
    for fold in range(5):
        seq_acc = cv_results_seq[fold]['accuracy']
        par_acc = cv_results_par[fold]['accuracy']
        print(f"Fold {fold}: Sequential accuracy = {seq_acc:.4f}, Parallel accuracy = {par_acc:.4f}")
    
    # Step 6: Parallel grid search
    print("\nStep 6: Parallel grid search...")
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 5, 10]
    }
    
    # Sequential grid search (for comparison)
    start_time = time.time()
    from sklearn.model_selection import ParameterGrid
    grid_results_seq = []
    for params in ParameterGrid(param_grid):
        # Perform cross-validation
        cv_results = []
        for fold in range(5):
            # Create train/test split
            from sklearn.model_selection import KFold
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            train_idx, test_idx = list(kf.split(X))[fold]
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = labels[train_idx], labels[test_idx]
            
            # Train and evaluate model
            model = rf_factory(**params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            from sklearn.metrics import accuracy_score
            accuracy = accuracy_score(y_test, y_pred)
            cv_results.append({'fold': fold, 'accuracy': accuracy})
        
        # Calculate mean metrics
        mean_accuracy = np.mean([r['accuracy'] for r in cv_results])
        grid_results_seq.append({
            'params': params,
            'mean_accuracy': mean_accuracy,
            'cv_results': cv_results
        })
    
    # Sort the results by mean accuracy
    grid_results_seq.sort(key=lambda x: x['mean_accuracy'], reverse=True)
    seq_time = time.time() - start_time
    print(f"Sequential grid search time: {seq_time:.2f} seconds")
    
    # Parallel grid search
    start_time = time.time()
    grid_results_par = qeeg.utils.parallel.parallel_grid_search(
        X, labels, rf_factory, param_grid, cv=5
    )
    par_time = time.time() - start_time
    print(f"Parallel grid search time: {par_time:.2f} seconds")
    print(f"Speedup: {seq_time / par_time:.2f}x")
    
    # Print grid search results
    print("\nGrid search results:")
    print("\nTop 3 parameter combinations (sequential):")
    for i in range(min(3, len(grid_results_seq))):
        params = grid_results_seq[i]['params']
        acc = grid_results_seq[i]['mean_accuracy']
        print(f"{i+1}. {params}: accuracy = {acc:.4f}")
    
    print("\nTop 3 parameter combinations (parallel):")
    for i in range(min(3, len(grid_results_par))):
        params = grid_results_par[i]['params']
        acc = grid_results_par[i]['mean_accuracy']
        print(f"{i+1}. {params}: accuracy = {acc:.4f}")
    
    # Step 7: Visualize speedups
    print("\nStep 7: Visualizing speedups...")
    
    # Create output directory if it doesn't exist
    os.makedirs('examples/output', exist_ok=True)
    
    # Create a bar chart of speedups
    operations = ['Preprocessing', 'Feature Extraction', 'Cross-Validation', 'Grid Search']
    speedups = [
        seq_time / par_time if 'seq_time' in locals() and 'par_time' in locals() else 0
        for seq_time, par_time in [
            (locals().get('seq_time', 0), locals().get('par_time', 0))
            for op in operations
        ]
    ]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(operations, speedups)
    
    # Add value labels on top of the bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            height,
            f'{height:.2f}x',
            ha='center',
            va='bottom',
            rotation=0
        )
    
    # Add labels and title
    ax.set_xlabel('Operation')
    ax.set_ylabel('Speedup (Sequential Time / Parallel Time)')
    ax.set_title('Parallel Processing Speedups')
    
    # Add a grid
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig('examples/output/parallel_speedups.png')
    print("Saved speedup chart to examples/output/parallel_speedups.png")
    
    print("\nParallel processing example completed!")

if __name__ == "__main__":
    main()
