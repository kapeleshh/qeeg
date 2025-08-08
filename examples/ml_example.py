"""
Machine learning example using the qeeg package.

This example demonstrates how to:
1. Load EEG data
2. Extract features
3. Train a machine learning model
4. Evaluate the model
5. Visualize the results
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
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
    print("Machine Learning Example for EEG Classification")
    print("==============================================")
    
    # Step 1: Load or create EEG data
    print("\nStep 1: Loading/creating EEG data...")
    raw_list, labels = create_simulated_eeg_dataset(n_samples=20)
    print(f"Created dataset with {len(raw_list)} samples ({np.sum(labels == 0)} normal, {np.sum(labels == 1)} abnormal)")
    
    # Step 2: Extract features
    print("\nStep 2: Extracting features...")
    features_list = []
    for raw in raw_list:
        # Preprocess the data
        raw_filtered = qeeg.preprocessing.filtering.bandpass_filter(raw, l_freq=1.0, h_freq=40.0)
        
        # Extract features
        features_dict = qeeg.ml.features.extract_all_features(
            raw_filtered,
            include_wavelet=False,  # Exclude wavelet features for speed
            include_connectivity=True
        )
        features_list.append(features_dict)
    
    # Step 3: Create feature matrices
    print("\nStep 3: Creating feature matrices...")
    # Get all feature names from the first sample
    all_feature_names = list(features_list[0].keys())
    print(f"Total number of features: {len(all_feature_names)}")
    
    # Create X matrix
    X = np.zeros((len(features_list), len(all_feature_names)))
    for i, features_dict in enumerate(features_list):
        for j, feature_name in enumerate(all_feature_names):
            X[i, j] = features_dict.get(feature_name, 0)
    
    # Step 4: Train and evaluate models
    print("\nStep 4: Training and evaluating models...")
    
    # Random Forest
    print("\nRandom Forest Classifier:")
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_pipeline, rf_metrics = qeeg.ml.classification.train_model(
        X, labels, rf_clf, test_size=0.3, n_features=20
    )
    print(f"Accuracy: {rf_metrics['accuracy']:.2f}")
    print(f"Precision: {rf_metrics['precision']:.2f}")
    print(f"Recall: {rf_metrics['recall']:.2f}")
    print(f"F1 Score: {rf_metrics['f1']:.2f}")
    
    # Support Vector Machine
    print("\nSupport Vector Machine Classifier:")
    svm_clf = SVC(probability=True, random_state=42)
    svm_pipeline, svm_metrics = qeeg.ml.classification.train_model(
        X, labels, svm_clf, test_size=0.3, n_features=20
    )
    print(f"Accuracy: {svm_metrics['accuracy']:.2f}")
    print(f"Precision: {svm_metrics['precision']:.2f}")
    print(f"Recall: {svm_metrics['recall']:.2f}")
    print(f"F1 Score: {svm_metrics['f1']:.2f}")
    
    # Step 5: Cross-validation
    print("\nStep 5: Performing cross-validation...")
    
    # Random Forest
    rf_cv_results = qeeg.ml.classification.cross_validate_model(
        X, labels, rf_clf, cv=5, n_features=20
    )
    print(f"Random Forest CV Accuracy: {rf_cv_results['mean_score']:.2f} ± {rf_cv_results['std_score']:.2f}")
    
    # Support Vector Machine
    svm_cv_results = qeeg.ml.classification.cross_validate_model(
        X, labels, svm_clf, cv=5, n_features=20
    )
    print(f"SVM CV Accuracy: {svm_cv_results['mean_score']:.2f} ± {svm_cv_results['std_score']:.2f}")
    
    # Step 6: Evaluate on the full dataset
    print("\nStep 6: Evaluating on the full dataset...")
    
    # Random Forest
    rf_eval = qeeg.ml.classification.evaluate_model(
        rf_pipeline, X, labels, class_names=['Normal', 'Abnormal']
    )
    
    # Support Vector Machine
    svm_eval = qeeg.ml.classification.evaluate_model(
        svm_pipeline, X, labels, class_names=['Normal', 'Abnormal']
    )
    
    # Step 7: Visualize results
    print("\nStep 7: Visualizing results...")
    
    # Create output directory if it doesn't exist
    os.makedirs('examples/output', exist_ok=True)
    
    # Plot confusion matrices
    rf_cm_fig = qeeg.ml.classification.plot_confusion_matrix(
        rf_eval['confusion_matrix'],
        class_names=['Normal', 'Abnormal'],
        title='Random Forest Confusion Matrix',
        show=False,
        figsize=(8, 6)
    )
    rf_cm_fig.savefig('examples/output/rf_confusion_matrix.png')
    print("Saved Random Forest confusion matrix to examples/output/rf_confusion_matrix.png")
    
    svm_cm_fig = qeeg.ml.classification.plot_confusion_matrix(
        svm_eval['confusion_matrix'],
        class_names=['Normal', 'Abnormal'],
        title='SVM Confusion Matrix',
        show=False,
        figsize=(8, 6)
    )
    svm_cm_fig.savefig('examples/output/svm_confusion_matrix.png')
    print("Saved SVM confusion matrix to examples/output/svm_confusion_matrix.png")
    
    # Plot ROC curves if available
    if 'roc_curve' in rf_eval:
        rf_roc_fig = qeeg.ml.classification.plot_roc_curve(
            rf_eval['roc_curve']['fpr'],
            rf_eval['roc_curve']['tpr'],
            rf_eval['roc_auc'],
            title='Random Forest ROC Curve',
            show=False,
            figsize=(8, 6)
        )
        rf_roc_fig.savefig('examples/output/rf_roc_curve.png')
        print("Saved Random Forest ROC curve to examples/output/rf_roc_curve.png")
    
    if 'roc_curve' in svm_eval:
        svm_roc_fig = qeeg.ml.classification.plot_roc_curve(
            svm_eval['roc_curve']['fpr'],
            svm_eval['roc_curve']['tpr'],
            svm_eval['roc_auc'],
            title='SVM ROC Curve',
            show=False,
            figsize=(8, 6)
        )
        svm_roc_fig.savefig('examples/output/svm_roc_curve.png')
        print("Saved SVM ROC curve to examples/output/svm_roc_curve.png")
    
    # Step 8: Save the models
    print("\nStep 8: Saving the models...")
    qeeg.ml.classification.save_model(rf_pipeline, 'examples/output/rf_model.joblib')
    print("Saved Random Forest model to examples/output/rf_model.joblib")
    
    qeeg.ml.classification.save_model(svm_pipeline, 'examples/output/svm_model.joblib')
    print("Saved SVM model to examples/output/svm_model.joblib")
    
    print("\nMachine learning example completed!")

if __name__ == "__main__":
    main()
