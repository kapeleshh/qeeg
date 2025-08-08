"""
Classification module for EEG data.

This module provides functions for classifying EEG data using machine learning.
"""

import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Any
import mne
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def create_pipeline(
    classifier: str = "svm",
    **kwargs
) -> Pipeline:
    """
    Create a classification pipeline.

    Parameters
    ----------
    classifier : str, optional
        Classifier to use, by default "svm"
    **kwargs
        Additional keyword arguments to pass to the classifier.

    Returns
    -------
    Pipeline
        Classification pipeline.

    Examples
    --------
    >>> from qeeg.ml import classification
    >>> pipeline = classification.create_pipeline(classifier="rf", n_estimators=100)
    """
    # Create the pipeline steps
    steps = [
        ('scaler', StandardScaler())
    ]
    
    # Add the classifier
    if classifier == "svm":
        steps.append(('classifier', SVC(probability=True, **kwargs)))
    elif classifier == "rf":
        steps.append(('classifier', RandomForestClassifier(**kwargs)))
    else:
        raise ValueError(f"Unknown classifier: {classifier}")
    
    # Create the pipeline
    pipeline = Pipeline(steps)
    
    return pipeline


def train_classifier(
    X: np.ndarray,
    y: np.ndarray,
    classifier: str = "svm",
    test_size: float = 0.2,
    random_state: int = 42,
    **kwargs
) -> Tuple[Pipeline, Dict[str, float]]:
    """
    Train a classifier on EEG features.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target vector.
    classifier : str, optional
        Classifier to use, by default "svm"
    test_size : float, optional
        Test set size, by default 0.2
    random_state : int, optional
        Random state, by default 42
    **kwargs
        Additional keyword arguments to pass to the classifier.

    Returns
    -------
    Tuple[Pipeline, Dict[str, float]]
        Trained pipeline and performance metrics.

    Examples
    --------
    >>> import numpy as np
    >>> from qeeg.ml import classification
    >>> X = np.random.randn(100, 10)
    >>> y = np.random.randint(0, 2, 100)
    >>> pipeline, metrics = classification.train_classifier(X, y)
    >>> print(metrics['accuracy'])
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Create the pipeline
    pipeline = create_pipeline(classifier=classifier, **kwargs)
    
    # Train the pipeline
    pipeline.fit(X_train, y_train)
    
    # Evaluate the pipeline
    y_pred = pipeline.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1': f1_score(y_test, y_pred, average='weighted'),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    
    return pipeline, metrics


def cross_validate(
    X: np.ndarray,
    y: np.ndarray,
    classifier: str = "svm",
    cv: int = 5,
    random_state: int = 42,
    **kwargs
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Cross-validate a classifier on EEG features.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target vector.
    classifier : str, optional
        Classifier to use, by default "svm"
    cv : int, optional
        Number of cross-validation folds, by default 5
    random_state : int, optional
        Random state, by default 42
    **kwargs
        Additional keyword arguments to pass to the classifier.

    Returns
    -------
    Dict[str, Union[float, np.ndarray]]
        Cross-validation metrics.

    Examples
    --------
    >>> import numpy as np
    >>> from qeeg.ml import classification
    >>> X = np.random.randn(100, 10)
    >>> y = np.random.randint(0, 2, 100)
    >>> metrics = classification.cross_validate(X, y)
    >>> print(metrics['accuracy'])
    """
    # Create the pipeline
    pipeline = create_pipeline(classifier=classifier, **kwargs)
    
    # Cross-validate the pipeline
    cv_scores = cross_val_score(pipeline, X, y, cv=cv)
    
    # Calculate metrics
    metrics = {
        'accuracy': np.mean(cv_scores),
        'std': np.std(cv_scores),
        'scores': cv_scores
    }
    
    return metrics


def classify_eeg(
    raw: mne.io.Raw,
    labels: np.ndarray,
    feature_types: List[str] = ["band_power", "statistical"],
    classifier: str = "svm",
    test_size: float = 0.2,
    random_state: int = 42,
    **kwargs
) -> Tuple[Pipeline, Dict[str, float]]:
    """
    Classify EEG data.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data.
    labels : np.ndarray
        Target vector.
    feature_types : List[str], optional
        List of feature types to extract, by default ["band_power", "statistical"]
    classifier : str, optional
        Classifier to use, by default "svm"
    test_size : float, optional
        Test set size, by default 0.2
    random_state : int, optional
        Random state, by default 42
    **kwargs
        Additional keyword arguments to pass to the classifier.

    Returns
    -------
    Tuple[Pipeline, Dict[str, float]]
        Trained pipeline and performance metrics.

    Examples
    --------
    >>> import mne
    >>> import numpy as np
    >>> from qeeg.ml import classification
    >>> raw = mne.io.read_raw_edf("sample.edf", preload=True)
    >>> labels = np.random.randint(0, 2, len(raw.times))
    >>> pipeline, metrics = classification.classify_eeg(raw, labels)
    >>> print(metrics['accuracy'])
    """
    from qeeg.ml.features import extract_features, create_feature_vector
    
    # Extract features
    features = extract_features(raw, feature_types=feature_types)
    
    # Create feature vector
    X = create_feature_vector(features)
    
    # Train the classifier
    pipeline, metrics = train_classifier(
        X, labels, classifier=classifier, test_size=test_size,
        random_state=random_state, **kwargs
    )
    
    return pipeline, metrics


def predict(
    raw: mne.io.Raw,
    pipeline: Pipeline,
    feature_types: List[str] = ["band_power", "statistical"]
) -> np.ndarray:
    """
    Predict labels for EEG data.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data.
    pipeline : Pipeline
        Trained classification pipeline.
    feature_types : List[str], optional
        List of feature types to extract, by default ["band_power", "statistical"]

    Returns
    -------
    np.ndarray
        Predicted labels.

    Examples
    --------
    >>> import mne
    >>> import numpy as np
    >>> from qeeg.ml import classification
    >>> raw_train = mne.io.read_raw_edf("train.edf", preload=True)
    >>> labels = np.random.randint(0, 2, len(raw_train.times))
    >>> pipeline, _ = classification.classify_eeg(raw_train, labels)
    >>> raw_test = mne.io.read_raw_edf("test.edf", preload=True)
    >>> predictions = classification.predict(raw_test, pipeline)
    >>> print(predictions.shape)
    """
    from qeeg.ml.features import extract_features, create_feature_vector
    
    # Extract features
    features = extract_features(raw, feature_types=feature_types)
    
    # Create feature vector
    X = create_feature_vector(features)
    
    # Predict labels
    y_pred = pipeline.predict(X)
    
    return y_pred
