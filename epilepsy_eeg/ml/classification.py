"""
Classification module for EEG data.

This module provides functions for training and evaluating machine learning models
for EEG data classification, particularly for epilepsy detection and neurological
condition assessment.
"""

import numpy as np
from typing import Tuple, Optional, Union, List, Dict, Any, Callable
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
import joblib
import os


def create_pipeline(
    classifier: BaseEstimator,
    n_features: Optional[int] = None,
    scaler: Optional[BaseEstimator] = None,
    **kwargs
) -> Pipeline:
    """
    Create a scikit-learn pipeline for EEG classification.

    Parameters
    ----------
    classifier : BaseEstimator
        The classifier to use.
    n_features : int or None, optional
        Number of features to select, by default None (no feature selection)
    scaler : BaseEstimator or None, optional
        The scaler to use, by default None (StandardScaler)
    **kwargs
        Additional keyword arguments to pass to the pipeline

    Returns
    -------
    Pipeline
        The scikit-learn pipeline.

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from epilepsy_eeg.ml import classification
    >>> clf = RandomForestClassifier(n_estimators=100, random_state=42)
    >>> pipeline = classification.create_pipeline(clf, n_features=20)
    """
    # Create the steps list
    steps = []
    
    # Add the scaler
    if scaler is None:
        scaler = StandardScaler()
    steps.append(('scaler', scaler))
    
    # Add feature selection if requested
    if n_features is not None:
        steps.append(('feature_selection', SelectKBest(f_classif, k=n_features)))
    
    # Add the classifier
    steps.append(('classifier', classifier))
    
    # Create the pipeline
    pipeline = Pipeline(steps, **kwargs)
    
    return pipeline


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    classifier: BaseEstimator,
    test_size: float = 0.2,
    random_state: int = 42,
    n_features: Optional[int] = None,
    **kwargs
) -> Tuple[Pipeline, Dict[str, float]]:
    """
    Train a machine learning model for EEG classification.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target vector.
    classifier : BaseEstimator
        The classifier to use.
    test_size : float, optional
        Proportion of the data to use for testing, by default 0.2
    random_state : int, optional
        Random state for reproducibility, by default 42
    n_features : int or None, optional
        Number of features to select, by default None (no feature selection)
    **kwargs
        Additional keyword arguments to pass to create_pipeline()

    Returns
    -------
    Tuple[Pipeline, Dict[str, float]]
        The trained pipeline and a dictionary of evaluation metrics.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from epilepsy_eeg.ml import classification
    >>> X = np.random.randn(100, 10)
    >>> y = np.random.randint(0, 2, 100)
    >>> clf = RandomForestClassifier(n_estimators=100, random_state=42)
    >>> pipeline, metrics = classification.train_model(X, y, clf)
    """
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Create the pipeline
    pipeline = create_pipeline(classifier, n_features=n_features, **kwargs)
    
    # Train the model
    pipeline.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = pipeline.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1': f1_score(y_test, y_pred, average='weighted')
    }
    
    return pipeline, metrics


def cross_validate_model(
    X: np.ndarray,
    y: np.ndarray,
    classifier: BaseEstimator,
    cv: int = 5,
    n_features: Optional[int] = None,
    scoring: str = 'accuracy',
    **kwargs
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Perform cross-validation of a machine learning model for EEG classification.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target vector.
    classifier : BaseEstimator
        The classifier to use.
    cv : int, optional
        Number of cross-validation folds, by default 5
    n_features : int or None, optional
        Number of features to select, by default None (no feature selection)
    scoring : str, optional
        Scoring metric to use, by default 'accuracy'
    **kwargs
        Additional keyword arguments to pass to create_pipeline()

    Returns
    -------
    Dict[str, Union[float, np.ndarray]]
        Dictionary of cross-validation results.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from epilepsy_eeg.ml import classification
    >>> X = np.random.randn(100, 10)
    >>> y = np.random.randint(0, 2, 100)
    >>> clf = RandomForestClassifier(n_estimators=100, random_state=42)
    >>> results = classification.cross_validate_model(X, y, clf)
    """
    # Create the pipeline
    pipeline = create_pipeline(classifier, n_features=n_features, **kwargs)
    
    # Create the cross-validation object
    cv_obj = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    # Perform cross-validation
    scores = cross_val_score(pipeline, X, y, cv=cv_obj, scoring=scoring)
    
    # Calculate results
    results = {
        'mean_score': np.mean(scores),
        'std_score': np.std(scores),
        'scores': scores
    }
    
    return results


def evaluate_model(
    pipeline: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
    class_names: Optional[List[str]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Evaluate a trained machine learning model for EEG classification.

    Parameters
    ----------
    pipeline : Pipeline
        The trained scikit-learn pipeline.
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target vector.
    class_names : List[str] or None, optional
        Names of the classes, by default None
    **kwargs
        Additional keyword arguments

    Returns
    -------
    Dict[str, Any]
        Dictionary of evaluation results.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from epilepsy_eeg.ml import classification
    >>> X = np.random.randn(100, 10)
    >>> y = np.random.randint(0, 2, 100)
    >>> clf = RandomForestClassifier(n_estimators=100, random_state=42)
    >>> pipeline, _ = classification.train_model(X, y, clf)
    >>> results = classification.evaluate_model(pipeline, X, y)
    """
    # Make predictions
    y_pred = pipeline.predict(X)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred, average='weighted'),
        'recall': recall_score(y, y_pred, average='weighted'),
        'f1': f1_score(y, y_pred, average='weighted'),
        'confusion_matrix': confusion_matrix(y, y_pred),
        'classification_report': classification_report(y, y_pred, target_names=class_names, output_dict=True)
    }
    
    # Calculate ROC curve and AUC if binary classification
    if len(np.unique(y)) == 2:
        try:
            y_score = pipeline.predict_proba(X)[:, 1]
            fpr, tpr, _ = roc_curve(y, y_score)
            metrics['roc_auc'] = auc(fpr, tpr)
            metrics['roc_curve'] = {'fpr': fpr, 'tpr': tpr}
        except:
            # Some classifiers don't support predict_proba
            pass
    
    return metrics


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
    normalize: bool = False,
    title: str = 'Confusion Matrix',
    cmap: str = 'Blues',
    show: bool = True,
    **kwargs
) -> plt.Figure:
    """
    Plot a confusion matrix.

    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix.
    class_names : List[str] or None, optional
        Names of the classes, by default None
    normalize : bool, optional
        Whether to normalize the confusion matrix, by default False
    title : str, optional
        Title of the plot, by default 'Confusion Matrix'
    cmap : str, optional
        Colormap to use, by default 'Blues'
    show : bool, optional
        Whether to show the plot, by default True
    **kwargs
        Additional keyword arguments to pass to plt.figure()

    Returns
    -------
    plt.Figure
        The matplotlib figure.

    Examples
    --------
    >>> import numpy as np
    >>> from epilepsy_eeg.ml import classification
    >>> cm = np.array([[10, 2], [3, 15]])
    >>> fig = classification.plot_confusion_matrix(cm, class_names=['Normal', 'Abnormal'])
    """
    # Create a figure
    fig, ax = plt.subplots(**kwargs)
    
    # Normalize the confusion matrix if requested
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    # Plot the confusion matrix
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.get_cmap(cmap))
    ax.figure.colorbar(im, ax=ax)
    
    # Set the axis labels
    if class_names is not None:
        ax.set_xticks(np.arange(len(class_names)))
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)
    
    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Loop over data dimensions and create text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    # Set the title and labels
    ax.set_title(title)
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    
    # Adjust the layout
    fig.tight_layout()
    
    # Show the plot if requested
    if show:
        plt.show()
    
    return fig


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    roc_auc: float,
    title: str = 'Receiver Operating Characteristic',
    show: bool = True,
    **kwargs
) -> plt.Figure:
    """
    Plot a ROC curve.

    Parameters
    ----------
    fpr : np.ndarray
        False positive rates.
    tpr : np.ndarray
        True positive rates.
    roc_auc : float
        Area under the ROC curve.
    title : str, optional
        Title of the plot, by default 'Receiver Operating Characteristic'
    show : bool, optional
        Whether to show the plot, by default True
    **kwargs
        Additional keyword arguments to pass to plt.figure()

    Returns
    -------
    plt.Figure
        The matplotlib figure.

    Examples
    --------
    >>> import numpy as np
    >>> from epilepsy_eeg.ml import classification
    >>> fpr = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    >>> tpr = np.array([0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 1])
    >>> roc_auc = 0.8
    >>> fig = classification.plot_roc_curve(fpr, tpr, roc_auc)
    """
    # Create a figure
    fig, ax = plt.subplots(**kwargs)
    
    # Plot the ROC curve
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    
    # Set the axis limits and labels
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc="lower right")
    
    # Adjust the layout
    fig.tight_layout()
    
    # Show the plot if requested
    if show:
        plt.show()
    
    return fig


def save_model(
    pipeline: Pipeline,
    filename: str,
    **kwargs
) -> None:
    """
    Save a trained machine learning model to a file.

    Parameters
    ----------
    pipeline : Pipeline
        The trained scikit-learn pipeline.
    filename : str
        Path to save the model to.
    **kwargs
        Additional keyword arguments to pass to joblib.dump()

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from epilepsy_eeg.ml import classification
    >>> X = np.random.randn(100, 10)
    >>> y = np.random.randint(0, 2, 100)
    >>> clf = RandomForestClassifier(n_estimators=100, random_state=42)
    >>> pipeline, _ = classification.train_model(X, y, clf)
    >>> classification.save_model(pipeline, 'model.joblib')
    """
    # Create the directory if it doesn't exist
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    
    # Save the model
    joblib.dump(pipeline, filename, **kwargs)


def load_model(
    filename: str,
    **kwargs
) -> Pipeline:
    """
    Load a trained machine learning model from a file.

    Parameters
    ----------
    filename : str
        Path to load the model from.
    **kwargs
        Additional keyword arguments to pass to joblib.load()

    Returns
    -------
    Pipeline
        The loaded scikit-learn pipeline.

    Examples
    --------
    >>> from epilepsy_eeg.ml import classification
    >>> pipeline = classification.load_model('model.joblib')
    """
    # Load the model
    pipeline = joblib.load(filename, **kwargs)
    
    return pipeline


def classify_eeg(
    raw: mne.io.Raw,
    pipeline: Pipeline,
    feature_extractor: Callable,
    class_names: Optional[List[str]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Classify EEG data using a trained machine learning model.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data.
    pipeline : Pipeline
        The trained scikit-learn pipeline.
    feature_extractor : Callable
        Function to extract features from the EEG data.
    class_names : List[str] or None, optional
        Names of the classes, by default None
    **kwargs
        Additional keyword arguments to pass to the feature extractor

    Returns
    -------
    Dict[str, Any]
        Dictionary of classification results.

    Examples
    --------
    >>> import mne
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from epilepsy_eeg.ml import classification, features
    >>> raw = mne.io.read_raw_edf("sample.edf", preload=True)
    >>> X = np.random.randn(100, 10)
    >>> y = np.random.randint(0, 2, 100)
    >>> clf = RandomForestClassifier(n_estimators=100, random_state=42)
    >>> pipeline, _ = classification.train_model(X, y, clf)
    >>> results = classification.classify_eeg(raw, pipeline, features.extract_all_features)
    """
    # Extract features
    features_dict = feature_extractor(raw, **kwargs)
    
    # Create a feature matrix
    X, feature_names = features.create_feature_matrix(features_dict)
    
    # Make a prediction
    y_pred = pipeline.predict([X])[0]
    
    # Get the class probabilities if available
    try:
        y_proba = pipeline.predict_proba([X])[0]
        class_probs = {
            class_names[i] if class_names else i: prob
            for i, prob in enumerate(y_proba)
        }
    except:
        class_probs = {}
    
    # Create the result dictionary
    result = {
        'prediction': y_pred,
        'prediction_name': class_names[y_pred] if class_names else y_pred,
        'probabilities': class_probs
    }
    
    return result
