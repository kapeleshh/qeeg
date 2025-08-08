"""
Custom exceptions for the epilepsy_eeg package.

This module defines custom exception classes for different types of errors
that can occur in the epilepsy_eeg package.
"""

class EEGError(Exception):
    """Base class for all epilepsy_eeg exceptions."""
    pass


class PreprocessingError(EEGError):
    """Exception raised for errors in preprocessing module."""
    pass


class AnalysisError(EEGError):
    """Exception raised for errors in analysis module."""
    pass


class VisualizationError(EEGError):
    """Exception raised for errors in visualization module."""
    pass


class MLError(EEGError):
    """Exception raised for errors in machine learning module."""
    pass


class ValidationError(EEGError):
    """Exception raised for input validation errors."""
    pass


class IOError(EEGError):
    """Exception raised for input/output errors."""
    pass


class ParallelError(EEGError):
    """Exception raised for errors in parallel processing."""
    pass
