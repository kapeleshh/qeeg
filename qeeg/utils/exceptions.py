"""
Custom exceptions for the qeeg package.

This module provides custom exception classes for more informative error messages
and better error handling throughout the package.
"""

class QEEGError(Exception):
    """Base class for all QEEG exceptions."""
    pass


class DataQualityError(QEEGError):
    """
    Raised when EEG data quality is insufficient for analysis.
    
    Parameters
    ----------
    message : str
        The error message
    channel : str, optional
        The channel name where the issue was detected
    metric : str, optional
        The metric that failed quality check
    value : float, optional
        The value of the metric
    threshold : float, optional
        The threshold that was exceeded
    """
    def __init__(self, message, channel=None, metric=None, value=None, threshold=None):
        self.channel = channel
        self.metric = metric
        self.value = value
        self.threshold = threshold
        
        if all(v is not None for v in [channel, metric, value, threshold]):
            message = f"{message} Channel '{channel}' has {metric}={value:.2f}, which exceeds threshold {threshold:.2f}."
            
        super().__init__(message)


class ProcessingError(QEEGError):
    """
    Raised when an error occurs during EEG data processing.
    
    Parameters
    ----------
    message : str
        The error message
    stage : str, optional
        The processing stage where the error occurred
    function : str, optional
        The function where the error occurred
    """
    def __init__(self, message, stage=None, function=None):
        self.stage = stage
        self.function = function
        
        if stage is not None and function is not None:
            message = f"Error during {stage} in {function}: {message}"
        elif stage is not None:
            message = f"Error during {stage}: {message}"
        elif function is not None:
            message = f"Error in {function}: {message}"
            
        super().__init__(message)


class ValidationError(QEEGError):
    """
    Raised when input validation fails.
    
    Parameters
    ----------
    message : str
        The error message
    parameter : str, optional
        The parameter that failed validation
    expected : str, optional
        The expected value or type
    received : str, optional
        The received value or type
    """
    def __init__(self, message, parameter=None, expected=None, received=None):
        self.parameter = parameter
        self.expected = expected
        self.received = received
        
        if all(v is not None for v in [parameter, expected, received]):
            message = f"{message} Parameter '{parameter}' expected {expected}, but received {received}."
            
        super().__init__(message)


class ConfigurationError(QEEGError):
    """
    Raised when there is an error in the configuration.
    
    Parameters
    ----------
    message : str
        The error message
    config_key : str, optional
        The configuration key that caused the error
    """
    def __init__(self, message, config_key=None):
        self.config_key = config_key
        
        if config_key is not None:
            message = f"Configuration error for '{config_key}': {message}"
            
        super().__init__(message)


class DependencyError(QEEGError):
    """
    Raised when a required dependency is missing or incompatible.
    
    Parameters
    ----------
    message : str
        The error message
    package : str, optional
        The package that is missing or incompatible
    required_version : str, optional
        The required version
    installed_version : str, optional
        The installed version
    """
    def __init__(self, message, package=None, required_version=None, installed_version=None):
        self.package = package
        self.required_version = required_version
        self.installed_version = installed_version
        
        if all(v is not None for v in [package, required_version, installed_version]):
            message = f"{message} Package '{package}' version {installed_version} does not meet requirement {required_version}."
        elif package is not None and required_version is not None:
            message = f"{message} Package '{package}' is required with version {required_version}."
        elif package is not None:
            message = f"{message} Package '{package}' is required but not installed."
            
        super().__init__(message)


def get_troubleshooting_info(exception):
    """
    Get troubleshooting information for a specific exception.
    
    Parameters
    ----------
    exception : Exception
        The exception to get troubleshooting information for
        
    Returns
    -------
    str
        Troubleshooting information
    """
    if isinstance(exception, DataQualityError):
        return _get_data_quality_troubleshooting(exception)
    elif isinstance(exception, ProcessingError):
        return _get_processing_troubleshooting(exception)
    elif isinstance(exception, ValidationError):
        return _get_validation_troubleshooting(exception)
    elif isinstance(exception, ConfigurationError):
        return _get_configuration_troubleshooting(exception)
    elif isinstance(exception, DependencyError):
        return _get_dependency_troubleshooting(exception)
    else:
        return "No specific troubleshooting information available for this error."


def _get_data_quality_troubleshooting(exception):
    """Get troubleshooting information for DataQualityError."""
    info = ["Data quality issues can be addressed by:"]
    
    if exception.metric == "noise":
        info.append("- Checking electrode placement and impedance")
        info.append("- Applying a notch filter to remove power line noise")
        info.append("- Using ICA to remove artifacts")
    elif exception.metric == "missing_data":
        info.append("- Interpolating missing data points")
        info.append("- Excluding the affected channel from analysis")
    elif exception.metric == "flat_signal":
        info.append("- Checking if the electrode is properly connected")
        info.append("- Verifying that the amplifier is functioning correctly")
    else:
        info.append("- Preprocessing the data to improve quality")
        info.append("- Checking electrode connections and impedance")
        info.append("- Filtering out noise and artifacts")
    
    return "\n".join(info)


def _get_processing_troubleshooting(exception):
    """Get troubleshooting information for ProcessingError."""
    info = ["Processing errors can be addressed by:"]
    
    if exception.stage == "filtering":
        info.append("- Checking filter parameters (cutoff frequencies, filter order)")
        info.append("- Ensuring data is preloaded before filtering")
        info.append("- Verifying that the sampling rate is appropriate for the filter")
    elif exception.stage == "artifact_removal":
        info.append("- Adjusting ICA parameters")
        info.append("- Trying a different artifact removal method")
        info.append("- Manually inspecting and marking bad segments")
    elif exception.stage == "feature_extraction":
        info.append("- Verifying that the data has been properly preprocessed")
        info.append("- Checking feature extraction parameters")
        info.append("- Ensuring sufficient data quality for feature extraction")
    else:
        info.append("- Checking input data format and quality")
        info.append("- Verifying function parameters")
        info.append("- Ensuring all prerequisites have been met")
    
    return "\n".join(info)


def _get_validation_troubleshooting(exception):
    """Get troubleshooting information for ValidationError."""
    info = ["Validation errors can be addressed by:"]
    
    if exception.parameter:
        info.append(f"- Checking the value provided for '{exception.parameter}'")
        if exception.expected and exception.received:
            info.append(f"- Ensuring the parameter is of type {exception.expected} (received {exception.received})")
    
    info.append("- Reviewing the function documentation for parameter requirements")
    info.append("- Verifying that all required parameters are provided")
    
    return "\n".join(info)


def _get_configuration_troubleshooting(exception):
    """Get troubleshooting information for ConfigurationError."""
    info = ["Configuration errors can be addressed by:"]
    
    if exception.config_key:
        info.append(f"- Checking the configuration value for '{exception.config_key}'")
        info.append(f"- Verifying that '{exception.config_key}' is properly formatted")
    
    info.append("- Reviewing the configuration documentation")
    info.append("- Ensuring all required configuration keys are present")
    
    return "\n".join(info)


def _get_dependency_troubleshooting(exception):
    """Get troubleshooting information for DependencyError."""
    info = ["Dependency errors can be addressed by:"]
    
    if exception.package:
        if exception.installed_version and exception.required_version:
            info.append(f"- Upgrading {exception.package} to version {exception.required_version} or compatible")
            info.append(f"  pip install {exception.package}>={exception.required_version}")
        else:
            info.append(f"- Installing the required package: {exception.package}")
            info.append(f"  pip install {exception.package}")
    
    info.append("- Checking for conflicts between package versions")
    info.append("- Creating a clean virtual environment for the project")
    
    return "\n".join(info)
