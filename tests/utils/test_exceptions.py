"""
Tests for the exceptions module.
"""

import pytest
from qeeg.utils.exceptions import (
    QEEGError,
    DataQualityError,
    ProcessingError,
    ValidationError,
    ConfigurationError,
    DependencyError,
    get_troubleshooting_info
)


def test_qeeg_error():
    """Test the base QEEGError class."""
    # Create a QEEGError
    error = QEEGError("Test error message")
    
    # Check that it's an instance of Exception
    assert isinstance(error, Exception)
    
    # Check the error message
    assert str(error) == "Test error message"


def test_data_quality_error():
    """Test the DataQualityError class."""
    # Create a DataQualityError with basic message
    error = DataQualityError("Data quality issue")
    
    # Check that it's an instance of QEEGError
    assert isinstance(error, QEEGError)
    
    # Check the error message
    assert str(error) == "Data quality issue"
    
    # Check attributes
    assert error.channel is None
    assert error.metric is None
    assert error.value is None
    assert error.threshold is None
    
    # Create a DataQualityError with all attributes
    error = DataQualityError(
        "Data quality issue",
        channel="Fp1",
        metric="noise",
        value=10.5,
        threshold=5.0
    )
    
    # Check the error message includes the details
    assert "Fp1" in str(error)
    assert "noise" in str(error)
    assert "10.50" in str(error)
    assert "5.00" in str(error)
    
    # Check attributes
    assert error.channel == "Fp1"
    assert error.metric == "noise"
    assert error.value == 10.5
    assert error.threshold == 5.0


def test_processing_error():
    """Test the ProcessingError class."""
    # Create a ProcessingError with basic message
    error = ProcessingError("Processing failed")
    
    # Check that it's an instance of QEEGError
    assert isinstance(error, QEEGError)
    
    # Check the error message
    assert str(error) == "Processing failed"
    
    # Check attributes
    assert error.stage is None
    assert error.function is None
    
    # Create a ProcessingError with stage
    error = ProcessingError("Processing failed", stage="filtering")
    
    # Check the error message includes the stage
    assert "filtering" in str(error)
    
    # Check attributes
    assert error.stage == "filtering"
    assert error.function is None
    
    # Create a ProcessingError with function
    error = ProcessingError("Processing failed", function="bandpass_filter")
    
    # Check the error message includes the function
    assert "bandpass_filter" in str(error)
    
    # Check attributes
    assert error.stage is None
    assert error.function == "bandpass_filter"
    
    # Create a ProcessingError with both stage and function
    error = ProcessingError(
        "Processing failed",
        stage="filtering",
        function="bandpass_filter"
    )
    
    # Check the error message includes both stage and function
    assert "filtering" in str(error)
    assert "bandpass_filter" in str(error)
    
    # Check attributes
    assert error.stage == "filtering"
    assert error.function == "bandpass_filter"


def test_validation_error():
    """Test the ValidationError class."""
    # Create a ValidationError with basic message
    error = ValidationError("Validation failed")
    
    # Check that it's an instance of QEEGError
    assert isinstance(error, QEEGError)
    
    # Check the error message
    assert str(error) == "Validation failed"
    
    # Check attributes
    assert error.parameter is None
    assert error.expected is None
    assert error.received is None
    
    # Create a ValidationError with all attributes
    error = ValidationError(
        "Validation failed",
        parameter="threshold",
        expected="float",
        received="str"
    )
    
    # Check the error message includes the details
    assert "threshold" in str(error)
    assert "float" in str(error)
    assert "str" in str(error)
    
    # Check attributes
    assert error.parameter == "threshold"
    assert error.expected == "float"
    assert error.received == "str"


def test_configuration_error():
    """Test the ConfigurationError class."""
    # Create a ConfigurationError with basic message
    error = ConfigurationError("Configuration issue")
    
    # Check that it's an instance of QEEGError
    assert isinstance(error, QEEGError)
    
    # Check the error message
    assert str(error) == "Configuration issue"
    
    # Check attributes
    assert error.config_key is None
    
    # Create a ConfigurationError with config_key
    error = ConfigurationError("Configuration issue", config_key="api_key")
    
    # Check the error message includes the config_key
    assert "api_key" in str(error)
    
    # Check attributes
    assert error.config_key == "api_key"


def test_dependency_error():
    """Test the DependencyError class."""
    # Create a DependencyError with basic message
    error = DependencyError("Dependency issue")
    
    # Check that it's an instance of QEEGError
    assert isinstance(error, QEEGError)
    
    # Check the error message
    assert str(error) == "Dependency issue"
    
    # Check attributes
    assert error.package is None
    assert error.required_version is None
    assert error.installed_version is None
    
    # Create a DependencyError with package
    error = DependencyError("Dependency issue", package="numpy")
    
    # Check the error message includes the package
    assert "numpy" in str(error)
    
    # Check attributes
    assert error.package == "numpy"
    assert error.required_version is None
    assert error.installed_version is None
    
    # Create a DependencyError with package and required_version
    error = DependencyError(
        "Dependency issue",
        package="numpy",
        required_version="1.20.0"
    )
    
    # Check the error message includes the package and required_version
    assert "numpy" in str(error)
    assert "1.20.0" in str(error)
    
    # Check attributes
    assert error.package == "numpy"
    assert error.required_version == "1.20.0"
    assert error.installed_version is None
    
    # Create a DependencyError with all attributes
    error = DependencyError(
        "Dependency issue",
        package="numpy",
        required_version="1.20.0",
        installed_version="1.19.0"
    )
    
    # Check the error message includes all details
    assert "numpy" in str(error)
    assert "1.20.0" in str(error)
    assert "1.19.0" in str(error)
    
    # Check attributes
    assert error.package == "numpy"
    assert error.required_version == "1.20.0"
    assert error.installed_version == "1.19.0"


def test_get_troubleshooting_info():
    """Test the get_troubleshooting_info function."""
    # Test with DataQualityError
    error = DataQualityError(
        "Data quality issue",
        channel="Fp1",
        metric="noise",
        value=10.5,
        threshold=5.0
    )
    
    info = get_troubleshooting_info(error)
    
    # Check that the info is a string
    assert isinstance(info, str)
    
    # Check that the info contains relevant troubleshooting advice
    assert "Data quality issues" in info
    
    # Test with ProcessingError
    error = ProcessingError(
        "Processing failed",
        stage="filtering",
        function="bandpass_filter"
    )
    
    info = get_troubleshooting_info(error)
    
    # Check that the info contains relevant troubleshooting advice
    assert "Processing errors" in info
    
    # Test with ValidationError
    error = ValidationError(
        "Validation failed",
        parameter="threshold",
        expected="float",
        received="str"
    )
    
    info = get_troubleshooting_info(error)
    
    # Check that the info contains relevant troubleshooting advice
    assert "Validation errors" in info
    
    # Test with ConfigurationError
    error = ConfigurationError("Configuration issue", config_key="api_key")
    
    info = get_troubleshooting_info(error)
    
    # Check that the info contains relevant troubleshooting advice
    assert "Configuration errors" in info
    
    # Test with DependencyError
    error = DependencyError(
        "Dependency issue",
        package="numpy",
        required_version="1.20.0",
        installed_version="1.19.0"
    )
    
    info = get_troubleshooting_info(error)
    
    # Check that the info contains relevant troubleshooting advice
    assert "Dependency errors" in info
    
    # Test with generic Exception
    error = Exception("Generic error")
    
    info = get_troubleshooting_info(error)
    
    # Check that the info contains a generic message
    assert "No specific troubleshooting information" in info


def test_data_quality_troubleshooting():
    """Test troubleshooting info for different data quality issues."""
    # Test with noise issue
    error = DataQualityError(
        "Data quality issue",
        channel="Fp1",
        metric="noise",
        value=10.5,
        threshold=5.0
    )
    
    info = get_troubleshooting_info(error)
    
    # Check that the info contains noise-specific advice
    assert "electrode placement" in info.lower() or "impedance" in info.lower()
    assert "notch filter" in info.lower() or "power line noise" in info.lower()
    
    # Test with missing data issue
    error = DataQualityError(
        "Data quality issue",
        channel="Fp1",
        metric="missing_data",
        value=10,
        threshold=5
    )
    
    info = get_troubleshooting_info(error)
    
    # Check that the info contains missing data-specific advice
    assert "interpolating" in info.lower() or "missing data" in info.lower()
    
    # Test with flat signal issue
    error = DataQualityError(
        "Data quality issue",
        channel="Fp1",
        metric="flat_signal",
        value=0,
        threshold=0.1
    )
    
    info = get_troubleshooting_info(error)
    
    # Check that the info contains flat signal-specific advice
    assert "electrode" in info.lower() and "connected" in info.lower()
    assert "amplifier" in info.lower() or "functioning" in info.lower()


def test_processing_troubleshooting():
    """Test troubleshooting info for different processing issues."""
    # Test with filtering issue
    error = ProcessingError(
        "Processing failed",
        stage="filtering",
        function="bandpass_filter"
    )
    
    info = get_troubleshooting_info(error)
    
    # Check that the info contains filtering-specific advice
    assert "filter parameters" in info.lower() or "cutoff frequencies" in info.lower()
    
    # Test with artifact removal issue
    error = ProcessingError(
        "Processing failed",
        stage="artifact_removal",
        function="remove_artifacts_ica"
    )
    
    info = get_troubleshooting_info(error)
    
    # Check that the info contains artifact removal-specific advice
    assert "ica parameters" in info.lower() or "artifact removal" in info.lower()
    
    # Test with feature extraction issue
    error = ProcessingError(
        "Processing failed",
        stage="feature_extraction",
        function="extract_features"
    )
    
    info = get_troubleshooting_info(error)
    
    # Check that the info contains feature extraction-specific advice
    assert "preprocessed" in info.lower() or "feature extraction" in info.lower()
