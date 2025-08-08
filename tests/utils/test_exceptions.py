"""
Tests for the exceptions module.
"""

import pytest
from epilepsy_eeg.utils.exceptions import (
    EEGError,
    PreprocessingError,
    AnalysisError,
    VisualizationError,
    MLError,
    ValidationError,
    IOError,
    ParallelError
)


def test_eeg_error():
    # Test that EEGError is a subclass of Exception
    assert issubclass(EEGError, Exception)
    
    # Test creating an EEGError
    error = EEGError("Test error message")
    assert str(error) == "Test error message"
    
    # Test raising an EEGError
    with pytest.raises(EEGError) as excinfo:
        raise EEGError("Test error message")
    assert str(excinfo.value) == "Test error message"


def test_preprocessing_error():
    # Test that PreprocessingError is a subclass of EEGError
    assert issubclass(PreprocessingError, EEGError)
    
    # Test creating a PreprocessingError
    error = PreprocessingError("Test preprocessing error")
    assert str(error) == "Test preprocessing error"
    
    # Test raising a PreprocessingError
    with pytest.raises(PreprocessingError) as excinfo:
        raise PreprocessingError("Test preprocessing error")
    assert str(excinfo.value) == "Test preprocessing error"
    
    # Test that PreprocessingError is caught by EEGError
    with pytest.raises(EEGError):
        raise PreprocessingError("Test preprocessing error")


def test_analysis_error():
    # Test that AnalysisError is a subclass of EEGError
    assert issubclass(AnalysisError, EEGError)
    
    # Test creating an AnalysisError
    error = AnalysisError("Test analysis error")
    assert str(error) == "Test analysis error"
    
    # Test raising an AnalysisError
    with pytest.raises(AnalysisError) as excinfo:
        raise AnalysisError("Test analysis error")
    assert str(excinfo.value) == "Test analysis error"
    
    # Test that AnalysisError is caught by EEGError
    with pytest.raises(EEGError):
        raise AnalysisError("Test analysis error")


def test_visualization_error():
    # Test that VisualizationError is a subclass of EEGError
    assert issubclass(VisualizationError, EEGError)
    
    # Test creating a VisualizationError
    error = VisualizationError("Test visualization error")
    assert str(error) == "Test visualization error"
    
    # Test raising a VisualizationError
    with pytest.raises(VisualizationError) as excinfo:
        raise VisualizationError("Test visualization error")
    assert str(excinfo.value) == "Test visualization error"
    
    # Test that VisualizationError is caught by EEGError
    with pytest.raises(EEGError):
        raise VisualizationError("Test visualization error")


def test_ml_error():
    # Test that MLError is a subclass of EEGError
    assert issubclass(MLError, EEGError)
    
    # Test creating an MLError
    error = MLError("Test ML error")
    assert str(error) == "Test ML error"
    
    # Test raising an MLError
    with pytest.raises(MLError) as excinfo:
        raise MLError("Test ML error")
    assert str(excinfo.value) == "Test ML error"
    
    # Test that MLError is caught by EEGError
    with pytest.raises(EEGError):
        raise MLError("Test ML error")


def test_validation_error():
    # Test that ValidationError is a subclass of EEGError
    assert issubclass(ValidationError, EEGError)
    
    # Test creating a ValidationError
    error = ValidationError("Test validation error")
    assert str(error) == "Test validation error"
    
    # Test raising a ValidationError
    with pytest.raises(ValidationError) as excinfo:
        raise ValidationError("Test validation error")
    assert str(excinfo.value) == "Test validation error"
    
    # Test that ValidationError is caught by EEGError
    with pytest.raises(EEGError):
        raise ValidationError("Test validation error")


def test_io_error():
    # Test that IOError is a subclass of EEGError
    assert issubclass(IOError, EEGError)
    
    # Test creating an IOError
    error = IOError("Test IO error")
    assert str(error) == "Test IO error"
    
    # Test raising an IOError
    with pytest.raises(IOError) as excinfo:
        raise IOError("Test IO error")
    assert str(excinfo.value) == "Test IO error"
    
    # Test that IOError is caught by EEGError
    with pytest.raises(EEGError):
        raise IOError("Test IO error")


def test_parallel_error():
    # Test that ParallelError is a subclass of EEGError
    assert issubclass(ParallelError, EEGError)
    
    # Test creating a ParallelError
    error = ParallelError("Test parallel error")
    assert str(error) == "Test parallel error"
    
    # Test raising a ParallelError
    with pytest.raises(ParallelError) as excinfo:
        raise ParallelError("Test parallel error")
    assert str(excinfo.value) == "Test parallel error"
    
    # Test that ParallelError is caught by EEGError
    with pytest.raises(EEGError):
        raise ParallelError("Test parallel error")


def test_error_chaining():
    # Test error chaining
    try:
        try:
            raise ValueError("Original error")
        except ValueError as e:
            raise PreprocessingError("Preprocessing failed") from e
    except PreprocessingError as e:
        assert str(e) == "Preprocessing failed"
        assert isinstance(e.__cause__, ValueError)
        assert str(e.__cause__) == "Original error"
