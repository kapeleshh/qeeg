"""
Validation utilities for EEG data processing.

This module provides functions for validating inputs to EEG processing functions,
ensuring data quality, and checking parameter values.
"""

import numpy as np
import mne
from typing import List, Dict, Any, Optional, Union, Tuple, Callable

from qeeg.utils.exceptions import ValidationError, DataQualityError


def validate_type(value: Any, expected_type: Union[type, Tuple[type, ...]], param_name: str = None):
    """
    Validate that a value is of the expected type.
    
    Parameters
    ----------
    value : Any
        The value to validate
    expected_type : type or tuple of types
        The expected type(s)
    param_name : str, optional
        The parameter name, by default None
        
    Raises
    ------
    ValidationError
        If the value is not of the expected type
        
    Examples
    --------
    >>> from qeeg.utils.validation import validate_type
    >>> 
    >>> # Validate a parameter
    >>> def process_data(data, threshold=0.5):
    >>>     validate_type(data, np.ndarray, 'data')
    >>>     validate_type(threshold, (int, float), 'threshold')
    >>>     # Process data
    """
    if not isinstance(value, expected_type):
        received_type = type(value).__name__
        if isinstance(expected_type, tuple):
            expected_type_str = ' or '.join(t.__name__ for t in expected_type)
        else:
            expected_type_str = expected_type.__name__
        
        raise ValidationError(
            f"Invalid type for {param_name or 'parameter'}.",
            parameter=param_name,
            expected=expected_type_str,
            received=received_type
        )


def validate_range(value: Union[int, float], min_value: Optional[Union[int, float]] = None,
                  max_value: Optional[Union[int, float]] = None, param_name: str = None):
    """
    Validate that a numeric value is within a specified range.
    
    Parameters
    ----------
    value : int or float
        The value to validate
    min_value : int or float, optional
        The minimum allowed value, by default None
    max_value : int or float, optional
        The maximum allowed value, by default None
    param_name : str, optional
        The parameter name, by default None
        
    Raises
    ------
    ValidationError
        If the value is outside the specified range
        
    Examples
    --------
    >>> from qeeg.utils.validation import validate_range
    >>> 
    >>> # Validate a parameter
    >>> def filter_data(data, low_freq=1.0, high_freq=40.0):
    >>>     validate_range(low_freq, 0.0, high_freq, 'low_freq')
    >>>     validate_range(high_freq, low_freq, None, 'high_freq')
    >>>     # Filter data
    """
    validate_type(value, (int, float), param_name)
    
    if min_value is not None and value < min_value:
        raise ValidationError(
            f"Value for {param_name or 'parameter'} is too small.",
            parameter=param_name,
            expected=f">= {min_value}",
            received=str(value)
        )
    
    if max_value is not None and value > max_value:
        raise ValidationError(
            f"Value for {param_name or 'parameter'} is too large.",
            parameter=param_name,
            expected=f"<= {max_value}",
            received=str(value)
        )


def validate_options(value: Any, options: List[Any], param_name: str = None):
    """
    Validate that a value is one of the allowed options.
    
    Parameters
    ----------
    value : Any
        The value to validate
    options : list
        The allowed options
    param_name : str, optional
        The parameter name, by default None
        
    Raises
    ------
    ValidationError
        If the value is not one of the allowed options
        
    Examples
    --------
    >>> from qeeg.utils.validation import validate_options
    >>> 
    >>> # Validate a parameter
    >>> def process_data(data, method='welch'):
    >>>     validate_options(method, ['welch', 'multitaper', 'fft'], 'method')
    >>>     # Process data
    """
    if value not in options:
        options_str = ', '.join(repr(opt) for opt in options)
        raise ValidationError(
            f"Invalid value for {param_name or 'parameter'}.",
            parameter=param_name,
            expected=f"one of [{options_str}]",
            received=repr(value)
        )


def validate_raw(raw: mne.io.Raw, checks: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Validate raw EEG data before processing.
    
    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data
    checks : list of str, optional
        List of validation checks to perform, by default None
        If None, all checks are performed.
        Available checks: 'type', 'preload', 'channels', 'sfreq', 'duration', 'data_quality'
        
    Returns
    -------
    dict
        Dictionary with validation results
        
    Raises
    ------
    ValidationError
        If the raw data fails validation
    DataQualityError
        If the data quality is insufficient
        
    Examples
    --------
    >>> import mne
    >>> from qeeg.utils.validation import validate_raw
    >>> 
    >>> # Load EEG data
    >>> raw = mne.io.read_raw_edf("sample.edf", preload=True)
    >>> 
    >>> # Validate the data
    >>> validation_results = validate_raw(raw)
    >>> print(f"Data duration: {validation_results['duration']:.2f} seconds")
    >>> print(f"Number of channels: {validation_results['n_channels']}")
    """
    # Default checks
    if checks is None:
        checks = ['type', 'preload', 'channels', 'sfreq', 'duration', 'data_quality']
    
    # Initialize results dictionary
    results = {}
    
    # Check type
    if 'type' in checks:
        validate_type(raw, mne.io.Raw, 'raw')
        results['type'] = 'mne.io.Raw'
    
    # Check if data is preloaded
    if 'preload' in checks:
        if not raw.preload:
            raise ValidationError(
                "Raw data must be preloaded before processing.",
                parameter='raw.preload',
                expected='True',
                received='False'
            )
        results['preload'] = True
    
    # Check channels
    if 'channels' in checks:
        n_channels = len(raw.ch_names)
        if n_channels == 0:
            raise ValidationError(
                "Raw data has no channels.",
                parameter='raw.ch_names',
                expected='at least 1 channel',
                received='0 channels'
            )
        results['n_channels'] = n_channels
        results['ch_names'] = raw.ch_names
    
    # Check sampling frequency
    if 'sfreq' in checks:
        sfreq = raw.info['sfreq']
        if sfreq <= 0:
            raise ValidationError(
                "Invalid sampling frequency.",
                parameter='raw.info[\'sfreq\']',
                expected='> 0',
                received=str(sfreq)
            )
        results['sfreq'] = sfreq
    
    # Check duration
    if 'duration' in checks:
        duration = raw.times[-1]
        if duration <= 0:
            raise ValidationError(
                "Invalid duration.",
                parameter='raw.times[-1]',
                expected='> 0',
                received=str(duration)
            )
        results['duration'] = duration
    
    # Check data quality
    if 'data_quality' in checks:
        # Get the data
        data = raw.get_data()
        
        # Check for NaN values
        nan_channels = []
        for i, ch_name in enumerate(raw.ch_names):
            if np.isnan(data[i]).any():
                nan_channels.append(ch_name)
        
        if nan_channels:
            raise DataQualityError(
                "Raw data contains NaN values.",
                channel=nan_channels[0] if len(nan_channels) == 1 else None,
                metric='nan_count',
                value=len(nan_channels),
                threshold=0
            )
        
        # Check for flat signals
        flat_channels = []
        for i, ch_name in enumerate(raw.ch_names):
            if np.std(data[i]) < 1e-10:
                flat_channels.append(ch_name)
        
        if flat_channels:
            raise DataQualityError(
                "Raw data contains flat signals.",
                channel=flat_channels[0] if len(flat_channels) == 1 else None,
                metric='std',
                value=0,
                threshold=1e-10
            )
        
        # Check for excessive noise
        noisy_channels = []
        for i, ch_name in enumerate(raw.ch_names):
            # Calculate signal-to-noise ratio (SNR)
            # This is a simple estimate based on the ratio of signal power to noise power
            # In practice, you would use a more sophisticated method
            signal = data[i]
            signal_power = np.var(signal)
            
            # Estimate noise power using the difference between adjacent samples
            noise = np.diff(signal)
            noise_power = np.var(noise)
            
            # Calculate SNR
            if noise_power > 0:
                snr = 10 * np.log10(signal_power / noise_power)
                if snr < 0:  # SNR below 0 dB indicates more noise than signal
                    noisy_channels.append((ch_name, snr))
        
        if noisy_channels:
            ch_name, snr = min(noisy_channels, key=lambda x: x[1])
            raise DataQualityError(
                "Raw data contains excessive noise.",
                channel=ch_name,
                metric='snr',
                value=snr,
                threshold=0
            )
        
        results['data_quality'] = 'good'
    
    return results


def validate_epochs(epochs: mne.Epochs, checks: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Validate epoched EEG data before processing.
    
    Parameters
    ----------
    epochs : mne.Epochs
        The epoched EEG data
    checks : list of str, optional
        List of validation checks to perform, by default None
        If None, all checks are performed.
        Available checks: 'type', 'preload', 'channels', 'sfreq', 'n_epochs', 'data_quality'
        
    Returns
    -------
    dict
        Dictionary with validation results
        
    Raises
    ------
    ValidationError
        If the epoched data fails validation
    DataQualityError
        If the data quality is insufficient
        
    Examples
    --------
    >>> import mne
    >>> from qeeg.utils.validation import validate_epochs
    >>> 
    >>> # Load EEG data and create epochs
    >>> raw = mne.io.read_raw_edf("sample.edf", preload=True)
    >>> events = mne.make_fixed_length_events(raw, duration=1.0)
    >>> epochs = mne.Epochs(raw, events, tmin=0, tmax=1.0, baseline=None, preload=True)
    >>> 
    >>> # Validate the epochs
    >>> validation_results = validate_epochs(epochs)
    >>> print(f"Number of epochs: {validation_results['n_epochs']}")
    >>> print(f"Number of channels: {validation_results['n_channels']}")
    """
    # Default checks
    if checks is None:
        checks = ['type', 'preload', 'channels', 'sfreq', 'n_epochs', 'data_quality']
    
    # Initialize results dictionary
    results = {}
    
    # Check type
    if 'type' in checks:
        validate_type(epochs, mne.Epochs, 'epochs')
        results['type'] = 'mne.Epochs'
    
    # Check if data is preloaded
    if 'preload' in checks:
        if not epochs.preload:
            raise ValidationError(
                "Epoched data must be preloaded before processing.",
                parameter='epochs.preload',
                expected='True',
                received='False'
            )
        results['preload'] = True
    
    # Check channels
    if 'channels' in checks:
        n_channels = len(epochs.ch_names)
        if n_channels == 0:
            raise ValidationError(
                "Epoched data has no channels.",
                parameter='epochs.ch_names',
                expected='at least 1 channel',
                received='0 channels'
            )
        results['n_channels'] = n_channels
        results['ch_names'] = epochs.ch_names
    
    # Check sampling frequency
    if 'sfreq' in checks:
        sfreq = epochs.info['sfreq']
        if sfreq <= 0:
            raise ValidationError(
                "Invalid sampling frequency.",
                parameter='epochs.info[\'sfreq\']',
                expected='> 0',
                received=str(sfreq)
            )
        results['sfreq'] = sfreq
    
    # Check number of epochs
    if 'n_epochs' in checks:
        n_epochs = len(epochs)
        if n_epochs == 0:
            raise ValidationError(
                "No epochs available.",
                parameter='len(epochs)',
                expected='> 0',
                received='0'
            )
        results['n_epochs'] = n_epochs
    
    # Check data quality
    if 'data_quality' in checks:
        # Get the data
        data = epochs.get_data()
        
        # Check for NaN values
        nan_epochs = []
        for i in range(len(epochs)):
            if np.isnan(data[i]).any():
                nan_epochs.append(i)
        
        if nan_epochs:
            raise DataQualityError(
                "Epoched data contains NaN values.",
                metric='nan_count',
                value=len(nan_epochs),
                threshold=0
            )
        
        # Check for flat signals
        flat_epochs = []
        for i in range(len(epochs)):
            if np.all(np.std(data[i], axis=1) < 1e-10):
                flat_epochs.append(i)
        
        if flat_epochs:
            raise DataQualityError(
                "Epoched data contains flat signals.",
                metric='std',
                value=0,
                threshold=1e-10
            )
        
        results['data_quality'] = 'good'
    
    return results


def validate_function_params(func: Callable, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate parameters for a function based on its annotations.
    
    Parameters
    ----------
    func : callable
        The function to validate parameters for
    params : dict
        Dictionary of parameter names and values
        
    Returns
    -------
    dict
        Dictionary with validated parameters
        
    Raises
    ------
    ValidationError
        If a parameter fails validation
        
    Examples
    --------
    >>> from qeeg.utils.validation import validate_function_params
    >>> 
    >>> # Define a function with annotations
    >>> def filter_data(raw, l_freq: float = 1.0, h_freq: float = 40.0, method: str = 'fir'):
    >>>     '''Filter EEG data.'''
    >>>     # Implementation
    >>> 
    >>> # Validate parameters
    >>> params = {'raw': raw, 'l_freq': 0.5, 'h_freq': 50.0, 'method': 'iir'}
    >>> validated_params = validate_function_params(filter_data, params)
    """
    import inspect
    
    # Get function signature
    sig = inspect.signature(func)
    
    # Initialize validated parameters dictionary
    validated_params = {}
    
    # Validate each parameter
    for param_name, param_value in params.items():
        # Check if parameter exists in function signature
        if param_name not in sig.parameters:
            raise ValidationError(
                f"Unknown parameter: {param_name}",
                parameter=param_name
            )
        
        # Get parameter from signature
        param = sig.parameters[param_name]
        
        # Check if parameter has a type annotation
        if param.annotation != inspect.Parameter.empty:
            # Get expected type from annotation
            expected_type = param.annotation
            
            # Handle Optional types
            if hasattr(expected_type, '__origin__') and expected_type.__origin__ is Union:
                # Check if Union includes NoneType
                if type(None) in expected_type.__args__:
                    # It's an Optional type
                    if param_value is None:
                        # None is allowed
                        validated_params[param_name] = param_value
                        continue
                    
                    # Get the non-None types
                    expected_types = tuple(t for t in expected_type.__args__ if t is not type(None))
                    
                    # Validate against the non-None types
                    try:
                        validate_type(param_value, expected_types, param_name)
                    except ValidationError:
                        # If validation fails, try to convert the value
                        converted = False
                        for t in expected_types:
                            try:
                                validated_params[param_name] = t(param_value)
                                converted = True
                                break
                            except (ValueError, TypeError):
                                pass
                        
                        if not converted:
                            raise
                else:
                    # It's a regular Union type
                    try:
                        validate_type(param_value, expected_type.__args__, param_name)
                    except ValidationError:
                        # If validation fails, try to convert the value
                        converted = False
                        for t in expected_type.__args__:
                            try:
                                validated_params[param_name] = t(param_value)
                                converted = True
                                break
                            except (ValueError, TypeError):
                                pass
                        
                        if not converted:
                            raise
            else:
                # Regular type
                try:
                    validate_type(param_value, expected_type, param_name)
                except ValidationError:
                    # If validation fails, try to convert the value
                    try:
                        validated_params[param_name] = expected_type(param_value)
                    except (ValueError, TypeError):
                        raise
        
        # If no type annotation or validation passed, use the original value
        if param_name not in validated_params:
            validated_params[param_name] = param_value
    
    return validated_params


def validate_montage(montage: Union[mne.channels.DigMontage, str], raw: Optional[mne.io.Raw] = None) -> bool:
    """
    Validate that a montage is compatible with the raw data.
    
    Parameters
    ----------
    montage : mne.channels.DigMontage or str
        The montage to validate
    raw : mne.io.Raw, optional
        The raw EEG data, by default None
        
    Returns
    -------
    bool
        True if the montage is valid
        
    Raises
    ------
    ValidationError
        If the montage is invalid or incompatible with the raw data
        
    Examples
    --------
    >>> import mne
    >>> from qeeg.utils.validation import validate_montage
    >>> 
    >>> # Load EEG data
    >>> raw = mne.io.read_raw_edf("sample.edf", preload=True)
    >>> 
    >>> # Validate montage
    >>> montage = mne.channels.make_standard_montage('standard_1020')
    >>> is_valid = validate_montage(montage, raw)
    """
    # Check montage type
    if isinstance(montage, str):
        try:
            montage = mne.channels.make_standard_montage(montage)
        except ValueError as e:
            raise ValidationError(
                f"Invalid montage name: {montage}",
                parameter='montage'
            ) from e
    elif not isinstance(montage, mne.channels.DigMontage):
        raise ValidationError(
            "Invalid montage type.",
            parameter='montage',
            expected='mne.channels.DigMontage or str',
            received=type(montage).__name__
        )
    
    # If raw is provided, check compatibility
    if raw is not None:
        # Check if all channels in raw are in montage
        montage_ch_names = set(montage.ch_names)
        raw_ch_names = set(raw.ch_names)
        
        missing_channels = raw_ch_names - montage_ch_names
        if missing_channels:
            # Some channels are missing from the montage
            # This is not necessarily an error, but we should warn about it
            import warnings
            warnings.warn(
                f"The following channels are not in the montage: {', '.join(missing_channels)}. "
                "These channels will not have position information."
            )
    
    return True


def validate_channel_names(ch_names: List[str], standard: str = '10-20') -> List[str]:
    """
    Validate that channel names conform to a standard naming convention.
    
    Parameters
    ----------
    ch_names : list of str
        List of channel names to validate
    standard : str, optional
        The standard naming convention, by default '10-20'
        Options: '10-20', '10-10', '10-5'
        
    Returns
    -------
    list of str
        List of non-standard channel names
        
    Examples
    --------
    >>> from qeeg.utils.validation import validate_channel_names
    >>> 
    >>> # Validate channel names
    >>> ch_names = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2', 'EKG']
    >>> non_standard = validate_channel_names(ch_names)
    >>> print(f"Non-standard channel names: {non_standard}")
    """
    # Define standard channel names for different systems
    standard_channels = {
        '10-20': {
            'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
            'T3', 'C3', 'Cz', 'C4', 'T4',
            'T5', 'P3', 'Pz', 'P4', 'T6',
            'O1', 'O2'
        },
        '10-10': {
            'Fp1', 'Fpz', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8',
            'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8',
            'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8',
            'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8',
            'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8',
            'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8',
            'PO7', 'PO3', 'POz', 'PO4', 'PO8',
            'O1', 'Oz', 'O2'
        },
        '10-5': {
            'Fp1', 'Fpz', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8',
            'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8',
            'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8',
            'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8',
            'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8',
            'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8',
            'PO7', 'PO3', 'POz', 'PO4', 'PO8',
            'O1', 'Oz', 'O2',
            # Additional 10-5 channels
            'Nz', 'Iz', 'NFpz', 'AFp1', 'AFp2',
            'AFF1', 'AFF2', 'AFF5', 'AFF6',
            'FFT7', 'FFT8', 'FFC1', 'FFC2', 'FFC3', 'FFC4', 'FFC5', 'FFC6',
            'FTT7', 'FTT8', 'FCC1', 'FCC2', 'FCC3', 'FCC4', 'FCC5', 'FCC6',
            'TTP7', 'TTP8', 'CCP1', 'CCP2', 'CCP3', 'CCP4', 'CCP5', 'CCP6',
            'TPP7', 'TPP8', 'CPP1', 'CPP2', 'CPP3', 'CPP4', 'CPP5', 'CPP6',
            'PPO1', 'PPO2', 'PPO5', 'PPO6',
            'POO1', 'POO2', 'POO5', 'POO6'
        }
    }
    
    # Validate standard
    if standard not in standard_channels:
        raise ValidationError(
            f"Invalid standard: {standard}",
            parameter='standard',
            expected=f"one of {list(standard_channels.keys())}",
            received=standard
        )
    
    # Check each channel name
    non_standard = []
    for ch_name in ch_names:
        # Skip common non-EEG channels
        if ch_name in ['EKG', 'ECG', 'EMG', 'EOG', 'VEOG', 'HEOG']:
            continue
        
        # Check if channel name is in the standard
        if ch_name not in standard_channels[standard]:
            non_standard.append(ch_name)
    
    return non_standard
