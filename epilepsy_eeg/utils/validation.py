"""
Validation module for input validation and error checking.

This module provides functions for validating inputs to functions
throughout the epilepsy_eeg package.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
import mne
import inspect

from epilepsy_eeg.utils.logging import get_logger
from epilepsy_eeg.utils.exceptions import ValidationError

# Get logger
logger = get_logger(__name__)


def validate_type(
    value: Any,
    expected_type: Any,
    param_name: str,
    allow_none: bool = False
) -> None:
    """
    Validate that a value is of the expected type.
    
    Parameters
    ----------
    value : any
        The value to validate
    expected_type : type or tuple of types
        The expected type(s)
    param_name : str
        The name of the parameter being validated
    allow_none : bool, optional
        Whether to allow None values, by default False
    
    Raises
    ------
    ValidationError
        If the value is not of the expected type
    
    Examples
    --------
    >>> from epilepsy_eeg.utils.validation import validate_type
    >>> validate_type(10, int, 'my_param')  # No error
    >>> validate_type('hello', (int, float), 'my_param')  # Raises ValidationError
    """
    if allow_none and value is None:
        return
    
    if not isinstance(value, expected_type):
        if isinstance(expected_type, tuple):
            type_names = [t.__name__ for t in expected_type]
            expected_type_str = f"one of ({', '.join(type_names)})"
        else:
            expected_type_str = expected_type.__name__
        
        actual_type_str = type(value).__name__
        
        raise ValidationError(
            f"Parameter '{param_name}' must be of type {expected_type_str}, "
            f"but got {actual_type_str}"
        )


def validate_range(
    value: Union[int, float],
    min_value: Optional[Union[int, float]] = None,
    max_value: Optional[Union[int, float]] = None,
    param_name: str = "",
    inclusive_min: bool = True,
    inclusive_max: bool = True
) -> None:
    """
    Validate that a numeric value is within a specified range.
    
    Parameters
    ----------
    value : int or float
        The value to validate
    min_value : int, float, or None, optional
        The minimum allowed value, by default None (no minimum)
    max_value : int, float, or None, optional
        The maximum allowed value, by default None (no maximum)
    param_name : str, optional
        The name of the parameter being validated, by default ""
    inclusive_min : bool, optional
        Whether the minimum value is inclusive, by default True
    inclusive_max : bool, optional
        Whether the maximum value is inclusive, by default True
    
    Raises
    ------
    ValidationError
        If the value is not within the specified range
    
    Examples
    --------
    >>> from epilepsy_eeg.utils.validation import validate_range
    >>> validate_range(5, 0, 10, 'my_param')  # No error
    >>> validate_range(10, 0, 10, 'my_param', inclusive_max=False)  # Raises ValidationError
    """
    if not isinstance(value, (int, float)):
        raise ValidationError(f"Parameter '{param_name}' must be numeric")
    
    if min_value is not None:
        if inclusive_min:
            if value < min_value:
                raise ValidationError(
                    f"Parameter '{param_name}' must be greater than or equal to {min_value}, "
                    f"but got {value}"
                )
        else:
            if value <= min_value:
                raise ValidationError(
                    f"Parameter '{param_name}' must be greater than {min_value}, "
                    f"but got {value}"
                )
    
    if max_value is not None:
        if inclusive_max:
            if value > max_value:
                raise ValidationError(
                    f"Parameter '{param_name}' must be less than or equal to {max_value}, "
                    f"but got {value}"
                )
        else:
            if value >= max_value:
                raise ValidationError(
                    f"Parameter '{param_name}' must be less than {max_value}, "
                    f"but got {value}"
                )


def validate_options(
    value: Any,
    options: List[Any],
    param_name: str = "",
    allow_none: bool = False
) -> None:
    """
    Validate that a value is one of the allowed options.
    
    Parameters
    ----------
    value : any
        The value to validate
    options : list
        The list of allowed options
    param_name : str, optional
        The name of the parameter being validated, by default ""
    allow_none : bool, optional
        Whether to allow None values, by default False
    
    Raises
    ------
    ValidationError
        If the value is not one of the allowed options
    
    Examples
    --------
    >>> from epilepsy_eeg.utils.validation import validate_options
    >>> validate_options('apple', ['apple', 'banana', 'orange'], 'fruit')  # No error
    >>> validate_options('grape', ['apple', 'banana', 'orange'], 'fruit')  # Raises ValidationError
    """
    if allow_none and value is None:
        return
    
    if value not in options:
        options_str = ", ".join(str(opt) for opt in options)
        raise ValidationError(
            f"Parameter '{param_name}' must be one of [{options_str}], "
            f"but got {value}"
        )


def validate_raw(
    raw: mne.io.Raw,
    param_name: str = "raw",
    require_preload: bool = True
) -> None:
    """
    Validate that a value is a valid MNE Raw object.
    
    Parameters
    ----------
    raw : mne.io.Raw
        The Raw object to validate
    param_name : str, optional
        The name of the parameter being validated, by default "raw"
    require_preload : bool, optional
        Whether to require that the data is preloaded, by default True
    
    Raises
    ------
    ValidationError
        If the value is not a valid MNE Raw object
    
    Examples
    --------
    >>> import mne
    >>> from epilepsy_eeg.utils.validation import validate_raw
    >>> raw = mne.io.read_raw_edf("sample.edf", preload=True)
    >>> validate_raw(raw)  # No error
    """
    validate_type(raw, mne.io.Raw, param_name)
    
    if require_preload and not raw.preload:
        raise ValidationError(
            f"Parameter '{param_name}' must be preloaded. "
            f"Use raw.load_data() before calling this function."
        )


def validate_epochs(
    epochs: mne.Epochs,
    param_name: str = "epochs",
    require_preload: bool = True
) -> None:
    """
    Validate that a value is a valid MNE Epochs object.
    
    Parameters
    ----------
    epochs : mne.Epochs
        The Epochs object to validate
    param_name : str, optional
        The name of the parameter being validated, by default "epochs"
    require_preload : bool, optional
        Whether to require that the data is preloaded, by default True
    
    Raises
    ------
    ValidationError
        If the value is not a valid MNE Epochs object
    
    Examples
    --------
    >>> import mne
    >>> from epilepsy_eeg.utils.validation import validate_epochs
    >>> raw = mne.io.read_raw_edf("sample.edf", preload=True)
    >>> events = mne.make_fixed_length_events(raw, duration=1.0)
    >>> epochs = mne.Epochs(raw, events, tmin=0, tmax=1.0, preload=True)
    >>> validate_epochs(epochs)  # No error
    """
    validate_type(epochs, mne.Epochs, param_name)
    
    if require_preload and not epochs.preload:
        raise ValidationError(
            f"Parameter '{param_name}' must be preloaded. "
            f"Use epochs.load_data() before calling this function."
        )


def validate_array(
    array: np.ndarray,
    ndim: Optional[int] = None,
    shape: Optional[Tuple[int, ...]] = None,
    dtype: Optional[np.dtype] = None,
    param_name: str = ""
) -> None:
    """
    Validate that a value is a valid NumPy array with the specified properties.
    
    Parameters
    ----------
    array : np.ndarray
        The array to validate
    ndim : int or None, optional
        The expected number of dimensions, by default None (any number of dimensions)
    shape : tuple or None, optional
        The expected shape, by default None (any shape)
    dtype : np.dtype or None, optional
        The expected data type, by default None (any data type)
    param_name : str, optional
        The name of the parameter being validated, by default ""
    
    Raises
    ------
    ValidationError
        If the value is not a valid NumPy array with the specified properties
    
    Examples
    --------
    >>> import numpy as np
    >>> from epilepsy_eeg.utils.validation import validate_array
    >>> arr = np.array([[1, 2], [3, 4]])
    >>> validate_array(arr, ndim=2, shape=(2, 2), dtype=np.int64, param_name='my_array')  # No error
    """
    validate_type(array, np.ndarray, param_name)
    
    if ndim is not None and array.ndim != ndim:
        raise ValidationError(
            f"Parameter '{param_name}' must have {ndim} dimensions, "
            f"but got {array.ndim}"
        )
    
    if shape is not None:
        if len(shape) != array.ndim:
            raise ValidationError(
                f"Parameter '{param_name}' must have shape {shape}, "
                f"but got {array.shape}"
            )
        
        for i, (s1, s2) in enumerate(zip(array.shape, shape)):
            if s2 != -1 and s1 != s2:
                raise ValidationError(
                    f"Parameter '{param_name}' must have shape {shape}, "
                    f"but got {array.shape}"
                )
    
    if dtype is not None and array.dtype != dtype:
        raise ValidationError(
            f"Parameter '{param_name}' must have dtype {dtype}, "
            f"but got {array.dtype}"
        )


def validate_function_params(func: Callable, **kwargs) -> Dict[str, Any]:
    """
    Validate that the provided keyword arguments are valid for the function.
    
    Parameters
    ----------
    func : callable
        The function to validate parameters for
    **kwargs
        The keyword arguments to validate
    
    Returns
    -------
    Dict[str, Any]
        Dictionary of valid keyword arguments
    
    Raises
    ------
    ValidationError
        If any of the keyword arguments are not valid for the function
    
    Examples
    --------
    >>> from epilepsy_eeg.utils.validation import validate_function_params
    >>> def my_func(a, b=1, c=2):
    ...     return a + b + c
    >>> valid_kwargs = validate_function_params(my_func, a=10, b=20)
    >>> print(valid_kwargs)
    {'a': 10, 'b': 20}
    """
    # Get the function signature
    sig = inspect.signature(func)
    
    # Check for unknown parameters
    valid_params = set(sig.parameters.keys())
    for param_name in kwargs:
        if param_name not in valid_params:
            raise ValidationError(
                f"Unknown parameter '{param_name}' for function '{func.__name__}'. "
                f"Valid parameters are: {', '.join(valid_params)}"
            )
    
    return kwargs


def validate_frequency_bands(
    bands: Dict[str, Tuple[float, float]],
    param_name: str = "frequency_bands"
) -> None:
    """
    Validate that a dictionary of frequency bands is valid.
    
    Parameters
    ----------
    bands : Dict[str, Tuple[float, float]]
        Dictionary mapping band names to (fmin, fmax) tuples
    param_name : str, optional
        The name of the parameter being validated, by default "frequency_bands"
    
    Raises
    ------
    ValidationError
        If the frequency bands are not valid
    
    Examples
    --------
    >>> from epilepsy_eeg.utils.validation import validate_frequency_bands
    >>> bands = {'alpha': (8, 13), 'beta': (13, 30)}
    >>> validate_frequency_bands(bands)  # No error
    """
    validate_type(bands, dict, param_name)
    
    for band_name, (fmin, fmax) in bands.items():
        validate_type(band_name, str, f"{param_name} key")
        validate_type((fmin, fmax), tuple, f"{param_name}[{band_name}]")
        
        if len((fmin, fmax)) != 2:
            raise ValidationError(
                f"Each frequency band in '{param_name}' must be a tuple of (fmin, fmax), "
                f"but got {(fmin, fmax)} for band '{band_name}'"
            )
        
        validate_type(fmin, (int, float), f"{param_name}[{band_name}][0]")
        validate_type(fmax, (int, float), f"{param_name}[{band_name}][1]")
        
        if fmin < 0:
            raise ValidationError(
                f"Minimum frequency in '{param_name}[{band_name}]' must be non-negative, "
                f"but got {fmin}"
            )
        
        if fmax <= fmin:
            raise ValidationError(
                f"Maximum frequency in '{param_name}[{band_name}]' must be greater than "
                f"minimum frequency, but got fmin={fmin}, fmax={fmax}"
            )


def validate_picks(
    picks: Union[str, List[str], List[int], None],
    info: mne.Info,
    param_name: str = "picks"
) -> None:
    """
    Validate that picks are valid for the given MNE Info object.
    
    Parameters
    ----------
    picks : str, list, or None
        The picks to validate
    info : mne.Info
        The MNE Info object
    param_name : str, optional
        The name of the parameter being validated, by default "picks"
    
    Raises
    ------
    ValidationError
        If the picks are not valid
    
    Examples
    --------
    >>> import mne
    >>> from epilepsy_eeg.utils.validation import validate_picks
    >>> raw = mne.io.read_raw_edf("sample.edf", preload=True)
    >>> validate_picks('eeg', raw.info)  # No error
    >>> validate_picks(['Fp1', 'Fp2'], raw.info)  # No error if these channels exist
    """
    if picks is None:
        return
    
    if isinstance(picks, str):
        if picks not in ['all', 'data', 'eeg', 'meg', 'misc']:
            raise ValidationError(
                f"String value for '{param_name}' must be one of "
                f"['all', 'data', 'eeg', 'meg', 'misc'], but got '{picks}'"
            )
    elif isinstance(picks, list):
        if len(picks) == 0:
            raise ValidationError(f"Parameter '{param_name}' cannot be an empty list")
        
        if all(isinstance(p, int) for p in picks):
            # Check that all indices are valid
            for p in picks:
                if p < 0 or p >= len(info['ch_names']):
                    raise ValidationError(
                        f"Channel index {p} in '{param_name}' is out of bounds "
                        f"(0 to {len(info['ch_names']) - 1})"
                    )
        elif all(isinstance(p, str) for p in picks):
            # Check that all channel names exist
            for p in picks:
                if p not in info['ch_names']:
                    raise ValidationError(
                        f"Channel name '{p}' in '{param_name}' does not exist in the data"
                    )
        else:
            raise ValidationError(
                f"Parameter '{param_name}' must be a list of all integers or all strings"
            )
    else:
        raise ValidationError(
            f"Parameter '{param_name}' must be None, a string, or a list of "
            f"channel indices or names"
        )


def input_validator(func: Callable) -> Callable:
    """
    Decorator to validate function inputs based on type annotations.
    
    Parameters
    ----------
    func : callable
        The function to decorate
    
    Returns
    -------
    callable
        The decorated function
    
    Examples
    --------
    >>> from epilepsy_eeg.utils.validation import input_validator
    >>> @input_validator
    ... def add_numbers(a: int, b: int) -> int:
    ...     return a + b
    >>> add_numbers(1, 2)  # No error
    3
    >>> add_numbers(1, '2')  # Raises ValidationError
    """
    sig = inspect.signature(func)
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Bind the arguments to the function signature
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        
        # Validate each parameter
        for param_name, param_value in bound_args.arguments.items():
            param = sig.parameters[param_name]
            
            # Skip validation for parameters without annotations
            if param.annotation is param.empty:
                continue
            
            # Get the expected type from the annotation
            expected_type = param.annotation
            
            # Handle Optional types
            allow_none = False
            if hasattr(expected_type, "__origin__") and expected_type.__origin__ is Union:
                if type(None) in expected_type.__args__:
                    allow_none = True
                    # Filter out None from the expected types
                    expected_type = tuple(t for t in expected_type.__args__ if t is not type(None))
            
            # Validate the parameter
            validate_type(param_value, expected_type, param_name, allow_none=allow_none)
        
        # Call the function
        return func(*args, **kwargs)
    
    return wrapper
