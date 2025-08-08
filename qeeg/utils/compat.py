"""
Compatibility module for external dependencies.

This module provides functions for checking and managing compatibility
with external dependencies used by the qeeg package.
"""

import importlib
import sys
from typing import Dict, List, Optional, Tuple, Union
import warnings
from packaging import version

from qeeg.utils.logging import get_logger
from qeeg.utils.exceptions import EEGError

# Get logger
logger = get_logger(__name__)

# Define minimum required versions for dependencies
MINIMUM_VERSIONS = {
    'numpy': '1.20.0',
    'scipy': '1.6.0',
    'matplotlib': '3.3.0',
    'mne': '1.0.0',
    'pandas': '1.2.0',
    'scikit-learn': '0.24.0',
    'networkx': '2.5.0',
    'joblib': '1.0.0',
    'pytest': '6.0.0',
    'pytest-cov': '2.12.0',
}

# Define recommended versions for dependencies
RECOMMENDED_VERSIONS = {
    'numpy': '1.22.0',
    'scipy': '1.8.0',
    'matplotlib': '3.5.0',
    'mne': '1.2.0',
    'pandas': '1.4.0',
    'scikit-learn': '1.0.0',
    'networkx': '2.8.0',
    'joblib': '1.1.0',
    'pytest': '7.0.0',
    'pytest-cov': '3.0.0',
}


def get_installed_version(package_name: str) -> Optional[str]:
    """
    Get the installed version of a package.
    
    Parameters
    ----------
    package_name : str
        Name of the package
    
    Returns
    -------
    str or None
        Installed version, or None if package is not installed
    
    Examples
    --------
    >>> from qeeg.utils.compat import get_installed_version
    >>> version = get_installed_version('numpy')
    >>> print(f"Installed NumPy version: {version}")
    """
    try:
        if package_name == 'scikit-learn':
            # scikit-learn is imported as sklearn
            module = importlib.import_module('sklearn')
        else:
            module = importlib.import_module(package_name)
        
        # Try different version attributes
        for attr in ['__version__', 'version', '__VERSION__']:
            if hasattr(module, attr):
                return getattr(module, attr)
        
        # If no version attribute is found
        return "unknown"
    except ImportError:
        return None


def check_version(
    package_name: str,
    min_version: Optional[str] = None,
    recommended_version: Optional[str] = None
) -> Tuple[bool, bool]:
    """
    Check if the installed version of a package meets minimum and recommended versions.
    
    Parameters
    ----------
    package_name : str
        Name of the package
    min_version : str or None, optional
        Minimum required version, by default None (use MINIMUM_VERSIONS)
    recommended_version : str or None, optional
        Recommended version, by default None (use RECOMMENDED_VERSIONS)
    
    Returns
    -------
    Tuple[bool, bool]
        (meets_minimum, meets_recommended)
    
    Examples
    --------
    >>> from qeeg.utils.compat import check_version
    >>> meets_min, meets_rec = check_version('numpy')
    >>> if meets_min:
    ...     print("NumPy meets minimum version requirement")
    >>> if meets_rec:
    ...     print("NumPy meets recommended version requirement")
    """
    # Get installed version
    installed_version = get_installed_version(package_name)
    
    if installed_version is None:
        logger.warning(f"Package {package_name} is not installed")
        return False, False
    
    if installed_version == "unknown":
        logger.warning(f"Could not determine version for {package_name}")
        return True, True  # Assume it's compatible
    
    # Get minimum and recommended versions
    if min_version is None and package_name in MINIMUM_VERSIONS:
        min_version = MINIMUM_VERSIONS[package_name]
    
    if recommended_version is None and package_name in RECOMMENDED_VERSIONS:
        recommended_version = RECOMMENDED_VERSIONS[package_name]
    
    # Check versions
    meets_minimum = True
    meets_recommended = True
    
    if min_version is not None:
        try:
            meets_minimum = version.parse(installed_version) >= version.parse(min_version)
        except Exception as e:
            logger.warning(f"Error comparing versions for {package_name}: {str(e)}")
    
    if recommended_version is not None:
        try:
            meets_recommended = version.parse(installed_version) >= version.parse(recommended_version)
        except Exception as e:
            logger.warning(f"Error comparing versions for {package_name}: {str(e)}")
    
    return meets_minimum, meets_recommended


def check_dependencies(
    dependencies: Optional[List[str]] = None,
    raise_error: bool = False
) -> Dict[str, Dict[str, Union[str, bool]]]:
    """
    Check if all dependencies meet minimum and recommended versions.
    
    Parameters
    ----------
    dependencies : List[str] or None, optional
        List of dependencies to check, by default None (check all in MINIMUM_VERSIONS)
    raise_error : bool, optional
        Whether to raise an error if minimum versions are not met, by default False
    
    Returns
    -------
    Dict[str, Dict[str, Union[str, bool]]]
        Dictionary with dependency status
    
    Examples
    --------
    >>> from qeeg.utils.compat import check_dependencies
    >>> status = check_dependencies()
    >>> for package, info in status.items():
    ...     print(f"{package}: installed={info['installed_version']}, "
    ...           f"minimum={info['meets_minimum']}, "
    ...           f"recommended={info['meets_recommended']}")
    """
    if dependencies is None:
        dependencies = list(MINIMUM_VERSIONS.keys())
    
    results = {}
    all_minimum_met = True
    
    for package in dependencies:
        installed_version = get_installed_version(package)
        meets_minimum, meets_recommended = check_version(package)
        
        results[package] = {
            'installed_version': installed_version if installed_version is not None else "not installed",
            'minimum_version': MINIMUM_VERSIONS.get(package, "not specified"),
            'recommended_version': RECOMMENDED_VERSIONS.get(package, "not specified"),
            'meets_minimum': meets_minimum,
            'meets_recommended': meets_recommended
        }
        
        if not meets_minimum:
            all_minimum_met = False
            message = (f"Package {package} version {installed_version} does not meet "
                      f"minimum requirement {MINIMUM_VERSIONS.get(package)}")
            logger.warning(message)
            warnings.warn(message, UserWarning)
    
    if not all_minimum_met and raise_error:
        raise EEGError("Some dependencies do not meet minimum version requirements")
    
    return results


def get_mne_version() -> Optional[str]:
    """
    Get the installed MNE version.
    
    Returns
    -------
    str or None
        Installed MNE version, or None if not installed
    
    Examples
    --------
    >>> from qeeg.utils.compat import get_mne_version
    >>> version = get_mne_version()
    >>> print(f"Installed MNE version: {version}")
    """
    return get_installed_version('mne')


def check_mne_version(min_version: str = '1.0.0') -> bool:
    """
    Check if the installed MNE version meets the minimum requirement.
    
    Parameters
    ----------
    min_version : str, optional
        Minimum required version, by default '1.0.0'
    
    Returns
    -------
    bool
        True if MNE meets the minimum version requirement
    
    Examples
    --------
    >>> from qeeg.utils.compat import check_mne_version
    >>> if check_mne_version('1.0.0'):
    ...     print("MNE meets minimum version requirement")
    """
    meets_minimum, _ = check_version('mne', min_version=min_version)
    return meets_minimum


def get_sklearn_version() -> Optional[str]:
    """
    Get the installed scikit-learn version.
    
    Returns
    -------
    str or None
        Installed scikit-learn version, or None if not installed
    
    Examples
    --------
    >>> from qeeg.utils.compat import get_sklearn_version
    >>> version = get_sklearn_version()
    >>> print(f"Installed scikit-learn version: {version}")
    """
    return get_installed_version('scikit-learn')


def check_sklearn_version(min_version: str = '0.24.0') -> bool:
    """
    Check if the installed scikit-learn version meets the minimum requirement.
    
    Parameters
    ----------
    min_version : str, optional
        Minimum required version, by default '0.24.0'
    
    Returns
    -------
    bool
        True if scikit-learn meets the minimum version requirement
    
    Examples
    --------
    >>> from qeeg.utils.compat import check_sklearn_version
    >>> if check_sklearn_version('0.24.0'):
    ...     print("scikit-learn meets minimum version requirement")
    """
    meets_minimum, _ = check_version('scikit-learn', min_version=min_version)
    return meets_minimum


def print_dependency_status() -> None:
    """
    Print the status of all dependencies.
    
    Examples
    --------
    >>> from qeeg.utils.compat import print_dependency_status
    >>> print_dependency_status()
    """
    status = check_dependencies()
    
    print("\nDependency Status:")
    print("-" * 80)
    print(f"{'Package':<15} {'Installed':<15} {'Minimum':<15} {'Recommended':<15} {'Status'}")
    print("-" * 80)
    
    for package, info in status.items():
        if info['meets_minimum'] and info['meets_recommended']:
            status_str = "✓ (recommended)"
        elif info['meets_minimum']:
            status_str = "✓ (minimum)"
        else:
            status_str = "✗ (outdated)"
        
        print(f"{package:<15} {info['installed_version']:<15} "
              f"{info['minimum_version']:<15} {info['recommended_version']:<15} {status_str}")
    
    print("-" * 80)
