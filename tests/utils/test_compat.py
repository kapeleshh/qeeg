"""
Tests for the compatibility module.
"""

import pytest
from unittest.mock import patch, MagicMock
import importlib

from epilepsy_eeg.utils.compat import (
    get_installed_version,
    check_version,
    check_dependencies,
    get_mne_version,
    check_mne_version,
    get_sklearn_version,
    check_sklearn_version,
    print_dependency_status,
    MINIMUM_VERSIONS,
    RECOMMENDED_VERSIONS
)
from epilepsy_eeg.utils.exceptions import EEGError


def test_get_installed_version():
    # Test with a module that exists
    with patch('importlib.import_module') as mock_import:
        mock_module = MagicMock()
        mock_module.__version__ = '1.0.0'
        mock_import.return_value = mock_module
        
        version = get_installed_version('numpy')
        assert version == '1.0.0'
    
    # Test with a module that doesn't have __version__
    with patch('importlib.import_module') as mock_import:
        mock_module = MagicMock()
        # No __version__ attribute
        mock_import.return_value = mock_module
        
        version = get_installed_version('some_module')
        assert version == 'unknown'
    
    # Test with a module that doesn't exist
    with patch('importlib.import_module', side_effect=ImportError):
        version = get_installed_version('nonexistent_module')
        assert version is None
    
    # Test with scikit-learn (special case)
    with patch('importlib.import_module') as mock_import:
        mock_module = MagicMock()
        mock_module.__version__ = '1.0.0'
        mock_import.return_value = mock_module
        
        version = get_installed_version('scikit-learn')
        assert version == '1.0.0'
        mock_import.assert_called_with('sklearn')


def test_check_version():
    # Test with a module that meets minimum and recommended versions
    with patch('epilepsy_eeg.utils.compat.get_installed_version', return_value='1.2.0'):
        meets_min, meets_rec = check_version('numpy', min_version='1.0.0', recommended_version='1.1.0')
        assert meets_min is True
        assert meets_rec is True
    
    # Test with a module that meets minimum but not recommended version
    with patch('epilepsy_eeg.utils.compat.get_installed_version', return_value='1.0.5'):
        meets_min, meets_rec = check_version('numpy', min_version='1.0.0', recommended_version='1.1.0')
        assert meets_min is True
        assert meets_rec is False
    
    # Test with a module that doesn't meet minimum version
    with patch('epilepsy_eeg.utils.compat.get_installed_version', return_value='0.9.0'):
        meets_min, meets_rec = check_version('numpy', min_version='1.0.0', recommended_version='1.1.0')
        assert meets_min is False
        assert meets_rec is False
    
    # Test with a module that doesn't exist
    with patch('epilepsy_eeg.utils.compat.get_installed_version', return_value=None):
        meets_min, meets_rec = check_version('nonexistent_module', min_version='1.0.0', recommended_version='1.1.0')
        assert meets_min is False
        assert meets_rec is False
    
    # Test with a module that has unknown version
    with patch('epilepsy_eeg.utils.compat.get_installed_version', return_value='unknown'):
        meets_min, meets_rec = check_version('some_module', min_version='1.0.0', recommended_version='1.1.0')
        assert meets_min is True  # Assume compatible
        assert meets_rec is True  # Assume compatible
    
    # Test with default versions from MINIMUM_VERSIONS and RECOMMENDED_VERSIONS
    with patch('epilepsy_eeg.utils.compat.get_installed_version', return_value='1.0.0'):
        with patch.dict('epilepsy_eeg.utils.compat.MINIMUM_VERSIONS', {'numpy': '0.9.0'}):
            with patch.dict('epilepsy_eeg.utils.compat.RECOMMENDED_VERSIONS', {'numpy': '1.1.0'}):
                meets_min, meets_rec = check_version('numpy')
                assert meets_min is True
                assert meets_rec is False
    
    # Test with version comparison error
    with patch('epilepsy_eeg.utils.compat.get_installed_version', return_value='1.0.0'):
        with patch('packaging.version.parse', side_effect=Exception('Test error')):
            meets_min, meets_rec = check_version('numpy', min_version='1.0.0', recommended_version='1.1.0')
            assert meets_min is True  # Default to True on error
            assert meets_rec is True  # Default to True on error


def test_check_dependencies():
    # Test with all dependencies meeting minimum versions
    with patch('epilepsy_eeg.utils.compat.check_version', return_value=(True, False)):
        with patch('epilepsy_eeg.utils.compat.get_installed_version', return_value='1.0.0'):
            results = check_dependencies(['numpy', 'scipy'])
            assert len(results) == 2
            assert all(info['meets_minimum'] for info in results.values())
            assert not any(info['meets_recommended'] for info in results.values())
    
    # Test with some dependencies not meeting minimum versions
    with patch('epilepsy_eeg.utils.compat.check_version', side_effect=[(True, False), (False, False)]):
        with patch('epilepsy_eeg.utils.compat.get_installed_version', return_value='1.0.0'):
            results = check_dependencies(['numpy', 'scipy'])
            assert results['numpy']['meets_minimum'] is True
            assert results['scipy']['meets_minimum'] is False
    
    # Test with raise_error=True and all dependencies meeting minimum versions
    with patch('epilepsy_eeg.utils.compat.check_version', return_value=(True, False)):
        with patch('epilepsy_eeg.utils.compat.get_installed_version', return_value='1.0.0'):
            results = check_dependencies(['numpy', 'scipy'], raise_error=True)
            assert len(results) == 2
    
    # Test with raise_error=True and some dependencies not meeting minimum versions
    with patch('epilepsy_eeg.utils.compat.check_version', side_effect=[(True, False), (False, False)]):
        with patch('epilepsy_eeg.utils.compat.get_installed_version', return_value='1.0.0'):
            with pytest.raises(EEGError):
                check_dependencies(['numpy', 'scipy'], raise_error=True)
    
    # Test with default dependencies from MINIMUM_VERSIONS
    with patch('epilepsy_eeg.utils.compat.check_version', return_value=(True, False)):
        with patch('epilepsy_eeg.utils.compat.get_installed_version', return_value='1.0.0'):
            with patch.dict('epilepsy_eeg.utils.compat.MINIMUM_VERSIONS', {'numpy': '0.9.0', 'scipy': '0.9.0'}):
                results = check_dependencies()
                assert len(results) == 2
                assert 'numpy' in results
                assert 'scipy' in results


def test_get_mne_version():
    # Test with MNE installed
    with patch('epilepsy_eeg.utils.compat.get_installed_version', return_value='1.0.0'):
        version = get_mne_version()
        assert version == '1.0.0'
    
    # Test with MNE not installed
    with patch('epilepsy_eeg.utils.compat.get_installed_version', return_value=None):
        version = get_mne_version()
        assert version is None


def test_check_mne_version():
    # Test with MNE meeting minimum version
    with patch('epilepsy_eeg.utils.compat.check_version', return_value=(True, False)):
        result = check_mne_version('1.0.0')
        assert result is True
    
    # Test with MNE not meeting minimum version
    with patch('epilepsy_eeg.utils.compat.check_version', return_value=(False, False)):
        result = check_mne_version('1.0.0')
        assert result is False


def test_get_sklearn_version():
    # Test with scikit-learn installed
    with patch('epilepsy_eeg.utils.compat.get_installed_version', return_value='1.0.0'):
        version = get_sklearn_version()
        assert version == '1.0.0'
    
    # Test with scikit-learn not installed
    with patch('epilepsy_eeg.utils.compat.get_installed_version', return_value=None):
        version = get_sklearn_version()
        assert version is None


def test_check_sklearn_version():
    # Test with scikit-learn meeting minimum version
    with patch('epilepsy_eeg.utils.compat.check_version', return_value=(True, False)):
        result = check_sklearn_version('0.24.0')
        assert result is True
    
    # Test with scikit-learn not meeting minimum version
    with patch('epilepsy_eeg.utils.compat.check_version', return_value=(False, False)):
        result = check_sklearn_version('0.24.0')
        assert result is False


def test_print_dependency_status(capsys):
    # Test printing dependency status
    with patch('epilepsy_eeg.utils.compat.check_dependencies') as mock_check:
        mock_check.return_value = {
            'numpy': {
                'installed_version': '1.20.0',
                'minimum_version': '1.20.0',
                'recommended_version': '1.22.0',
                'meets_minimum': True,
                'meets_recommended': False
            },
            'scipy': {
                'installed_version': '1.8.0',
                'minimum_version': '1.6.0',
                'recommended_version': '1.8.0',
                'meets_minimum': True,
                'meets_recommended': True
            }
        }
        
        print_dependency_status()
        
        # Check that the output contains the expected information
        captured = capsys.readouterr()
        assert 'Dependency Status' in captured.out
        assert 'numpy' in captured.out
        assert 'scipy' in captured.out
        assert '1.20.0' in captured.out
        assert '1.8.0' in captured.out
        assert '✓ (minimum)' in captured.out
        assert '✓ (recommended)' in captured.out


def test_minimum_versions():
    # Test that MINIMUM_VERSIONS contains the expected packages
    assert 'numpy' in MINIMUM_VERSIONS
    assert 'scipy' in MINIMUM_VERSIONS
    assert 'matplotlib' in MINIMUM_VERSIONS
    assert 'mne' in MINIMUM_VERSIONS
    assert 'pandas' in MINIMUM_VERSIONS
    assert 'scikit-learn' in MINIMUM_VERSIONS
    assert 'networkx' in MINIMUM_VERSIONS


def test_recommended_versions():
    # Test that RECOMMENDED_VERSIONS contains the expected packages
    assert 'numpy' in RECOMMENDED_VERSIONS
    assert 'scipy' in RECOMMENDED_VERSIONS
    assert 'matplotlib' in RECOMMENDED_VERSIONS
    assert 'mne' in RECOMMENDED_VERSIONS
    assert 'pandas' in RECOMMENDED_VERSIONS
    assert 'scikit-learn' in RECOMMENDED_VERSIONS
    assert 'networkx' in RECOMMENDED_VERSIONS
    
    # Test that recommended versions are greater than or equal to minimum versions
    for package in MINIMUM_VERSIONS:
        if package in RECOMMENDED_VERSIONS:
            from packaging import version
            min_ver = version.parse(MINIMUM_VERSIONS[package])
            rec_ver = version.parse(RECOMMENDED_VERSIONS[package])
            assert rec_ver >= min_ver
