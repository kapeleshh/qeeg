"""
Tests for the logging module.
"""

import os
import tempfile
import logging
import pytest
from unittest.mock import patch, MagicMock

from epilepsy_eeg.utils.logging import setup_logger, get_logger


def test_setup_logger():
    # Test creating a logger with default settings
    logger = setup_logger(name='test_logger')
    assert logger.name == 'test_logger'
    assert logger.level == logging.INFO
    assert len(logger.handlers) == 1
    assert isinstance(logger.handlers[0], logging.StreamHandler)
    
    # Test creating a logger with custom level
    logger = setup_logger(name='test_logger_debug', level=logging.DEBUG)
    assert logger.level == logging.DEBUG
    
    # Test creating a logger with no console output
    logger = setup_logger(name='test_logger_no_console', console=False)
    assert len(logger.handlers) == 0


def test_setup_logger_with_file():
    # Create a temporary file for logging
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        log_file = temp_file.name
    
    try:
        # Test creating a logger with file output
        logger = setup_logger(name='test_file_logger', log_file=log_file)
        assert len(logger.handlers) == 2
        assert isinstance(logger.handlers[1], logging.FileHandler)
        
        # Write a log message
        logger.info('Test log message')
        
        # Check that the message was written to the file
        with open(log_file, 'r') as f:
            log_content = f.read()
        assert 'Test log message' in log_content
    finally:
        # Clean up the temporary file
        if os.path.exists(log_file):
            os.remove(log_file)


def test_setup_logger_with_nonexistent_directory():
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a log file path in a subdirectory that doesn't exist yet
        log_dir = os.path.join(temp_dir, 'logs')
        log_file = os.path.join(log_dir, 'test.log')
        
        # Test creating a logger with a log file in a nonexistent directory
        logger = setup_logger(name='test_dir_logger', log_file=log_file)
        
        # Check that the directory was created
        assert os.path.exists(log_dir)
        
        # Write a log message
        logger.info('Test log message')
        
        # Check that the message was written to the file
        with open(log_file, 'r') as f:
            log_content = f.read()
        assert 'Test log message' in log_content


def test_get_logger():
    # Test getting a logger
    logger1 = get_logger('test_get_logger')
    assert logger1.name == 'test_get_logger'
    
    # Test getting the same logger again
    logger2 = get_logger('test_get_logger')
    assert logger1 is logger2  # Should be the same object


def test_logger_handlers_not_duplicated():
    # Test that handlers are not duplicated when setting up a logger multiple times
    logger1 = setup_logger(name='test_duplicate')
    n_handlers1 = len(logger1.handlers)
    
    # Set up the same logger again
    logger2 = setup_logger(name='test_duplicate')
    n_handlers2 = len(logger2.handlers)
    
    # Check that the number of handlers didn't change
    assert n_handlers1 == n_handlers2


def test_logger_levels():
    # Test different logging levels
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        log_file = temp_file.name
    
    try:
        # Create a logger with DEBUG level
        logger = setup_logger(name='test_levels', level=logging.DEBUG, log_file=log_file)
        
        # Write log messages at different levels
        logger.debug('Debug message')
        logger.info('Info message')
        logger.warning('Warning message')
        logger.error('Error message')
        logger.critical('Critical message')
        
        # Check that all messages were written to the file
        with open(log_file, 'r') as f:
            log_content = f.read()
        assert 'Debug message' in log_content
        assert 'Info message' in log_content
        assert 'Warning message' in log_content
        assert 'Error message' in log_content
        assert 'Critical message' in log_content
        
        # Create a new log file
        os.remove(log_file)
        
        # Create a logger with WARNING level
        logger = setup_logger(name='test_levels_warning', level=logging.WARNING, log_file=log_file)
        
        # Write log messages at different levels
        logger.debug('Debug message')
        logger.info('Info message')
        logger.warning('Warning message')
        logger.error('Error message')
        logger.critical('Critical message')
        
        # Check that only WARNING and above messages were written to the file
        with open(log_file, 'r') as f:
            log_content = f.read()
        assert 'Debug message' not in log_content
        assert 'Info message' not in log_content
        assert 'Warning message' in log_content
        assert 'Error message' in log_content
        assert 'Critical message' in log_content
    finally:
        # Clean up the temporary file
        if os.path.exists(log_file):
            os.remove(log_file)


def test_logger_formatting():
    # Test logger formatting
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        log_file = temp_file.name
    
    try:
        # Create a logger
        logger = setup_logger(name='test_format', log_file=log_file)
        
        # Write a log message
        logger.info('Test format message')
        
        # Check the format of the log message
        with open(log_file, 'r') as f:
            log_content = f.read()
        
        # Check that the log message contains the expected parts
        assert 'test_format' in log_content  # Logger name
        assert 'INFO' in log_content  # Log level
        assert 'Test format message' in log_content  # Message
    finally:
        # Clean up the temporary file
        if os.path.exists(log_file):
            os.remove(log_file)
