"""
Logging module for the epilepsy_eeg package.

This module provides functions for setting up and using loggers
throughout the epilepsy_eeg package.
"""

import logging
import os
import sys
from typing import Optional


def setup_logger(
    name: str = 'epilepsy_eeg',
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    console: bool = True
) -> logging.Logger:
    """
    Set up and return a logger with the given name and level.

    Parameters
    ----------
    name : str, optional
        Name of the logger, by default 'epilepsy_eeg'
    level : int, optional
        Logging level, by default logging.INFO
    log_file : str or None, optional
        Path to log file, by default None (no file logging)
    console : bool, optional
        Whether to log to console, by default True

    Returns
    -------
    logging.Logger
        Configured logger instance.

    Examples
    --------
    >>> from epilepsy_eeg.utils.logging import setup_logger
    >>> logger = setup_logger(name='my_module', level=logging.DEBUG)
    >>> logger.debug('Debug message')
    >>> logger.info('Info message')
    >>> logger.warning('Warning message')
    >>> logger.error('Error message')
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    if logger.handlers:
        logger.handlers = []
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Add console handler if requested
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Add file handler if log file is specified
    if log_file:
        # Create directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


# Create default logger
logger = setup_logger()


def get_logger(name: str = 'epilepsy_eeg') -> logging.Logger:
    """
    Get a logger with the given name.

    Parameters
    ----------
    name : str, optional
        Name of the logger, by default 'epilepsy_eeg'

    Returns
    -------
    logging.Logger
        Logger instance.

    Examples
    --------
    >>> from epilepsy_eeg.utils.logging import get_logger
    >>> logger = get_logger('my_module')
    >>> logger.info('Info message')
    """
    return logging.getLogger(name)
