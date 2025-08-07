# Contributing to Epilepsy-EEG

Thank you for considering contributing to Epilepsy-EEG! This document provides guidelines and instructions for contributing to this project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Environment](#development-environment)
- [Branching Strategy](#branching-strategy)
- [Commit Guidelines](#commit-guidelines)
- [Pull Request Process](#pull-request-process)
- [Testing](#testing)
- [Documentation](#documentation)
- [Style Guide](#style-guide)

## Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/your-username/epilepsy-eeg.git
   cd epilepsy-eeg
   ```
3. Set up the development environment (see below)
4. Create a branch for your changes
5. Make your changes
6. Push your branch to your fork
7. Submit a pull request

## Development Environment

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install the package in development mode:
   ```bash
   pip install -e ".[dev]"
   ```

3. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Branching Strategy

- `main`: The main branch contains the latest stable release
- `develop`: The development branch contains the latest development changes
- Feature branches: Create a new branch for each feature or bugfix

Name your branches according to the following convention:
- `feature/your-feature-name`: For new features
- `bugfix/issue-number-description`: For bug fixes
- `docs/what-you-are-documenting`: For documentation changes

## Commit Guidelines

- Use the present tense ("Add feature" not "Added feature")
- Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit the first line to 72 characters or less
- Reference issues and pull requests liberally after the first line
- Consider starting the commit message with an applicable emoji:
  - ✨ `:sparkles:` when adding a new feature
  - 🐛 `:bug:` when fixing a bug
  - 📚 `:books:` when adding or updating documentation
  - ♻️ `:recycle:` when refactoring code
  - 🧪 `:test_tube:` when adding tests
  - 🔧 `:wrench:` when updating configuration files

## Pull Request Process

1. Update the README.md or documentation with details of changes if appropriate
2. Add or update tests for any new or modified functionality
3. Ensure all tests pass
4. Update the version number in relevant files following [Semantic Versioning](https://semver.org/)
5. You may merge the Pull Request once you have the sign-off of at least one other developer

## Testing

- Write tests for all new features and bug fixes
- Run tests before submitting a pull request:
  ```bash
  pytest
  ```
- Aim for high test coverage:
  ```bash
  pytest --cov=epilepsy_eeg
  ```

## Documentation

- Update documentation for any new or modified functionality
- Follow the [NumPy docstring format](https://numpydoc.readthedocs.io/en/latest/format.html)
- Build and check the documentation:
  ```bash
  cd docs
  make html
  ```

## Style Guide

This project follows the [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide. Additionally:

- Use 4 spaces for indentation
- Use docstrings for all public modules, functions, classes, and methods
- Keep line length to 88 characters (compatible with Black)
- Run linting tools before submitting:
  ```bash
  flake8 epilepsy_eeg
  black epilepsy_eeg
  ```

Thank you for contributing to Epilepsy-EEG!
