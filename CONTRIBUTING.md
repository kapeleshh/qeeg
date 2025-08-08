# Contributing to QEEG

Thank you for considering contributing to QEEG! This document provides guidelines and instructions for contributing to this project.

## Code of Conduct

Please be respectful and considerate of others when contributing to this project. We aim to foster an inclusive and welcoming community.

## Development Process

1. **Fork the Repository**
   - Fork the repository on GitHub
   - Clone your fork locally: `git clone https://github.com/yourusername/qeeg.git`
   - Add the original repository as upstream: `git remote add upstream https://github.com/originalusername/qeeg.git`

2. **Create a Feature Branch**
   - Create a branch for your feature: `git checkout -b feature-name`
   - Make your changes in this branch

3. **Development Environment**
   - Set up a virtual environment: `python -m venv venv`
   - Activate it: `source venv/bin/activate` (or `venv\Scripts\activate` on Windows)
   - Install development dependencies: `pip install -e ".[dev]"`

4. **Make Your Changes**
   - Write code that follows our style guidelines
   - Add or update tests as necessary
   - Update documentation for any API changes

5. **Run Tests**
   - Ensure all tests pass: `pytest`
   - Check code style: `flake8 qeeg tests`
   - Format code: `black qeeg tests`

6. **Submit a Pull Request**
   - Push your changes to your fork: `git push origin feature-name`
   - Submit a pull request from your branch to the main repository
   - Describe your changes in detail in the pull request

## Code Style

We follow PEP 8 with a line length of 88 characters. Please use Black for formatting:

```bash
pip install black
black qeeg tests
```

For imports, we use isort:

```bash
pip install isort
isort qeeg tests
```

## Testing

All new features should include tests. Run the test suite with:

```bash
pytest
```

For coverage reports:

```bash
pytest --cov=qeeg tests/
```

## Documentation

Please update documentation for any changes to the API. We use NumPy-style docstrings:

```python
def function_name(param1, param2):
    """
    Short description of the function.

    Parameters
    ----------
    param1 : type
        Description of param1
    param2 : type
        Description of param2

    Returns
    -------
    type
        Description of return value
    """
    # Function implementation
```

## Versioning

We use semantic versioning (SemVer) for this project:

- MAJOR version for incompatible API changes
- MINOR version for backwards-compatible functionality additions
- PATCH version for backwards-compatible bug fixes

During the 0.x phase, minor version increases may include breaking changes.

## Issue Reporting

When reporting issues, please include:

- A clear description of the issue
- Steps to reproduce
- Expected behavior
- Actual behavior
- Version information (Python version, package version, OS)
- Any relevant logs or error messages

## Feature Requests

For feature requests, please describe:

- The problem you're trying to solve
- Your proposed solution
- Any alternatives you've considered
- Any relevant context or examples

Thank you for contributing to QEEG!
