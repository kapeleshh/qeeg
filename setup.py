from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="epilepsy_eeg",
    version="0.1.0",
    author="EEG Analysis Team",
    author_email="example@example.com",
    description="A package for EEG analysis focused on epilepsy detection and neurological condition assessment",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kapeleshh/epilepsy-eeg",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.6.0",
        "matplotlib>=3.3.0",
        "mne>=1.0.0",
        "pandas>=1.2.0",
        "scikit-learn>=0.24.0",
        "networkx>=2.5.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.12.0",
            "flake8>=3.9.0",
            "black>=21.5b2",
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=0.5.0",
        ],
    },
)
