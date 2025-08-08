from setuptools import setup, find_packages

setup(
    name="qeeg",
    version="0.1.0",
    description="Quantitative EEG analysis toolkit for neurological conditions",
    author="Kapeleshh KS",
    author_email="kapeleshh@gmail.com",
    url="https://github.com/kapeleshh/qeeg",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "mne>=1.0.0",
        "nibabel>=3.2.0",
        "nilearn>=0.8.0",
        "pywavelets>=1.3.0",
        "psutil>=5.9.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "qeeg=qeeg.cli.commands:main",
        ],
    },
)
