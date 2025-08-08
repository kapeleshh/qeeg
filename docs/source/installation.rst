Installation
===========

Requirements
-----------

QEEG requires the following dependencies:

* Python 3.8 or higher
* NumPy (>= 1.20.0)
* SciPy (>= 1.7.0)
* Matplotlib (>= 3.4.0)
* MNE (>= 1.0.0)
* Nibabel (>= 3.2.0)
* Nilearn (>= 0.8.0)
* PyWavelets (>= 1.3.0)
* psutil (>= 5.9.0)

Installing from PyPI
-------------------

The recommended way to install QEEG is from PyPI using pip:

.. code-block:: bash

    pip install qeeg

Installing from Source
---------------------

You can also install QEEG from source:

.. code-block:: bash

    git clone https://github.com/kapeleshh/qeeg.git
    cd qeeg
    pip install -e .

Development Installation
-----------------------

For development, you should install the development dependencies:

.. code-block:: bash

    git clone https://github.com/kapeleshh/qeeg.git
    cd qeeg
    pip install -e ".[dev]"
    # or
    pip install -e . -r requirements-dev.txt
