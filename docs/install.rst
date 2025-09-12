Installation Instructions
=========================

Let's see how to install the package. The first thing to do is ensure your Python environment
is set up correctly. To create a new environment:

Pre-Installation
^^^^^^^^^^^^^^^^

First, create a minimal environment with only Python installed in it. We recommend using `Python virtual environments <https://docs.python.org/3.10/tutorial/venv.html>`__.

.. code:: sh

    python3 -m venv /path/to/virtual/environment

Activate your new environment.

.. code:: sh

    source /path/to/virtual/environment/bin/activate

There are two primary ways to install this package -- from PyPI or source. The preferred method is to install from PyPI:

Install from PyPI
^^^^^^^^^^^^^^^^^

.. code:: sh

    pip install 'qiskit-addon-sqd'


Install from Source
^^^^^^^^^^^^^^^^^^^

Users who wish to develop in the repository or run the notebooks locally may want to install from source.

If so, the first step is to clone the ``qiskit-addon-sqd`` repository.

.. code:: sh

    git clone git@github.com:Qiskit/qiskit-addon-sqd.git

Next, install the Rust toolchain, upgrade pip, and enter the repository. Refer to the `Rust documentation <https://www.rust-lang.org/tools/install>`__
for instructions on installing the toolchain.

.. code:: sh
    
    ### <INSTALL RUST HERE> ###
    pip install --upgrade pip
    cd qiskit-addon-sqd

The next step is to install ``qiskit-addon-sqd`` to the virtual environment. If you plan on running the notebooks, install the
notebook dependencies in order to run all the visualizations in the notebooks. If you plan on developing in the repository, you
may want to install the ``dev`` dependencies.

Adjust the options below to suit your needs.

.. code:: sh

    pip install tox notebook -e '.[notebook-dependencies,dev]'

If you installed the notebook dependencies, you can get started by running the notebooks in the docs.

.. code::

    cd docs/
    jupyter lab
