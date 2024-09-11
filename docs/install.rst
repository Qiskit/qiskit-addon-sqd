Installation Instructions
=========================

Let's see how to install the package. The first
thing to do is choose how you're going to run and install the
package. There are two primary ways to do this:

- :ref:`Option 1`
- :ref:`Option 2`

Pre-Installation
^^^^^^^^^^^^^^^^

First, create a minimal environment with only Python installed in it. We recommend using `Python virtual environments <https://docs.python.org/3.10/tutorial/venv.html>`__.

.. code:: sh

    python3 -m venv /path/to/virtual/environment

Activate your new environment.

.. code:: sh

    source /path/to/virtual/environment/bin/activate


.. _Option 1:

Option 1: Install from PyPI
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The most straightforward way to install the ``qiskit-addon-sqd`` package is via ``PyPI``.

.. code:: sh

    pip install 'qiskit-addon-sqd'


.. _Option 2:

Option 2: Install from Source
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Users who wish to develop in the repository or run the notebooks locally may want to install from source.

If so, the first step is to clone the ``qiskit-addon-sqd`` repository.

.. code:: sh

    git clone git@github.com:Qiskit/qiskit-addon-sqd.git

Next, upgrade pip and enter the repository.

.. code:: sh

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
