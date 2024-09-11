########################################################
Qiskit addon: sample-based quantum diagonalization (SQD)
########################################################

Qiskit addons are a collection of modular tools for building utility-scale workloads powered by Qiskit.

This package contains the Qiskit addon for sample-based quantum diagonalization -- a technique for finding eigenvalues and eigenvectors of quantum operators, such as a quantum system Hamiltonian, using quantum and distributed classical computing together.

Classical distributed computing is used to process samples obtained from a quantum processor, and to project and diagonalize a target Hamiltonian in a subspace spanned by them. This allows SQD to be robust to samples corrupted by quantum noise and deal with large Hamiltonians, such as chemistry Hamiltonians with millions of interaction terms, beyond the reach of any exact diagonalization methods.  

The SQD tool can target Hamiltonians expressed as linear combination of Pauli operators, or second-quantized fermionic operators. The input samples are obtained by quantum circuits defined by the user, which are believed to be good representations of eigenstates (e.g. the ground state) of a target operator. The convergence rate of SQD as a function of the number of samples improves with the sparseness of the target eigenstate. 

The projection and diagonalization steps are performed by a classical solver. We provide here two generic solvers, one for fermionic systems and another for qubit systems. Other solvers that might be more efficient for specific systems can be interfaced by the users.

Documentation
-------------

All documentation is available `here <https://qiskit.github.io/qiskit-addon-sqd/>`_.

Installation
------------

We encourage installing this package via ``pip``, when possible:

.. code-block:: bash

   pip install 'qiskit-addon-sqd'


For more installation information refer to the `installation instructions <install.rst>`_ in the documentation.

System sizes and computational requirements
-------------------------------------------

The computational cost of SQD is dominated by the eigenstate solver calls. At each step of the self-consistent configuration recovery iteration, `n_batches` of eigenstate solver calls are performed. The different calls are embarrassingly parallel. In this `tutorial <tutorials/01_chemistry_hamiltonian.ipynb>`_, those calls are inside a `for` loop. **It is highly recommended to perform these calls in parallel**.

The :func:`qiskit_addon_sqd.fermion.solve_fermion` function is multithreaded and capable of handling systems with ~25 spacial orbitals and ~10 electrons with subspace dimensions of ~$10^7$, using ~10-30 cores.

Choosing subspace dimensions
----------------------------

The choice of the subspace dimension affects the accuracy and runtime of the eigenstate solver. The larger the subspace the more accurate the calculation, at the cost of increasing the runtime and memory requirements. The optimal subspace size of a given system is not known, thus a convergence study with the subspace dimension may be performed, as described in this `guide <how_tos/choose_subspace_dimension.ipynb>`_.

The subspace dimension is set indirectly
----------------------------------------

In this package, the user controls the number of bitstrings contained in each subspace with the `samples_per_batch` argument in :func:`.qiskit_addon_sqd.subsampling.postselect_and_subsample`. The value of this argument sets an upper bound to the subspace dimension in the case of quantum chemistry applications. See this `example <how_tos/select_open_closed_shell.ipynb>`_ for more details.

Deprecation Policy
------------------

We follow `semantic versioning <https://semver.org/>`_ and are guided by the principles in
`Qiskit's deprecation policy <https://github.com/Qiskit/qiskit/blob/main/DEPRECATION.md>`_.
We may occasionally make breaking changes in order to improve the user experience.
When possible, we will keep old interfaces and mark them as deprecated, as long as they can co-exist with the
new ones.
Each substantial improvement, breaking change, or deprecation will be documented in the
release notes.

Contributing
------------

The source code is available `on GitHub <https://github.com/Qiskit/qiskit-addon-sqd>`_.

The developer guide is located at `CONTRIBUTING.md <https://github.com/Qiskit/qiskit-addon-sqd/blob/main/CONTRIBUTING.md>`_
in the root of this project's repository.
By participating, you are expected to uphold Qiskit's `code of conduct <https://github.com/Qiskit/qiskit/blob/main/CODE_OF_CONDUCT.md>`_.

We use `GitHub issues <https://github.com/Qiskit/qiskit-addon-sqd/issues/new/choose>`_ for tracking requests and bugs.

License
-------

`Apache License 2.0 <https://github.com/Qiskit/qiskit-addon-sqd/blob/main/LICENSE.txt>`_

.. _references:

References
----------

[1] Javier Robledo-Moreno, et al., `Chemistry Beyond Exact Solutions on a Quantum-Centric Supercomputer <https://arxiv.org/abs/2405.05068>`_, arXiv:2405.05068 [quant-ph].

[2] Keita Kanno, et al., `Quantum-Selected Configuration Interaction: classical diagonalization of Hamiltonians in subspaces selected by quantum computers <https://arxiv.org/abs/2302.11320>`_, arXiv:2302.11320 [quant-ph].

.. toctree::
  :hidden:
   
   Documentation Home <self>
   Installation Instructions <install>
   Tutorials <tutorials/index>
   How-To Guides <how_tos/index>
   API Reference <apidocs/qiskit_addon_sqd>
   GitHub <https://github.com/qiskit/qiskit-addon-sqd>
   Release Notes <release-notes>
