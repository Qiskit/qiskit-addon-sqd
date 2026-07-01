##########################################
Sample-based quantum diagonalization (SQD)
##########################################

This package implements sample-based quantum diagonalization (SQD) --- a technique for finding eigenvalues and eigenvectors of quantum operators, such as a quantum system Hamiltonian `[1–6] <references_>`_. It can target Hamiltonians expressed as linear combinations of Pauli operators or second-quantized fermionic operators. SQD-based workflows can be run on current quantum computers and have been shown to scale to problem sizes beyond what was possible with variational methods --- and even beyond the reach of exact classical diagonalization methods.

SQD-based workflows involve first preparing one or more quantum states on a quantum device and sampling from them. Then, classical distributed computing is used to process those noisy samples. This processing occurs iteratively in two steps: first, a configuration recovery step corrects noisy samples using information about the input problem; second, the Hamiltonian is projected and diagonalized in the subspace spanned by those samples. These steps are repeated self-consistently until convergence. The result is an approximated lowest eigenvalue (energy) and lowest energy eigenstate of a given Hamiltonian. SQD is robust to samples corrupted by quantum noise; in fact, as long as a useful signal can be retrieved out of the quantum computer, the outcome of SQD will be insensitive to noisy bitstrings.

``qiskit-addon-sqd`` can be used to classically process samples drawn from a variety of quantum circuits in practice. For example:

    1. A variational circuit ansatz with parameters chosen such that sampling the circuit produces electronic configurations on which the target wavefunction (for example, the ground state) has significant support. This is appealing for chemistry applications where Hamiltonians can have millions of interaction terms `[1] <ref1_>`_. For an example of this approach applied to chemistry using an LUCJ circuit, see the `tutorial for approximating the ground state energy of the N2 molecule <https://quantum.cloud.ibm.com/docs/tutorials/sample-based-quantum-diagonalization>`_.

    2. A set of Krylov basis states are prepared over increasing time intervals. Assuming a good initial state and sparsity of the ground state, this approach is proven to converge efficiently. As one needs to prepare Trotterized time evolution circuits on a quantum device, this approach is best for applications to lattice models `[2] <ref2_>`_ instead of complex many-body Hamiltonians like those for quantum chemistry. For an example of this approach applied to fermionic lattice Hamiltonians, see the `tutorial for approximating the ground state energy of a simplified single-impurity Anderson model <https://quantum.cloud.ibm.com/docs/tutorials/sample-based-krylov-quantum-diagonalization>`_.

    3. A set of Krylov basis states implemented with randomized compilation of the time evolution operator. This approach yields shorter-depth circuits compared to Trotter-based decompositions of the time evolution and so can be used for quantum chemistry Hamiltonians. This technique has been applied to the ground state energy of polycyclic aromatic hydrocarbons `[6] <ref6_>`_.

Getting started
---------------

A simple guide to help you get started quickly with this package is available in the :doc:`quickstart guide <guides/quickstart>`.

Use case examples
-----------------

The sample-based quantum diagonalization technique can be used to implement a diverse set of workflows. Some examples of where it has been used include:

- Electronic structure calculations of the ground state energies of iron-sulfur clusters `[ref] <https://arxiv.org/abs/2405.05068>`__
- Estimating ground state energies to predict band gaps of dielectrics `[ref] <https://arxiv.org/abs/2503.10901>`__
- Estimating low-lying molecular excited states `[ref] <https://arxiv.org/abs/2411.00468>`__
- Simulating interactions between molecules `[ref] <https://arxiv.org/abs/2410.09209>`__
- Studying solute-solvent interactions in simulations of electronic structure `[ref] <https://arxiv.org/abs/2502.10189>`__
- Open-shell analysis of molecular dissociation `[ref] <https://arxiv.org/abs/2411.04827>`__
- Modeling reaction pathways for photochemistry problems `[ref] <https://arxiv.org/abs/2510.00484>`__
- Combining with entanglement forging to study reaction pathways for materials degradation via hydrogen abstraction `[ref] <https://arxiv.org/abs/2508.08229>`__
- Constructing subspaces from quantum Krylov basis states to perform ground state simulations of impurity models for fermionic systems `[ref] <https://arxiv.org/abs/2501.09702>`__
- Using qDRIFT randomized compilation to lower the overhead of implementing Krylov basis states for chemistry Hamiltonians [ref] and with application to a molecule exhibiting a half-Mobius electronic topology `[ref] <https://arxiv.org/abs/2603.08696>`__
- Using SQD as a solver in the context of embedding and fragmentation methods for molecular systems, such as to compute energies of hydrogen rings and cyclohexane `[ref] <https://arxiv.org/abs/2411.09861>`__ and of bi- and tri-metallic complexes `[ref] <https://arxiv.org/abs/2512.14936>`__, study the oxygen reduction reaction `[ref] <https://arxiv.org/abs/2503.10923>`__, estimate energies of proteins `[ref] <https://arxiv.org/abs/2512.17130>`__ and protein–ligand complexes `[ref] <https://arxiv.org/abs/2605.01138>`__, and predict tritium speciation `[ref] <https://arxiv.org/abs/2606.30402>`__.

Technical discussion
--------------------

System sizes and computational requirements
"""""""""""""""""""""""""""""""""""""""""""

The computational cost of SQD is dominated by the eigenstate solver calls. At each step of the self-consistent configuration recovery iteration, `n_batches` of eigenstate solver calls are performed. The different calls are embarrassingly parallel. In this `tutorial <https://quantum.cloud.ibm.com/docs/tutorials/sample-based-quantum-diagonalization>`_, those calls are inside a `for` loop. **It is highly recommended to perform these calls in parallel**.

The :func:`qiskit_addon_sqd.fermion.solve_fermion` function is multithreaded and capable of handling systems with ~25 spacial orbitals and ~10 electrons with subspace dimensions of ~$10^7$, using ~10-30 cores.

Choosing subspace dimensions
""""""""""""""""""""""""""""

The choice of the subspace dimension affects the accuracy and runtime of the eigenstate solver. The larger the subspace, the more accurate the calculation, at the cost of increasing the runtime and memory requirements. The optimal subspace size of a given system is not known; thus, a convergence study with the subspace dimension can be performed, as described in this `guide <guides/choose_subspace_dimension.ipynb>`_.

The subspace dimension is set indirectly
""""""""""""""""""""""""""""""""""""""""

In this package, the user controls the number of bitstrings contained in each subspace with the `samples_per_batch` argument in :func:`.qiskit_addon_sqd.subsampling.postselect_and_subsample`. The value of this argument sets an upper bound to the subspace dimension in the case of quantum chemistry applications. See this `example <guides/select_open_closed_shell.ipynb>`_ for more details.

Solvers
"""""""

This package contains the functionality for the classical processing of user-provided samples. It can target Hamiltonians expressed as linear combinations of Pauli operators or second-quantized fermionic operators. The projection and diagonalization steps are performed by a classical solver. We provide here two generic solvers, one for fermionic systems and another for qubit systems. Other solvers that might be more efficient for specific systems can be interfaced by the users.

Contributing
------------

The source code is available `on GitHub <https://github.com/Qiskit/qiskit-addon-sqd>`_.

The developer guide is located at `CONTRIBUTING.md <https://github.com/Qiskit/qiskit-addon-sqd/blob/main/CONTRIBUTING.md>`_
in the root of this project's repository.
By participating, you are expected to uphold Qiskit's `code of conduct <https://github.com/Qiskit/qiskit/blob/main/CODE_OF_CONDUCT.md>`_.

We use `GitHub issues <https://github.com/Qiskit/qiskit-addon-sqd/issues/new/choose>`_ for tracking requests and bugs.

Citing this package
-------------------

If you use this package in your research, use the `CITATION.bib <https://github.com/Qiskit/qiskit-addon-sqd/blob/main/CITATION.bib>`_ file in this project's repository to cite the appropriate reference(s).

License
-------

`Apache License 2.0 <https://github.com/Qiskit/qiskit-addon-sqd/blob/main/LICENSE.txt>`_

Deprecation Policy
------------------

We follow `semantic versioning <https://semver.org/>`_. We may occasionally make breaking changes in order to
improve the user experience. When possible, we will keep old interfaces and mark them as deprecated, as long
as they can co-exist with the new ones. Each substantial improvement, breaking change, or deprecation will be
documented in the `release notes <https://quantum.cloud.ibm.com/docs/api/qiskit-addon-sqd/release-notes>`_.

.. _references:

References
----------

.. _ref1:

1. Javier Robledo-Moreno, et al., `Chemistry Beyond Exact Solutions on a Quantum-Centric Supercomputer <https://arxiv.org/abs/2405.05068>`_, arXiv:2405.05068 [quant-ph].

.. _ref2:

2. Jeffery Yu, et al., `Quantum-Centric Algorithm for Sample-Based Krylov Diagonalization <https://arxiv.org/abs/2501.09702>`_, arXiv:2501.09702 [quant-ph].

.. _ref3:

3. Keita Kanno, et al., `Quantum-Selected Configuration Interaction: classical diagonalization of Hamiltonians in subspaces selected by quantum computers <https://arxiv.org/abs/2302.11320>`_, arXiv:2302.11320 [quant-ph].

.. _ref4:

4. Kenji Sugisaki, et al., `Hamiltonian simulation-based quantum-selected configuration interaction for large-scale electronic structure calculations with a quantum computer <https://arxiv.org/abs/2412.07218>`_, arXiv:2412.07218 [quant-ph].

.. _ref5:

5. Mathias Mikkelsen, Yuya O. Nakagawa, `Quantum-selected configuration interaction with time-evolved state <https://arxiv.org/abs/2412.13839>`_, arXiv:2412.13839 [quant-ph].

.. _ref6:

6. Samuele Piccinelli, et al., `Quantum chemistry with provable convergence via randomized sample-based Krylov quantum diagonalization <https://arxiv.org/abs/2508.02578>`_, arXiv:2508.02578 [quant-ph].

.. toctree::
   :hidden:

   Documentation home <self>
   Installation instructions <install>
   Guides <guides/overview>
   GitHub <https://github.com/Qiskit/qiskit-addon-sqd>

.. toctree::
   :hidden:
   :caption: Tutorials

   Sample-based quantum diagonalization on a chemistry Hamiltonian <https://quantum.cloud.ibm.com/docs/tutorials/sample-based-quantum-diagonalization>
   Sample-based Krylov quantum diagonalization of a Fermionic lattice model <https://quantum.cloud.ibm.com/docs/tutorials/sample-based-krylov-quantum-diagonalization>

.. toctree::
   :hidden:
   :caption: Learning

   Quantum diagonalization algorithms <https://quantum.cloud.ibm.com/learning/courses/quantum-diagonalization-algorithms>

.. toctree::
   :hidden:
   :caption: API reference

   Python API reference <https://quantum.cloud.ibm.com/docs/api/qiskit-addon-sqd>
   Release notes <release-notes>
