<!-- SHIELDS -->
<div align="left">

  [![Release](https://img.shields.io/pypi/v/qiskit-addon-sqd.svg?label=Release)](https://github.com/Qiskit/qiskit-addon-sqd/releases)
  ![Platform](https://img.shields.io/badge/%F0%9F%92%BB_Platform-Linux%20%7C%20macOS-blue)
  [![Python](https://img.shields.io/pypi/pyversions/qiskit-addon-sqd?label=Python&logo=python)](https://www.python.org/)
  [![Qiskit](https://img.shields.io/badge/Qiskit%20-%20%3E%3D1.2%20-%20%236133BD?logo=Qiskit)](https://github.com/Qiskit/qiskit)
<br />
  [![Docs (stable)](https://img.shields.io/badge/%F0%9F%93%84%20Docs-stable-blue.svg)](https://qiskit.github.io/qiskit-addon-sqd/)
  <!-- [![DOI](https://zenodo.org/badge/DOI/TODO](https://zenodo.org/doi/TODO -->
  [![License](https://img.shields.io/github/license/Qiskit/qiskit-addon-sqd?label=License)](LICENSE.txt)
  [![Downloads](https://img.shields.io/pypi/dm/qiskit-addon-sqd.svg?label=Downloads)](https://pypi.org/project/qiskit-addon-sqd/)
  [![Tests](https://github.com/Qiskit/qiskit-addon-sqd/actions/workflows/test_latest_versions.yml/badge.svg)](https://github.com/Qiskit/qiskit-addon-sqd/actions/workflows/test_latest_versions.yml)

# Qiskit addon: sample-based quantum diagonalization (SQD)

### Table of contents

* [About](#about)
* [Documentation](#documentation)
* [Installation](#installation)
* [Computational requirements](#computational-requirements)
* [Deprecation Policy](#deprecation-policy)
* [Contributing](#contributing)
* [License](#license)
* [References](#references)

----------------------------------------------------------------------------------------------------

### About

[Qiskit addons](https://quantum.cloud.ibm.com/docs/guides/addons) are a collection of modular tools for building utility-scale workloads powered by Qiskit.

This package contains the Qiskit addon for sample-based quantum diagonalization (SQD) -- a technique for finding eigenvalues and eigenvectors of quantum operators, such as a quantum system Hamiltonian, using quantum and distributed classical computing together [1-5]. This technique can be run on current quantum computers and has been shown to scale to problem sizes beyond what was possible with variational methods and even beyond the reach of exact classical diagonalization methods [1,2].

SQD-based workflows involve first preparing one or more quantum states on a quantum device and sampling from them. Then, classical distributed computing is used to process those noisy samples. This processing occurs iteratively in two steps: First, a configuration recovery step corrects noisy samples using information about the input problem and second, the Hamiltonian is projected and diagonalized in the subspace spanned by those samples. These steps are repeated self-consistently until convergence. The result is an approximated lowest eigenvalue (energy) and lowest energy eigenstate of a given Hamiltonian. SQD is robust to samples corrupted by quantum noise; in fact, as long as a useful signal can be retrieved out of the quantum computer, the outcome of SQD will be insensitive to noisy bitstrings.

SQD can be used in various ways in practice. For example, we can use two categories of quantum circuits to sample from:
  
  1. A variational circuit ansatz with parameters chosen such that sampling the circuit produces electronic configurations on which the target wavefunction (i.e., the ground state) has significant support. This is appealing for chemistry applications where Hamiltonians can have millions of interaction terms [1]. For an example of this approach applied to chemistry see the [tutorial for approximating the ground state energy of the N2 molecule](https://qiskit.github.io/qiskit-addon-sqd/tutorials/01_chemistry_hamiltonian.html).

  2. A set of Krylov basis states are prepared over increasing time intervals. Assuming a good initial state and sparsity of the ground state, this approach is proven to converge efficiently. As one needs to prepare Trotterized time evolution circuits on a quantum device, this approach is best for applications to lattice models [2]. For an example of this approach applied to Fermionic lattice Hamiltonians, see the [tutorial for approximating the ground state energy of a simplified single-impurity Anderson model](https://qiskit.github.io/qiskit-addon-sqd/tutorials/02_fermionic_lattice_hamiltonian.html).

  This package contains the functionality for the classical processing of user-provided samples. It can target Hamiltonians expressed as linear combinations of Pauli operators or second-quantized Fermionic operators. The projection and diagonalization steps are performed by a classical solver. We provide here two generic solvers, one for Fermionic systems and another for qubit systems. Other solvers that might be more efficient for specific systems can be interfaced by the users.

----------------------------------------------------------------------------------------------------

### Documentation

All documentation is available at https://qiskit.github.io/qiskit-addon-sqd/.

----------------------------------------------------------------------------------------------------

### Installation

We encourage installing this package via `pip`, when possible:

```bash
pip install 'qiskit-addon-sqd'
```

For more installation information refer to these [installation instructions](docs/install.rst).

----------------------------------------------------------------------------------------------------

### Computational requirements

The computational cost of SQD is dominated by the eigenstate solver calls. At each step of the self-consistent configuration recovery iteration, `n_batches` of eigenstate solver calls are performed. The different calls are embarrassingly parallel. In this [tutorial](docs/tutorials/01_chemistry_hamiltonian.ipynb), those calls are inside a `for` loop. **It is highly recommended to perform these calls in parallel**.

The [`qiskit_addon_sqd.fermion.solve_fermion()`](qiskit_addon_sqd/fermion.py) function is multithreaded and capable of handling systems with ~25 spacial orbitals and ~10 electrons with subspace dimensions of ~$10^7$, using ~10-30 cores.

##### Choosing subspace dimensions

The choice of the subspace dimension affects the accuracy and runtime of the eigenstate solver. The larger the subspace the more accurate the calculation, at the cost of increasing the runtime and memory requirements. The optimal subspace size for a given system is not known, thus a convergence study with the subspace dimension may be performed as described in this [example](docs/how_tos/choose_subspace_dimension.ipynb).

##### The subspace dimension is set indirectly

In this package, the user controls the number of bitstrings (see the `samples_per_batch` argument in [`qiskit_addon_sqd.subsampling.postselect_and_subsample()`](qiskit_addon_sqd/subsampling.py)) contained in each subspace. The value of this argument sets an upper bound to the subspace dimension in the case of quantum chemistry applications. See this [example](docs/how_tos/select_open_closed_shell.ipynb) for more details.

----------------------------------------------------------------------------------------------------

### Deprecation Policy

We follow [semantic versioning](https://semver.org/) and are guided by the principles in
[Qiskit's deprecation policy](https://github.com/Qiskit/qiskit/blob/main/DEPRECATION.md).
We may occasionally make breaking changes in order to improve the user experience.
When possible, we will keep old interfaces and mark them as deprecated, as long as they can co-exist with the
new ones.
Each substantial improvement, breaking change, or deprecation will be documented in the
[release notes](https://qiskit.github.io/qiskit-addon-sqd/release-notes.html).

----------------------------------------------------------------------------------------------------

### Contributing

The source code is available [on GitHub](https://github.com/Qiskit/qiskit-addon-sqd).

The developer guide is located at [CONTRIBUTING.md](https://github.com/Qiskit/qiskit-addon-sqd/blob/main/CONTRIBUTING.md)
in the root of this project's repository.
By participating, you are expected to uphold Qiskit's [code of conduct](https://github.com/Qiskit/qiskit/blob/main/CODE_OF_CONDUCT.md).

We use [GitHub issues](https://github.com/Qiskit/qiskit-addon-sqd/issues/new/choose) for tracking requests and bugs.

----------------------------------------------------------------------------------------------------

### License

[Apache License 2.0](LICENSE.txt)

----------------------------------------------------------------------------------------------------

### References

[1] Javier Robledo-Moreno, et al., [Chemistry Beyond Exact Solutions on a Quantum-Centric Supercomputer](https://arxiv.org/abs/2405.05068), arXiv:2405.05068 [quant-ph].

[2] Jeffery Yu, et al., [Quantum-Centric Algorithm for Sample-Based Krylov Diagonalization](https://arxiv.org/abs/2501.09702), arXiv:2501.09702 [quant-ph].

[3] Keita Kanno, et al., [Quantum-Selected Configuration Interaction: classical diagonalization of Hamiltonians in subspaces selected by quantum computers](https://arxiv.org/abs/2302.11320), arXiv:2302.11320 [quant-ph].

[4] Kenji Sugisaki, et al., [Hamiltonian simulation-based quantum-selected configuration interaction for large-scale electronic structure calculations with a quantum computer](https://arxiv.org/abs/2412.07218), arXiv:2412.07218 [quant-ph].

[5] Mathias Mikkelsen, Yuya O. Nakagawa, [Quantum-selected configuration interaction with time-evolved state](https://arxiv.org/abs/2412.13839), arXiv:2412.13839 [quant-ph].
