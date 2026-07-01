<!-- SHIELDS -->
<div align="left">

  [![Release](https://img.shields.io/pypi/v/qiskit-addon-sqd.svg?label=Release)](https://github.com/Qiskit/qiskit-addon-sqd/releases)
  ![Platform](https://img.shields.io/badge/%F0%9F%92%BB_Platform-Linux%20%7C%20macOS-blue)
  [![Python](https://img.shields.io/pypi/pyversions/qiskit-addon-sqd?label=Python&logo=python)](https://www.python.org/)
  [![Qiskit](https://img.shields.io/badge/Qiskit%20-%20%3E%3D1.2%20-%20%236133BD?logo=Qiskit)](https://github.com/Qiskit/qiskit)
<br />
  <!-- [![DOI](https://zenodo.org/badge/DOI/TODO](https://zenodo.org/doi/TODO -->
  [![License](https://img.shields.io/github/license/Qiskit/qiskit-addon-sqd?label=License)](LICENSE.txt)
  [![Downloads](https://img.shields.io/pypi/dm/qiskit-addon-sqd.svg?label=Downloads)](https://pypi.org/project/qiskit-addon-sqd/)
  [![Tests](https://github.com/Qiskit/qiskit-addon-sqd/actions/workflows/test_latest_versions.yml/badge.svg)](https://github.com/Qiskit/qiskit-addon-sqd/actions/workflows/test_latest_versions.yml)

# Sample-based quantum diagonalization (SQD)

This package implements sample-based quantum diagonalization (SQD) -- a technique for finding eigenvalues and eigenvectors of quantum operators, such as a quantum system Hamiltonian [[1-6]](#references). It can target Hamiltonians expressed as linear combinations of Pauli operators or second-quantized fermionic operators. SQD-based workflows can be run on current quantum computers and have been shown to scale to problem sizes beyond what was possible with variational methods — and even beyond the reach of exact classical diagonalization methods.

SQD-based workflows involve first preparing one or more quantum states on a quantum device and sampling from them. Then, classical distributed computing is used to process those noisy samples. This processing occurs iteratively in two steps: first, a configuration recovery step corrects noisy samples using information about the input problem; second, the Hamiltonian is projected and diagonalized in the subspace spanned by those samples. These steps are repeated self-consistently until convergence. The result is an approximated lowest eigenvalue (energy) and lowest energy eigenstate of a given Hamiltonian. SQD is robust to samples corrupted by quantum noise; in fact, as long as a useful signal can be retrieved out of the quantum computer, the outcome of SQD will be insensitive to noisy bitstrings.

`qiskit-addon-sqd` can be used to classically process samples drawn from a variety of quantum circuits in practice. For example:
  
  1. A variational circuit ansatz with parameters chosen such that sampling the circuit produces electronic configurations on which the target wavefunction (for example, the ground state) has significant support. This is appealing for chemistry applications where Hamiltonians can have millions of interaction terms [[1]](#references). For an example of this approach applied to chemistry using an LUCJ circuit see the [tutorial for approximating the ground state energy of the N2 molecule](https://quantum.cloud.ibm.com/docs/tutorials/sample-based-quantum-diagonalization).

  2. A set of Krylov basis states prepared over increasing time intervals. Assuming a good initial state and sparsity of the ground state, this approach is proven to converge efficiently. As one needs to prepare Trotterized time evolution circuits on a quantum device, this approach is best for applications to lattice models [[2]](#references) instead of complex many-body Hamiltonians like those for quantum chemistry. For an example of this approach applied to Fermionic lattice Hamiltonians, see the [tutorial for approximating the ground state energy of a simplified single-impurity Anderson model](https://quantum.cloud.ibm.com/docs/tutorials/sample-based-krylov-quantum-diagonalization).
  
  3. A set of Kyrlov basis states implemented with randomized compilation of the time evolution operator. This approach yields shorter-depth circuits compared to Trotter-based decompositions of the time evolution and so can be used for quantum chemistry Hamiltonians. This technique has been applied to the ground state energy of polycyclic aromatic hydrocarbons [[6]](#references).

----------------------------------------------------------------------------------------------------

### Documentation

All documentation is available at https://quantum.cloud.ibm.com/docs/addons/qiskit-addon-sqd.

----------------------------------------------------------------------------------------------------

### Installation

We encourage installing this package via `pip`, when possible:

```bash
pip install 'qiskit-addon-sqd'
```

For more installation information refer to these [installation instructions](docs/install.rst).

----------------------------------------------------------------------------------------------------

### Getting started

A simple guide to help you get started quickly with this package is available [here][docs/guides/quickstart.ipynb).

----------------------------------------------------------------------------------------------------

### Use case examples

The sample-based quantum diagonalization technique can be used to implement a diverse set of workflows. Some examples of where it has been used include:

- [Electronic structure calculations of the ground state energies of iron-sulfur clusters](https://arxiv.org/abs/2405.05068)
- [Estimating ground state energies to predict band gaps of dielectrics](https://arxiv.org/abs/2503.10901)
- [Estimating low-lying molecular excited states](https://arxiv.org/abs/2411.00468)
- [Simulating interactions between molecules](https://arxiv.org/abs/2410.09209)
- [Studying solute-solvent interactions in simulations of electronic structure](https://arxiv.org/abs/2502.10189)
- [Open-shell analysis of molecular dissociation](https://arxiv.org/abs/2411.04827)
- [Modeling reaction pathways for photochemistry problems.](https://arxiv.org/abs/2510.00484)
- [Combining with entanglement forging to study reaction pathways for materials degradation via hydrogen abstraction](https://arxiv.org/abs/2508.08229)
- [Constructing subspaces from quantum Krylov basis states to perform ground state simulations of impurity models for Fermionic systems](https://arxiv.org/abs/2501.09702)
- [Using qDRIFT randomized compilation to lower the overhead of implementing Krylov basis states for chemistry Hamiltonians](https://arxiv.org/abs/2508.02578) and [with application to a molecule exhibiting a half-Mobius electronic topology](https://arxiv.org/abs/2603.08696)
- [Using SQD as a solver used in the context of embedding and fragmentation methods for molecular systems, such as to compute energies of hydrogen rings and cyclohexane](https://arxiv.org/abs/2411.09861) and [of bi- and tri-metallic complexes](https://arxiv.org/abs/2512.14936), [study oxygen reduction reaction](https://arxiv.org/abs/2503.10923), [estimate energies of proteins](https://arxiv.org/abs/2512.17130) and [protein–ligand complexes](https://arxiv.org/abs/2605.01138), and to [predict tritium speciation](https://arxiv.org/abs/2606.30402).

----------------------------------------------------------------------------------------------------
### Technical discussion


#### Computational requirements

The computational cost of SQD is dominated by the eigenstate solver calls. At each step of the self-consistent configuration recovery iteration, `n_batches` of eigenstate solver calls are performed. The different calls are embarrassingly parallel. In this [tutorial](https://quantum.cloud.ibm.com/docs/tutorials/sample-based-quantum-diagonalization), those calls are inside a `for` loop. **It is highly recommended to perform these calls in parallel**.

The [`qiskit_addon_sqd.fermion.solve_fermion()`](qiskit_addon_sqd/fermion.py) function is multithreaded and capable of handling systems with ~25 spatial orbitals and ~10 electrons with subspace dimensions of ~$10^7$, using ~10-30 cores.

#### Choosing subspace dimensions

The choice of the subspace dimension affects the accuracy and runtime of the eigenstate solver. The larger the subspace the more accurate the calculation, at the cost of increasing the runtime and memory requirements. The optimal subspace size for a given system is not known, thus a convergence study with the subspace dimension may be performed as described in this [example](docs/how_tos/choose_subspace_dimension.ipynb).

#### The subspace dimension is set indirectly

In this package, the user controls the number of bitstrings (see the `samples_per_batch` argument in [`qiskit_addon_sqd.subsampling.postselect_and_subsample()`](qiskit_addon_sqd/subsampling.py)) contained in each subspace. The value of this argument sets an upper bound to the subspace dimension in the case of quantum chemistry applications. See this [example](docs/how_tos/select_open_closed_shell.ipynb) for more details.

#### Solvers

This package contains the functionality for the classical processing of user-provided samples. It can target Hamiltonians expressed as linear combinations of Pauli operators or second-quantized Fermionic operators. The projection and diagonalization steps are performed by a classical solver. We provide here two generic solvers, one for Fermionic systems and another for qubit systems. Other solvers that might be more efficient for specific systems can be interfaced by the users.

----------------------------------------------------------------------------------------------------

### Contributing

The source code is available [on GitHub](https://github.com/Qiskit/qiskit-addon-sqd).

The developer guide is located at [CONTRIBUTING.md](https://github.com/Qiskit/qiskit-addon-sqd/blob/main/CONTRIBUTING.md)
in the root of this project's repository.
By participating, you are expected to uphold Qiskit's [code of conduct](https://github.com/Qiskit/qiskit/blob/main/CODE_OF_CONDUCT.md).

----------------------------------------------------------------------------------------------------

### Citing this package

If you use this package in your research, use the [CITATION.bib](CITATION.bib) file in this project’s repository to cite the appropriate reference(s).

----------------------------------------------------------------------------------------------------

### License

[Apache License 2.0](LICENSE.txt)

----------------------------------------------------------------------------------------------------

### Deprecation policy

We follow [semantic versioning](https://semver.org/). We may occasionally make breaking changes in
order to improve the user experience. When possible, we will keep old interfaces and mark them as
deprecated, as long as they can co-exist with the new ones. Each substantial improvement, breaking
change, or deprecation will be documented in the [release notes](https://quantum.cloud.ibm.com/docs/api/qiskit-addon-sqd/release-notes).

----------------------------------------------------------------------------------------------------

### References

[1] Javier Robledo-Moreno, et al., [Chemistry Beyond Exact Solutions on a Quantum-Centric Supercomputer](https://arxiv.org/abs/2405.05068), arXiv:2405.05068 [quant-ph].

[2] Jeffery Yu, et al., [Quantum-Centric Algorithm for Sample-Based Krylov Diagonalization](https://arxiv.org/abs/2501.09702), arXiv:2501.09702 [quant-ph].

[3] Keita Kanno, et al., [Quantum-Selected Configuration Interaction: classical diagonalization of Hamiltonians in subspaces selected by quantum computers](https://arxiv.org/abs/2302.11320), arXiv:2302.11320 [quant-ph].

[4] Kenji Sugisaki, et al., [Hamiltonian simulation-based quantum-selected configuration interaction for large-scale electronic structure calculations with a quantum computer](https://arxiv.org/abs/2412.07218), arXiv:2412.07218 [quant-ph].

[5] Mathias Mikkelsen, Yuya O. Nakagawa, [Quantum-selected configuration interaction with time-evolved state](https://arxiv.org/abs/2412.13839), arXiv:2412.13839 [quant-ph].

[6] Samuele Piccinelli, et al., [Quantum chemistry with provable convergence via randomized sample-based Krylov quantum diagonalization](https://arxiv.org/abs/2508.02578), arXiv:2508.02578 [quant-ph].
