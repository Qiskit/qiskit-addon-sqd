# This code is a Qiskit project.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# Reminder: update the RST file in docs/apidocs when adding new interfaces.
"""Functions for handling quantum samples."""

import numpy as np
from qiskit.quantum_info import SparsePauliOp
from scipy.sparse import coo_matrix, spmatrix
from scipy.sparse.linalg import eigsh

from qiskit_addon_sqd._accelerate import (  # type: ignore[attr-defined]
    connected_elements_and_amplitudes,
    generate_sparse_elements,
)


def solve_qubit(
    bitstring_matrix: np.ndarray,
    hamiltonian: SparsePauliOp,
    **scipy_kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """Find the energies and eigenstates of a Hamiltonian projected into a subspace.

    The subspace is defined by a collection of computational basis states which
    are specified by the bitstrings (rows) in the ``bitstring_matrix``. The ``bitstring_matrix``
    will be sorted and de-duplicated in this function, as the underlying solver
    requires that structure.

    .. note::

        This function supports systems of up to 128 qubits.

    This function calls `scipy.sparse.linalg.eigsh <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigsh.html#eigsh>`_ for the diagonalization.

    Args:
        bitstring_matrix: A 2D array of ``bool`` representations of bit
            values such that each row represents a single bitstring. This set of
            bitstrings specifies the subspace into which the ``hamiltonian`` will be
            projected and diagonalized.
        hamiltonian: A Hamiltonian specified as a Pauli operator.
        **scipy_kwargs: Keyword arguments to be passed to `scipy.sparse.linalg.eigsh <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigsh.html#eigsh>`_.

    Returns:
        - 1D array with the eigenvalues
        - 2D array with the eigenvectors. Each column represents an eigenvector.

    Raises:
        ValueError: Bitstrings (rows) in ``bitstring_matrix`` must have length <= ``128``.

    """
    if bitstring_matrix.shape[1] > 128:
        raise ValueError("Bitstrings (rows) in bitstring_matrix must have length <= 128.")
    # Get a sparse representation of the projected operator
    d, _ = bitstring_matrix.shape
    ham_proj = project_operator_to_subspace(bitstring_matrix, hamiltonian)

    energies, eigenstates = eigsh(ham_proj, **scipy_kwargs)

    return energies, eigenstates


def project_operator_to_subspace(
    bitstring_matrix: np.ndarray,
    hamiltonian: SparsePauliOp,
) -> spmatrix:
    """Project a Pauli operator onto a Hilbert subspace defined by the computational basis states (rows) in ``bitstring_matrix``.

    The output sparse matrix, ``A``, represents an ``NxN`` matrix s.t. ``N`` is the number of unique rows
    in ``bitstring_matrix``. The rows of ``A`` represent the input configurations, and the columns
    represent the connected component associated with the configuration in the corresponding row. The
    non-zero elements of the matrix represent the complex amplitudes associated with the connected components.

    .. note::

        This function supports systems of up to 128 qubits.

    Args:
        bitstring_matrix: A 2D array of ``bool`` representations of bit
            values such that each row represents a single bitstring. This set of
            bitstrings specifies the subspace into which the ``hamiltonian`` will be
            projected and diagonalized.
        hamiltonian: A Pauli operator to project onto a Hilbert subspace defined by ``bitstring_matrix``.

    Return:
        A `scipy.sparse.coo_matrix <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.html#coo-matrix>`_ representing the operator projected in the subspace. The rows
        represent the input configurations, and the columns represent the connected component associated with the
        configuration in the corresponding row. The non-zero elements of the matrix represent the complex amplitudes
        associated with the pairs of connected components.

    Raises:
        ValueError: Bitstrings (rows) in ``bitstring_matrix`` must have length <= ``128``.

    """
    if bitstring_matrix.shape[1] > 128:
        raise ValueError("Bitstrings (rows) in bitstring_matrix must have length <= 128.")
    bitstring_matrix = np.unique(bitstring_matrix, axis=0)
    num_samples, num_qubits = bitstring_matrix.shape
    num_ham_terms = len(hamiltonian.coeffs)
    diags = np.zeros((num_ham_terms, num_qubits), dtype=bool)
    signs = np.zeros((num_ham_terms, num_qubits), dtype=bool)
    imags = np.zeros((num_ham_terms, num_qubits), dtype=bool)

    for i, pauli in enumerate(hamiltonian.paulis):
        diags[i] = np.logical_not(pauli.x)[::-1]
        signs[i] = pauli.z[::-1]
        imags[i] = np.logical_and(pauli.x, pauli.z)[::-1]

    connected_bss, amplitudes = connected_elements_and_amplitudes(
        bitstring_matrix, diags, signs, imags
    )
    idx_map_keys = np.array([_bitarray_to_u64_pair(row) for row in bitstring_matrix])
    operator = _build_operator(connected_bss, idx_map_keys, amplitudes, hamiltonian, num_samples)

    return operator


def _bitarray_to_u64_pair(bitarr):
    """Pack a bit array into two u64 chunks, matching Rust logic."""
    n_bits = len(bitarr)
    chunk1 = np.uint64(0)
    chunk2 = np.uint64(0)

    for idx, b in enumerate(bitarr):
        rev_idx = n_bits - 1 - idx
        if rev_idx < 64:
            chunk1 <<= 1
            if b:
                chunk1 |= 1
        else:
            chunk2 <<= 1
            if b:
                chunk2 |= 1

    return (chunk1, chunk2)


def _build_operator(connected_bss, idx_map_keys, amplitudes, hamiltonian, d):
    """Project a Hamiltonian onto a subspace."""
    # Create batches
    connected_bss = np.asarray(connected_bss, dtype=bool)
    amplitudes = np.asarray(amplitudes, dtype=np.complex128)
    elements, rows, cols = generate_sparse_elements(
        idx_map_keys, connected_bss, amplitudes, hamiltonian.coeffs
    )
    return coo_matrix((elements, (rows, cols)), (d, d)).tocsr()
