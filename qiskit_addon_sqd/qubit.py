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

"""
Functions for handling quantum samples.

.. currentmodule:: qiskit_addon_sqd.qubit

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   solve_qubit
   project_operator_to_subspace
   sort_and_remove_duplicates
   matrix_elements_from_pauli
   sort_and_remove_duplicates
"""

from typing import Any

import jax.numpy as jnp
import numpy as np
from jax import Array, config, jit, vmap
from numpy.typing import NDArray
from qiskit.quantum_info import Pauli, SparsePauliOp
from scipy.sparse import coo_matrix, spmatrix
from scipy.sparse.linalg import eigsh

config.update("jax_enable_x64", True)  # To deal with large integers


def solve_qubit(
    bitstring_matrix: np.ndarray,
    hamiltonian: SparsePauliOp,
    *,
    verbose: bool = False,
    **scipy_kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Find the energies and eigenstates of a Hamiltonian projected into a subspace.

    The subspace is defined by a collection of computational basis states which
    are specified by the bitstrings (rows) in the ``bitstring_matrix``.

    This function calls `scipy.sparse.linalg.eigsh <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigsh.html#eigsh>`_ for the diagonalization.

    Args:
        bitstring_matrix: A 2D array of ``bool`` representations of bit
            values such that each row represents a single bitstring. This set of
            bitstrings specifies the subspace into which the ``hamiltonian`` will be
            projected and diagonalized.
        hamiltonian: A Hamiltonian specified as a Pauli operator.
        verbose: Whether to print the stage of the subroutine.
        **scipy_kwargs: Keyword arguments to be passed to `scipy.sparse.linalg.eigsh <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigsh.html#eigsh>`_.

    Returns:
        - 1D array with the eigenvalues
        - 2D array with the eigenvectors. Each column represents an eigenvector.

    Raises:
        ValueError: Bitstrings (rows) in ``bitstring_matrix`` must have length < ``64``.
    """
    if bitstring_matrix.shape[1] > 63:
        raise ValueError("Bitstrings (rows) in bitstring_matrix must have length < 64.")

    # Projection requires the bitstring matrix be sorted in ascending order by their unsigned integer representation
    bitstring_matrix = sort_and_remove_duplicates(bitstring_matrix)

    # Get a sparse representation of the projected operator
    d, _ = bitstring_matrix.shape
    ham_proj = project_operator_to_subspace(bitstring_matrix, hamiltonian, verbose=verbose)

    if verbose:
        print("Diagonalizing Hamiltonian in the subspace...")
    energies, eigenstates = eigsh(ham_proj, **scipy_kwargs)

    return energies, eigenstates


def project_operator_to_subspace(
    bitstring_matrix: np.ndarray,
    hamiltonian: SparsePauliOp,
    *,
    verbose: bool = False,
) -> spmatrix:
    """
    Projects a Pauli operator into a subspace.

    The subspace is defined by a collection of computational basis states, which
    are specified by the bitstrings (rows) in ``bitstring_matrix``.

    Args:
        bitstring_matrix: A 2D array of ``bool`` representations of bit
            values such that each row represents a single bitstring. This set of
            bitstrings specifies the subspace into which the ``hamiltonian`` will be
            projected and diagonalized.
        hamiltonian: A Hamiltonian specified as a Pauli operator.
        verbose: whether to print the stage of the subroutine.

    Return:
        A `scipy.sparse.coo_matrix <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.html#coo-matrix>`_ representing the operator projected in the subspace.

    Raises:
        ValueError: Bitstrings (rows) in ``bitstring_matrix`` must have length < ``64``.
    """
    if bitstring_matrix.shape[1] > 63:
        raise ValueError("Bitstrings (rows) in bitstring_matrix must have length < 64.")

    d, _ = bitstring_matrix.shape
    operator = coo_matrix((d, d), dtype="complex128")

    for i, pauli in enumerate(hamiltonian.paulis):
        coefficient = hamiltonian.coeffs[i]
        if verbose:
            (
                print(
                    f"Projecting term {i+1} out of {hamiltonian.size}: {coefficient} * "
                    + "".join(pauli.to_label())
                    + " ..."
                )
            )

        matrix_elements, row_coords, col_coords = matrix_elements_from_pauli(
            bitstring_matrix, pauli
        )

        operator += coefficient * coo_matrix((matrix_elements, (row_coords, col_coords)), (d, d))

    return operator


def sort_and_remove_duplicates(bitstring_matrix: np.ndarray, inplace: bool = True) -> np.ndarray:
    """
    Sort a bitstring matrix and remove duplicate entries.

    The lowest bitstring values will be placed in the lowest-indexed rows.

    Args:
        bitstring_matrix: A 2D array of ``bool`` representations of bit
            values such that each row represents a single bitstring.
        inplace: Whether to modify the input array in place.

    Returns:
        Sorted version of ``bitstring_matrix`` without repeated rows.
    """
    if not inplace:
        bitstring_matrix = bitstring_matrix.copy()

    bsmat_asints = _int_conversion_from_bts_matrix_vmap(bitstring_matrix)

    _, indices = np.unique(bsmat_asints, return_index=True)

    return bitstring_matrix[indices, :]


def matrix_elements_from_pauli(
    bitstring_matrix: np.ndarray, pauli: Pauli
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Find the matrix elements of a Pauli operator in the subspace defined by the bitstrings.

    .. note::
       The bitstrings in the ``bitstring_matrix`` must be unique and sorted in ascending order
       according to their unsigned integer representation. Otherwise the projection will return wrong
       results. This function does not explicitly check for uniqueness and order because
       this can be rather time consuming. See :func:`qiskit_addon_sqd.qubit.sort_and_remove_duplicates`
       for a simple way to ensure your bitstring matrix is well-formatted.

    .. note::
       This function relies on ``jax`` to efficiently perform some calculations. ``jax``
       converts the bit arrays to ``int64_t``, which means the bit arrays in
       ``bitstring_matrix`` may not have length greater than ``63``.

    Args:
        bitstring_matrix: A 2D array of ``bool`` representations of bit
            values such that each row represents a single bitstring.
            The bitstrings in the matrix must be sorted according to
            their unsigned integer representations. Otherwise the projection will return
            wrong results.
        pauli: A Pauli operator.

    Returns:
        A 1D array corresponding to the nonzero matrix elements
        A 1D array corresponding to the row indices of the elements
        A 1D array corresponding to the column indices of the elements

    Raises:
        ValueError: Bitstrings (rows) in ``bitstring_matrix`` must have length < ``64``.
    """
    if bitstring_matrix.shape[1] > 63:
        raise ValueError("Bitstrings (rows) in bitstring_matrix must have length < 64.")

    d, n_qubits = bitstring_matrix.shape
    row_array = np.arange(d)

    diag, sign, imag = _pauli_to_bool(pauli.to_label()[::-1])

    int_array_rows = _int_conversion_from_bts_matrix_vmap(bitstring_matrix)

    bs_mat_conn, matrix_elements = _connected_elements_and_amplitudes_bool_vmap(
        bitstring_matrix, diag, sign, imag
    )

    int_array_cols = _int_conversion_from_bts_matrix_vmap(bs_mat_conn)

    indices = np.isin(int_array_cols, int_array_rows, assume_unique=True, kind="sort")

    matrix_elements = matrix_elements[indices]
    row_array = row_array[indices]
    int_array_cols = int_array_cols[indices]

    col_array = np.searchsorted(int_array_rows, int_array_cols)

    return matrix_elements, row_array, col_array


def _pauli_to_bool(pauli_str: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Transform sequences of Pauli strings into arrays.

    A Pauli operator will be transformed into 3 arrays which represent
    the diagonal terms of the Pauli operator.

    Args:
        pauli_str: A Pauli string such that index ``0`` corresponds to qubit ``0``.

    Returns:
        A 3-tuple:
            - A mask signifying the diagonal Pauli terms (I, Z).
            - A mask signifying whether there is a change in sign between the two rows
                of the Pauli matrix (Y, Z).
            - A mask signifying whether the Pauli matrix elements are purely imaginary.
    """
    diag = []
    sign = []
    imag = []
    for p in pauli_str:
        if p == "I":
            diag.append(True)
            sign.append(False)
            imag.append(False)
        if p == "X":
            diag.append(False)
            sign.append(False)
            imag.append(False)
        if p == "Y":
            diag.append(False)
            sign.append(True)
            imag.append(True)
        if p == "Z":
            diag.append(True)
            sign.append(True)
            imag.append(False)

    return np.array(diag), np.array(sign), np.array(imag)


def _connected_elements_and_amplitudes_bool(
    bitstring_matrix: np.ndarray, diag: np.ndarray, sign: np.ndarray, imag: np.ndarray
) -> tuple[NDArray[np.bool_], Array]:
    """
    Find the connected element to computational basis state |X>.

    Given a Pauli operator represented by ``{diag, sign, imag}``.
    Args:
        bitstring_matrix: A 1D array of ``bool`` representations of bits.
        diag: ``bool`` whether the Pauli operator is diagonal. Only ``True``
            for I and Z.
        sign: ``bool`` Whether there is a change of sign in the matrix elements
            of the different rows of the Pauli operators. Only True for Y and Z.
        imag: ``bool`` whether the matrix elements of the Pauli operator are
            purely imaginary

    Returns:
        A matrix of bitstrings where each row is the connected element to the
            input the matrix element.
    """
    bitstring_matrix_mask: NDArray[np.bool_] = bitstring_matrix == diag
    return bitstring_matrix_mask, jnp.prod(
        (-1) ** (jnp.logical_and(bitstring_matrix, sign))
        * jnp.array(1j, dtype="complex64") ** (imag)
    )


"""Same as ``_connected_elements_and_amplitudes_jnp_bool()`` but allows to deal
with 2D arrays of bitstrings through the ``vmap`` transformation of Jax. Also
JIT compiled.
"""
_connected_elements_and_amplitudes_bool_vmap = jit(
    vmap(_connected_elements_and_amplitudes_bool, (0, None, None, None), 0)
)


def _int_conversion_from_bts_array(bit_array: np.ndarray) -> Any:
    """
    Convert a bit array to an integer representation.

    NOTE: This can only handle up to 63 qubits. Then the integer will overflow

    Args:
        bit_array: A 1D array of ``bool`` representations of bit values.

    Returns:
        Integer representation of the bit array.
    """
    n_qubits = len(bit_array)
    bitarray_asint = 0.0
    for i in range(n_qubits):
        bitarray_asint = bitarray_asint + bit_array[i] * 2 ** (n_qubits - 1 - i)

    return bitarray_asint.astype("longlong")  # type: ignore


_int_conversion_from_bts_matrix_vmap = jit(vmap(_int_conversion_from_bts_array, 0, 0))
