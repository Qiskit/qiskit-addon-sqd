# This code is a Qiskit project.

# (C) Copyright IBM 2024.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Functions for handling quantum samples.

.. currentmodule:: qiskit_addon_sqd.qubit

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   matrix_elements_from_pauli_string
   sort_and_remove_duplicates
"""

from collections.abc import Sequence
from typing import Any

import jax.numpy as jnp
import numpy as np
from jax import Array, config, jit, vmap
from numpy.typing import NDArray

config.update("jax_enable_x64", True)  # To deal with large integers


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

    base_10 = _base_10_conversion_from_bts_matrix_vmap(bitstring_matrix)

    _, indices = np.unique(base_10, return_index=True)

    return bitstring_matrix[indices, :]


def matrix_elements_from_pauli_string(
    bitstring_matrix: np.ndarray, pauli_str: Sequence[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Find the matrix elements of a Pauli operator in the subspace defined by the bitstrings.

    .. note::
       The bitstrings in the ``bitstring_matrix`` must be sorted and unique according
       to their base-10 representation. Otherwise the projection will return wrong
       results. We do not explicitly check for uniqueness and order because this
       can be rather time consuming.

    .. note::
       This function relies on ``jax`` to efficiently perform some calculations. ``jax``
       converts the bit arrays to ``int64_t``, which means the bit arrays in
       ``bitstring_matrix`` may not have length greater than ``63``.

    Args:
        bitstring_matrix: A 2D array of ``bool`` representations of bit
            values such that each row represents a single bitstring.
            The bitstrings in the matrix must be sorted according to
            their base-10 representation. Otherwise the projection will return
            wrong results.
        pauli_str: A length-N sequence of single-qubit Pauli strings representing
            an N-qubit Pauli operator. The Pauli term for qubit ``i`` should be
            in ``pauli_str[i]`` (e.g. ``qiskit.quantum_info.Pauli("XYZ") = ["Z", "Y", "X"]``).

    Returns:
        First array corresponds to the nonzero matrix elements
        Second array corresponds to the row indices of the elements
        Third array corresponds to the column indices of the elements

    Raises:
        ValueError: Input bit arrays must have length < ``64``.
    """
    d, n_qubits = bitstring_matrix.shape
    row_array = np.arange(d)

    if n_qubits > 63:
        raise ValueError("Bit arrays must have length < 64.")

    diag, sign, imag = _pauli_str_to_bool(pauli_str)

    base_10_array_rows = _base_10_conversion_from_bts_matrix_vmap(bitstring_matrix)

    bs_mat_conn, matrix_elements = _connected_elements_and_amplitudes_bool_vmap(
        bitstring_matrix, diag, sign, imag
    )

    base_10_array_cols = _base_10_conversion_from_bts_matrix_vmap(bs_mat_conn)

    indices = np.isin(base_10_array_cols, base_10_array_rows, assume_unique=True, kind="sort")

    matrix_elements = matrix_elements[indices]
    row_array = row_array[indices]
    base_10_array_cols = base_10_array_cols[indices]

    col_array = np.searchsorted(base_10_array_rows, base_10_array_cols)

    return matrix_elements, row_array, col_array


def _pauli_str_to_bool(
    pauli_str: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Transform sequences of Pauli strings into arrays.

    An N-qubit Pauli string will be transformed into 3 arrays which represent
    the diagonal terms of the Pauli operator.

    Args:
        pauli_str: A sequence of single-qubit Pauli strings.

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
        if p == "I" or p == "i":
            diag.append(True)
            sign.append(False)
            imag.append(False)
        if p == "X" or p == "x":
            diag.append(False)
            sign.append(False)
            imag.append(False)
        if p == "Y" or p == "y":
            diag.append(False)
            sign.append(True)
            imag.append(True)
        if p == "Z" or p == "z":
            diag.append(True)
            sign.append(True)
            imag.append(False)

    return np.array(diag), np.array(sign), np.array(imag)


def _connected_elements_and_amplitudes_bool(
    bit_array: np.ndarray, diag: np.ndarray, sign: np.ndarray, imag: np.ndarray
) -> tuple[NDArray[np.bool_], Array]:
    """
    Find the connected element to computational basis state |X>.

    Given a Pauli operator represented by ``{diag, sign, imag}``.
    Args:
        bit_array: A 1D array of ``bool`` representations of bits.
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
    bit_array_mask: NDArray[np.bool_] = bit_array == diag
    return bit_array_mask, jnp.prod(
        (-1) ** (jnp.logical_and(bit_array, sign)) * jnp.array(1j, dtype="complex64") ** (imag)
    )


"""Same as ``_connected_elements_and_amplitudes_jnp_bool()`` but allows to deal
with 2D arrays of bitstrings through the ``vmap`` transformation of Jax. Also
JIT compiled.
"""
_connected_elements_and_amplitudes_bool_vmap = jit(
    vmap(_connected_elements_and_amplitudes_bool, (0, None, None, None), 0)
)


def _base_10_conversion_from_bts_array(bit_array: np.ndarray) -> Any:
    """
    Convert a bit array to a base-10 representation.

    NOTE: This can only handle up to 63 qubits. Then the integer will overflow

    Args:
        bit_array: A 1D array of ``bool`` representations of bit values.

    Returns:
        Base-10 representation of the bit array.
    """
    n_qubits = len(bit_array)
    base_10_array = 0.0
    for i in range(n_qubits):
        base_10_array = base_10_array + bit_array[i] * 2 ** (n_qubits - 1 - i)

    return base_10_array.astype("longlong")  # type: ignore


_base_10_conversion_from_bts_matrix_vmap = jit(vmap(_base_10_conversion_from_bts_array, 0, 0))
