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

"""Tests for the qubit module."""

import unittest

import numpy as np
import pytest
from qiskit.quantum_info import Operator, Pauli, SparsePauliOp
from qiskit_addon_sqd.qubit import (
    matrix_elements_from_pauli,
    project_operator_to_subspace,
    solve_qubit,
    sort_and_remove_duplicates,
)
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh


class TestQubit(unittest.TestCase):
    def test_solve_qubit(self):
        with self.subTest("Basic test"):
            op = SparsePauliOp("XZIY")
            bs_mat = np.array(
                [
                    [0, 0, 0, 0],
                    [0, 0, 0, 1],
                    [0, 0, 1, 0],
                    [0, 0, 1, 1],
                    [0, 1, 0, 0],
                    [1, 0, 0, 0],
                    [1, 1, 0, 0],
                ]
            )
            # Flip sign on 1's corresponding to Z operator terms
            # Add an imaginary factor to all qubit msmts corresponding to Y terms
            # Take product of all terms to get component amplitude
            amps_test = np.array([(-1j), (1j)])
            rows_test = np.array([5, 1])
            cols_test = np.array([1, 5])
            num_configs = bs_mat.shape[0]
            coo_matrix_test = coo_matrix(
                (amps_test, (rows_test, cols_test)), (num_configs, num_configs)
            )

            scipy_kwargs = {"k": 1, "which": "SA"}
            energies_test, _ = eigsh(coo_matrix_test, **scipy_kwargs)
            e, _ = solve_qubit(bs_mat, op, **scipy_kwargs)
            self.assertTrue(np.allclose(energies_test, e))
        with self.subTest("64 qubits"):
            op = SparsePauliOp("Z" * 64)
            bs_mat = np.array([[1] * 64])
            with pytest.raises(ValueError) as e_info:
                solve_qubit(bs_mat, op)
            assert (
                e_info.value.args[0]
                == "Bitstrings (rows) in bitstring_matrix must have length < 64."
            )

    def test_project_operator_to_subspace(self):
        with self.subTest("Basic test"):
            op = SparsePauliOp("XZIY", coeffs=[0.5])
            bs_mat = np.array(
                [
                    [0, 0, 0, 0],
                    [0, 0, 0, 1],
                    [0, 0, 1, 0],
                    [0, 0, 1, 1],
                    [0, 1, 0, 0],
                    [1, 0, 0, 0],
                    [1, 1, 0, 0],
                ]
            )
            # Flip sign on 1's corresponding to Z operator terms
            # Add an imaginary factor to all qubit msmts corresponding to Y terms
            # Take product of all terms to get component amplitude
            amps_test = np.array([(-0.5j), (0.5j)])
            rows_test = np.array([5, 1])
            cols_test = np.array([1, 5])
            num_configs = bs_mat.shape[0]
            coo_matrix_test = coo_matrix(
                (amps_test, (rows_test, cols_test)), (num_configs, num_configs)
            )
            coo_mat = project_operator_to_subspace(bs_mat, op)
            # Compare the dense matrices rather than the raw ``.data`` arrays, which are
            # sensitive to the (arbitrary) storage order of the sparse representation.
            self.assertTrue(np.allclose(coo_matrix_test.toarray(), coo_mat.toarray()))
            self.assertEqual(coo_matrix_test.shape, coo_mat.shape)
        with self.subTest("64 qubits"):
            op = SparsePauliOp("Z" * 64)
            bs_mat = np.array([[1] * 64])
            with pytest.raises(ValueError) as e_info:
                project_operator_to_subspace(bs_mat, op)
            assert (
                e_info.value.args[0]
                == "Bitstrings (rows) in bitstring_matrix must have length < 64."
            )

    def test_matrix_elements_from_pauli(self):
        with self.subTest("Basic test"):
            pauli = Pauli("XZ")
            bs_mat = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
            # Flip sign on 1's corresponding to Z operator terms
            # Add an imaginary factor to all qubit msmts corresponding to Y terms
            # Take product of all terms to get component amplitude
            amps_test = np.array([(1 + 0j), (-1 + 0j), (1 + 0j), (-1 + 0j)])
            # Rows index the connected element <conn| P |input>; cols index the input
            rows_test = np.array([2, 3, 0, 1])
            # All columns represent an input configuration
            cols_test = np.array([0, 1, 2, 3])

            amps, rows, cols = matrix_elements_from_pauli(bs_mat, pauli)

            self.assertTrue((amps_test == amps).all())
            self.assertTrue((rows_test == rows).all())
            self.assertTrue((cols_test == cols).all())
        with self.subTest("All Paulis"):
            pauli = Pauli("XZIY")
            bs_mat = np.array(
                [
                    [0, 0, 0, 0],
                    [0, 0, 0, 1],
                    [0, 0, 1, 0],
                    [0, 0, 1, 1],
                    [0, 1, 0, 0],
                    [1, 0, 0, 0],
                    [1, 1, 0, 0],
                ]
            )
            # Flip sign on 1's corresponding to Z operator terms
            # Add an imaginary factor to all qubit msmts corresponding to Y terms
            # Take product of all terms to get component amplitude
            amps_test = np.array([(-1j), (1j)])
            rows_test = np.array([5, 1])
            cols_test = np.array([1, 5])
            amps, rows, cols = matrix_elements_from_pauli(bs_mat, pauli)
            self.assertTrue(np.allclose(amps_test, amps))
            self.assertTrue(np.allclose(rows_test, rows))
            self.assertTrue(np.allclose(cols_test, cols))
        with self.subTest("64 qubits"):
            pauli = Pauli("Z" * 64)
            bs_mat = np.array([[1] * 64])
            with pytest.raises(ValueError) as e_info:
                matrix_elements_from_pauli(bs_mat, pauli)
            assert (
                e_info.value.args[0]
                == "Bitstrings (rows) in bitstring_matrix must have length < 64."
            )

    def test_pauli_projection_matches_dense_matrix(self):
        """Projecting a Pauli onto the full computational subspace must reproduce its dense matrix.

        This guards against a transpose of the projected operator. Because a matrix and
        its transpose are isospectral, an eigenvalue-only check cannot catch it; and since
        X, Z (and I) are symmetric, only terms with an *odd* number of Y operators expose
        the bug (Y^T = -Y).
        """

        def full_subspace(num_qubits: int) -> np.ndarray:
            dim = 2**num_qubits
            return np.array(
                [[(i >> (num_qubits - 1 - k)) & 1 for k in range(num_qubits)] for i in range(dim)],
                dtype=bool,
            )

        # Includes odd-Y terms (Y, XY, YX, XYZ) that are wrong under a transpose,
        # plus symmetric/even-Y controls (X, Z, XX, YY) that are correct either way.
        for label in ["X", "Y", "Z", "XX", "XY", "YX", "YY", "XYZ"]:
            with self.subTest(pauli=label):
                pauli = Pauli(label)
                bs_mat = full_subspace(len(label))
                op = SparsePauliOp([label], coeffs=[1.0])

                projected = project_operator_to_subspace(bs_mat, op).toarray()
                expected = Operator(pauli).to_matrix()

                self.assertTrue(
                    np.allclose(projected, expected),
                    msg=f"Projected {label} does not match its dense matrix:\n{projected}\n!=\n{expected}",
                )

    def test_sort_and_remove_duplicates(self):
        with self.subTest("Basic test"):
            bs_mat = np.array([[0, 0], [1, 0], [0, 1], [0, 0], [1, 1]])
            test_mat = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

            new_mat = sort_and_remove_duplicates(bs_mat)
            self.assertTrue((test_mat == new_mat).all())
