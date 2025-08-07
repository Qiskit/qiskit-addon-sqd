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
from qiskit.quantum_info import SparsePauliOp
from qiskit_addon_sqd.qubit import (
    project_operator_to_subspace,
    solve_qubit,
)
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh


class TestQubit(unittest.TestCase):
    def test_solve_qubit(self):
        with self.subTest("Basic test"):
            op = SparsePauliOp("XZIY")
            bs_mat = np.array(
                [
                    [False, False, False, False],
                    [False, False, False, True],
                    [False, False, True, False],
                    [False, False, True, True],
                    [False, True, False, False],
                    [True, False, False, False],
                    [True, True, False, False],
                ]
            )
            # Flip sign on 1's corresponding to Z operator terms
            # Add an imaginary factor to all qubit msmts corresponding to Y terms
            # Take product of all terms to get component amplitude
            amps_test = np.array([(-1j), (1j)])
            rows_test = np.array([1, 5])
            cols_test = np.array([5, 1])
            num_configs = bs_mat.shape[0]
            coo_matrix_test = coo_matrix(
                (amps_test, (rows_test, cols_test)), (num_configs, num_configs)
            )

            scipy_kwargs = {"k": 1, "which": "SA"}
            energies_test, eigenstates_test = eigsh(coo_matrix_test, **scipy_kwargs)
            e, ev = solve_qubit(bs_mat, op, **scipy_kwargs)
            self.assertTrue(np.allclose(energies_test, e))

    def test_project_operator_to_subspace(self):
        with self.subTest("Basic test"):
            op = SparsePauliOp("XZIY", coeffs=[0.5])
            bs_mat = np.array(
                [
                    [False, False, False, False],
                    [False, False, False, True],
                    [False, False, True, False],
                    [False, False, True, True],
                    [False, True, False, False],
                    [True, False, False, False],
                    [True, True, False, False],
                ]
            )
            # Flip sign on 1's corresponding to Z operator terms
            # Add an imaginary factor to all qubit msmts corresponding to Y terms
            # Take product of all terms to get component amplitude
            amps_test = np.array([(-0.5j), (0.5j)])
            rows_test = np.array([1, 5])
            cols_test = np.array([5, 1])
            num_configs = bs_mat.shape[0]
            coo_matrix_test = coo_matrix(
                (amps_test, (rows_test, cols_test)), (num_configs, num_configs)
            )
            coo_mat = project_operator_to_subspace(bs_mat, op)
            self.assertTrue(np.allclose(coo_matrix_test.data, coo_mat.data))
            self.assertEqual(coo_matrix_test.shape, coo_mat.shape)
