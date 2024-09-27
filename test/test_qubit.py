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
from qiskit.quantum_info import Pauli
from qiskit_addon_sqd.qubit import matrix_elements_from_pauli


class TestQubit(unittest.TestCase):
    def test_matrix_elements_from_pauli(self):
        with self.subTest("Basic test"):
            pauli = Pauli("XZ")
            bs_mat = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
            # Flip sign on 1's corresponding to diagonal operator terms
            # Add an imaginary factor to all qubit msmts corresponding to Y terms
            # Take product of all terms to get component amplitude
            amps_test = np.array([(1 + 0j), (-1 + 0j), (1 + 0j), (-1 + 0j)])
            # All rows represent a connected component to the operator
            rows_test = np.array([0, 1, 2, 3])
            # All columns represent a connected component to the operator
            cols_test = np.array([2, 3, 0, 1])

            amps, rows, cols = matrix_elements_from_pauli(bs_mat, pauli)

            self.assertTrue((amps_test == amps).all())
            self.assertTrue((rows_test == rows).all())
            self.assertTrue((cols_test == cols).all())
