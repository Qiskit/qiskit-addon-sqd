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

"""Tests for configuration_recovery submodule."""

import unittest

import numpy as np
import pytest
from qiskit_addon_sqd.configuration_recovery import (
    post_select_by_hamming_weight,
    recover_configurations,
)


class TestConfigurationRecovery(unittest.TestCase):
    def setUp(self):
        self.small_mat = np.array(
            [[False, False, True, False, False, False], [False, False, True, True, False, False]]
        )

    def test_post_select_by_hamming_weight(self):
        with self.subTest("Empty test"):
            ham_l = 1
            ham_r = 0
            empty_mat = np.empty((0, 6))
            bs_mask = post_select_by_hamming_weight(
                empty_mat, hamming_right=ham_r, hamming_left=ham_l
            )
            self.assertEqual(0, bs_mask.size)
        with self.subTest("Basic test"):
            ham_l = 1
            ham_r = 0
            expected = np.array([True, False])
            bs_mask = post_select_by_hamming_weight(
                self.small_mat, hamming_right=ham_r, hamming_left=ham_l
            )
            self.assertTrue((expected == bs_mask).all())
        with self.subTest("Bad hamming"):
            ham_l = 0
            ham_r = -1
            with pytest.raises(ValueError) as e_info:
                post_select_by_hamming_weight(
                    self.small_mat, hamming_right=ham_r, hamming_left=ham_l
                )
            assert e_info.value.args[0] == "Hamming weights must be non-negative integers."

    def test_recover_configurations(self):
        with self.subTest("Empty test"):
            num_orbs = 6
            ham_l = 1
            ham_r = 0
            empty_mat = np.empty((0, num_orbs))
            empty_probs = np.empty((0,))
            occs = [False] * num_orbs
            mat_rec, probs_rec = recover_configurations(
                empty_mat, empty_probs, occs, num_elec_a=ham_r, num_elec_b=ham_l
            )
            self.assertEqual(0, mat_rec.size)
            self.assertEqual(0, probs_rec.size)
        with self.subTest("Basic test. Zeros to ones."):
            bs_mat = np.array([[False, False, False, False]])
            probs = np.array([1.0])
            occs = [1.0, 1.0, 1.0, 1.0]
            num_a = 2
            num_b = 2
            expected_mat = np.array([[True, True, True, True]])
            expected_probs = np.array([1.0])
            mat_rec, probs_rec = recover_configurations(
                bs_mat, probs, occs, num_a, num_b, rand_seed=4224
            )
            self.assertTrue((expected_mat == mat_rec).all())
            self.assertTrue((expected_probs == probs_rec).all())
        with self.subTest("Basic test. Ones to zeros."):
            bs_mat = np.array([[True, True, True, True]])
            probs = np.array([1.0])
            occs = [0.0, 0.0, 0.0, 0.0]
            num_a = 0
            num_b = 0
            expected_mat = np.array([[False, False, False, False]])
            expected_probs = np.array([1.0])
            mat_rec, probs_rec = recover_configurations(
                bs_mat, probs, occs, num_a, num_b, rand_seed=4224
            )
            self.assertTrue((expected_mat == mat_rec).all())
            self.assertTrue((expected_probs == probs_rec).all())
        with self.subTest("Basic test. Mismatching orbitals."):
            bs_mat = np.array([[True, True, True, True]])
            probs = np.array([1.0])
            occs = [0.0, 1.0, 0.0, 0.0]
            num_a = 0
            num_b = 1
            expected_mat = np.array([[False, True, False, False]])
            expected_probs = np.array([1.0])
            mat_rec, probs_rec = recover_configurations(
                bs_mat, probs, occs, num_a, num_b, rand_seed=4224
            )
            self.assertTrue((expected_mat == mat_rec).all())
            self.assertTrue((expected_probs == probs_rec).all())
        with self.subTest("Test with more than 72 bits. Ones to zeros."):
            n_bits = 74
            rng = np.random.default_rng(554)
            bs_mat = rng.integers(2, size=(1, n_bits), dtype=bool)
            probs = np.array([1.0])
            occs = np.zeros(n_bits)
            num_a = 0
            num_b = 0
            expected_mat = np.zeros((1, n_bits), dtype=bool)
            expected_probs = np.array([1.0])
            mat_rec, probs_rec = recover_configurations(
                bs_mat, probs, occs, num_a, num_b, rand_seed=4224
            )
            self.assertTrue((expected_mat == mat_rec).all())
            self.assertTrue((expected_probs == probs_rec).all())
        with self.subTest("Bad hamming."):
            bs_mat = np.array([[True, True, True, True]])
            probs = np.array([1.0])
            occs = [0.0, 0.0, 0.0, 0.0]
            num_a = 0
            num_b = -1
            with pytest.raises(ValueError) as e_info:
                recover_configurations(bs_mat, probs, occs, num_a, num_b, rand_seed=4224)
            assert (
                e_info.value.args[0]
                == "The numbers of electrons must be specified as non-negative integers."
            )
