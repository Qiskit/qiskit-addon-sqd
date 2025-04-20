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
        # small_mat has 2 bitstrings of length 6
        self.small_mat = np.array(
            [
                [False, False, True, False, False, False],
                [False, False, True, True, False, False],
            ],
            dtype=bool,
        )

    def test_post_select_by_hamming_weight(self):
        # tuple-based (left, right) tests
        with self.subTest("Empty tuple case"):
            mask = post_select_by_hamming_weight(np.empty((0, 6), dtype=bool), hamming=(1, 0))
            self.assertEqual(0, mask.size)

        with self.subTest("Basic tuple case"):
            expected = np.array([True, False])
            mask = post_select_by_hamming_weight(self.small_mat, hamming=(1, 0))
            np.testing.assert_array_equal(mask, expected)

        with self.subTest("Bad tuple hamming"):
            with pytest.raises(ValueError) as exc:
                post_select_by_hamming_weight(self.small_mat, hamming=(0, -1))
            assert exc.value.args[0] == "Hamming weights must be non-negative integers."

        # int-based tests
        with self.subTest("Empty int case"):
            mask = post_select_by_hamming_weight(np.empty((0, 6), dtype=bool), hamming=1)
            self.assertEqual(0, mask.size)

        with self.subTest("Basic int case"):
            expected = np.array([True, False])
            mask = post_select_by_hamming_weight(self.small_mat, hamming=1)
            np.testing.assert_array_equal(mask, expected)

        with self.subTest("Bad int hamming"):
            with pytest.raises(ValueError) as exc:
                post_select_by_hamming_weight(self.small_mat, hamming=-1)
            assert exc.value.args[0] == "Hamming weights must be non-negative integers."

    def test_recover_configurations(self):
        # tuple-based (left, right) tests
        with self.subTest("Empty tuple case"):
            empty_mat = np.empty((0, 6), dtype=bool)
            empty_probs = np.empty((0,), dtype=float)
            occs = (np.array([0.5] * 6), np.array([0.5] * 6))
            mat_rec, probs_rec = recover_configurations(
                empty_mat, empty_probs, occs, hamming=(0, 1)
            )
            self.assertEqual(0, mat_rec.size)
            self.assertEqual(0, probs_rec.size)

        with self.subTest("Basic tuple zeros→ones"):
            bs_mat = np.array([[False, False, False, False]], dtype=bool)
            probs = np.array([1.0], dtype=float)
            occs = (np.array([1.0] * 4), np.array([1.0] * 4))
            mat_rec, probs_rec = recover_configurations(
                bs_mat, probs, occs, hamming=(2, 2), rand_seed=4224
            )
            np.testing.assert_array_equal(mat_rec, np.ones((1, 4), dtype=bool))
            np.testing.assert_allclose(probs_rec, [1.0])

        with self.subTest("Basic tuple ones→zeros"):
            bs_mat = np.array([[True, True, True, True]], dtype=bool)
            probs = np.array([1.0], dtype=float)
            occs = (np.array([0.0] * 4), np.array([0.0] * 4))
            mat_rec, probs_rec = recover_configurations(
                bs_mat, probs, occs, hamming=(0, 0), rand_seed=4224
            )
            np.testing.assert_array_equal(mat_rec, np.zeros((1, 4), dtype=bool))
            np.testing.assert_allclose(probs_rec, [1.0])

        with self.subTest("Bad tuple hamming"):
            bs_mat = np.array([[True, True, True, True]], dtype=bool)
            probs = np.array([1.0], dtype=float)
            occs = (np.array([0.0] * 4), np.array([0.0] * 4))
            with pytest.raises(ValueError) as exc:
                recover_configurations(bs_mat, probs, occs, hamming=(0, -1))
            assert exc.value.args[0] == "Hamming weights must be non-negative integers."

        # int-based tests
        with self.subTest("Empty int case"):
            empty_mat = np.empty((0, 4), dtype=bool)
            empty_probs = np.empty((0,), dtype=float)
            occs = (np.array([0.5] * 4),)
            mat_rec, probs_rec = recover_configurations(empty_mat, empty_probs, occs, hamming=2)
            self.assertEqual(0, mat_rec.size)
            self.assertEqual(0, probs_rec.size)

        with self.subTest("Basic int zeros→ones"):
            bs_mat = np.array([[False, False, False, False]], dtype=bool)
            probs = np.array([1.0], dtype=float)
            occs = (np.array([1.0] * 4),)
            mat_rec, probs_rec = recover_configurations(
                bs_mat, probs, occs, hamming=4, rand_seed=4224
            )
            np.testing.assert_array_equal(mat_rec, np.ones((1, 4), dtype=bool))
            np.testing.assert_allclose(probs_rec, [1.0])

        with self.subTest("Basic int ones→zeros"):
            bs_mat = np.array([[True, True, True, True]], dtype=bool)
            probs = np.array([1.0], dtype=float)
            occs = (np.array([0.0] * 4),)
            mat_rec, probs_rec = recover_configurations(
                bs_mat, probs, occs, hamming=0, rand_seed=4224
            )
            np.testing.assert_array_equal(mat_rec, np.zeros((1, 4), dtype=bool))
            np.testing.assert_allclose(probs_rec, [1.0])

        with self.subTest("Bad int hamming"):
            bs_mat = np.array([[True, True, True, True]], dtype=bool)
            probs = np.array([1.0], dtype=float)
            occs = (np.array([0.0] * 4),)
            with pytest.raises(ValueError) as exc:
                recover_configurations(bs_mat, probs, occs, hamming=-1)
                assert exc.value.args[0] == "Hamming weight must be a non-negative integer."
