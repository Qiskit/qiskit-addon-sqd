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
        pass
