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

"""Unit tests for subsampling module."""

import unittest

import numpy as np
import pytest
from qiskit_addon_sqd.subsampling import postselect_and_subsample, subsample


class TestSubsampling(unittest.TestCase):
    def setUp(self):
        # 4 qubit full sampling
        self.bitstring_matrix = np.array(
            [
                [False, False, False, False],
                [False, False, False, True],
                [False, False, True, False],
                [False, False, True, True],
                [False, True, False, False],
                [False, True, False, True],
                [False, True, True, False],
                [False, True, True, True],
                [True, False, False, False],
                [True, False, False, True],
                [True, False, True, False],
                [True, False, True, True],
                [True, True, False, False],
                [True, True, False, True],
                [True, True, True, False],
                [True, True, True, True],
            ]
        )
        self.uniform_probs = np.array(
            [1 / self.bitstring_matrix.shape[0] for _ in self.bitstring_matrix]
        )

    def test_subsample(self):
        with self.subTest("Basic test"):
            samples_per_batch = 2
            num_batches = 10
            batches = subsample(
                self.bitstring_matrix, self.uniform_probs, samples_per_batch, num_batches
            )
            self.assertEqual(num_batches, len(batches))
            for batch in batches:
                self.assertEqual(samples_per_batch, batch.shape[0])
        with self.subTest("Test probability specification"):
            samples_per_batch = 2
            num_batches = 10
            batches = subsample(
                self.bitstring_matrix, self.uniform_probs, samples_per_batch, num_batches
            )
            self.assertEqual(num_batches, len(batches))
            for batch in batches:
                self.assertEqual(samples_per_batch, batch.shape[0])
        with self.subTest("Full sampling"):
            samples_per_batch = 20
            num_batches = 1
            batches = subsample(
                self.bitstring_matrix, self.uniform_probs, samples_per_batch, num_batches
            )
            self.assertEqual(num_batches, len(batches))
            for batch in batches:
                self.assertEqual(self.bitstring_matrix.shape[0], batch.shape[0])

        with self.subTest("Non-positive batch size"):
            samples_per_batch = 0
            num_batches = 10
            with pytest.raises(ValueError) as e_info:
                subsample(
                    self.bitstring_matrix,
                    self.uniform_probs,
                    samples_per_batch,
                    num_batches,
                )
            assert (
                e_info.value.args[0]
                == "Samples per batch must be specified with a positive integer."
            )
        with self.subTest("Non-positive num batches"):
            samples_per_batch = 1
            num_batches = 0
            with pytest.raises(ValueError) as e_info:
                subsample(
                    self.bitstring_matrix,
                    self.uniform_probs,
                    samples_per_batch,
                    num_batches,
                )
            assert (
                e_info.value.args[0]
                == "The number of batches must be specified with a positive integer."
            )
        with self.subTest("Mismatching probs"):
            samples_per_batch = 1
            num_batches = 1
            with pytest.raises(ValueError) as e_info:
                subsample(
                    self.bitstring_matrix,
                    np.array([]),
                    samples_per_batch,
                    num_batches,
                )
            assert (
                e_info.value.args[0]
                == "The number of elements in the probabilities array must match the number of rows in the bitstring matrix."
            )
        with self.subTest("Empty matrix"):
            samples_per_batch = 1
            num_batches = 1
            batches = subsample(
                np.array([]),
                np.array([]),
                samples_per_batch,
                num_batches,
            )
            self.assertEqual(num_batches, len(batches))
            self.assertEqual(0, batches[0].shape[0])

    def test_postselect_and_subsample(self):
        with self.subTest("Basic test"):
            samples_per_batch = 2
            num_batches = 10
            hamming_left = 1
            hamming_right = 1
            partition_len = self.bitstring_matrix.shape[1] // 2
            batches = postselect_and_subsample(
                self.bitstring_matrix,
                self.uniform_probs,
                hamming_right=hamming_right,
                hamming_left=hamming_left,
                samples_per_batch=samples_per_batch,
                num_batches=num_batches,
            )
            self.assertEqual(num_batches, len(batches))
            for batch in batches:
                self.assertEqual(samples_per_batch, batch.shape[0])
                for bitstring in batch:
                    self.assertEqual(hamming_left, np.sum(bitstring[:partition_len]))
                    self.assertEqual(hamming_right, np.sum(bitstring[partition_len:]))
        with self.subTest("Zero hamming"):
            samples_per_batch = 2
            num_batches = 10
            hamming_left = 0
            hamming_right = 0
            partition_len = self.bitstring_matrix.shape[1] // 2
            batches = postselect_and_subsample(
                self.bitstring_matrix,
                self.uniform_probs,
                hamming_right=hamming_right,
                hamming_left=hamming_left,
                samples_per_batch=samples_per_batch,
                num_batches=num_batches,
            )
            self.assertEqual(num_batches, len(batches))
            for batch in batches:
                self.assertEqual(1, batch.shape[0])
                bitstring = batch[0]
                self.assertEqual(hamming_left, np.sum(bitstring[:partition_len]))
                self.assertEqual(hamming_right, np.sum(bitstring[partition_len:]))
        with self.subTest("Empty after postselection"):
            samples_per_batch = 2
            num_batches = 10
            hamming_left = 0
            hamming_right = 0
            partition_len = self.bitstring_matrix.shape[1] // 2
            batches = postselect_and_subsample(
                self.bitstring_matrix[1:],
                self.uniform_probs[1:],
                hamming_right=hamming_right,
                hamming_left=hamming_left,
                samples_per_batch=samples_per_batch,
                num_batches=num_batches,
            )
            self.assertEqual(num_batches, len(batches))
            for batch in batches:
                self.assertEqual(0, batch.shape[0])
        with self.subTest("Negative hamming"):
            samples_per_batch = 2
            num_batches = 10
            hamming_left = -1
            hamming_right = -1
            with pytest.raises(ValueError) as e_info:
                postselect_and_subsample(
                    self.bitstring_matrix,
                    self.uniform_probs,
                    hamming_right=hamming_right,
                    hamming_left=hamming_left,
                    samples_per_batch=samples_per_batch,
                    num_batches=num_batches,
                )
            assert (
                e_info.value.args[0]
                == "Hamming weight must be specified with a non-negative integer."
            )
        with self.subTest("Non-positive batch size"):
            samples_per_batch = 0
            num_batches = 10
            hamming_left = 1
            hamming_right = 1
            with pytest.raises(ValueError) as e_info:
                postselect_and_subsample(
                    self.bitstring_matrix,
                    self.uniform_probs,
                    hamming_right=hamming_right,
                    hamming_left=hamming_left,
                    samples_per_batch=samples_per_batch,
                    num_batches=num_batches,
                )
            assert (
                e_info.value.args[0]
                == "Samples per batch must be specified with a positive integer."
            )
        with self.subTest("Non-positive num batches"):
            samples_per_batch = 1
            num_batches = 0
            hamming_left = 1
            hamming_right = 1
            with pytest.raises(ValueError) as e_info:
                postselect_and_subsample(
                    self.bitstring_matrix,
                    self.uniform_probs,
                    hamming_right=hamming_right,
                    hamming_left=hamming_left,
                    samples_per_batch=samples_per_batch,
                    num_batches=num_batches,
                )
            assert (
                e_info.value.args[0]
                == "The number of batches must be specified with a positive integer."
            )
        with self.subTest("Mismatching probs"):
            samples_per_batch = 1
            num_batches = 1
            hamming_left = 1
            hamming_right = 1
            with pytest.raises(ValueError) as e_info:
                postselect_and_subsample(
                    self.bitstring_matrix,
                    np.array([]),
                    hamming_right=hamming_right,
                    hamming_left=hamming_left,
                    samples_per_batch=samples_per_batch,
                    num_batches=num_batches,
                )
            assert (
                e_info.value.args[0]
                == "The number of elements in the probabilities array must match the number of rows in the bitstring matrix."
            )
        with self.subTest("Empty matrix"):
            samples_per_batch = 1
            num_batches = 1
            hamming_left = 1
            hamming_right = 1
            batches = postselect_and_subsample(
                np.array([]),
                np.array([]),
                hamming_right=hamming_right,
                hamming_left=hamming_left,
                samples_per_batch=samples_per_batch,
                num_batches=num_batches,
            )
            self.assertEqual(num_batches, len(batches))
            self.assertEqual(0, batches[0].shape[0])
