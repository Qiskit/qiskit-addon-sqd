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
from qiskit_addon_sqd.subsampling import (
    partition_subsample,
    postselect_and_subsample,
    postselect_by_hamming_right_and_left,
    subsample,
)


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

    def test_postselect_by_hamming_right_and_left(self):
        bitstrings = np.array(
            [
                [1, 1, 1, 0, 1, 0, 0, 1],
                [0, 1, 1, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 1, 0, 0],
                [0, 1, 0, 1, 1, 1, 0, 0],
            ]
        )
        probs = np.array([0.1, 0.2, 0.4, 0.3])
        bitstrings_post, probs_post = postselect_by_hamming_right_and_left(
            bitstrings, probs, hamming_right=2, hamming_left=3
        )
        expected_bitstrings = np.array(
            [
                [1, 1, 1, 0, 1, 0, 0, 1],
                [0, 1, 1, 1, 1, 1, 0, 0],
            ]
        )
        expected_probs = np.array([0.2, 0.8])
        np.testing.assert_array_equal(bitstrings_post, expected_bitstrings)
        np.testing.assert_allclose(probs_post, expected_probs)

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

    def test_partition_subsample(self):
        def as_rows(batch):
            """Represent the rows of a batch as hashable tuples."""
            return [tuple(bitstring) for bitstring in batch]

        with self.subTest("Batches are disjoint"):
            batches = partition_subsample(
                self.bitstring_matrix, self.uniform_probs, 2, 5, rand_seed=1234
            )
            self.assertEqual(5, len(batches))
            rows = [row for batch in batches for row in as_rows(batch)]
            self.assertEqual(len(rows), len(set(rows)))
        with self.subTest("Batch sizes"):
            batches = partition_subsample(
                self.bitstring_matrix, self.uniform_probs, 2, 5, rand_seed=1234
            )
            for batch in batches:
                self.assertEqual(2, batch.shape[0])
        with self.subTest("Pool exhausts the input"):
            # 4 * 4 is exactly the number of bitstrings, so the batches partition the
            # whole input rather than a subset of it.
            batches = partition_subsample(
                self.bitstring_matrix, self.uniform_probs, 4, 4, rand_seed=1234
            )
            rows = [row for batch in batches for row in as_rows(batch)]
            self.assertEqual(sorted(rows), sorted(as_rows(self.bitstring_matrix)))
        with self.subTest("Pool larger than the input"):
            # The pool is capped at the number of bitstrings available, so the batches
            # come out smaller than samples_per_batch but remain a partition.
            batches = partition_subsample(
                self.bitstring_matrix, self.uniform_probs, 10, 4, rand_seed=1234
            )
            self.assertEqual(4, len(batches))
            rows = [row for batch in batches for row in as_rows(batch)]
            self.assertEqual(sorted(rows), sorted(as_rows(self.bitstring_matrix)))
            for batch in batches:
                self.assertEqual(4, batch.shape[0])
        with self.subTest("Uneven division"):
            # A pool of 3 * 5 = 15 over 5 batches divides evenly, but 7 over 3 does not;
            # round-robin keeps the sizes within one of each other.
            batches = partition_subsample(
                self.bitstring_matrix, self.uniform_probs, 7, 3, rand_seed=1234
            )
            sizes = sorted(batch.shape[0] for batch in batches)
            self.assertEqual(16, sum(sizes))
            self.assertLessEqual(sizes[-1] - sizes[0], 1)
        with self.subTest("Empty input"):
            batches = partition_subsample(np.array([]), np.array([]), 2, 5)
            self.assertEqual(5, len(batches))
            for batch in batches:
                self.assertEqual(0, batch.shape[0])
        with self.subTest("Mismatched probabilities"):
            with pytest.raises(ValueError) as e_info:
                partition_subsample(self.bitstring_matrix, self.uniform_probs[1:], 2, 5)
            assert (
                e_info.value.args[0]
                == "The number of elements in the probabilities array must match the number of rows in the bitstring matrix."
            )
        with self.subTest("Non-positive batch size"):
            with pytest.raises(ValueError) as e_info:
                partition_subsample(self.bitstring_matrix, self.uniform_probs, 0, 5)
            assert (
                e_info.value.args[0]
                == "Samples per batch must be specified with a positive integer."
            )
        with self.subTest("Non-positive num batches"):
            with pytest.raises(ValueError) as e_info:
                partition_subsample(self.bitstring_matrix, self.uniform_probs, 2, 0)
            assert (
                e_info.value.args[0]
                == "The number of batches must be specified with a positive integer."
            )

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
