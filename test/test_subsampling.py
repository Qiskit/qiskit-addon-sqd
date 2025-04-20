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
        # 4-qubit full sampling
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
            ],
            dtype=bool,
        )
        n = self.bitstring_matrix.shape[0]
        self.uniform_probs = np.full(n, 1 / n, dtype=float)

    def test_subsample(self):
        with self.subTest("Basic test"):
            batches = subsample(self.bitstring_matrix, self.uniform_probs, 2, 10)
            self.assertEqual(10, len(batches))
            for batch in batches:
                self.assertEqual(2, batch.shape[0])

        with self.subTest("Full sampling"):
            batches = subsample(self.bitstring_matrix, self.uniform_probs, 20, 1)
            self.assertEqual(1, len(batches))
            self.assertEqual(self.bitstring_matrix.shape[0], batches[0].shape[0])

        with self.subTest("Non-positive batch size"):
            with pytest.raises(ValueError) as exc:
                subsample(self.bitstring_matrix, self.uniform_probs, 0, 10)
            assert (
                exc.value.args[0] == "Samples per batch must be specified with a positive integer."
            )

        with self.subTest("Non-positive num batches"):
            with pytest.raises(ValueError) as exc:
                subsample(self.bitstring_matrix, self.uniform_probs, 1, 0)
            assert (
                exc.value.args[0]
                == "The number of batches must be specified with a positive integer."
            )

        with self.subTest("Mismatching probs"):
            with pytest.raises(ValueError) as exc:
                subsample(self.bitstring_matrix, np.array([]), 1, 1)
            assert (
                exc.value.args[0]
                == "The number of elements in the probabilities array must match the number of rows in the bitstring matrix."
            )

        with self.subTest("Empty matrix"):
            empty_bs = np.empty((0, 4), dtype=bool)
            empty_p = np.empty((0,), dtype=float)
            batches = subsample(empty_bs, empty_p, 1, 1)
            self.assertEqual(1, len(batches))
            self.assertEqual(0, batches[0].shape[0])

    def test_postselect_and_subsample(self):
        partition = self.bitstring_matrix.shape[1] // 2

        # tuple-based tests
        with self.subTest("Basic tuple case"):
            batches = postselect_and_subsample(
                self.bitstring_matrix,
                self.uniform_probs,
                hamming=(1, 1),
                samples_per_batch=2,
                num_batches=10,
            )
            self.assertEqual(10, len(batches))
            for batch in batches:
                self.assertEqual(2, batch.shape[0])
                for bs in batch:
                    self.assertEqual(1, np.sum(bs[:partition]))
                    self.assertEqual(1, np.sum(bs[partition:]))

        with self.subTest("Zero tuple hamming"):
            batches = postselect_and_subsample(
                self.bitstring_matrix,
                self.uniform_probs,
                hamming=(0, 0),
                samples_per_batch=2,
                num_batches=10,
            )
            self.assertEqual(10, len(batches))
            for batch in batches:
                self.assertEqual(1, batch.shape[0])
                bs = batch[0]
                self.assertEqual(0, np.sum(bs[:partition]))
                self.assertEqual(0, np.sum(bs[partition:]))

        with self.subTest("Empty after postselection tuple"):
            batches = postselect_and_subsample(
                self.bitstring_matrix[1:],
                self.uniform_probs[1:],
                hamming=(0, 0),
                samples_per_batch=2,
                num_batches=5,
            )
            self.assertEqual(5, len(batches))
            for batch in batches:
                self.assertEqual(0, batch.shape[0])

        with self.subTest("Negative tuple hamming"):
            with pytest.raises(ValueError) as exc:
                postselect_and_subsample(
                    self.bitstring_matrix,
                    self.uniform_probs,
                    hamming=(-1, 0),
                    samples_per_batch=2,
                    num_batches=5,
                )
            assert exc.value.args[0] == "Hamming weights must be non-negative integers."

        # int-based tests
        with self.subTest("Basic int case"):
            batches = postselect_and_subsample(
                self.bitstring_matrix,
                self.uniform_probs,
                hamming=1,
                samples_per_batch=3,
                num_batches=4,
            )
            self.assertEqual(4, len(batches))
            for batch in batches:
                self.assertEqual(3, batch.shape[0])
                for bs in batch:
                    self.assertEqual(1, np.sum(bs))

        with self.subTest("Empty after postselection int"):
            batches = postselect_and_subsample(
                self.bitstring_matrix,
                self.uniform_probs,
                hamming=5,
                samples_per_batch=2,
                num_batches=3,
            )
            self.assertEqual(3, len(batches))
            for batch in batches:
                self.assertEqual(0, batch.shape[0])

        with self.subTest("Negative int hamming"):
            with pytest.raises(ValueError) as exc:
                postselect_and_subsample(
                    self.bitstring_matrix,
                    self.uniform_probs,
                    hamming=-1,
                    samples_per_batch=2,
                    num_batches=3,
                )
            assert exc.value.args[0] == "Hamming weight must be a non-negative integer."
