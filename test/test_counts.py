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

"""Tests for the counts module."""

import unittest

import pytest
from qiskit_addon_sqd.counts import (
    counts_to_arrays,
    generate_counts_bipartite_hamming,
    generate_counts_uniform,
    normalize_counts_dict,
)


class TestCounts(unittest.TestCase):
    def setUp(self):
        self.max_val = 16
        self.counts = {format(i, "04b"): 100 for i in range(self.max_val)}

    def test_counts_to_arrays(self):
        with self.subTest("Basic test"):
            bitstring_matrix, probs = counts_to_arrays(self.counts)
            self.assertEqual(self.max_val, bitstring_matrix.shape[0])
            self.assertEqual(4, bitstring_matrix.shape[1])
            self.assertEqual(self.max_val, len(probs))
            uniform_prob = 1 / bitstring_matrix.shape[0]
            for p in probs:
                self.assertEqual(uniform_prob, p)
        with self.subTest("Null test"):
            counts = {}
            bitstring_matrix, probs = counts_to_arrays(counts)
            self.assertEqual((0,), bitstring_matrix.shape)
            self.assertEqual((0,), probs.shape)

    def test_generate_counts_uniform(self):
        with self.subTest("Basic test"):
            num_samples = 10
            num_bits = 4
            counts = generate_counts_uniform(num_samples, num_bits)
            self.assertLessEqual(len(counts), num_samples)
            for bs in counts:
                self.assertEqual(num_bits, len(bs))
        with self.subTest("Non-positive num_bits"):
            num_samples = 10
            num_bits = 0
            with pytest.raises(ValueError) as e_info:
                generate_counts_uniform(num_samples, num_bits)
            self.assertEqual(
                "The number of bits must be specified with a positive integer.",
                e_info.value.args[0],
            )
        with self.subTest("Non-positive num_samples"):
            num_samples = 0
            num_bits = 4
            with pytest.raises(ValueError) as e_info:
                generate_counts_uniform(num_samples, num_bits)
            self.assertEqual(
                "The number of samples must be specified with a positive integer.",
                e_info.value.args[0],
            )

    def test_generate_counts_bipartite_hamming(self):
        with self.subTest("Basic test"):
            num_samples = 10
            num_bits = 8
            hamming_left = 3
            hamming_right = 2
            counts = generate_counts_bipartite_hamming(
                num_samples, num_bits, hamming_right=hamming_right, hamming_left=hamming_left
            )
            self.assertLessEqual(len(counts), num_samples)
            for bs in counts:
                ham_l = sum([b == "1" for b in bs[: num_bits // 2]])
                ham_r = sum([b == "1" for b in bs[num_bits // 2 :]])
                self.assertEqual(num_bits, len(bs))
                self.assertEqual(hamming_left, ham_l)
                self.assertEqual(hamming_right, ham_r)
        with self.subTest("Uneven num bits"):
            num_samples = 10
            num_bits = 7
            hamming_left = 3
            hamming_right = 2
            with pytest.raises(ValueError) as e_info:
                generate_counts_bipartite_hamming(
                    num_samples, num_bits, hamming_right=hamming_right, hamming_left=hamming_left
                )
            self.assertEqual(
                "The number of bits must be specified with an even integer.", e_info.value.args[0]
            )
        with self.subTest("Non-positive num_samples"):
            num_samples = 0
            num_bits = 8
            hamming_left = 3
            hamming_right = 2
            with pytest.raises(ValueError) as e_info:
                generate_counts_bipartite_hamming(
                    num_samples, num_bits, hamming_right=hamming_right, hamming_left=hamming_left
                )
            self.assertEqual(
                "The number of samples must be specified with a positive integer.",
                e_info.value.args[0],
            )
        with self.subTest("Non-positive num_bits"):
            num_samples = 10
            num_bits = 0
            hamming_left = 3
            hamming_right = 2
            with pytest.raises(ValueError) as e_info:
                generate_counts_bipartite_hamming(
                    num_samples, num_bits, hamming_right=hamming_right, hamming_left=hamming_left
                )
            self.assertEqual(
                "The number of bits must be specified with a positive integer.",
                e_info.value.args[0],
            )
        with self.subTest("Negative hamming"):
            num_samples = 10
            num_bits = 8
            hamming_left = -1
            hamming_right = -1
            with pytest.raises(ValueError) as e_info:
                generate_counts_bipartite_hamming(
                    num_samples, num_bits, hamming_right=hamming_right, hamming_left=hamming_left
                )
            self.assertEqual(
                "Hamming weights must be specified as non-negative integers.", e_info.value.args[0]
            )

    def test_normalize_counts(self):
        with self.subTest("Basic test"):
            counts_norm = normalize_counts_dict(self.counts)
            uniform_prob = 1 / self.max_val
            for prob in counts_norm.values():
                self.assertEqual(uniform_prob, prob)
        with self.subTest("Null test"):
            counts = {}
            counts_norm = normalize_counts_dict(counts)
            self.assertEqual(counts_norm, counts)
