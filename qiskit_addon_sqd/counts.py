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

# Reminder: update the RST file in docs/apidocs when adding new interfaces.
"""Functions for transforming counts dictionaries."""

from __future__ import annotations

import numpy as np


def counts_to_arrays(counts: dict[str, float | int]) -> tuple[np.ndarray, np.ndarray]:
    """Convert a counts dictionary into a bitstring matrix and a probability array.

    Args:
        counts: The counts dictionary to convert

    Returns:
        - A 2D array representing the sampled bitstrings. Each row represents a
          bitstring, and each element is a ``bool`` representation of the
          bit's value
        - A 1D array containing the probability with which each bitstring was sampled

    """
    if not counts:
        return np.array([]), np.array([])
    prob_dict = normalize_counts_dict(counts)
    bs_mat = np.array([[bit == "1" for bit in bitstring] for bitstring in prob_dict])
    freq_arr = np.array(list(prob_dict.values()))

    return bs_mat, freq_arr


def generate_counts_uniform(
    num_samples: int, num_bits: int, rand_seed: np.random.Generator | int | None = None
) -> dict[str, int]:
    """Generate a bitstring counts dictionary of samples drawn from the uniform distribution.

    Args:
        num_samples: The number of samples to draw
        num_bits: The number of bits in the bitstrings
        rand_seed: A seed for controlling randomness

    Returns:
        A dictionary mapping bitstrings of length ``num_bits`` to the
        number of times they were sampled.

    Raises:
        ValueError: ``num_samples`` and ``num_bits`` must be positive integers.

    """
    if num_samples < 1:
        raise ValueError("The number of samples must be specified with a positive integer.")
    if num_bits < 1:
        raise ValueError("The number of bits must be specified with a positive integer.")

    rng = np.random.default_rng(rand_seed)

    sample_dict: dict[str, int] = {}
    # Use numpy to generate a random matrix of bit values and
    # convert it to a dictionary of bitstring samples
    bts_matrix = rng.choice([0, 1], size=(num_samples, num_bits))
    for i in range(num_samples):
        bts_arr = bts_matrix[i, :].astype("int")
        bts = "".join("1" if bit else "0" for bit in bts_arr)
        sample_dict[bts] = sample_dict.get(bts, 0) + 1

    return sample_dict


def generate_counts_bipartite_hamming(
    num_samples: int,
    num_bits: int,
    *,
    hamming_right: int,
    hamming_left: int,
    rand_seed: np.random.Generator | int | None = None,
) -> dict[str, int]:
    """Generate a bitstring counts dictionary with specified bipartite hamming weight.

    Args:
        num_samples: The number of samples to draw
        num_bits: The number of bits in the bitstrings
        hamming_right: The hamming weight on the right half of each bitstring
        hamming_left: The hamming weight on the left half of each bitstring
        rand_seed: A seed for controlling randomness

    Returns:
        A dictionary mapping bitstrings to the number of times they were sampled.
        Each half of each bitstring in the output dictionary will have a hamming
        weight as specified by the inputs.

    Raises:
        ValueError: ``num_bits`` and ``num_samples`` must be positive integers.
        ValueError: Hamming weights must be specified as non-negative integers.
        ValueError: ``num_bits`` must be even.

    """
    if num_bits % 2 != 0:
        raise ValueError("The number of bits must be specified with an even integer.")
    if num_samples < 1:
        raise ValueError("The number of samples must be specified with a positive integer.")
    if num_bits < 1:
        raise ValueError("The number of bits must be specified with a positive integer.")
    if hamming_left < 0 or hamming_right < 0:
        raise ValueError("Hamming weights must be specified as non-negative integers.")

    rng = np.random.default_rng(rand_seed)

    sample_dict: dict[str, int] = {}
    for _ in range(num_samples):
        # Pick random bits to flip such that the left and right hamming weights are correct
        up_flips = rng.choice(np.arange(num_bits // 2), hamming_right, replace=False).astype("int")
        dn_flips = rng.choice(np.arange(num_bits // 2), hamming_left, replace=False).astype("int")

        # Create a bitstring with the chosen bits flipped
        bts_arr = np.zeros(num_bits, dtype=int)
        bts_arr[dn_flips] = 1
        bts_arr[up_flips + num_bits // 2] = 1
        bts = "".join("1" if bit else "0" for bit in bts_arr)

        # Add the bitstring to the sample dict
        sample_dict[bts] = sample_dict.get(bts, 0) + 1

    return sample_dict


def normalize_counts_dict(counts: dict[str, float | int]) -> dict[str, float]:
    """Convert a counts dictionary into a probability dictionary."""
    if not counts:
        return counts

    total_counts = sum(counts.values())

    return {bs: count / total_counts for bs, count in counts.items()}
