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

from collections.abc import Mapping

import numpy as np
from qiskit.primitives import BitArray


def counts_to_arrays(counts: Mapping[str, float | int]) -> tuple[np.ndarray, np.ndarray]:
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


def bit_array_to_arrays(bit_array: BitArray) -> tuple[np.ndarray, np.ndarray]:
    """Convert a bit array into a bitstring matrix and a probability array.

    Args:
        bit_array: The bit array to convert

    Returns:
        - A 2D array representing the sampled bitstrings. Each row represents a
          bitstring, and each element is a ``bool`` representation of the
          bit's value
        - A 1D array containing the probability with which each bitstring was sampled
    """
    # TODO can use bit_array.to_bool_array() when it's available
    bool_array = np.unpackbits(bit_array.array, axis=-1)[..., -bit_array.num_bits :].astype(bool)
    bitstrings, counts = np.unique(bool_array, axis=0, return_counts=True)
    probs = counts / bit_array.num_shots
    return bitstrings, probs


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


def generate_bit_array_uniform(
    num_samples: int, num_bits: int, rand_seed: np.random.Generator | int | None = None
) -> BitArray:
    """Generate a bit array of samples drawn from the uniform distribution.

    Args:
        num_samples: The number of samples to draw
        num_bits: The number of bits in the bitstrings
        rand_seed: A seed for controlling randomness

    Returns:
        The sampled bit array.

    Raises:
        ValueError: ``num_samples`` and ``num_bits`` must be positive integers.
    """
    rng = np.random.default_rng(rand_seed)
    return BitArray.from_bool_array(rng.integers(2, size=(num_samples, num_bits), dtype=bool))


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


def normalize_counts_dict(counts: Mapping[str, float | int]) -> Mapping[str, float]:
    """Convert a counts dictionary into a probability dictionary."""
    if not counts:
        return counts

    total_counts = sum(counts.values())

    return {bs: count / total_counts for bs, count in counts.items()}


def bitstring_matrix_to_integers(bitstring_matrix: np.ndarray) -> np.ndarray:
    """Convert a bitstring matrix to an array of integers."""
    n_bitstrings, n_bits = bitstring_matrix.shape

    if n_bits < 64:
        dtype: type = int
    else:
        # If 64 orbitals or more, use Python unbounded integer type
        dtype = object
        bitstring_matrix = bitstring_matrix.astype(object)

    result = np.zeros(n_bitstrings, dtype=dtype)
    for i in range(n_bits):
        result += bitstring_matrix[:, i] * (1 << (n_bits - 1 - i))

    return result
