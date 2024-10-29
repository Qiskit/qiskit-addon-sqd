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
"""Functions for performing self-consistent configuration recovery."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import numpy as np


def post_select_by_hamming_weight(
    bitstring_matrix: np.ndarray, *, hamming_right: int, hamming_left: int
) -> np.ndarray:
    """Post-select bitstrings based on the hamming weight of each half.

    Args:
        bitstring_matrix: A 2D array of ``bool`` representations of bit
            values such that each row represents a single bitstring
        hamming_right: The target hamming weight of the right half of bitstrings
        hamming_left: The target hamming weight of the left half of bitstrings

    Returns:
        A mask signifying which samples (rows) were selected from the input matrix.

    """
    if hamming_left < 0 or hamming_right < 0:
        raise ValueError("Hamming weights must be non-negative integers.")
    num_bits = bitstring_matrix.shape[1]

    # Find the bitstrings with correct hamming weight on both halves
    up_keepers = np.sum(bitstring_matrix[:, num_bits // 2 :], axis=1) == hamming_right
    down_keepers = np.sum(bitstring_matrix[:, : num_bits // 2], axis=1) == hamming_left
    correct_bs_mask = np.array(np.logical_and(up_keepers, down_keepers))

    return correct_bs_mask


def recover_configurations(
    bitstring_matrix: np.ndarray,
    probabilities: Sequence[float],
    avg_occupancies: np.ndarray,
    num_elec_a: int,
    num_elec_b: int,
    rand_seed: np.random.Generator | int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Refine bitstrings based on average orbital occupancy and a target hamming weight.

    This function refines each bit in isolation in an attempt to transform the Hilbert space
    represented by the input ``bitstring_matrix`` into a space closer to that which supports
    the ground state.

    .. note::

        This function makes the assumption that bit ``i`` represents the spin-down orbital
        corresponding to the spin-up orbital in bit ``i + N`` where ``N`` is the number of
        spatial orbitals and ``i < N``.

    .. note::

        The output configurations may not necessarily have correct hamming weight, as each bit
        is flipped in isolation from the other bits in the bitstring.

    Args:
        bitstring_matrix: A 2D array of ``bool`` representations of bit
            values such that each row represents a single bitstring
        probabilities: A 1D array specifying a probability distribution over the bitstrings
        avg_occupancies: A 1D array containing the mean occupancy of each orbital. It is assumed
            that ``avg_occupancies[i]`` corresponds to the orbital represented by column
            ``i`` in ``bitstring_matrix``.
        num_elec_a: The number of spin-up electrons in the system.
        num_elec_b: The number of spin-down electrons in the system.
        rand_seed: A seed for controlling randomness

    Returns:
        A refined bitstring matrix and an updated probability array.

    References:
        [1]: J. Robledo-Moreno, et al., `Chemistry Beyond Exact Solutions on a Quantum-Centric Supercomputer <https://arxiv.org/abs/2405.05068>`_,
             arXiv:2405.05068 [quant-ph].

    """
    rng = np.random.default_rng(rand_seed)

    if num_elec_a < 0 or num_elec_b < 0:
        raise ValueError("The numbers of electrons must be specified as non-negative integers.")

    # First, we need to flip the orbitals such that

    corrected_dict: defaultdict[str, float] = defaultdict(float)
    for bitstring, freq in zip(bitstring_matrix, probabilities):
        bs_corrected = _bipartite_bitstring_correcting(
            bitstring,
            avg_occupancies,
            num_elec_a,
            num_elec_b,
            rng=rng,
        )
        bs_str = "".join("1" if bit else "0" for bit in bs_corrected)
        corrected_dict[bs_str] += freq
    bs_mat_out = np.array([[bit == "1" for bit in bs] for bs in corrected_dict])
    freqs_out = np.array([f for f in corrected_dict.values()])
    freqs_out = np.abs(freqs_out) / np.sum(np.abs(freqs_out))

    return bs_mat_out, freqs_out


def _p_flip_0_to_1(ratio_exp: float, occ: float, eps: float = 0.01) -> float:  # pragma: no cover
    """Calculate the probability of flipping a bit from 0 to 1.

    This function will more aggressively flip bits which are in disagreement
    with the occupation information.

    Args:
        ratio_exp: The ratio of 1's expected in the set of bits
        occ: The occupancy of a particular bit, based estimated ground
            state found at the end of each configuration recovery iteration.
        eps: A value for controlling how aggressively to flip bits

    Returns:
        The probability with which to flip the bit

    """
    # Occupancy is < than naive expectation.
    # Flip 0s to 1 with small (~eps) probability in this case
    if occ < ratio_exp:
        return occ * eps / ratio_exp

    # Occupancy is >= naive expectation.
    # The probability to flip the bit increases linearly from ``eps`` to
    # ``~1.0`` as the occupation deviates further from the expected ratio
    if ratio_exp == 1.0:
        return eps
    slope = (1 - eps) / (1 - ratio_exp)
    intercept = 1 - slope
    return occ * slope + intercept


def _p_flip_1_to_0(ratio_exp: float, occ: float, eps: float = 0.01) -> float:  # pragma: no cover
    """Calculate the probability of flipping a bit from 1 to 0.

    This function will more aggressively flip bits which are in disagreement
    with the occupation information.

    Args:
        ratio_exp: The ratio of 1's expected in the set of bits
        occ: The occupancy of a particular bit, based estimated ground
            state found at the end of each configuration recovery iteration.
        eps: A value for controlling how aggressively to flip bits

    Returns:
        The probability with which to flip the bit

    """
    # Occupancy is < naive expectation.
    # The probability to flip the bit increases linearly from ``eps`` to
    # ``~1.0`` as the occupation deviates further from the expected ratio
    if occ < 1.0 - ratio_exp:
        slope = (1.0 - eps) / (1.0 - ratio_exp)
        return 1.0 - occ * slope

    # Occupancy is >= naive expectation.
    # Flip 1s to 0 with small (~eps) probability in this case
    if ratio_exp == 0.0:
        return 1 - eps
    slope = -eps / ratio_exp
    intercept = eps / ratio_exp
    return occ * slope + intercept


def _bipartite_bitstring_correcting(
    bit_array: np.ndarray,
    avg_occupancies: np.ndarray,
    hamming_right: int,
    hamming_left: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Use occupancy information and target hamming weight to correct a bitstring.

    Args:
        bit_array: A 1D array of ``bool`` representations of bit values
        avg_occupancies: A 1D array containing the mean occupancy of each orbital.
        hamming_right: The target hamming weight used for the right half of the bitstring
        hamming_left: The target hamming weight used for the left half of the bitstring
        rng: A random number generator

    Returns:
        A corrected bitstring

    """
    # This function must not mutate the input arrays.
    bit_array = bit_array.copy()

    # The number of bits should be even
    num_bits = bit_array.shape[0]
    partition_size = num_bits // 2

    # Get the probability of flipping each bit, separated into LEFT and RIGHT subsystems,
    # based on the avg occupancy of each bit and the target hamming weight
    probs_left = np.zeros(partition_size)
    probs_right = np.zeros(partition_size)
    for i in range(partition_size):
        if bit_array[i]:
            probs_left[i] = _p_flip_1_to_0(hamming_left / partition_size, avg_occupancies[i], 0.01)
        else:
            probs_left[i] = _p_flip_0_to_1(hamming_left / partition_size, avg_occupancies[i], 0.01)

        if bit_array[i + partition_size]:
            probs_right[i] = _p_flip_1_to_0(
                hamming_right / partition_size, avg_occupancies[i], 0.01
            )
        else:
            probs_right[i] = _p_flip_0_to_1(
                hamming_right / partition_size, avg_occupancies[i], 0.01
            )

    # Normalize
    probs_left = np.absolute(probs_left)
    probs_right = np.absolute(probs_right)
    probs_left = probs_left / np.sum(probs_left)
    probs_right = probs_right / np.sum(probs_right)

    ######################## Handle LEFT bits ########################

    # Get difference between # of 1s and expected # of 1s in LEFT bits
    n_left = np.sum(bit_array[:partition_size])
    n_diff = n_left - hamming_left

    # Too many electrons in LEFT bits
    if n_diff > 0:
        indices_occupied = np.where(bit_array[:partition_size])[0]
        # Get the probabilities that each 1 should be flipped to 0
        p_choice = probs_left[bit_array[:partition_size]] / np.sum(
            probs_left[bit_array[:partition_size]]
        )
        # Correct the hamming by probabilistically flipping some bits to flip to 0
        indices_to_flip = rng.choice(
            indices_occupied, size=round(n_diff), replace=False, p=p_choice
        )
        bit_array[:partition_size][indices_to_flip] = False

    # too few electrons in LEFT bits
    if n_diff < 0:
        indices_empty = np.where(np.logical_not(bit_array[:partition_size]))[0]
        # Get the probabilities that each 0 should be flipped to 1
        p_choice = probs_left[np.logical_not(bit_array[:partition_size])] / np.sum(
            probs_left[np.logical_not(bit_array[:partition_size])]
        )
        # Correct the hamming by probabilistically flipping some bits to flip to 1
        indices_to_flip = rng.choice(
            indices_empty, size=round(np.abs(n_diff)), replace=False, p=p_choice
        )
        bit_array[:partition_size][indices_to_flip] = np.logical_not(
            bit_array[:partition_size][indices_to_flip]
        )

    ######################## Handle RIGHT bits ########################

    # Get difference between # of 1s and expected # of 1s in RIGHT bits
    n_right = np.sum(bit_array[partition_size:])
    n_diff = n_right - hamming_right

    # too many electrons in RIGHT bits
    if n_diff > 0:
        indices_occupied = np.where(bit_array[partition_size:])[0]
        # Get the probabilities that each 1 should be flipped to 0
        p_choice = probs_right[bit_array[partition_size:]] / np.sum(
            probs_right[bit_array[partition_size:]]
        )
        # Correct the hamming by probabilistically flipping some bits to flip to 0
        indices_to_flip = rng.choice(
            indices_occupied, size=round(n_diff), replace=False, p=p_choice
        )
        bit_array[partition_size:][indices_to_flip] = np.logical_not(
            bit_array[partition_size:][indices_to_flip]
        )

    # too few electrons in RIGHT bits
    if n_diff < 0:
        indices_empty = np.where(np.logical_not(bit_array[partition_size:]))[0]
        # Get the probabilities that each 1 should be flipped to 0
        p_choice = probs_right[np.logical_not(bit_array[partition_size:])] / np.sum(
            probs_right[np.logical_not(bit_array[partition_size:])]
        )
        # Correct the hamming by probabilistically flipping some bits to flip to 1
        indices_to_flip = rng.choice(
            indices_empty, size=round(np.abs(n_diff)), replace=False, p=p_choice
        )
        bit_array[partition_size:][indices_to_flip] = np.logical_not(
            bit_array[partition_size:][indices_to_flip]
        )

    return bit_array
