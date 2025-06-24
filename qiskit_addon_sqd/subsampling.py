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
"""Functions for creating batches of samples from a bitstring matrix."""

from __future__ import annotations

import numpy as np
from qiskit.utils.deprecation import deprecate_func

from .configuration_recovery import post_select_by_hamming_weight


@deprecate_func(
    since="0.12.0",
    package_name="qiskit-addon-sqd",
    removal_timeline="no earlier than v0.13.0",
    additional_msg=(
        "Instead, use the ``postselect_by_hamming_right_and_left`` and ``subsample`` functions."
    ),
)
def postselect_and_subsample(
    bitstring_matrix: np.ndarray,
    probabilities: np.ndarray,
    *,
    hamming_right: int,
    hamming_left: int,
    samples_per_batch: int,
    num_batches: int,
    rand_seed: np.random.Generator | int | None = None,
) -> list[np.ndarray]:
    """Subsample batches of bit arrays with correct hamming weight from an input ``bitstring_matrix``.

    Bitstring samples with incorrect hamming weight on either their left or right half will not
    be sampled.

    Each individual batch will be sampled without replacement from the input ``bitstring_matrix``.
    Samples will be replaced after creation of each batch, so different batches may contain
    identical samples.

    Args:
        bitstring_matrix: A 2D array of ``bool`` representations of bit
            values such that each row represents a single bitstring.
        probabilities: A 1D array specifying a probability distribution over the bitstrings
        hamming_right: The target hamming weight for the right half of sampled bitstrings
        hamming_left: The target hamming weight for the left half of sampled bitstrings
        samples_per_batch: The number of samples to draw for each batch
        num_batches: The number of batches to generate
        rand_seed: A seed to control random behavior

    Returns:
        A list of bitstring matrices with correct hamming weight subsampled from the input bitstring matrix

    Raises:
        ValueError: The number of elements in ``probabilities`` must equal the number of rows in ``bitstring_matrix``.
        ValueError: Hamming weights must be non-negative integers.
        ValueError: Samples per batch and number of batches must be positive integers.

    """
    num_bitstrings = len(bitstring_matrix)
    if num_bitstrings == 0:
        return [np.array([])] * num_batches
    if len(probabilities) != num_bitstrings:
        raise ValueError(
            "The number of elements in the probabilities array must match the number of rows in the bitstring matrix."
        )
    if hamming_left < 0 or hamming_right < 0:
        raise ValueError("Hamming weight must be specified with a non-negative integer.")

    rng = np.random.default_rng(rand_seed)

    # Post-select only bitstrings with correct hamming weight
    mask_postsel = post_select_by_hamming_weight(
        bitstring_matrix, hamming_right=hamming_right, hamming_left=hamming_left
    )
    bs_mat_postsel = bitstring_matrix[mask_postsel]
    probs_postsel = probabilities[mask_postsel]
    probs_postsel = np.abs(probs_postsel) / np.sum(np.abs(probs_postsel))

    if len(probs_postsel) == 0:
        return [np.array([])] * num_batches

    return subsample(bs_mat_postsel, probs_postsel, samples_per_batch, num_batches, rand_seed=rng)


def postselect_by_hamming_right_and_left(
    bitstring_matrix: np.ndarray,
    probabilities: np.ndarray,
    *,
    hamming_right: int,
    hamming_left: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Postselect bitstrings based on desired Hamming weight on right and left halves.

    Args:
        bitstring_matrix: A 2D array of ``bool`` representations of bit
            values such that each row represents a single bitstring.
        probabilities: A 1D array specifying a probability distribution over the bitstrings
        hamming_right: The target hamming weight for the right half of sampled bitstrings
        hamming_left: The target hamming weight for the left half of sampled bitstrings

    Returns:
        Postselected bitstring matrix and probabilities. The new bitstring matrix contains
        only those bitstrings from the original matrix that have the desired Hamming weight
        on the right and left halves, and the new probabilities are constructed by taking
        the original probabilities corresponding to the postselected bitstrings and rescaling
        them to sum to one.

    Raises:
        ValueError: Hamming weights must be non-negative integers.
        ValueError: The number of columns in ``bitstring_matrix`` must be even.
        ValueError: The number of elements in ``probabilities`` must equal the number of rows in ``bitstring_matrix``.
    """
    if hamming_left < 0 or hamming_right < 0:
        raise ValueError("Hamming weight must be specified with a non-negative integer.")

    n_bitstrings, n_bits = bitstring_matrix.shape
    if n_bits % 2:
        raise ValueError(f"The length of the bitstrings must be even. Instead, got {n_bits}.")
    if len(probabilities) != n_bitstrings:
        raise ValueError(
            "The number of elements in the probabilities array must match the number of rows in the bitstring matrix."
        )

    norb = n_bits // 2
    valid_right = np.sum(bitstring_matrix[:, norb:], axis=1) == hamming_right
    valid_left = np.sum(bitstring_matrix[:, :norb], axis=1) == hamming_left
    valid_indices = np.logical_and(valid_right, valid_left)

    bitstrings_post = bitstring_matrix[valid_indices]
    probs_post = probabilities[valid_indices]
    probs_post /= np.sum(probs_post)

    return bitstrings_post, probs_post


def subsample(
    bitstring_matrix: np.ndarray,
    probabilities: np.ndarray,
    samples_per_batch: int,
    num_batches: int,
    rand_seed: np.random.Generator | int | None = None,
) -> list[np.ndarray]:
    """Subsample batches of bit arrays from an input ``bitstring_matrix``.

    Each individual batch will be sampled without replacement from the input ``bitstring_matrix``.
    Samples will be replaced after creation of each batch, so different batches may contain
    identical samples.

    Args:
        bitstring_matrix: A 2D array of ``bool`` representations of bit
            values such that each row represents a single bitstring.
        probabilities: A 1D array specifying a probability distribution over the bitstrings
        samples_per_batch: The number of samples to draw for each batch
        num_batches: The number of batches to generate
        rand_seed: A seed to control random behavior

    Returns:
        A list of bitstring matrices subsampled from the input bitstring matrix.

    Raises:
        ValueError: The number of elements in ``probabilities`` must equal the number of rows in ``bitstring_matrix``.
        ValueError: Samples per batch and number of batches must be positive integers.

    """
    if bitstring_matrix.shape[0] < 1:
        return [np.array([])] * num_batches
    if len(probabilities) != bitstring_matrix.shape[0]:
        raise ValueError(
            "The number of elements in the probabilities array must match the number of rows in the bitstring matrix."
        )
    if samples_per_batch < 1:
        raise ValueError("Samples per batch must be specified with a positive integer.")
    if num_batches < 1:
        raise ValueError("The number of batches must be specified with a positive integer.")

    rng = np.random.default_rng(rand_seed)

    num_bitstrings = bitstring_matrix.shape[0]

    # If the number of requested samples is >= the number of bitstrings, return
    # num_batches copies of the input array.
    randomly_sample = True
    if samples_per_batch >= num_bitstrings:
        randomly_sample = False
        indices = np.arange(num_bitstrings).astype("int")

    # Create batches of samples
    batches = []
    for _ in range(num_batches):
        if randomly_sample:
            indices = rng.choice(
                np.arange(num_bitstrings).astype("int"),
                samples_per_batch,
                replace=False,
                p=probabilities,
            )

        batches.append(bitstring_matrix[indices])

    return batches
