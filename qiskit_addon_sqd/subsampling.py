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

import warnings

import numpy as np

from .configuration_recovery import post_select_by_hamming_weight


def postselect_and_subsample(
    bitstring_matrix: np.ndarray,
    probabilities: np.ndarray,
    *args,
    hamming: int | tuple[int, int] = None,
    hamming_right: int | None = None,
    hamming_left: int | None = None,
    samples_per_batch: int,
    num_batches: int,
    rand_seed: np.random.Generator | int | None = None,
) -> list[np.ndarray]:
    """Subsample batches of bitstrings with correct Hamming weight.

    Uses `post_select_by_hamming_weight` under the hood: if `hamming` is an int, it
    selects by total weight; if a (left, right) tuple, it selects by left/right halves.

    Each batch is drawn without replacement from the post-selected pool, but batches
    themselves may repeat samples.

    Args:
        bitstring_matrix: 2D bool array where each row is a bitstring.
        probabilities: 1D array of sampling probabilities (must sum > 0).
        hamming: If int, the target total Hamming weight;
                 if (hamming_left, hamming_right), the per-half targets.
        hamming_right: The target hamming weight for the right half of sampled bitstrings
        hamming_left: The target hamming weight for the left half of sampled bitstrings
        samples_per_batch: Number of samples per batch (must be > 0).
        num_batches: Number of batches to generate (must be > 0).
        rand_seed: Seed or np.random.Generator for RNG.

    Returns:
        A list of `num_batches` numpy arrays, each of shape
        (samples_per_batch, num_bits), containing sampled bitstrings.

    Raises:
        ValueError: if `probabilities` length ≠ number of rows in `bitstring_matrix`.
        ValueError: if any Hamming target is negative.
        ValueError: if `samples_per_batch` or `num_batches` is not positive.
        TypeError: if `hamming` is neither int nor 2-tuple of ints.
    """
    num_bitstrings = bitstring_matrix.shape[0]
    if num_bitstrings == 0:
        return [np.empty((0, bitstring_matrix.shape[1]), dtype=bool) for _ in range(num_batches)]

    if probabilities.shape[0] != num_bitstrings:
        raise ValueError(
            "The number of elements in probabilities must match the number of bitstrings."
        )

    # Backwards-compatible: accept hamming_right / hamming_left
    if hamming is None:
        if hamming_right is not None and hamming_left is not None:
            hamming = (hamming_right, hamming_left)
        elif len(args) == 1 and isinstance(args[0], int):
            hamming = args[0]
        elif len(args) == 2 and all(isinstance(x, int) for x in args):
            hamming = (args[0], args[1])
        else:
            raise TypeError("Must specify hamming or (hamming_right, hamming_left).")

    if hamming_left is not None or hamming_right is not None:
        warnings.warn(
            "Using `num_elec_a, num_elec_b` is deprecated; "
            "please switch to `hamming=(left, right)`.",
            DeprecationWarning,
            stacklevel=2,
        )

    # Validate hamming
    if isinstance(hamming, tuple):
        if len(hamming) != 2 or not all(isinstance(h, int) for h in hamming):
            raise TypeError("`hamming` tuple must be two ints (hamming_left, hamming_right).")
        if hamming[0] < 0 or hamming[1] < 0:
            raise ValueError("Hamming weights must be non-negative integers.")
    elif isinstance(hamming, int):
        if hamming < 0:
            raise ValueError("Hamming weight must be a non-negative integer.")
    else:
        raise TypeError("`hamming` must be an int or a tuple of two ints.")

    if samples_per_batch <= 0 or num_batches <= 0:
        raise ValueError("samples_per_batch and num_batches must be positive integers.")

    rng = np.random.default_rng(rand_seed)

    # Post-select with the unified selector
    mask = post_select_by_hamming_weight(bitstring_matrix, hamming)
    bs_sel = bitstring_matrix[mask]
    probs_sel = probabilities[mask]
    if probs_sel.size == 0:
        return [np.empty((0, bitstring_matrix.shape[1]), dtype=bool) for _ in range(num_batches)]
    probs_sel = np.abs(probs_sel) / np.sum(np.abs(probs_sel))

    # Delegate to subsample
    return subsample(bs_sel, probs_sel, samples_per_batch, num_batches, rand_seed=rng)


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
