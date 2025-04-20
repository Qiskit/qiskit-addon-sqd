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

import warnings
from collections import defaultdict
from collections.abc import Sequence

import numpy as np


def post_select_by_hamming_weight(
    bitstring_matrix: np.ndarray, hamming: int | tuple[int, int]
) -> np.ndarray:
    """Post-select bitstrings based on Hamming weight.

    Args:
        bitstring_matrix: A 2D array of bools where each row is a bitstring.
        hamming: If int, the target Hamming weight of the whole string.
                 If tuple (left, right), the target Hamming weights
                 of the left and right halves, respectively.

    Returns:
        A 1D bool mask indicating which rows match the criterion.
    """
    # Validate and unpack hamming specification
    if isinstance(hamming, tuple):
        left, right = hamming
        if left < 0 or right < 0:
            raise ValueError("Hamming weights must be non-negative integers.")
    elif isinstance(hamming, int):
        if hamming < 0:
            raise ValueError("Hamming weights must be non-negative integers.")
    else:
        raise TypeError("`hamming` must be an int or a tuple of two ints.")

    num_bits = bitstring_matrix.shape[1]

    if isinstance(hamming, tuple):
        # Split the bitstrings in half
        half = num_bits // 2
        left_keep = np.sum(bitstring_matrix[:, :half], axis=1) == left
        right_keep = np.sum(bitstring_matrix[:, half:], axis=1) == right
        correct_bs_mask = np.logical_and(left_keep, right_keep)
    else:
        # Single total Hamming weight over the entire string
        correct_bs_mask = np.sum(bitstring_matrix, axis=1) == hamming

    return correct_bs_mask


def recover_configurations(
    bitstring_matrix: np.ndarray,
    probabilities: Sequence[float],
    avg_occupancies: np.ndarray | tuple[np.ndarray, ...],
    *args,
    hamming: int | tuple[int, int] = None,
    num_elec_a: int | None = None,
    num_elec_b: int | None = None,
    rand_seed: np.random.Generator | int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Refine bitstrings based on average occupancy and a target Hamming weight.

    This function refines each bit in isolation in an attempt to transform the Hilbert space
    represented by the input ``bitstring_matrix`` into a space closer to that which supports
    the ground state.

    .. note::
        - If ``hamming`` is a 2-tuple (num_elec_a, num_elec_b), bit ``i`` represents the
          spin-down orbital corresponding to the spin-up orbital in bit ``i + N`` where
          ``N`` is the number of spatial orbitals (bipartite spin case).
        - If ``hamming`` is an int, bit ``i`` represents qubit site ``i`` (single-site case).

    Args:
        bitstring_matrix: A 2D array of ``bool`` values where each row is one bitstring.
        probabilities: A 1D sequence of floats giving a probability for each row.
        avg_occupancies: Either
            * a 1D ``np.ndarray`` of length 2N (deprecated bipartite form), or
            * a length-2 tuple ``(occ_up, occ_down)`` of 1D arrays each of length N, or
            * a single 1D array of site occupancies (for the single-site case).
        hamming: If an ``int``, the target total Hamming weight;
                 if a ``(num_elec_a, num_elec_b)`` tuple, the bipartite target weights.
        num_elec_a: The number of spin-up electrons in the system.
        num_elec_b: The number of spin-down electrons in the system.
        rand_seed: Seed or ``np.random.Generator`` for stochastic routines.

    Returns:
        A tuple ``(bs_mat_out, freqs_out)`` where
          - ``bs_mat_out`` is the refined bitstring matrix (shape MxL),
          - ``freqs_out``   is the normalized frequency array (length M).

    References:
        [1]: J. Robledo-Moreno et al., _Chemistry Beyond Exact Solutions on a Quantum-Centric Supercomputer_, arXiv:2405.05068 [quant-ph].
    """
    rng = np.random.default_rng(rand_seed)

    # Back-compat: if caller passed (num_elec_a, num_elec_b) positionally:
    if hamming is None:
        if num_elec_a is not None and num_elec_b is not None:
            hamming = (num_elec_a, num_elec_b)
        elif len(args) == 1 and isinstance(args[0], int):
            hamming = args[0]
        elif len(args) == 2 and all(isinstance(x, int) for x in args):
            hamming = (args[0], args[1])
        else:
            raise TypeError("Must specify `hamming` or `num_elec_a, num_elec_b`.")

    if num_elec_a is not None or num_elec_b is not None:
        warnings.warn(
            "Using `num_elec_a, num_elec_b` is deprecated; "
            "please switch to `hamming=(left, right)`.",
            DeprecationWarning,
            stacklevel=2,
        )

    # Handle deprecated 1D avg_occupancies for bipartite case
    if isinstance(avg_occupancies, np.ndarray):
        if isinstance(hamming, tuple):
            warnings.warn(
                "Passing avg_occupancies as a 1D array is deprecated for bipartite spins; "
                "please supply a length-2 tuple of (occ_up, occ_down).",
                DeprecationWarning,
                stacklevel=2,
            )
            norb = bitstring_matrix.shape[1] // 2
            avg_occupancies = (
                np.flip(avg_occupancies[norb:]),
                np.flip(avg_occupancies[:norb]),
            )
        else:
            warnings.warn(
                "Passing avg_occupancies as a bare array is accepted but wrapping it "
                "in a length-1 tuple is preferred for the single-site case.",
                DeprecationWarning,
                stacklevel=2,
            )
            avg_occupancies = (avg_occupancies,)

    # Validate avg_occupancies length vs. hamming type
    if isinstance(avg_occupancies, tuple):
        if isinstance(hamming, tuple) and len(avg_occupancies) != 2:
            warnings.warn(
                f"avg_occupancies should be length-2 for bipartite spins; got {len(avg_occupancies)}.",
                DeprecationWarning,
                stacklevel=2,
            )
        if isinstance(hamming, int) and len(avg_occupancies) != 1:
            warnings.warn(
                f"avg_occupancies should be length-1 for single-site; got {len(avg_occupancies)}.",
                DeprecationWarning,
                stacklevel=2,
            )

    # Flatten occupancy into bit-order
    occs_array = np.flip(np.array(avg_occupancies)).flatten()

    # Refine each bitstring using the unified corrector
    corrected: defaultdict[str, float] = defaultdict(float)
    for bs, freq in zip(bitstring_matrix, probabilities):
        bs2 = _bitstring_correcting(bs, occs_array, hamming, rng=rng)
        key = "".join("1" if bit else "0" for bit in bs2)
        corrected[key] += freq

    # Reconstruct outputs
    bs_mat_out = np.array([[c == "1" for c in key] for key in corrected])
    freqs_out = np.array(list(corrected.values()), dtype=float)
    freqs_out /= np.sum(np.abs(freqs_out))

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
    # Flip 0s to 1 with small (<eps) probability in this case
    if occ < ratio_exp:
        return occ * eps / ratio_exp

    # Occupancy is >= naive expectation.
    # The probability weight to flip the bit increases linearly from ``eps`` to
    # ``1.0`` as the occupation deviates further from the expected ratio
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
    # The probability weight to flip the bit decreases linearly from ``1.0`` to
    # ``eps`` as the occupation increases towards the expected ratio
    if occ < ratio_exp:
        slope = -(1.0 - eps) / ratio_exp
        return 1.0 + occ * slope

    # Occupancy is >= naive expectation.
    # Flip 1s to 0 with small (<eps) probability in this case
    if ratio_exp == 0.0:
        return 1 - eps
    slope = -eps / (1 - ratio_exp)
    intercept = eps / (1 - ratio_exp)
    return occ * slope + intercept


def _bitstring_correcting(
    bit_array: np.ndarray,
    avg_occupancies: np.ndarray,
    hamming: int | tuple[int, int],
    rng: np.random.Generator,
) -> np.ndarray:
    """Use occupancy information and target Hamming weight(s) to correct a bitstring.

    This function must not mutate the input arrays.

    Args:
        bit_array: A 1D array of bools representing the bitstring.
        avg_occupancies: A 1D array containing the mean occupancy of each site/orbital.
        hamming:
            - If int: the total Hamming weight target for the entire bitstring.
            - If tuple (hamming_left, hamming_right): target weights for left and right halves.
        rng: A numpy random number generator.

    Returns:
        A corrected bitstring (new numpy array of bools).
    """
    bit_array = bit_array.copy()
    num_bits = bit_array.shape[0]

    # Determine whether we're in the bipartite (left/right) or single-case
    if isinstance(hamming, tuple):
        hamming_left, hamming_right = hamming
        if hamming_left < 0 or hamming_right < 0:
            raise ValueError("Hamming weights must be non-negative integers.")
        partition_size = num_bits // 2
        is_bipartite = True
    else:
        hamming_weight = hamming
        if hamming_weight < 0:
            raise ValueError("Hamming weight must be a non-negative integer.")
        partition_size = num_bits
        is_bipartite = False

    # Compute flip probabilities for left half (or full string in single-case)
    probs_left = np.zeros(partition_size)
    for i in range(partition_size):
        target = (hamming_left if is_bipartite else hamming_weight) / partition_size
        occ = avg_occupancies[i]
        if bit_array[i]:
            probs_left[i] = _p_flip_1_to_0(target, occ, 0.01)
        else:
            probs_left[i] = _p_flip_0_to_1(target, occ, 0.01)

    # If bipartite, compute for right half too
    if is_bipartite:
        probs_right = np.zeros(partition_size)
        for i in range(partition_size):
            target = hamming_right / partition_size
            occ = avg_occupancies[i + partition_size]
            if bit_array[i + partition_size]:
                probs_right[i] = _p_flip_1_to_0(target, occ, 0.01)
            else:
                probs_right[i] = _p_flip_0_to_1(target, occ, 0.01)

    # Normalize probabilities
    probs_left = np.abs(probs_left)
    probs_left /= probs_left.sum()
    if is_bipartite:
        probs_right = np.abs(probs_right)
        probs_right /= probs_right.sum()

    # Correct LEFT (or full) partition
    n_left = np.sum(bit_array[:partition_size])
    diff_left = n_left - (hamming_left if is_bipartite else hamming_weight)
    if diff_left > 0:
        occupied = np.where(bit_array[:partition_size])[0]
        p_choice = probs_left[bit_array[:partition_size]]
        p_choice /= p_choice.sum()
        to_flip = rng.choice(occupied, size=round(diff_left), replace=False, p=p_choice)
        bit_array[:partition_size][to_flip] = False
    elif diff_left < 0:
        empty = np.where(~bit_array[:partition_size])[0]
        p_choice = probs_left[~bit_array[:partition_size]]
        p_choice /= p_choice.sum()
        to_flip = rng.choice(empty, size=round(-diff_left), replace=False, p=p_choice)
        bit_array[:partition_size][to_flip] = True

    # If bipartite, correct RIGHT partition
    if is_bipartite:
        n_right = np.sum(bit_array[partition_size:])
        diff_right = n_right - hamming_right
        if diff_right > 0:
            occ = np.where(bit_array[partition_size:])[0]
            p_choice = probs_right[bit_array[partition_size:]]
            p_choice /= p_choice.sum()
            to_flip = rng.choice(occ, size=round(diff_right), replace=False, p=p_choice)
            bit_array[partition_size:][to_flip] = False
        elif diff_right < 0:
            empty = np.where(~bit_array[partition_size:])[0]
            p_choice = probs_right[~bit_array[partition_size:]]
            p_choice /= p_choice.sum()
            to_flip = rng.choice(empty, size=round(-diff_right), replace=False, p=p_choice)
            bit_array[partition_size:][to_flip] = True

    return bit_array
