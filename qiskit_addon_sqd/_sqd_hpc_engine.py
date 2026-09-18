# This code is a Qiskit project.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The ``sqd-hpc`` acceleration engine: a compiled C++/nanobind backend.

Entry-point target for ``coheriq.engines.qiskit_addon_sqd:sqd-hpc``.  On import
it registers a compiled implementation of ``recover_configurations`` (backed by
the C++ implementation in qiskit-addon-sqd-hpc) with coheriq, then materializes
the engine.  The heavy lifting lives in the compiled :mod:`qiskit_addon_sqd._accel`
extension; this module only wires it into coheriq.

Kept out of the package ``__init__`` so that importing ``qiskit_addon_sqd`` has
no side effect -- only loading this entry-point module registers the engine, and
only when coheriq loads it via ``coheriq.enable_engine("qiskit_addon_sqd",
"sqd-hpc")`` (or the ``QISKIT_ADDON_SQD_ENGINE=sqd-hpc`` environment variable).

The import of :mod:`qiskit_addon_sqd._accel` below fails loudly if the compiled
extension is not present (for example, on a platform for which only the
universal wheel was installed).  Selecting this engine on such a platform is a
misconfiguration, and coheriq surfaces the resulting ``ImportError``.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
from coheriq import AccelerationEngine

# Importing the package guarantees the domain is registered and materialized
# before we construct the engine below (the engine looks the domain up by name).
import qiskit_addon_sqd  # noqa: F401  pylint: disable=unused-import

from . import _accel

_engine = AccelerationEngine("qiskit_addon_sqd", "sqd-hpc")


@_engine.override(name="recover_configurations")
def recover_configurations(
    bitstring_matrix: np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    avg_occupancies: tuple[np.ndarray, np.ndarray],
    num_elec_a: int,
    num_elec_b: int,
    rand_seed: np.random.Generator | int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Refine bitstrings via the compiled qiskit-addon-sqd-hpc implementation.

    This mirrors the public
    :func:`~qiskit_addon_sqd.configuration_recovery.recover_configurations` and
    is dispatched to in its place when the ``sqd-hpc`` engine is active.  It
    handles the same argument validation and the deprecated 1D-occupancies form,
    then hands the work to the C++ extension.

    The extension operates in the C++ bit convention of
    ``Qiskit::addon::sqd::recover_configurations``: within each row the first
    ``norb`` columns are the alpha (spin-up) orbitals ``0..norb-1`` and the next
    ``norb`` columns are the beta (spin-down) orbitals ``0..norb-1``.  The public
    numpy layout differs, so we permute columns on the way in and back out.
    """
    rng = np.random.default_rng(rand_seed)

    occ_dims = len(np.array(avg_occupancies).shape)
    if occ_dims == 1:
        warnings.warn(
            "Passing avg_occupancies as a 1D array is deprecated. Pass a length-2 tuple containing the spin-up and spin-down occupancies respectively.",
            DeprecationWarning,
            stacklevel=2,
        )
        norb = bitstring_matrix.shape[1] // 2
        avg_occupancies = (np.flip(avg_occupancies[norb:]), np.flip(avg_occupancies[:norb]))

    if num_elec_a < 0 or num_elec_b < 0:
        raise ValueError("The numbers of electrons must be specified as non-negative integers.")

    bitstring_matrix = np.asarray(bitstring_matrix)
    if bitstring_matrix.size == 0:
        # Match the empty-input return of the pure-Python implementation, which
        # builds its outputs from an empty dict (both are 1D, size-0 arrays).
        return np.array([]), np.array([])

    num_bits = bitstring_matrix.shape[1]
    norb = num_bits // 2

    occ_alpha = np.ascontiguousarray(avg_occupancies[0], dtype=np.float64)
    occ_beta = np.ascontiguousarray(avg_occupancies[1], dtype=np.float64)

    # Column permutation from the public numpy layout to the C++ convention.
    # Numpy left half (cols 0..norb-1) is beta with orbital index reversed;
    # numpy right half (cols norb..2*norb-1) is alpha with orbital index
    # reversed (see the pure-Python reference for the occupancy flattening).
    #   C++ alpha orbital j (C++ col j)        <- numpy col (2*norb - 1 - j)
    #   C++ beta  orbital j (C++ col norb + j) <- numpy col (norb - 1 - j)
    to_cpp = np.empty(num_bits, dtype=np.intp)
    j = np.arange(norb)
    to_cpp[j] = 2 * norb - 1 - j
    to_cpp[norb + j] = norb - 1 - j

    mat_cpp = np.ascontiguousarray(bitstring_matrix[:, to_cpp], dtype=np.uint8)
    probs = np.ascontiguousarray(probabilities, dtype=np.float64)
    seed = int(rng.integers(np.iinfo(np.uint64).max, dtype=np.uint64))

    out_cpp, out_probs = _accel.recover_configurations(
        mat_cpp, probs, occ_alpha, occ_beta, int(num_elec_a), int(num_elec_b), seed
    )

    # Invert the permutation to restore the public numpy layout.
    from_cpp = np.empty(num_bits, dtype=np.intp)
    from_cpp[to_cpp] = np.arange(num_bits)
    bs_mat_out = out_cpp[:, from_cpp].astype(bool)

    return bs_mat_out, out_probs


_engine.materialize()
