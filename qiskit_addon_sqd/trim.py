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

# Reminder: update the RST file in docs/apidocs when adding new interfaces.
"""Trim SQD, a schedule that screens more configurations than it diagonalizes.

:func:`~qiskit_addon_sqd.fermion.diagonalize_fermionic_hamiltonian` diagonalizes each
subsampled batch and keeps the best result. *Trim SQD* instead treats the batches as
candidates to be screened: each is diagonalized, trimmed down to the configurations
carrying the largest weight, and the survivors of all the batches are merged and
diagonalized together. The energy of that merged diagonalization is what the iteration
reports, and its own trimmed output seeds the next one.

The point is that screening is cheaper than diagonalizing. Each batch is roughly the
size of the merged subspace, so a set of candidates several times larger than the
affordable subspace can be ranked for the cost of a few extra diagonalizations of that
size.

This is expressed as a :class:`~qiskit_addon_sqd.fermion.SubspacePolicy`, so it describes
the schedule and leaves the diagonalizations themselves to whatever ``sci_solver`` the
caller passes.
"""

from __future__ import annotations

import numpy as np

from .fermion import (
    SCIResult,
    SubspaceRequest,
    _select_carryover,
    _unique_with_order_preserved,
    batch_to_ci_strings,
)
from .subsampling import partition_subsample


class TrimPolicy:
    """A :class:`~qiskit_addon_sqd.fermion.SubspacePolicy` that screens disjoint batches.

    Each iteration diagonalizes every batch, keeps the highest-weight CI strings of each,
    merges them, and performs one further diagonalization over the merged subspace. That
    diagonalization's energy is what the iteration reports, and its own trimmed CI strings
    seed the next one.

    Pass an instance as the ``policy`` argument of
    :func:`~qiskit_addon_sqd.fermion.diagonalize_fermionic_hamiltonian`::

        result = diagonalize_fermionic_hamiltonian(
            hcore,
            eri,
            bit_array,
            samples_per_batch=300,
            norb=norb,
            nelec=nelec,
            num_batches=5,
            policy=TrimPolicy(trim_ratio=0.1),
        )

    The cost of an iteration is ``num_batches + 1`` diagonalizations rather than
    ``num_batches``, and the merged subspace is bounded by what the batches retain.

    The batches are drawn as one pool of ``samples_per_batch * num_batches`` bitstrings
    divided among them, so that no *bitstring* appears in more than one. Screening selects
    among the batches, so it is only as useful as the batches differ: batches drawn
    independently would overlap heavily, which shrinks the pool of distinct candidates and
    gives a bitstring appearing in several batches several chances to survive the trim.
    Because that follows from what this schedule does, it is not something to configure.

    What this makes disjoint is the configurations. The per-spin CI string arrays derived
    from them can still share strings between batches, since two configurations may agree
    on one of their halves.

    Note:
        A subspace here is spanned by the Cartesian product of a spin-alpha and a
        spin-beta CI string array, so trimming to a given number of *configurations*
        retains close to that many strings in each spin sector, whose product is larger.
        ``max_strings_per_trim`` and ``max_carryover`` bound the strings per sector, and the ``max_dim`` argument of
        :func:`~qiskit_addon_sqd.fermion.diagonalize_fermionic_hamiltonian` bounds the
        subspace that the next iteration builds from them.

    Note:
        Whether screening improves the result depends on the system and the scale. It is
        worth comparing against a run without it, which is what passing
        ``num_batches=1`` amounts to: the single batch is diagonalized, trimmed and
        diagonalized again, with no cross-batch selection. Screening has the most to
        offer when the sampled configurations greatly outnumber what can be
        diagonalized at once, and the least when a subspace of the affordable size
        already captures most of the wavefunction.
    """

    def __init__(
        self,
        *,
        trim_ratio: float = 0.1,
        carryover_threshold: float = 1e-4,
        max_strings_per_trim: int | None = None,
        max_carryover: int | None = None,
    ) -> None:
        """Initialize the policy.

        This policy performs two kinds of trim, and they are bounded separately. The
        *screening* trims shrink each batch of the first round, once per batch; the
        *carryover* trim shrinks the merged subspace to what seeds the next iteration. Only
        the screening trims use a ratio; the carryover uses the same amplitude cutoff as
        :class:`~qiskit_addon_sqd.fermion.StandardPolicy`.

        Args:
            trim_ratio: The fraction of each batch's CI strings, per spin sector, that
                survive screening and enter the merged subspace. Ranked by summed squared
                amplitude over the other spin sector.
            carryover_threshold: Amplitude cutoff for the CI strings that seed the next
                iteration, applied exactly as
                :class:`~qiskit_addon_sqd.fermion.StandardPolicy` applies it, with the same
                default. Note that this bounds the weight discarded rather than the number
                of strings kept, so the merged subspace can grow from one iteration to the
                next; ``max_carryover`` is the only hard bound on it.
            max_strings_per_trim: Ceiling on the number of CI strings, per spin sector, that
                any one screening trim retains. Where it binds it lowers the effective
                ratio, so a trim keeps
                ``min(ceil(trim_ratio * len(strings)), max_strings_per_trim)`` strings in
                each sector. Defaults to ``None``, leaving ``trim_ratio`` to decide.

                Reach for it when a ratio would retain more than intended:
                ``symmetrize_spin`` merges the two spin sectors into a single list and so
                roughly doubles what each batch offers, and a large ``samples_per_batch``
                does the same.

                **This is not a bound on a subspace.** It applies once per batch, and the
                contributions are merged, so the merged subspace can hold up to
                ``num_batches`` times as many strings per sector -- in practice close to
                that, since the batches have little in common. The ``max_dim`` argument of
                :func:`~qiskit_addon_sqd.fermion.diagonalize_fermionic_hamiltonian` is what
                bounds a subspace.
            max_carryover: Ceiling on the number of CI strings, per spin sector, that seed
                the next iteration, applied to the marginal-weight ranking of whatever
                survived ``carryover_threshold``. Unlike ``max_strings_per_trim`` this
                governs a single trim of the merged subspace, so it bounds the carryover as
                a whole.

        Note:
            Neither ceiling constrains a solver that selects its own carryover through
            :attr:`~qiskit_addon_sqd.fermion.SCIResult.carryover`, whose selection is used
            as given.

        Raises:
            ValueError: ``trim_ratio`` was not in ``(0, 1]``, or a ceiling was less than
                one.
        """
        if not 0 < trim_ratio <= 1:
            raise ValueError(
                f"trim_ratio must be greater than zero and at most one. Got {trim_ratio}."
            )
        for name, limit in (
            ("max_strings_per_trim", max_strings_per_trim),
            ("max_carryover", max_carryover),
        ):
            if limit is not None and limit < 1:
                raise ValueError(f"{name} must be at least one. Got {limit}.")

        self.trim_ratio = trim_ratio
        self.carryover_threshold = carryover_threshold
        self.max_strings_per_trim = max_strings_per_trim
        self.max_carryover = max_carryover

    def refine(
        self, results: list[SCIResult], round_index: int
    ) -> list[tuple[np.ndarray, np.ndarray]] | None:
        """Merge what the batches retained, then stop after diagonalizing it.

        The screening round's results are trimmed to their highest-weight CI strings and
        merged into a single subspace, which the loop diagonalizes as the second round.
        """
        if round_index:
            return None
        kept_a, kept_b = [], []
        for result in results:
            strings_a, strings_b = _select_carryover(
                result, ratio=self.trim_ratio, max_strings=self.max_strings_per_trim
            )
            kept_a.append(strings_a)
            kept_b.append(strings_b)
        # Merge with the order preserved, not by sorting: the batches were each ranked by
        # weight, and a string ranked highly by its own batch has to stay early, since that
        # is what a later truncation to max_dim keeps.
        return [
            (
                _unique_with_order_preserved(np.concatenate(kept_a)),
                _unique_with_order_preserved(np.concatenate(kept_b)),
            )
        ]

    def select_result(self, results: list[SCIResult]) -> SCIResult:
        """Report the merged diagonalization, which is the only result of the last round."""
        (result,) = results
        return result

    def select_carryover(
        self, result: SCIResult, *, symmetrize_spin: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """Carry over the highest-weight CI strings of the merged diagonalization.

        This selects the carryover exactly as
        :class:`~qiskit_addon_sqd.fermion.StandardPolicy` does, by amplitude cutoff, and
        with the same default. ``max_carryover`` additionally bounds the count, which the
        threshold alone does not. The treatment of ``symmetrize_spin`` is the same too: it
        ranks the two sectors together before merging them.
        """
        return _select_carryover(
            result,
            threshold=self.carryover_threshold,
            max_strings=self.max_carryover,
            symmetrize_spin=symmetrize_spin,
        )

    def prepare_subspaces(self, request: SubspaceRequest) -> list[tuple[np.ndarray, np.ndarray]]:
        """Divide one pool of bitstrings among the batches, so that they are disjoint."""
        batches = partition_subsample(
            request.bitstrings,
            request.probabilities,
            samples_per_batch=request.samples_per_batch,
            num_batches=request.num_batches,
            rand_seed=request.rng,
        )
        return [
            batch_to_ci_strings(
                batch,
                request.norb,
                request.carryover_strings_a,
                request.carryover_strings_b,
                symmetrize_spin=request.symmetrize_spin,
            )
            for batch in batches
        ]
