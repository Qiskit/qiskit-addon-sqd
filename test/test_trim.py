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

"""Tests for the trim module."""

from __future__ import annotations

import numpy as np
import pytest
from qiskit_addon_sqd.counts import generate_bit_array_uniform
from qiskit_addon_sqd.fermion import (
    SCIResult,
    SubspacePolicy,
    diagonalize_fermionic_hamiltonian,
    solve_sci_batch,
)
from qiskit_addon_sqd.subsampling import partition_subsample
from qiskit_addon_sqd.trim import TrimPolicy


def _hubbard_integrals(norb: int, u: float = 2.0):
    """One- and two-body tensors of a 1-D Hubbard chain, as a cheap test Hamiltonian."""
    one_body = np.zeros((norb, norb))
    for p in range(norb - 1):
        one_body[p, p + 1] = one_body[p + 1, p] = -1.0
    two_body = np.zeros((norb,) * 4)
    for p in range(norb):
        two_body[p, p, p, p] = u
    return one_body, two_body


def _run(one_body, two_body, *, policy, seed, spy=None, **kwargs):
    """Run the loop with a TrimPolicy, optionally spying on the solver."""
    rng = np.random.default_rng(seed)
    norb = one_body.shape[0]
    bit_array = generate_bit_array_uniform(2_000, 2 * norb, rand_seed=rng)
    return diagonalize_fermionic_hamiltonian(
        one_body,
        two_body,
        bit_array,
        norb=norb,
        policy=policy,
        sci_solver=spy,
        seed=rng,
        **kwargs,
    )


def test_trim_policy_conforms_to_the_subspace_policy_protocol():
    """The policy satisfies the interface structurally, not by inheriting from it."""
    assert isinstance(TrimPolicy(), SubspacePolicy)
    assert SubspacePolicy not in TrimPolicy.__mro__


def test_trim_policy_partitions_its_batches():
    """The batches are disjoint, which is what makes screening worth doing.

    This is the property that previously had to be requested separately from choosing the
    trim schedule, and could therefore be forgotten. It now follows from the policy.
    """
    norb, nelec = 6, (3, 3)
    one_body, two_body = _hubbard_integrals(norb)
    seen: list[np.ndarray] = []
    original = partition_subsample

    def spy_partition(*args, **kwargs):
        batches = original(*args, **kwargs)
        seen.extend(batches)
        return batches

    import qiskit_addon_sqd.trim as trim_module

    trim_module.partition_subsample = spy_partition
    try:
        _run(
            one_body,
            two_body,
            policy=TrimPolicy(trim_ratio=0.5),
            seed=31,
            samples_per_batch=20,
            nelec=nelec,
            num_batches=4,
            max_iterations=1,
        )
    finally:
        trim_module.partition_subsample = original

    assert seen, "TrimPolicy should draw its batches with partition_subsample"
    rows = [tuple(bitstring) for batch in seen for bitstring in batch]
    assert len(rows) == len(set(rows)), "the batches should share no bitstring"


def test_screening_round_passes_every_batch_in_one_call():
    """No matter how many batches there are, the solver sees them together.

    The screening round hands the solver every batch at once so that a distributed solver
    can diagonalize them concurrently, one group of processes per batch, rather than giving
    each batch every process in turn.
    """
    norb, nelec = 6, (3, 3)
    one_body, two_body = _hubbard_integrals(norb)

    for num_batches in (1, 2, 5):
        call_sizes: list[int] = []

        def spy(ci_strings, *args, sizes=call_sizes, **kwargs):
            sizes.append(len(ci_strings))
            return solve_sci_batch(ci_strings, *args, **kwargs)

        _run(
            one_body,
            two_body,
            policy=TrimPolicy(trim_ratio=0.5),
            seed=31,
            spy=spy,
            samples_per_batch=20,
            nelec=nelec,
            num_batches=num_batches,
            max_iterations=1,
        )

        # Exactly two calls per iteration: the screening round with all the batches,
        # and the merged round with one subspace.
        assert call_sizes == [num_batches, 1], f"num_batches={num_batches}"


def test_trim_policy_reports_the_merged_diagonalization():
    """The energy an iteration reports is the merged round's, not the best batch's."""
    norb, nelec = 6, (3, 3)
    one_body, two_body = _hubbard_integrals(norb)
    energies: list[list[float]] = []

    def spy(ci_strings, *args, **kwargs):
        results = solve_sci_batch(ci_strings, *args, **kwargs)
        energies.append([result.energy for result in results])
        return results

    result = _run(
        one_body,
        two_body,
        policy=TrimPolicy(trim_ratio=0.5),
        seed=17,
        spy=spy,
        samples_per_batch=20,
        nelec=nelec,
        num_batches=3,
        max_iterations=1,
    )

    batch_energies, merged_energies = energies
    assert len(merged_energies) == 1
    assert result.energy == merged_energies[0]
    # The merged subspace contains what the batches retained, so it does at least as well.
    assert result.energy <= min(batch_energies) + 1e-10


def test_trim_policy_carryover_reaches_the_next_subspace():
    """The strings the merged diagonalization retained seed the next iteration."""
    norb, nelec = 6, (3, 3)
    one_body, two_body = _hubbard_integrals(norb)

    merged_calls: list[tuple[np.ndarray, np.ndarray]] = []
    batch_calls: list[tuple[np.ndarray, np.ndarray]] = []

    def spy(ci_strings, *args, **kwargs):
        if len(ci_strings) == 1:
            merged_calls.append(ci_strings[0])
        else:
            batch_calls.extend(ci_strings)
        return solve_sci_batch(ci_strings, *args, **kwargs)

    _run(
        one_body,
        two_body,
        policy=TrimPolicy(trim_ratio=0.5),
        seed=99,
        spy=spy,
        samples_per_batch=20,
        nelec=nelec,
        num_batches=2,
        max_iterations=3,
    )

    assert len(merged_calls) >= 2, "the loop should have run more than one iteration"
    first_merged_a, first_merged_b = merged_calls[0]
    later_a: set[int] = set()
    later_b: set[int] = set()
    for strs_a, strs_b in batch_calls[2:]:
        later_a.update(int(s) for s in strs_a)
        later_b.update(int(s) for s in strs_b)
    # At least some of the retained strings survive into the next iteration's batches.
    assert later_a & {int(s) for s in first_merged_a}
    assert later_b & {int(s) for s in first_merged_b}


def test_max_strings_per_trim_bounds_each_batch():
    """max_strings_per_trim caps the strings each batch contributes, per spin sector."""
    norb, nelec = 6, (3, 3)
    one_body, two_body = _hubbard_integrals(norb)
    merged: list[tuple[np.ndarray, np.ndarray]] = []

    def spy(ci_strings, *args, **kwargs):
        if len(ci_strings) == 1:
            merged.append(ci_strings[0])
        return solve_sci_batch(ci_strings, *args, **kwargs)

    _run(
        one_body,
        two_body,
        policy=TrimPolicy(trim_ratio=1.0, max_strings_per_trim=3),
        seed=5,
        spy=spy,
        samples_per_batch=20,
        nelec=nelec,
        num_batches=3,
        max_iterations=1,
    )

    assert merged
    for strs_a, strs_b in merged:
        # Three batches contributing at most three strings each, before deduplication.
        assert len(strs_a) <= 9
        assert len(strs_b) <= 9


def test_trim_policy_respects_a_solver_supplied_carryover():
    """A solver that trimmed its own subspace has its selection used as given."""
    norb, nelec = 6, (3, 3)
    one_body, two_body = _hubbard_integrals(norb)
    trim_to = 2

    def solver(ci_strings, *args, **kwargs):
        results = solve_sci_batch(ci_strings, *args, **kwargs)
        return [
            SCIResult(
                energy=result.energy,
                sci_state=result.sci_state,
                orbital_occupancies=result.orbital_occupancies,
                carryover=(
                    result.sci_state.ci_strs_a[:trim_to],
                    result.sci_state.ci_strs_b[:trim_to],
                ),
            )
            for result in results
        ]

    merged: list[tuple[np.ndarray, np.ndarray]] = []

    def spy(ci_strings, *args, **kwargs):
        if len(ci_strings) == 1:
            merged.append(ci_strings[0])
        return solver(ci_strings, *args, **kwargs)

    _run(
        one_body,
        two_body,
        policy=TrimPolicy(trim_ratio=1.0),
        seed=23,
        spy=spy,
        samples_per_batch=20,
        nelec=nelec,
        num_batches=2,
        max_iterations=1,
    )

    assert merged
    for strs_a, strs_b in merged:
        # Two batches, each contributing the trim_to strings it selected itself.
        assert len(strs_a) <= 2 * trim_to
        assert len(strs_b) <= 2 * trim_to


def test_trim_policy_requires_a_state_or_carryover():
    """Ranking the CI strings needs either an eigenvector or the solver's own choice."""
    result = SCIResult(
        energy=-1.0,
        sci_state=None,
        orbital_occupancies=(np.zeros(4), np.zeros(4)),
    )
    with pytest.raises(ValueError, match="neither"):
        TrimPolicy().refine([result], 0)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"trim_ratio": 0.0}, "trim_ratio must be greater than zero"),
        ({"trim_ratio": 1.5}, "trim_ratio must be greater than zero"),
        ({"carryover_ratio": 0.0}, "carryover_ratio must be greater than zero"),
        ({"max_strings_per_trim": 0}, "max_strings_per_trim must be at least one"),
        ({"max_carryover": 0}, "max_carryover must be at least one"),
    ],
)
def test_trim_policy_validation(kwargs, match):
    """Out-of-range settings are rejected at construction."""
    with pytest.raises(ValueError, match=match):
        TrimPolicy(**kwargs)


def test_carryover_ratio_defaults_to_trim_ratio():
    """Leaving carryover_ratio unset reuses trim_ratio."""
    assert TrimPolicy(trim_ratio=0.25).carryover_ratio == 0.25
    assert TrimPolicy(trim_ratio=0.25, carryover_ratio=0.5).carryover_ratio == 0.5


def test_merged_subspace_is_spin_symmetric():
    """Every subspace obeys symmetrize_spin, including the merged one.

    The merged subspace is built from what the batches retained, per spin sector. Two
    independent unions of per-sector strings need not be equal, so the merged subspace
    would not be spin-symmetric if nothing re-imposed the constraint. The loop applies the
    caller's shape constraints to a policy's refined subspaces for this reason.
    """
    norb, nelec = 6, (3, 3)
    one_body, two_body = _hubbard_integrals(norb)
    violations: list[tuple[int, int, int]] = []

    def spy(ci_strings, *args, **kwargs):
        for strs_a, strs_b in ci_strings:
            if not np.array_equal(strs_a, strs_b):
                violations.append((len(ci_strings), len(strs_a), len(strs_b)))
        return solve_sci_batch(ci_strings, *args, **kwargs)

    _run(
        one_body,
        two_body,
        policy=TrimPolicy(trim_ratio=0.5),
        seed=24,
        spy=spy,
        samples_per_batch=20,
        nelec=nelec,
        num_batches=3,
        max_iterations=4,
        symmetrize_spin=True,
        max_dim=12,
    )

    assert not violations, f"subspaces with unequal spin sectors: {violations}"
