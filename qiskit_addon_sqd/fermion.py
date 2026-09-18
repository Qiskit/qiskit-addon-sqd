# This code is a Qiskit project.
#
# (C) Copyright IBM 2024, 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# Reminder: update the RST file in docs/apidocs when adding new interfaces.
"""Functions for the study of fermionic systems."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, cast, runtime_checkable

import numpy as np
from jax import Array, config, grad, jit, vmap
from jax import numpy as jnp
from jax.scipy.linalg import expm
from pyscf import fci
from pyscf.fci.selected_ci import (
    _as_SCIvector,
    make_rdm1,
    make_rdm1s,
    make_rdm2,
    make_rdm2s,
    spin_square,
)
from qiskit.primitives import BitArray
from qiskit.utils.deprecation import deprecate_func
from scipy import linalg as LA

from .configuration_recovery import recover_configurations
from .counts import bit_array_to_arrays, bitstring_matrix_to_integers
from .processes import broadcast, is_control_process
from .subsampling import postselect_by_hamming_right_and_left, subsample

config.update("jax_enable_x64", True)  # To deal with large integers

_ORBITAL_OPTIMIZATION_DEPRECATION = dict(
    since="0.13.0",
    package_name="qiskit-addon-sqd",
    removal_timeline="no earlier than v0.15.0",
    additional_msg=(
        "Orbital optimization is now supported by the ffsim package. See "
        "https://quantum.cloud.ibm.com/docs/addons/qiskit-addon-sqd/guides/optimize-orbitals "
        "for a guide on how to optimize the Hamiltonian basis with ffsim."
    ),
)


@dataclass(frozen=True)
class SCIState:
    """The amplitudes and determinants describing a quantum state."""

    amplitudes: np.ndarray
    """An :math:`M \\times N` array where :math:`M =` len(``ci_strs_a``)
    and :math:`N` = len(``ci_strs_b``). ``amplitudes[i][j]`` is the
    amplitude of the determinant pair (``ci_strs_a[i]``, ``ci_strs_b[j]``).
    """

    ci_strs_a: np.ndarray
    """The alpha determinants."""

    ci_strs_b: np.ndarray
    """The beta determinants."""

    norb: int
    """The number of spatial orbitals."""

    nelec: tuple[int, int]
    """The numbers of alpha and beta electrons."""

    def __post_init__(self):
        """Validate dimensions of inputs."""
        object.__setattr__(
            self, "amplitudes", np.asarray(self.amplitudes)
        )  # Convert to ndarray if not already
        if self.amplitudes.shape != (len(self.ci_strs_a), len(self.ci_strs_b)):
            raise ValueError(
                f"'amplitudes' shape must be ({len(self.ci_strs_a)}, {len(self.ci_strs_b)}) "
                f"but got {self.amplitudes.shape}"
            )

    def save(self, filename):
        """Save the SCIState object to an .npz file."""
        np.savez(
            filename,
            amplitudes=self.amplitudes,
            ci_strs_a=self.ci_strs_a,
            ci_strs_b=self.ci_strs_b,
            norb=self.norb,
            nelec=self.nelec,
        )

    @classmethod
    def load(cls, filename):
        """Load an SCIState object from an .npz file."""
        with np.load(filename) as data:
            return cls(
                data["amplitudes"],
                data["ci_strs_a"],
                data["ci_strs_b"],
                norb=data["norb"],
                nelec=tuple(data["nelec"]),
            )

    def rdm(self, rank: int = 1, spin_summed: bool = False) -> np.ndarray:
        """Compute reduced density matrix."""
        # Reason for type: ignore: mypy can't tell the return type of the
        # PySCF functions
        sci_vector = _as_SCIvector(self.amplitudes, (self.ci_strs_a, self.ci_strs_b))
        if rank == 1:
            if spin_summed:
                return make_rdm1(sci_vector, self.norb, self.nelec)  # type: ignore
            return make_rdm1s(sci_vector, self.norb, self.nelec)  # type: ignore
        if rank == 2:
            if spin_summed:
                return make_rdm2(sci_vector, self.norb, self.nelec)  # type: ignore
            return make_rdm2s(sci_vector, self.norb, self.nelec)  # type: ignore
        raise NotImplementedError(
            f"Computing the rank {rank} reduced density matrix is currently not supported."
        )

    def spin_square(self) -> float:
        """Return spin squared."""
        sci_vector = _as_SCIvector(self.amplitudes, (self.ci_strs_a, self.ci_strs_b))
        spin_squared, _ = spin_square(sci_vector, norb=self.norb, nelec=self.nelec)
        return cast(float, spin_squared)

    def orbital_occupancies(self) -> tuple[np.ndarray, np.ndarray]:
        """Average orbital occupancies."""
        dm_a, dm_b = self.rdm(rank=1, spin_summed=False)
        return np.diagonal(dm_a), np.diagonal(dm_b)


@dataclass(frozen=True)
class SCIResult:
    """Result of an SCI calculation."""

    energy: float
    """The SCI energy."""

    sci_state: SCIState | None
    """The SCI state.

    This may be ``None`` when the solver does not materialize the eigenvector. In that
    case, the solver must supply :attr:`carryover`, since the configuration recovery
    loop would otherwise derive it from the amplitudes.
    """

    orbital_occupancies: tuple[np.ndarray, np.ndarray]
    """The average orbital occupancies."""

    rdm1: np.ndarray | None = None
    """Spin-summed 1-particle reduced density matrix."""

    rdm2: np.ndarray | None = None
    """Spin-summed 2-particle reduced density matrix."""

    carryover: tuple[np.ndarray, np.ndarray] | None = None
    """The CI strings to carry over into the next iteration, chosen by the solver.

    A pair ``(strings_a, strings_b)`` of arrays of spin-alpha and spin-beta CI strings,
    each ordered by descending marginal weight. This is the same shape as a subspace,
    because that is what the strings are used to build.

    Set this when the solver ranks and selects determinants itself, which an eigensolver
    that trims its own subspace is well placed to do. When it is ``None``,
    :func:`diagonalize_fermionic_hamiltonian` selects the carryover from
    :attr:`sci_state` using its ``carryover_threshold`` argument, as it always has.

    Note that the next subspace is spanned by the *Cartesian product* of the two arrays,
    so it is generally larger than the set of configurations that were ranked. Selecting
    the 150 highest-weight configurations of a subspace typically involves close to 150
    distinct spin-alpha strings and 150 distinct spin-beta ones, whose product spans
    roughly 22,500 configurations. Use the ``max_dim`` argument of
    :func:`diagonalize_fermionic_hamiltonian` to bound the result.
    """


@dataclass(frozen=True)
class _LoopConfig:
    """Loop-invariant configuration for the configuration recovery loop.

    These values are computed once before the loop and do not change between
    iterations. Bundling them keeps the per-iteration helper calls concise.

    NOTE: The elements stored in this dataclass do not mutate *except* for the
    random number generator (rng), which will change state when used for
    configuration recovery and subsampling.
    """

    raw_bitstrings: np.ndarray
    raw_probs: np.ndarray
    n_alpha: int
    n_beta: int
    samples_per_batch: int
    num_batches: int
    norb: int
    symmetrize_spin: bool
    include_a: np.ndarray
    include_b: np.ndarray
    max_dim_a: int | None
    max_dim_b: int | None
    energy_tol: float
    occupancies_tol: float
    rng: np.random.Generator


@dataclass(frozen=True)
class _IterationState:
    """State produced by processing the results of one configuration recovery iteration."""

    best_result: SCIResult
    current_result: SCIResult
    current_occupancies: tuple[np.ndarray, np.ndarray]
    carryover_strings_a: np.ndarray
    carryover_strings_b: np.ndarray
    converged: bool


# Bound on the diagonalization rounds a policy may request within one iteration. This
# exists only to turn a policy that never stops into an error rather than a hang, so it
# is set well above what any real schedule needs.
_MAX_ROUNDS = 16


@dataclass(frozen=True)
class SubspaceRequest:
    """The inputs a :class:`SubspacePolicy` builds an iteration's subspaces from."""

    bitstrings: np.ndarray
    """The postselected or recovered bitstrings, as a 2D array of ``bool`` with one
    bitstring per row and the alpha part concatenated on the right-hand side, like this:
    ``[b_N, ..., b_0, a_N, ..., a_0]``."""

    probabilities: np.ndarray
    """A probability for each of the bitstrings."""

    carryover_strings_a: np.ndarray
    """The spin-alpha CI strings carried over from the previous iteration, in descending
    order of weight. Empty on the first iteration."""

    carryover_strings_b: np.ndarray
    """The spin-beta CI strings carried over from the previous iteration. Equal to
    :attr:`carryover_strings_a` when spin symmetrization was requested."""

    norb: int
    """The number of spatial orbitals."""

    nelec: tuple[int, int]
    """The numbers of alpha and beta electrons."""

    samples_per_batch: int
    """The number of bitstrings each batch should hold."""

    num_batches: int
    """The number of batches to build."""

    rng: np.random.Generator
    """The loop's random number generator.

    Draw from this rather than constructing another, and do not reseed it: the ``seed``
    argument of :func:`diagonalize_fermionic_hamiltonian` is what makes a run
    reproducible, and it controls this generator alone.
    """

    iteration: int
    """The index of the current configuration recovery iteration, counting from zero."""

    symmetrize_spin: bool
    """Whether the two spin sectors share a single list of CI strings.

    This is a constraint on the shape of a subspace, which the loop enforces on whatever
    a policy returns. It appears here because merging the sectors is not something that
    can be applied afterwards: the merge happens *before* the strings are ranked, so that
    the ranking is over both sectors at once, and concatenating two separately ranked
    lists would order them differently. A policy that ranks strings must therefore know
    about it. :func:`batch_to_ci_strings` takes it for the same reason.
    """


@runtime_checkable
class SubspacePolicy(Protocol):
    """The schedule of one configuration recovery iteration.

    A policy decides how an iteration spends the diagonalizations it is given: how the
    sampled configurations become subspaces, whether the results of one round of
    diagonalizations are refined into a further round, which result the iteration
    reports, and which CI strings seed the next iteration.

    Pass an implementation as the ``policy`` argument of
    :func:`diagonalize_fermionic_hamiltonian`. The default,
    :class:`StandardPolicy`, diagonalizes the subsampled batches and keeps the best
    result. :class:`~qiskit_addon_sqd.trim.TrimPolicy` screens the batches and
    diagonalizes their merged survivors.

    A policy chooses *which strings* a subspace holds. The constraints on the *shape* of
    a subspace belong to the caller, so the loop applies them to whatever a policy
    returns: the configurations requested through ``include_configurations`` go into
    every subspace, spin symmetrization is enforced when it was asked for, and each spin
    sector is truncated to ``max_dim``. A policy neither receives nor can override them.

    Note:
        A policy produces only data. It is never given an ``sci_solver`` and never
        performs a diagonalization itself: :meth:`refine` returns the subspaces to
        diagonalize and the loop calls the solver with them.

        This matters under multi-process execution. Unlike an ``sci_solver``, which every
        process enters together, **a policy runs on the control process alone** and its
        return values are broadcast to the others, so those return values must be
        picklable. A policy must therefore not call
        :func:`qiskit_addon_sqd.processes.broadcast`,
        :func:`~qiskit_addon_sqd.processes.barrier`, or any other collective operation:
        the other processes are not executing it, and the call would hang.
    """

    def prepare_subspaces(self, request: SubspaceRequest) -> list[tuple[np.ndarray, np.ndarray]]:
        """Build the subspaces to diagonalize first.

        Args:
            request: The bitstrings, carryover strings, and batch sizes to build from.

        Returns:
            One ``(strings_a, strings_b)`` pair per subspace, each spanning the Cartesian
            product of the two arrays.
        """
        ...

    def refine(
        self, results: list[SCIResult], round_index: int
    ) -> list[tuple[np.ndarray, np.ndarray]] | None:
        """Decide whether the results of a round lead to a further round.

        Args:
            results: The results of the round just completed, one per subspace.
            round_index: The index of that round, counting from zero.

        Returns:
            The subspaces for a further round, or ``None`` to stop and report this
            round's results. Returning subspaces on every call raises ``ValueError``
            once the round limit is reached.
        """
        ...

    def select_result(self, results: list[SCIResult]) -> SCIResult:
        """Choose the result an iteration reports, from those of its final round.

        The chosen result is what convergence is judged on, what becomes the returned
        result if its energy is the lowest seen, and what :meth:`select_carryover`
        receives.
        """
        ...

    def select_carryover(
        self, result: SCIResult, *, symmetrize_spin: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """Choose the CI strings that seed the next iteration.

        Args:
            result: The result chosen by :meth:`select_result`.
            symmetrize_spin: Whether the two spin sectors share one list of strings. As
                with :attr:`SubspaceRequest.symmetrize_spin`, this is passed in because
                the sectors are merged before being ranked, so it cannot be applied
                afterwards.

        Returns:
            The spin-alpha and spin-beta strings, in descending order of weight, since
            that is the order a truncation to ``max_dim`` keeps.
        """
        ...


class StandardPolicy:
    """The default :class:`SubspacePolicy`: diagonalize the batches, keep the best.

    Each iteration subsamples ``num_batches`` batches of ``samples_per_batch``
    bitstrings, diagonalizes each, and reports the lowest energy among them. The strings
    of that result whose amplitudes exceed ``carryover_threshold`` seed the next
    iteration.

    This is the behavior of :func:`diagonalize_fermionic_hamiltonian` when no ``policy``
    is given, and passing an instance explicitly is equivalent to omitting it.
    """

    def __init__(self, *, carryover_threshold: float = 1e-4) -> None:
        """Initialize the policy.

        Args:
            carryover_threshold: Threshold for carrying over bitstrings with large CI
                weight. The CI strings with amplitude above this value are carried over.
        """
        self.carryover_threshold = carryover_threshold

    def prepare_subspaces(self, request: SubspaceRequest) -> list[tuple[np.ndarray, np.ndarray]]:
        """Subsample the batches independently and build a subspace from each."""
        batches = subsample(
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

    def refine(
        self,
        results: list[SCIResult],  # pylint: disable=unused-argument
        round_index: int,  # pylint: disable=unused-argument
    ) -> list[tuple[np.ndarray, np.ndarray]] | None:
        """Stop after the first round, which diagonalized every batch.

        The arguments are unused because this schedule never asks for a second round; they
        are part of the :class:`SubspacePolicy` interface.
        """
        return None

    def select_result(self, results: list[SCIResult]) -> SCIResult:
        """Report the batch that reached the lowest energy."""
        return min(results, key=lambda result: result.energy)

    def select_carryover(
        self, result: SCIResult, *, symmetrize_spin: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """Carry over the strings whose amplitudes exceed the threshold."""
        return _select_carryover_by_threshold(
            result, self.carryover_threshold, symmetrize_spin=symmetrize_spin
        )


def diagonalize_fermionic_hamiltonian(
    one_body_tensor: np.ndarray,
    two_body_tensor: np.ndarray,
    bit_array: BitArray | np.ndarray,
    samples_per_batch: int,
    norb: int,
    nelec: tuple[int, int],
    *,
    num_batches: int = 1,
    energy_tol: float = 1e-8,
    occupancies_tol: float = 1e-5,
    max_iterations: int = 100,
    sci_solver: Callable[
        [list[tuple[np.ndarray, np.ndarray]], np.ndarray, np.ndarray, int, tuple[int, int]],
        list[SCIResult],
    ]
    | None = None,
    policy: SubspacePolicy | None = None,
    symmetrize_spin: bool = False,
    max_dim: int | tuple[int, int] | None = None,
    include_configurations: list[int] | tuple[list[int], list[int]] | np.ndarray | None = None,
    initial_occupancies: tuple[np.ndarray, np.ndarray] | None = None,
    carryover_threshold: float | None = None,
    callback: Callable[[list[SCIResult]], None] | None = None,
    seed: int | np.random.Generator | None = None,
) -> SCIResult:
    """Run the sample-based quantum diagonalization (SQD) algorithm.

    Args:
        one_body_tensor: The one-body tensor of the Hamiltonian.
        two_body_tensor: The two-body tensor of the Hamiltonian.
        bit_array: Array of sampled bitstrings, provided as either a Qiskit
            :class:`~qiskit.primitives.BitArray` or a two-dimensional NumPy boolean
            array. Each bitstring should have both the alpha part and beta part
            concatenated together, with the alpha part concatenated on the right-hand
            side, like this: ``[b_N, ..., b_0, a_N, ..., a_0]``.
        samples_per_batch: The number of bitstrings to include in each subsampled batch
            of bitstrings.
        norb: The number of spatial orbitals.
        nelec: The numbers of alpha and beta electrons.
        num_batches: The number of batches to subsample in each configuration recovery
            iteration. This argument indirectly controls the dimensions of the
            diagonalization subspaces. A higher value will yield larger subspace dimensions.
        energy_tol: Numerical tolerance for convergence of the energy. If the change in
            energy between iterations is smaller than this value, then the configuration
            recovery loop will exit, if the occupancies have also converged
            (see the ``occupancies_tol`` argument).
        occupancies_tol: Numerical tolerance for convergence of the average orbital
            occupancies. If the maximum change in absolute value of the average occupancy
            of an orbital between iterations is smaller than this value, then the
            configuration recovery loop will exit, if the energy has also converged
            (see the ``energy_tol`` argument).
        max_iterations: Limit on the number of configuration recovery iterations.
        sci_solver: Selected configuration interaction solver function.

            Inputs:

            - List of pairs (strings_a, strings_b) of arrays of spin-alpha CI strings
              and spin-beta CI strings whose Cartesian product give the basis of the
              subspace in which to perform a diagonalization. Every subspace of an
              iteration is passed in a single call, so that a solver may diagonalize
              them concurrently rather than one after another.
            - One-body tensor of the Hamiltonian.
            - Two-body tensor of the Hamiltonian.
            - The number of spatial orbitals.
            - A pair (n_alpha, n_beta) indicating the numbers of alpha and beta
              electrons.

            Output: List of :class:`SCIResult`, one per subspace passed in.

            A solver that selects its own carryover CI strings can return them in
            :attr:`SCIResult.carryover`, in which case they are used in place of the
            ones this function would select using ``carryover_threshold``. Such a solver
            may leave :attr:`SCIResult.sci_state` as ``None``, so that it never has to
            return the eigenvector. :class:`~qiskit_addon_sqd.trim.TrimPolicy` does
            both.

            A solver may also return fewer results than it was given subspaces, which is
            how a solver that merges its subspaces reports the single diagonalization it
            performed over them.

            See the note below for the semantics that apply when this function is
            invoked collectively from multiple processes.
        policy: The schedule of each iteration: how the sampled configurations become
            subspaces, whether one round of diagonalizations leads to another, which
            result the iteration reports, and which CI strings seed the next one. See
            :class:`SubspacePolicy`.

            Defaults to :class:`StandardPolicy`, which diagonalizes the subsampled
            batches and keeps the best result.
            :class:`~qiskit_addon_sqd.trim.TrimPolicy` screens the batches instead and
            diagonalizes their merged survivors.

            A policy returns the
            subspaces and this function calls ``sci_solver`` with them. The two are
            independent, so any policy works with any solver.
        symmetrize_spin: Whether to always merge spin-alpha and spin-beta CI strings
            into a single list, so that the diagonalization subspace is invariant with
            respect to the exchange of spin alpha with spin beta. This requires the
            numbers of alpha and beta electrons to be equal, as well as a single
            ``max_dim`` shared by both spin sectors; otherwise, an error is raised.
            The invariance ensures that the returned state does not mix components of
            even and odd total spin, but it does *not* guarantee that the state is an
            eigenvector of the total spin operator :math:`S^2`. Note that merging the
            two lists increases the number of CI strings in each spin sector by up to a
            factor of two, so the dimension of the diagonalization subspace can grow by
            up to a factor of four (less when the lists overlap, which is typical). This
            growth is still subject to the ``max_dim`` limit, if one is set.
        max_dim: Limit on the dimension of the spin sectors of the SCI subspace.
            It can be either:

            - A tuple ``(max_dim_a, max_dim_b)`` of integers giving separate limits for the
              spin-alpha and spin-beta sectors. In this case, the dimension of the
              SCI subspace won't exceed ``max_dim_a * max_dim_b``.
            - A single integer specifying a limit that will be used for both the
              spin-alpha and spin-beta sectors. In this case, the dimension of the
              SCI subspace won't exceed ``max_dim**2``.
            - ``None``, in which case no limit is set.

            Note that the dimension limit is set on the spin-sector(s), while the
            full dimension of the SCI subspace is the product of the dimensions of the
            individual spin sectors.
        include_configurations: Configurations to always include in the diagonalization
            subspace. You can specify either a single list of single-spin strings to
            use for both spin sectors, or a pair (alpha_strings, beta_strings) of lists
            of single-spin strings, one for each spin.
        initial_occupancies: Initial guess for the average occupancies of the orbitals.
        carryover_threshold: Threshold for carrying over bitstrings with large CI
            weight from one iteration of configuration recovery to the next.
            All single-spin CI strings associated with configurations whose coefficient
            has absolute value greater than this threshold will be included in the
            diagonalization subspace for the next iteration. A smaller threshold will
            retain more configurations, leading to a larger subspace and hence a more
            costly diagonalization. Defaults to ``1e-4``.

            This is a setting of the default policy, which chooses the carryover by
            thresholding amplitudes. Another policy chooses it in its own way, so passing
            both this and ``policy`` raises ``ValueError``; pass the threshold to
            :class:`StandardPolicy` instead if you are constructing one explicitly.
        callback: A callback function to be called after each configuration recovery
            iteration. The function will be passed the output of the sci_solver
            function, which is a list of (energy, sci_state, occupancies) triplets,
            where each triplet contains the result of a diagonalization.
        seed: A seed for the pseudorandom number generator.

    Returns:
        The estimate of the energy and the SCI state with that energy.

    Note:
        This function supports collective multi-process (SPMD) execution using
        MPI, with a single thread controlling each process
        (``MPI_THREAD_FUNNELED`` or lower). When it is invoked collectively from
        all processes, the arguments must agree across processes, and the
        following semantics apply:

        - The ``sci_solver`` step is the only collective operation: all
          processes participate in it, so an ``sci_solver`` implementation can
          distribute work across every process. The remaining steps of the
          configuration-recovery loop (preparing the CI strings, processing the
          diagonalization results, and checking convergence) have no distributed
          implementation and are performed on the control process alone.
        - The ``callback``, if provided, is invoked on the control process only.
        - The return value is the same on every process: the final result is
          broadcast from the control process to all ranks.
        - Only the results that a collective ``sci_solver`` returns on the control
          process are consumed; what it returns on the other processes is ignored
          entirely and need not be meaningful. This lets a solver leave each
          diagonalization's energy, state and occupancies on whichever process
          computed them, rather than gathering them onto every rank. A solver that
          does so must still ensure the *control* process receives a result for
          every subspace, gathering across process groups if it distributed them.

        Whether the calling program should be launched under MPI depends on the
        ``sci_solver`` in use. A collective ``sci_solver``, as described above,
        expects the program to run collectively across all processes. Other
        implementations require the calling program to run as a single process,
        outside any MPI/SPMD environment.  Some of these implementations might
        manage their own parallelism internally, for example, by launching
        ``mpirun`` themselves. The default ``sci_solver`` is not distributed
        and falls into this latter category.

        A collective ``sci_solver`` implementation is not required to raise an
        exception on a single process when it encounters an error; instead, it
        can follow fail-stop semantics for the execution context as a whole,
        aborting all processes collectively. (This describes what such an
        implementation is permitted to do, not the behavior of the default
        ``sci_solver``.)

        Because every subspace of an iteration arrives in one ``sci_solver`` call,
        a collective implementation may divide the processes into groups and
        diagonalize several subspaces at once, one group per subspace, rather than
        giving every subspace all of the processes in turn. How to do so is the
        implementation's own concern: this package does not divide the processes,
        and nothing in its interface describes them. An implementation that does
        divide them is responsible for confining its collective operations to the
        group that is performing a given diagonalization, and must not use
        :func:`qiskit_addon_sqd.processes.broadcast` or
        :func:`qiskit_addon_sqd.processes.barrier` for that purpose, as those
        operate over every process rather than over a group.

        How many times per iteration the solver is called, and with how many
        subspaces, is determined by ``policy``. With
        :class:`~qiskit_addon_sqd.trim.TrimPolicy`, for instance, it is called twice:
        once with every batch, and once with the single merged subspace. A solver that
        divides the processes therefore needs only one such division, which it can build
        on the first call and reuse, rather than rebuilding it every time. Note also that
        the number of subspaces is the same on every process, since they are broadcast
        before the solver is called. That matters because dividing the processes is
        itself a collective operation: every process must take part and agree on how the
        division is made. A solver may safely base that decision on the number of
        subspaces it received, but not on anything that could differ between processes.
    """
    if max_iterations < 1:
        raise ValueError("Maximum number of iterations must be at least 1.")

    n_alpha, n_beta = nelec
    if symmetrize_spin and n_alpha != n_beta:
        raise ValueError(
            "Spin symmetrization is only possible if the numbers of alpha and beta "
            f"electrons are equal. Instead, got {n_alpha} and {n_beta}."
        )

    if max_dim is None:
        max_dim_a = max_dim_b = None
    elif isinstance(max_dim, tuple):
        max_dim_a, max_dim_b = max_dim
    else:
        max_dim_a = max_dim_b = max_dim
    if symmetrize_spin and max_dim_a != max_dim_b:
        raise ValueError(
            "When requesting spin symmetrization, the maximum dimension must be "
            "the same for both spin alpha and spin beta. "
            f"Instead, got {max_dim_a} and {max_dim_b}"
        )

    if include_configurations is None:
        include_a: list[int] | np.ndarray = np.array([], dtype=int)
        include_b: list[int] | np.ndarray = np.array([], dtype=int)
    elif isinstance(include_configurations, tuple):
        include_a, include_b = include_configurations
    else:
        include_a = include_configurations
        include_b = include_configurations

    rng = np.random.default_rng(seed)
    current_occupancies = initial_occupancies
    best_result = None
    current_result = None
    if sci_solver is None:
        sci_solver = solve_sci_batch
    if policy is None:
        policy = StandardPolicy(
            carryover_threshold=1e-4 if carryover_threshold is None else carryover_threshold
        )
    elif carryover_threshold is not None:
        raise ValueError(
            "carryover_threshold applies only to the default policy, which chooses the "
            "carryover by thresholding amplitudes. A policy chooses it in its own way, so "
            "pass the threshold to the policy instead. Got "
            f"carryover_threshold={carryover_threshold}."
        )

    include_a = np.unique(include_a)
    include_b = np.unique(include_b)
    carryover_strings_a = np.array([], dtype=np.int64)
    carryover_strings_b = np.array([], dtype=np.int64)

    # Convert the samples into bitstring and probability arrays
    if isinstance(bit_array, BitArray):
        raw_bitstrings, raw_probs = bit_array_to_arrays(bit_array)
    else:
        raw_bitstrings, counts = np.unique(bit_array, axis=0, return_counts=True)
        raw_probs = counts / len(bit_array)

    # Bundle the loop-invariant configuration once, so the per-iteration helper
    # calls only need to pass the values that change between iterations.
    config = _LoopConfig(
        raw_bitstrings=raw_bitstrings,
        raw_probs=raw_probs,
        n_alpha=n_alpha,
        n_beta=n_beta,
        samples_per_batch=samples_per_batch,
        num_batches=num_batches,
        norb=norb,
        symmetrize_spin=symmetrize_spin,
        include_a=include_a,
        include_b=include_b,
        max_dim_a=max_dim_a,
        max_dim_b=max_dim_b,
        energy_tol=energy_tol,
        occupancies_tol=occupancies_tol,
        rng=rng,
    )

    # Run configuration recovery loop
    #
    # In distributed (SPMD) mode, the control process orchestrates the loop and
    # performs procedures that do not have a distributed implementation, while
    # all ranks participate in sci_solver calls for collective operations.
    for iteration in range(max_iterations):
        # Ask the policy for the subspaces to diagonalize, and enforce the caller's
        # constraints on their shape. The policy has no distributed implementation, so
        # only the control process runs it; the result is then broadcast to all ranks for
        # the MPI collective operations in sci_solver.
        if is_control_process():
            bitstrings, probs = _recover_or_postselect(config, current_occupancies)
            request = SubspaceRequest(
                bitstrings=bitstrings,
                probabilities=probs,
                carryover_strings_a=carryover_strings_a,
                carryover_strings_b=carryover_strings_b,
                norb=norb,
                nelec=nelec,
                samples_per_batch=samples_per_batch,
                num_batches=num_batches,
                rng=rng,
                iteration=iteration,
                symmetrize_spin=symmetrize_spin,
            )
            ci_strings = [
                _apply_shape_constraints(config, strs_a, strs_b)
                for strs_a, strs_b in policy.prepare_subspaces(request)
            ]
        else:
            ci_strings = None
        ci_strings = broadcast(ci_strings, root=0)

        # Run the diagonalizations, one round at a time. Every process enters this loop
        # and must agree on how many times, because sci_solver is collective. Only the
        # control process knows whether the policy asked for another round, so that
        # decision is broadcast and every rank breaks on the same value; deciding it
        # locally would let one rank enter a collective the others had left.
        for round_index in range(_MAX_ROUNDS):
            results = sci_solver(ci_strings, one_body_tensor, two_body_tensor, norb, nelec)
            if is_control_process():
                refined = policy.refine(results, round_index)
                if refined is not None:
                    refined = [
                        _apply_shape_constraints(config, strs_a, strs_b)
                        for strs_a, strs_b in refined
                    ]
            else:
                refined = None
            refined = broadcast(refined, root=0)
            if refined is None:
                break
            ci_strings = refined
        else:
            raise ValueError(
                "The policy asked for another round of diagonalizations after "
                f"{_MAX_ROUNDS} of them, which is more than any schedule should need. "
                "Its refine method must eventually return None. Got "
                f"policy={policy!r}."
            )

        # Call callback function if provided (only on the control process)
        if callback is not None and is_control_process():
            callback(results)

        # Process results: update best result, check convergence, compute
        # carryover. This has no distributed implementation, so only the control
        # process performs it; the resulting state is then broadcast to all ranks.
        if is_control_process():
            state = _process_sci_results(
                config,
                policy,
                results,
                best_result,
                current_result,
                current_occupancies,
            )
        else:
            state = None
        state = broadcast(state, root=0)

        best_result = state.best_result
        if state.converged:
            break
        current_result = state.current_result
        current_occupancies = state.current_occupancies
        carryover_strings_a = state.carryover_strings_a
        carryover_strings_b = state.carryover_strings_b

    # best_result is not None because there must have been at least one iteration
    return cast(SCIResult, best_result)


def _unique_with_order_preserved(vals: np.ndarray) -> np.ndarray:
    """Return unique values of an array while preserving the original order."""
    _, indices = np.unique(vals, return_index=True)
    indices.sort()
    return vals[indices]


def _recover_or_postselect(
    config: _LoopConfig,
    current_occupancies: tuple[np.ndarray, np.ndarray] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Postselect or recover the configurations an iteration draws its subspaces from.

    This step is the same whatever schedule builds the subspaces, so it stays outside
    the subspace policy.
    """
    if current_occupancies is None:
        # If we don't have average orbital occupancy information, simply postselect
        # bitstrings with the correct numbers of spin-up and spin-down electrons
        bitstrings, probs = postselect_by_hamming_right_and_left(
            config.raw_bitstrings,
            config.raw_probs,
            hamming_right=config.n_alpha,
            hamming_left=config.n_beta,
        )
        if not bitstrings.size:
            raise ValueError(
                "The input bit array did not contain any valid bitstrings. "
                "Either pass a bit array that contains at least one valid bitstring "
                "(with the correct right and left Hamming weights), or specify a value for initial_occupancies."
            )
    else:
        # If we do have average orbital occupancy information, use it to refine the
        # full set of noisy configurations
        bitstrings, probs = recover_configurations(
            config.raw_bitstrings,
            config.raw_probs,
            current_occupancies,
            config.n_alpha,
            config.n_beta,
            rand_seed=config.rng,
        )
    return bitstrings, probs


def batch_to_ci_strings(
    batch: np.ndarray,
    norb: int,
    carryover_strings_a: np.ndarray | None = None,
    carryover_strings_b: np.ndarray | None = None,
    *,
    symmetrize_spin: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a batch of bitstrings into a pair of per-spin CI string arrays.

    The strings of each spin sector are returned in descending order of the number of
    times they were sampled, with any carryover strings ahead of them. That order is
    what a later truncation to a maximum dimension keeps, so it matters.

    This is the step a subspace policy needs in order to turn subsampled bitstrings into
    a subspace, and it is exposed so that a policy does not have to reimplement it.

    Args:
        batch: A 2D array of ``bool`` bitstrings, one per row, with the alpha part
            concatenated on the right-hand side, like this:
            ``[b_N, ..., b_0, a_N, ..., a_0]``.
        norb: The number of spatial orbitals.
        carryover_strings_a: Spin-alpha CI strings to place ahead of the sampled ones,
            in the order they should be kept. Defaults to none.
        carryover_strings_b: The same for spin beta. When ``symmetrize_spin`` is set,
            this is ignored, since the two sectors hold the same strings.
        symmetrize_spin: Whether to merge the two spin sectors into a single list of
            strings, used for both. The merge happens before the strings are ranked, so
            that the ranking is over both sectors at once rather than within each.

    Returns:
        The spin-alpha and spin-beta CI string arrays. When ``symmetrize_spin`` is set,
        the two are the same array.
    """
    # Get the single-spin bitstrings and counts.
    samples_a, counts_a = np.unique(
        bitstring_matrix_to_integers(batch[:, norb:]), return_counts=True
    )
    samples_b, counts_b = np.unique(
        bitstring_matrix_to_integers(batch[:, :norb]), return_counts=True
    )
    empty = np.array([], dtype=np.int64)
    if carryover_strings_a is None:
        carryover_strings_a = empty
    if carryover_strings_b is None:
        carryover_strings_b = empty

    if symmetrize_spin:
        # Merge the bitstrings for spin alpha and spin beta.
        samples = np.concatenate((samples_a, samples_b))
        counts = np.concatenate((counts_a, counts_b))
        # Sort the single-spin bitstrings in descending order by marginal probability.
        samples = samples[np.argsort(counts)[::-1]]
        # Note that in this case, carryover_strings_a and carryover_strings_b are equal.
        strs_a = strs_b = np.concatenate((carryover_strings_a, samples))
    else:
        # Sort the single-spin bitstrings in descending order by marginal probability.
        samples_a = samples_a[np.argsort(counts_a)[::-1]]
        samples_b = samples_b[np.argsort(counts_b)[::-1]]
        strs_a = np.concatenate((carryover_strings_a, samples_a))
        strs_b = np.concatenate((carryover_strings_b, samples_b))
    return strs_a, strs_b


def _apply_shape_constraints(
    config: _LoopConfig,
    strings_a: np.ndarray,
    strings_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Enforce the caller's constraints on the shape of a subspace.

    A subspace policy chooses which strings a subspace holds; these constraints are the
    caller's, so the loop applies them to whatever the policy returns. They are the
    explicitly requested configurations, which go into every subspace, and the maximum
    dimension of each spin sector.
    """
    if config.symmetrize_spin:
        # Prioritize explicitly requested bitstrings, then everything the policy chose.
        # In this case, max_dim_a and max_dim_b are equal.
        strs = np.concatenate((config.include_a, config.include_b, strings_a))
        strs_a = strs_b = _unique_with_order_preserved(strs)[: config.max_dim_a]
    else:
        strs_a = np.concatenate((config.include_a, strings_a))
        strs_b = np.concatenate((config.include_b, strings_b))
        # Truncate bitstrings to the maximum dimension.
        strs_a = _unique_with_order_preserved(strs_a)[: config.max_dim_a]
        strs_b = _unique_with_order_preserved(strs_b)[: config.max_dim_b]
    strs_a = np.sort(strs_a)
    strs_b = np.sort(strs_b)
    return strs_a, strs_b


def _process_sci_results(
    config: _LoopConfig,
    policy: SubspacePolicy,
    results: list[SCIResult],
    best_result: SCIResult | None,
    current_result: SCIResult | None,
    current_occupancies: tuple[np.ndarray, np.ndarray] | None,
) -> _IterationState:
    """Process the diagonalization results of one configuration recovery iteration.

    This is the second half of one configuration recovery iteration: it updates the
    best result seen so far, checks for convergence, and (when not converged) computes
    the carryover strings for the next iteration.
    """
    # Let the policy choose which of the final round's results the iteration reports.
    best_result_in_batch = policy.select_result(results)

    # Check if the energy is the lowest seen so far
    if best_result is None or best_result_in_batch.energy < best_result.energy:
        best_result = best_result_in_batch

    # Check convergence
    if (
        current_result is not None
        and abs(current_result.energy - best_result_in_batch.energy) < config.energy_tol
        and np.linalg.norm(
            # Reason for type: ignore: mypy thinks current_occupancies can be None
            np.ravel(current_occupancies) - np.ravel(best_result_in_batch.orbital_occupancies),  # type: ignore
            ord=np.inf,
        )
        < config.occupancies_tol
    ):
        # Converged: carry the pre-existing state through unchanged, since the caller
        # will stop iterating and these fields will not be used again.
        return _IterationState(
            best_result=best_result,
            current_result=current_result,
            current_occupancies=cast("tuple[np.ndarray, np.ndarray]", current_occupancies),
            carryover_strings_a=np.array([], dtype=np.int64),
            carryover_strings_b=np.array([], dtype=np.int64),
            converged=True,
        )
    current_result = best_result_in_batch
    current_occupancies = current_result.orbital_occupancies

    # Let the policy choose the strings that seed the next iteration. A solver that chose
    # them itself reports them in the result; whether to defer to that is the policy's
    # decision, and the ones built into this package do. Either way the strings still have
    # to be merged across spin sectors, which is a constraint on the shape of the next
    # subspace rather than part of the choosing.
    carryover_strings_a, carryover_strings_b = policy.select_carryover(
        current_result, symmetrize_spin=config.symmetrize_spin
    )
    if config.symmetrize_spin:
        # Idempotent, so a policy that already merged the sectors is unaffected.
        carryover_strings_a, carryover_strings_b = _symmetrize_carryover(
            carryover_strings_a, carryover_strings_b
        )

    return _IterationState(
        best_result=best_result,
        current_result=current_result,
        current_occupancies=current_occupancies,
        carryover_strings_a=carryover_strings_a,
        carryover_strings_b=carryover_strings_b,
        converged=False,
    )


def _symmetrize_carryover(
    carryover_strings_a: np.ndarray,
    carryover_strings_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Merge the two spin sectors' carryover strings into one list used for both.

    The order of the input is preserved, so a caller that has already ranked its strings
    keeps that ranking.
    """
    merged = _unique_with_order_preserved(
        np.concatenate((carryover_strings_a, carryover_strings_b))
    )
    return merged, merged


def _select_carryover_by_threshold(
    result: SCIResult,
    threshold: float,
    *,
    symmetrize_spin: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Choose the carryover strings by thresholding the eigenvector amplitudes.

    A solver that selected the carryover itself has already done this work, so its choice
    is taken as-is. Note that ``threshold`` does not apply in that case: the solver, not
    this function, decided how many determinants survive.

    The strings are returned in descending order of marginal weight, which is the order a
    later truncation keeps. When ``symmetrize_spin`` is set, the two sectors are ranked
    together rather than separately, so that the ranking is over both at once.

    Raises:
        ValueError: The result has neither an SCI state nor a carryover.
    """
    if result.carryover is not None:
        return result.carryover

    sci_state = result.sci_state
    if sci_state is None:
        raise ValueError(
            "The solver returned a result with neither an SCI state nor a carryover, so "
            "there is nothing to build the next iteration's subspace from. A solver "
            "that does not return an SCI state must set SCIResult.carryover."
        )
    flattened = sci_state.amplitudes.reshape(-1)
    absolute_vals = np.abs(flattened)
    indices = np.argsort(absolute_vals)
    carryover_index = np.searchsorted(absolute_vals, threshold, sorter=indices)
    carryover_indices = indices[carryover_index:]
    _, n_strings_b = sci_state.amplitudes.shape
    alpha_indices, beta_indices = np.divmod(carryover_indices, n_strings_b)
    alpha_indices = np.unique(alpha_indices)
    beta_indices = np.unique(beta_indices)
    carryover_strings_a = sci_state.ci_strs_a[alpha_indices]
    carryover_strings_b = sci_state.ci_strs_b[beta_indices]
    # Sort carryover strings in descending order by marginal weight
    weights_a = np.sum(np.abs(sci_state.amplitudes[alpha_indices]) ** 2, axis=1)
    weights_b = np.sum(np.abs(sci_state.amplitudes[:, beta_indices]) ** 2, axis=0)
    if symmetrize_spin:
        # Rank the two sectors together rather than separately, then merge. Ranking
        # before merging is what makes this one operation: concatenating two separately
        # ranked lists would interleave them differently.
        carryover_strings = np.concatenate((carryover_strings_a, carryover_strings_b))
        weights = np.concatenate((weights_a, weights_b))
        carryover_strings = carryover_strings[np.argsort(weights)[::-1]]
        return _symmetrize_carryover(carryover_strings, np.array([], dtype=np.int64))
    return (
        carryover_strings_a[np.argsort(weights_a)[::-1]],
        carryover_strings_b[np.argsort(weights_b)[::-1]],
    )


def solve_sci_batch(
    ci_strings: list[tuple[np.ndarray, np.ndarray]],
    one_body_tensor: np.ndarray,
    two_body_tensor: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    *,
    spin_sq: float | None = None,
    **kwargs,
) -> list[SCIResult]:
    """Diagonalize Hamiltonian in subspaces.

    Args:
        ci_strings: List of pairs (strings_a, strings_b) of arrays of spin-alpha CI
            strings and spin-beta CI strings whose Cartesian product give the basis of
            the subspace in which to perform a diagonalization.
        one_body_tensor: The one-body tensor of the Hamiltonian.
        two_body_tensor: The two-body tensor of the Hamiltonian.
        norb: The number of spatial orbitals.
        nelec: The numbers of alpha and beta electrons.
        spin_sq: Target value for the total spin squared for the ground state.
            If ``None``, no spin will be imposed.
        **kwargs: Keyword arguments to pass to `pyscf.fci.selected_ci.kernel_fixed_space <https://pyscf.org/pyscf_api_docs/pyscf.fci.html#pyscf.fci.selected_ci.kernel_fixed_space>`_

    Returns:
        The results of the diagonalizations in the subspaces given by ci_strings.
    """
    return [
        solve_sci(
            ci_strs,
            one_body_tensor,
            two_body_tensor,
            norb=norb,
            nelec=nelec,
            spin_sq=spin_sq,
            **kwargs,
        )
        for ci_strs in ci_strings
    ]


def solve_sci(
    ci_strings: tuple[np.ndarray, np.ndarray],
    one_body_tensor: np.ndarray,
    two_body_tensor: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    *,
    spin_sq: float | None = None,
    **kwargs,
) -> SCIResult:
    """Diagonalize Hamiltonian in subspace defined by CI strings.

    Args:
        ci_strings: Pair (strings_a, strings_b) of arrays of spin-alpha CI
            strings and spin-beta CI strings whose Cartesian product give the basis of
            the subspace in which to perform a diagonalization.
        one_body_tensor: The one-body tensor of the Hamiltonian.
        two_body_tensor: The two-body tensor of the Hamiltonian.
        norb: The number of spatial orbitals.
        nelec: The numbers of alpha and beta electrons.
        spin_sq: Target value for the total spin squared for the ground state.
            If ``None``, no spin will be imposed.
        **kwargs: Keyword arguments to pass to `pyscf.fci.selected_ci.kernel_fixed_space <https://pyscf.org/pyscf_api_docs/pyscf.fci.html#pyscf.fci.selected_ci.kernel_fixed_space>`_

    Returns:
        The diagonalization result.
    """
    norb, _ = one_body_tensor.shape

    myci = fci.selected_ci.SelectedCI()
    if spin_sq is not None:
        myci = fci.addons.fix_spin_(myci, ss=spin_sq)

    # The energy returned from this function is not guaranteed to be
    # the energy of the returned wavefunction when the spin^2 deviates
    # from the value requested. We will calculate the energy from the
    # RDMs below and ignore this value to be safe.
    _, sci_vec = fci.selected_ci.kernel_fixed_space(
        myci, one_body_tensor, two_body_tensor, norb, nelec, ci_strs=ci_strings, **kwargs
    )
    # Calculate the average occupancy of each orbital
    dm1s = myci.make_rdm1s(sci_vec, norb, nelec)
    occupancies = (np.diagonal(dm1s[0]), np.diagonal(dm1s[1]))
    # Calculate energy from RDMs
    dm1 = myci.make_rdm1(sci_vec, norb, nelec)
    dm2 = myci.make_rdm2(sci_vec, norb, nelec)
    energy = np.einsum("pr,pr->", dm1, one_body_tensor) + 0.5 * np.einsum(
        "prqs,prqs->", dm2, two_body_tensor
    )
    # Construct SCIState
    sci_state = SCIState(
        amplitudes=np.array(sci_vec),
        ci_strs_a=sci_vec._strs[0],
        ci_strs_b=sci_vec._strs[1],
        norb=norb,
        nelec=nelec,
    )
    # Return result
    return SCIResult(energy, sci_state, orbital_occupancies=occupancies, rdm1=dm1, rdm2=dm2)


def solve_fermion(
    bitstring_matrix: tuple[np.ndarray, np.ndarray] | np.ndarray,
    /,
    hcore: np.ndarray,
    eri: np.ndarray,
    *,
    open_shell: bool = False,
    spin_sq: float | None = None,
    shift: float = 0.1,
    **kwargs,
) -> tuple[float, SCIState, tuple[np.ndarray, np.ndarray], float]:
    """Approximate the ground state given molecular integrals and a set of electronic configurations.

    Args:
        bitstring_matrix: A set of configurations defining the subspace onto which the Hamiltonian
            will be projected and diagonalized.

            This may be specified in two ways:

            - Bitstring matrix: A 2D ``numpy.ndarray`` of ``bool`` values, where each row represents a bitstring.
              The spin-up configurations should occupy column indices ``(N, N/2]``, and the spin-down configurations
              should occupy column indices ``(N/2, 0]``, where ``N`` is the number of qubits.

            - CI strings: A tuple of two sequences containing integer representations of spin-up and spin-down
              determinants, respectively. The expected format is ``([a_str_0, ..., a_str_N], [b_str_0, ..., b_str_M])``.

        hcore: Core Hamiltonian matrix representing single-electron integrals
        eri: Electronic repulsion integrals representing two-electron integrals
        open_shell: A flag specifying whether configurations from the left and right
            halves of the bitstrings should be kept separate. If ``False``, CI strings
            from the left and right halves of the bitstrings are combined into a single
            set of unique configurations and used for both the alpha and beta subspaces.
        spin_sq: Target value for the total spin squared for the ground state, :math:`S^2 = s(s + 1)`.
            If ``None``, no spin will be imposed.
        shift: Level shift for states which have different spin. :math:`(H + shift * S^2)|ψ> = E|ψ>`
        **kwargs: Keyword arguments to pass to `pyscf.fci.selected_ci.kernel_fixed_space <https://pyscf.org/pyscf_api_docs/pyscf.fci.html#pyscf.fci.selected_ci.kernel_fixed_space>`_

    Returns:
        - Minimum energy from SCI calculation
        - The SCI ground state
        - Tuple containing orbital occupancies for spin-up and spin-down orbitals. Formatted as: ``(array([occ_a_0, ..., occ_a_N]), array([occ_b_0, ..., occ_b_N]))``
        - Expectation value of spin-squared

    """
    # Format inputs
    if isinstance(bitstring_matrix, tuple):
        ci_strs = bitstring_matrix
    else:
        ci_strs = bitstring_matrix_to_ci_strs(bitstring_matrix, open_shell=open_shell)
    ci_strs = _check_ci_strs(ci_strs)

    # Get hamming weights of each half of the first CI str. All CI strs should share the same hamming weight
    num_up = format(ci_strs[0][0], "b").count("1")
    num_dn = format(ci_strs[1][0], "b").count("1")

    # Number of molecular orbitals
    norb = hcore.shape[0]
    # Call the projection + eigenstate finder
    myci = fci.selected_ci.SelectedCI()
    if spin_sq is not None:
        myci = fci.addons.fix_spin_(myci, ss=spin_sq, shift=shift)
    # The energy returned from this function is not guaranteed to be
    # the energy of the returned wavefunction when the spin^2 deviates
    # from the value requested. We will calculate the energy from the
    # RDMs below and ignore this value to be safe.
    _, sci_vec = fci.selected_ci.kernel_fixed_space(
        myci,
        hcore,
        eri,
        norb,
        (num_up, num_dn),
        ci_strs,
        **kwargs,
    )

    # Calculate the avg occupancy of each orbital
    dm1s = myci.make_rdm1s(sci_vec, norb, (num_up, num_dn))
    avg_occupancy = (np.diagonal(dm1s[0]), np.diagonal(dm1s[1]))

    # Calculate energy from RDMs
    dm1 = myci.make_rdm1(sci_vec, norb, (num_up, num_dn))
    dm2 = myci.make_rdm2(sci_vec, norb, (num_up, num_dn))
    e_sci = np.einsum("pr,pr->", dm1, hcore) + 0.5 * np.einsum("prqs,prqs->", dm2, eri)

    # Compute total spin
    spin_squared = myci.spin_square(sci_vec, norb, (num_up, num_dn))[0]

    # Convert the PySCF SCIVector to internal format. We access a private field here,
    # so we assert that we expect the SCIVector output from kernel_fixed_space to
    # have its _strs field populated with alpha and beta strings.
    assert isinstance(sci_vec._strs[0], np.ndarray) and isinstance(sci_vec._strs[1], np.ndarray)
    assert sci_vec.shape == (len(sci_vec._strs[0]), len(sci_vec._strs[1]))
    sci_state = SCIState(
        amplitudes=np.array(sci_vec),
        ci_strs_a=sci_vec._strs[0],
        ci_strs_b=sci_vec._strs[1],
        norb=norb,
        nelec=(num_up, num_dn),
    )

    return e_sci, sci_state, avg_occupancy, spin_squared


@deprecate_func(**_ORBITAL_OPTIMIZATION_DEPRECATION)
def optimize_orbitals(
    bitstring_matrix: tuple[np.ndarray, np.ndarray] | np.ndarray,
    /,
    hcore: np.ndarray,
    eri: np.ndarray,
    k_flat: np.ndarray,
    *,
    open_shell: bool = False,
    spin_sq: float = 0.0,
    num_iters: int = 10,
    num_steps_grad: int = 10_000,
    learning_rate: float = 0.01,
    **kwargs,
) -> tuple[float, np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Optimize orbitals to produce a minimal ground state.

    The process involves iterating over 3 steps:

    For ``num_iters`` iterations:
        - Rotate the integrals with respect to the parameters, ``k_flat``
        - Diagonalize and approximate the groundstate energy and wavefunction amplitudes
        - Optimize ``k_flat`` using gradient descent and the wavefunction
          amplitudes found in Step 2

    Refer to `Sec. II A 4 <https://arxiv.org/pdf/2405.05068>`_ for more detailed
    discussion on this orbital optimization technique.

    Args:
        bitstring_matrix: A set of configurations defining the subspace onto which the Hamiltonian
            will be projected and diagonalized.

            This may be specified in two ways:

            - Bitstring matrix: A 2D ``numpy.ndarray`` of ``bool`` values, where each row represents a bitstring.
              The spin-up configurations should occupy column indices ``(N, N/2]``, and the spin-down configurations
              should occupy column indices ``(N/2, 0]``, where ``N`` is the number of qubits.

            - CI strings: A tuple of two sequences containing integer representations of spin-up and spin-down
              determinants, respectively. The expected format is ``([a_str_0, ..., a_str_N], [b_str_0, ..., b_str_M])``.

        hcore: Core Hamiltonian matrix representing single-electron integrals
        eri: Electronic repulsion integrals representing two-electron integrals
        k_flat: 1D array defining the orbital transform, ``K``. The array should specify the upper
            triangle of the anti-symmetric transform operator in row-major order, excluding the diagonal.
        open_shell: A flag specifying whether configurations from the left and right
            halves of the bitstrings should be kept separate. If ``False``, CI strings
            from the left and right halves of the bitstrings are combined into a single
            set of unique configurations and used for both the alpha and beta subspaces.
        spin_sq: Target value for the total spin squared for the ground state
        num_iters: The number of iterations of orbital optimization to perform
        num_steps_grad: The number of steps of gradient descent to perform
            during each optimization iteration
        learning_rate: The learning rate to use during gradient descent
        **kwargs: Keyword arguments to pass to `pyscf.fci.selected_ci.kernel_fixed_space <https://pyscf.org/pyscf_api_docs/pyscf.fci.html#pyscf.fci.selected_ci.kernel_fixed_space>`_

    Returns:
        - The groundstate energy found during the last optimization iteration
        - An optimized 1D array defining the orbital transform
        - Tuple containing orbital occupancies for spin-up and spin-down orbitals. Formatted as: ``(array([occ_a_0, ..., occ_a_N]), array([occ_b_0, ..., occ_b_N]))``

    """
    norb = hcore.shape[0]
    num_params = (norb**2 - norb) // 2
    if len(k_flat) != num_params:
        raise ValueError(
            f"k_flat must specify the upper triangle of the transform matrix. k_flat length is {len(k_flat)}. "
            f"Expected {num_params}."
        )
    if isinstance(bitstring_matrix, tuple):
        ci_strs = bitstring_matrix
    else:
        ci_strs = bitstring_matrix_to_ci_strs(bitstring_matrix, open_shell=open_shell)
    ci_strs = _check_ci_strs(ci_strs)

    num_up = format(ci_strs[0][0], "b").count("1")
    num_dn = format(ci_strs[1][0], "b").count("1")

    # TODO: Need metadata showing the optimization history
    ## hcore and eri in physicist ordering
    k_flat = k_flat.copy()
    eri_phys = np.asarray(eri.transpose(0, 2, 3, 1), order="C")  # physicist ordering
    for _ in range(num_iters):
        # Rotate integrals
        hcore_rot, eri_rot = rotate_integrals(hcore, eri_phys, k_flat)
        eri_rot_chem = np.asarray(eri_rot.transpose(0, 3, 1, 2), order="C")  # chemist ordering

        # Solve for ground state with respect to optimized integrals
        myci = fci.selected_ci.SelectedCI()
        myci = fci.addons.fix_spin_(myci, ss=spin_sq)
        e_qsci, amplitudes = fci.selected_ci.kernel_fixed_space(
            myci,
            hcore_rot,
            eri_rot_chem,
            norb,
            (num_up, num_dn),
            ci_strs,
            **kwargs,
        )

        # Generate the one and two-body reduced density matrices from latest wavefunction amplitudes
        dm1, dm2_chem = myci.make_rdm12(amplitudes, norb, (num_up, num_dn))
        dm2 = np.asarray(dm2_chem.transpose(0, 2, 3, 1), order="C")
        dm1a, dm1b = myci.make_rdm1s(amplitudes, norb, (num_up, num_dn))
        avg_occupancy = (np.diagonal(dm1a), np.diagonal(dm1b))

        # TODO: Expose the momentum parameter as an input option
        # Optimize the basis rotations
        _optimize_orbitals_sci(
            k_flat, learning_rate, 0.9, num_steps_grad, dm1, dm2, hcore, eri_phys
        )

    return e_qsci, k_flat, avg_occupancy


@deprecate_func(**_ORBITAL_OPTIMIZATION_DEPRECATION)
def rotate_integrals(
    hcore: np.ndarray, eri: np.ndarray, k_flat: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    r"""Perform a similarity transform on the integrals.

    The transformation is described as:

    .. math::

       \hat{\widetilde{H}} = \hat{U^{\dagger}}(k)\hat{H}\hat{U}(k)

    For more information on how :math:`\hat{U}` and :math:`\hat{U^{\dagger}}` are generated from ``k_flat``
    and applied to the one- and two-body integrals, refer to `Sec. II A 4 <https://arxiv.org/pdf/2405.05068>`_.

    Args:
        hcore: Core Hamiltonian matrix representing single-electron integrals
        eri: Electronic repulsion integrals representing two-electron integrals
        k_flat: 1D array defining the orbital transform, ``K``. The array should specify the upper
            triangle of the anti-symmetric transform operator in row-major order, excluding the diagonal.

    Returns:
        - The rotated core Hamiltonian matrix
        - The rotated ERI matrix

    """
    norb = hcore.shape[0]
    num_params = (norb**2 - norb) // 2
    if len(k_flat) != num_params:
        raise ValueError(
            f"k_flat must specify the upper triangle of the transform matrix. k_flat length is {len(k_flat)}. "
            f"Expected {num_params}."
        )
    K = _antisymmetric_matrix_from_upper_tri(k_flat, norb)
    U = LA.expm(K)
    hcore_rot = np.matmul(np.transpose(U), np.matmul(hcore, U))
    eri_rot = np.einsum("pqrs, pi, qj, rk, sl->ijkl", eri, U, U, U, U, optimize=True)

    return np.array(hcore_rot), np.array(eri_rot)


def bitstring_matrix_to_ci_strs(
    bitstring_matrix: np.ndarray, open_shell: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Convert bitstrings (rows) in a ``bitstring_matrix`` into integer representations of determinants.

    This function separates each bitstring in ``bitstring_matrix`` in half, translates them into
    integer representations, and finally appends them to their respective (spin-up or spin-down) lists.
    Those lists are sorted and output from this function.

    Args:
        bitstring_matrix: A 2D array of ``bool`` representations of bit
            values such that each row represents a single bitstring
        open_shell: A flag specifying whether unique configurations from the left and right
            halves of the bitstrings should be kept separate. If ``False``, configurations
            from the left and right halves of the bitstrings are combined into a single
            set of unique configurations. That combined set will be returned for both the left
            and right bitstrings.

    Returns:
        A length-2 tuple of determinant lists representing the right (spin-up) and left (spin-down)
        halves of the bitstrings, respectively.

    """
    norb = bitstring_matrix.shape[1] // 2

    ci_strs_left = np.unique(bitstring_matrix_to_integers(bitstring_matrix[:, :norb]))
    ci_strs_right = np.unique(bitstring_matrix_to_integers(bitstring_matrix[:, norb:]))

    if not open_shell:
        ci_strs_left = ci_strs_right = np.union1d(ci_strs_left, ci_strs_right)

    return ci_strs_right, ci_strs_left


def enlarge_batch_from_transitions(
    bitstring_matrix: np.ndarray, transition_operators: np.ndarray
) -> np.ndarray:
    """Apply the set of transition operators to the configurations represented in ``bitstring_matrix``.

    Args:
        bitstring_matrix: A 2D array of ``bool`` representations of bit
            values such that each row represents a single bitstring.
        transition_operators: A 1D or 2D array ``I``, ``+``, ``-``, and ``n`` strings
            representing the action of the identity, creation, annihilation, or number operators.
            Each row represents a transition operator.

    Returns:
        Bitstring matrix representing the augmented set of electronic configurations after applying
        the excitation operators.

    """
    diag, create, annihilate = _transition_str_to_bool(transition_operators)

    bitstring_matrix_augmented, mask = apply_excitations(bitstring_matrix, diag, create, annihilate)

    bitstring_matrix_augmented = bitstring_matrix_augmented[mask]

    return np.array(bitstring_matrix_augmented)


def _antisymmetric_matrix_from_upper_tri(k_flat: np.ndarray, k_dim: int) -> Array:
    """Create an anti-symmetric matrix given the upper triangle."""
    K = jnp.zeros((k_dim, k_dim))
    upper_indices = jnp.triu_indices(k_dim, k=1)
    lower_indices = jnp.tril_indices(k_dim, k=-1)
    K = K.at[upper_indices].set(k_flat)
    K = K.at[lower_indices].set(-k_flat)

    return K


def _check_ci_strs(
    ci_strs: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Make sure the hamming weight is consistent in all determinants."""
    addr_up, addr_dn = ci_strs
    addr_up_ham = format(addr_up[0], "b").count("1")
    for i, addr in enumerate(addr_up):
        ham = format(addr, "b").count("1")
        if ham != addr_up_ham:
            raise ValueError(
                f"Spin-up CI string in index 0 has hamming weight {addr_up_ham}, but CI string in "
                f"index {i} has hamming weight {ham}."
            )
    addr_dn_ham = format(addr_dn[0], "b").count("1")
    for i, addr in enumerate(addr_dn):
        ham = format(addr, "b").count("1")
        if ham != addr_dn_ham:
            raise ValueError(
                f"Spin-down CI string in index 0 has hamming weight {addr_dn_ham}, but CI string in "
                f"index {i} has hamming weight {ham}."
            )

    return np.sort(np.unique(addr_up)), np.sort(np.unique(addr_dn))


def _optimize_orbitals_sci(
    k_flat: np.ndarray,
    learning_rate: float,
    momentum: float,
    num_steps: int,
    dm1: np.ndarray,
    dm2: np.ndarray,
    hcore: np.ndarray,
    eri: np.ndarray,
) -> None:
    """Optimize orbital rotation parameters in-place using gradient descent.

    This procedure is described in `Sec. II A 4 <https://arxiv.org/pdf/2405.05068>`_.
    """
    prev_update = np.zeros(len(k_flat))
    for _ in range(num_steps):
        grad = _SCISCF_Energy_contract_grad(dm1, dm2, hcore, eri, k_flat)
        prev_update = learning_rate * grad + momentum * prev_update
        k_flat -= prev_update


def _SCISCF_Energy_contract(
    dm1: np.ndarray,
    dm2: np.ndarray,
    hcore: np.ndarray,
    eri: np.ndarray,
    k_flat: np.ndarray,
) -> Array:
    """Calculate gradient.

    The gradient can be calculated by contracting the bare one and two-body
    reduced density matrices with the gradients of the of the one and two-body
    integrals with respect to the rotation parameters, ``k_flat``.
    """
    K = _antisymmetric_matrix_from_upper_tri(k_flat, hcore.shape[0])
    U = expm(K)
    hcore_rot = jnp.matmul(jnp.transpose(U), jnp.matmul(hcore, U))
    eri_rot = jnp.einsum("pqrs, pi, qj, rk, sl->ijkl", eri, U, U, U, U)
    grad = jnp.sum(dm1 * hcore_rot) + jnp.sum(dm2 * eri_rot / 2.0)

    return grad


_SCISCF_Energy_contract_grad = jit(grad(_SCISCF_Energy_contract, argnums=4))


def _apply_excitation_single(
    single_bts: np.ndarray, diag: np.ndarray, create: np.ndarray, annihilate: np.ndarray
) -> tuple[Array, Array]:
    falses = jnp.array([False for _ in range(len(diag))])

    bts_ret = single_bts == diag
    create_crit = jnp.all(jnp.logical_or(diag, falses == jnp.logical_and(single_bts, create)))
    annihilate_crit = jnp.all(falses == jnp.logical_and(falses == single_bts, annihilate))

    include_crit = jnp.logical_and(create_crit, annihilate_crit)

    return bts_ret, include_crit


_apply_excitation = jit(vmap(_apply_excitation_single, (0, None, None, None), 0))

apply_excitations = jit(vmap(_apply_excitation, (None, 0, 0, 0), 0))


def _transition_str_to_bool(string_rep: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transform string representations of a transition operator into bool representation.

    Transform sequences of identity ("I"), creation ("+"), annihilation ("-"), and number ("n")
    characters into the internal representation used to apply the transitions into electronic
    configurations.

    Args:
        string_rep: A 1D or 2D array of ``I``, ``+``, ``-``, ``n`` strings representing
        the action of the identity, creation, annihilation, or number operators.

    Returns:
        A 3-tuple:
            - A mask signifying the diagonal terms (I).
            - A mask signifying whether there is a creation operator (+).
            - A mask signifying whether there is an annihilation operator (-).

    """
    diag = np.logical_or(string_rep == "I", string_rep == "n")
    create = np.logical_or(string_rep == "+", string_rep == "n")
    annihilate = np.logical_or(string_rep == "-", string_rep == "n")

    return diag, create, annihilate
