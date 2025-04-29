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
"""Functions for the study of fermionic systems."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from jax import Array, config, grad, jit, vmap
from jax import numpy as jnp
from jax.scipy.linalg import expm
from pyscf import fci
from scipy import linalg as LA

from qiskit_addon_sqd.configuration_recovery import recover_configurations
from qiskit_addon_sqd.counts import counts_to_arrays
from qiskit_addon_sqd.subsampling import postselect_and_subsample

config.update("jax_enable_x64", True)  # To deal with large integers


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
            filename, amplitudes=self.amplitudes, ci_strs_a=self.ci_strs_a, ci_strs_b=self.ci_strs_b
        )

    @classmethod
    def load(cls, filename):
        """Load an SCIState object from an .npz file."""
        with np.load(filename) as data:
            return cls(data["amplitudes"], data["ci_strs_a"], data["ci_strs_b"])


def run_sqd(
    one_body_tensor: np.ndarray,
    two_body_tensor: np.ndarray,
    # TODO take BitArray instead of counts
    # see https://github.com/Qiskit/qiskit-addon-sqd/issues/113
    counts: dict[str, int],
    subsample_size: int,
    norb: int,
    nelec: tuple[int, int],
    *,
    n_subsamples: int = 1,
    iterations: int = 1,
    sci_solver: Callable[
        [list[tuple[np.ndarray, np.ndarray]], np.ndarray, np.ndarray],
        list[tuple[float, SCIState, tuple[np.ndarray, np.ndarray]]],
    ]
    | None = None,
    symmetrize_spin: bool = False,
    include_configurations: list[int] | tuple[list[int], list[int]] | None = None,
    initial_occupancies: tuple[np.ndarray, np.ndarray] | None = None,
    carryover_threshold: float = 1e-4,
    callback: Callable[[np.ndarray, np.ndarray, np.ndarray], None] | None = None,
    seed: int | np.random.Generator | None = None,
) -> tuple[float, SCIState]:
    """Run SQD.

    Args:
        one_body_tensor: The one-body tensor of the Hamiltonian.
        two_body_tensor: The two-body tensor of the Hamiltonian.
        counts: The counts of sampled bitstrings. Each bitstring should have both the
            alpha part and beta part concatenated together, with the alpha part
            concatenated on the right-hand side.
        subsample_size: The number of bitstrings to include in each subsample.
        norb: The number of spatial orbitals.
        nelec: The numbers of alpha and beta electrons.
        n_subsamples: The number of subsamples to generate in each configuration recovery
            iteration. This argument indirectly controls the dimensions of the
            diagonalization subspaces. A higher value will yield larger subspace dimensions.
        iterations: Number of configuration recovery iterations.
        sci_solver: Selected configuration interaction solver function.

            Inputs:
            - List of pairs (strings_a, strings_b) of arrays of spin-alpha CI strings
              and spin-beta CI strings whose Cartesian product give the basis of the
              subspace in which to perform a diagonalization. A list is passed to allow
              the solver function to perform the diagonalizations in parallel.
            - One-body tensor of the Hamiltonian.
            - Two-body tensor of the Hamiltonian.
            - The number of spatial orbitals.
            - A pair (n_alpha, n_beta) indicating the numbers of alpha and beta
              electrons.

            Output: List of (energy, sci_state, occupancies) triplets, where each triplet
            contains the result of the corresponding diagonalization.
        symmetrize_spin: Whether to always merge spin-alpha and spin-beta CI strings
            into a single list, so that the diagonalization subspace is invariant with
            respect to the exchange of spin alpha with spin beta.
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
            costly diagonalization.
        callback: A callback function to be called after each configuration recovery
            iteration. The function will be passed the output of the sci_solver
            function, which is a list of (energy, sci_state, occupancies) triplets,
            where each triplet contains the result of a diagonalization.
        seed: A seed for the pseudorandom number generator.

    Returns:
        The estimate of the energy and the SCI state with that energy.
    """
    if iterations < 1:
        raise ValueError("Number of iterations must be at least 1.")

    n_alpha, n_beta = nelec
    if symmetrize_spin and n_alpha != n_beta:
        raise ValueError(
            "Spin symmetrization is only possible if the numbers of alpha and beta "
            f"electrons are equal. Instead, got {n_alpha} and {n_beta}."
        )

    rng = np.random.default_rng(seed)
    current_occupancies = initial_occupancies
    min_energy = float("inf")
    if sci_solver is None:
        sci_solver = solve_sci_batch

    if include_configurations is None:
        include_a = np.array([], dtype=np.int64)
        include_b = np.array([], dtype=np.int64)
    elif isinstance(include_configurations, tuple):
        include_a, include_b = include_configurations
    else:
        include_a = include_configurations
        include_b = include_configurations

    carryover_strings_a = np.array([], dtype=np.int64)
    carryover_strings_b = np.array([], dtype=np.int64)

    # Convert counts into bitstring and probability arrays
    raw_bitstrings, raw_probs = counts_to_arrays(counts)

    for _ in range(iterations):
        # On the first iteration, we have no orbital occupancy information from the
        # solver, so we begin with the full set of noisy configurations.
        if current_occupancies is None:
            bitstrings, probs = raw_bitstrings, raw_probs
        else:
            # If we have average orbital occupancy information, we use it to refine the
            # full set of noisy configurations
            bitstrings, probs = recover_configurations(
                raw_bitstrings, raw_probs, current_occupancies, n_alpha, n_beta, rand_seed=rng
            )

        # Postselect and subsample batches of bitstrings
        subsamples = postselect_and_subsample(
            bitstrings,
            probs,
            hamming_right=n_alpha,
            hamming_left=n_beta,
            samples_per_batch=subsample_size,
            num_batches=n_subsamples,
            rand_seed=rng,
        )

        # Convert bitstrings to CI strings and include requested and carryover strings
        ci_strings = []
        for subsample in subsamples:
            strs_a, strs_b = bitstring_matrix_to_ci_strs(subsample, open_shell=not symmetrize_spin)
            strs_a = np.union1d(strs_a, np.union1d(include_a, carryover_strings_a))
            strs_b = np.union1d(strs_b, np.union1d(include_b, carryover_strings_b))
            ci_strings.append((strs_a, strs_b))

        # Run diagonalization
        results = sci_solver(ci_strings, one_body_tensor, two_body_tensor, norb, nelec)
        energies, sci_states, occupancies = zip(*results)

        # Get best result from batch
        min_index = np.argmin(energies)
        current_occupancies = occupancies[min_index]
        sci_state = sci_states[min_index]

        # Carry over bitstrings with large CI weight
        flattened = sci_state.amplitudes.reshape(-1)
        absolute_vals = np.abs(flattened)
        indices = np.argsort(absolute_vals)
        carryover_index = np.searchsorted(absolute_vals, carryover_threshold, sorter=indices)
        carryover_indices = indices[carryover_index:]
        _, n_strings_b = sci_state.amplitudes.shape
        alpha_indices, beta_indices = np.divmod(carryover_indices, n_strings_b)
        carryover_strings_a = sci_state.ci_strs_a[alpha_indices]
        carryover_strings_b = sci_state.ci_strs_b[beta_indices]
        if symmetrize_spin:
            carryover_strings_a = carryover_strings_b = np.union1d(
                carryover_strings_a, carryover_strings_b
            )

        # Check if the energy is the lowest seen so far
        if energies[min_index] < min_energy:
            min_energy = energies[min_index]
            min_sci_state = sci_states[min_index]

        # Call callback function if provided
        if callback is not None:
            callback(results)

    return min_energy, min_sci_state


def solve_sci_batch(
    ci_strings: list[tuple[np.ndarray, np.ndarray]],
    one_body_tensor: np.ndarray,
    two_body_tensor: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    *,
    spin_sq: float | None = None,
    max_davidson: int = 100,
    verbose: int = 0,
) -> list[tuple[float, SCIState, tuple[np.ndarray, np.ndarray]]]:
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
        max_davidson: The maximum number of cycles of Davidson's algorithm.
        verbose: Level of output verbosity, as an integer ranging from 0 (least verbose)
            to 10 (most verbose).

    Returns:
        The results of the diagonalizations in the subspaces given by ci_strings,
        as a list of (energy, sci_state, occupancies) triplets.
    """
    norb, _ = one_body_tensor.shape

    myci = fci.selected_ci.SelectedCI()
    if spin_sq is not None:
        myci = fci.addons.fix_spin_(myci, ss=spin_sq)

    results = []
    for ci_strs in ci_strings:
        # The energy returned from this function is not guaranteed to be
        # the energy of the returned wavefunction when the spin^2 deviates
        # from the value requested. We will calculate the energy from the
        # RDMs below and ignore this value to be safe.
        _, sci_vec = fci.selected_ci.kernel_fixed_space(
            myci,
            one_body_tensor,
            two_body_tensor,
            norb,
            nelec,
            ci_strs=ci_strs,
            verbose=verbose,
            max_cycle=max_davidson,
        )
        # Calculate the average occupancy of each orbital
        dm1s = myci.make_rdm1s(sci_vec, norb, nelec)
        occupancy = (np.diagonal(dm1s[0]), np.diagonal(dm1s[1]))
        # Calculate energy from RDMs
        dm1 = myci.make_rdm1(sci_vec, norb, nelec)
        dm2 = myci.make_rdm2(sci_vec, norb, nelec)
        energy = np.einsum("pr,pr->", dm1, one_body_tensor) + 0.5 * np.einsum(
            "prqs,prqs->", dm2, two_body_tensor
        )
        # Construct SCIState
        sci_state = SCIState(
            amplitudes=np.array(sci_vec), ci_strs_a=sci_vec._strs[0], ci_strs_b=sci_vec._strs[1]
        )
        # Append results to list
        results.append((energy, sci_state, occupancy))

    return results


def solve_fermion(
    bitstring_matrix: tuple[np.ndarray, np.ndarray] | np.ndarray,
    /,
    hcore: np.ndarray,
    eri: np.ndarray,
    *,
    open_shell: bool = False,
    spin_sq: float | None = None,
    max_davidson: int = 100,
    verbose: int | None = None,
) -> tuple[float, SCIState, tuple[np.ndarray, np.ndarray], float]:
    """Approximate the ground state given molecular integrals and a set of electronic configurations.

    Args:
        bitstring_matrix: A set of configurations defining the subspace onto which the Hamiltonian
            will be projected and diagonalized.

            This may be specified in two ways:

            A bitstring matrix: A 2D ``numpy.ndarray`` of ``bool`` representations of bit values such that each row represents a single bitstring. The spin-up
            configurations should be specified by column indices in range ``(N, N/2]``, and the spin-down configurations should be specified by column
            indices in range ``(N/2, 0]``, where ``N`` is the number of qubits.

            CI strings: A length-2 tuple of sequences containing integer representations of the spin-up and spin-down determinants, respectively.
                The expected ordering is ``([a_str_0, ..., a_str_N], [b_str_0, ..., b_str_M])``.
        hcore: Core Hamiltonian matrix representing single-electron integrals
        eri: Electronic repulsion integrals representing two-electron integrals
        open_shell: A flag specifying whether configurations from the left and right
            halves of the bitstrings should be kept separate. If ``False``, CI strings
            from the left and right halves of the bitstrings are combined into a single
            set of unique configurations and used for both the alpha and beta subspaces.
        spin_sq: Target value for the total spin squared for the ground state.
            If ``None``, no spin will be imposed.
        max_davidson: The maximum number of cycles of Davidson's algorithm
        verbose: A verbosity level between 0 and 10

    Returns:
        - Minimum energy from SCI calculation
        - The SCI ground state
        - Tuple containing orbital occupancies for spin-up and spin-down orbitals. Formatted as: ``(array([occ_a_0, ..., occ_a_N]), array([occ_b_0, ..., occ_b_N]))``
        - Expectation value of spin-squared

    """
    if isinstance(bitstring_matrix, tuple):
        ci_strs = bitstring_matrix
    else:
        ci_strs = bitstring_matrix_to_ci_strs(bitstring_matrix, open_shell=open_shell)
    ci_strs = _check_ci_strs(ci_strs)

    num_up = format(ci_strs[0][0], "b").count("1")
    num_dn = format(ci_strs[1][0], "b").count("1")

    # Number of molecular orbitals
    norb = hcore.shape[0]
    # Call the projection + eigenstate finder
    myci = fci.selected_ci.SelectedCI()
    if spin_sq is not None:
        myci = fci.addons.fix_spin_(myci, ss=spin_sq)
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
        ci_strs=ci_strs,
        verbose=verbose,
        max_cycle=max_davidson,
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
        amplitudes=np.array(sci_vec), ci_strs_a=sci_vec._strs[0], ci_strs_b=sci_vec._strs[1]
    )

    return e_sci, sci_state, avg_occupancy, spin_squared


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
    max_davidson: int = 100,
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

            A bitstring matrix: A 2D ``numpy.ndarray`` of ``bool`` representations of bit values such that each row represents a single bitstring. The spin-up
            configurations should be specified by column indices in range ``(N, N/2]``, and the spin-down configurations should be specified by column
            indices in range ``(N/2, 0]``, where ``N`` is the number of qubits.

            CI strings: A length-2 tuple of sequences containing integer representations of the spin-up and spin-down determinants, respectively.
                The expected ordering is ``([a_str_0, ..., a_str_N], [b_str_0, ..., b_str_M])``.
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
        max_davidson: The maximum number of cycles of Davidson's algorithm to
            perform during diagonalization.
        num_steps_grad: The number of steps of gradient descent to perform
            during each optimization iteration
        learning_rate: The learning rate to use during gradient descent

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
            ci_strs=ci_strs,
            max_cycle=max_davidson,
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

    This function separates each bitstring in ``bitstring_matrix`` in half, flips the
    bits and translates them into integer representations, and finally appends them to
    their respective (spin-up or spin-down) lists. Those lists are sorted and output
    from this function.

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
    num_configs = bitstring_matrix.shape[0]

    ci_str_left = np.zeros(num_configs)
    ci_str_right = np.zeros(num_configs)
    bts_matrix_left = bitstring_matrix[:, :norb]
    bts_matrix_right = bitstring_matrix[:, norb:]

    # For performance, we accumulate the left and right CI strings together, column-wise,
    # across the two halves of the input bitstring matrix.
    for i in range(norb):
        ci_str_left[:] += bts_matrix_left[:, i] * 2 ** (norb - 1 - i)
        ci_str_right[:] += bts_matrix_right[:, i] * 2 ** (norb - 1 - i)

    ci_strs_right = np.unique(ci_str_right.astype("longlong"))
    ci_strs_left = np.unique(ci_str_left.astype("longlong"))

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
