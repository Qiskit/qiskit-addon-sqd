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

import warnings
from dataclasses import dataclass

import numpy as np
from jax import Array, config, grad, jit, vmap
from jax import numpy as jnp
from jax.scipy.linalg import expm
from pyscf import fci
from qiskit.utils import deprecate_func
from scipy import linalg as LA

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
            will be projected and diagonalized. This is a 2D array of ``bool`` representations of bit
            values such that each row represents a single bitstring. The spin-up configurations
            should be specified by column indices in range ``(N, N/2]``, and the spin-down
            configurations should be specified by column indices in range ``(N/2, 0]``, where ``N``
            is the number of qubits.

            (DEPRECATED) The configurations may also be specified by a length-2 tuple of sorted 1D
            arrays containing unsigned integer representations of the determinants. The two lists
            should represent the spin-up and spin-down orbitals, respectively.
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
        warnings.warn(
            "Passing the input determinants as integers is deprecated. Users should instead pass a bitstring matrix defining the subspace.",
            DeprecationWarning,
            stacklevel=2,
        )
        ci_strs = bitstring_matrix
    else:
        # This will become the default code path after the deprecation period.
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
            will be projected and diagonalized. This is a 2D array of ``bool`` representations of bit
            values such that each row represents a single bitstring. The spin-up configurations
            should be specified by column indices in range ``(N, N/2]``, and the spin-down
            configurations should be specified by column indices in range ``(N/2, 0]``, where ``N``
            is the number of qubits.

            (DEPRECATED) The configurations may also be specified by a length-2 tuple of sorted 1D
            arrays containing unsigned integer representations of the determinants. The
            two lists should represent the spin-up and spin-down orbitals, respectively.
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
        warnings.warn(
            "Passing a length-2 tuple of determinant lists to define the spin-up/down subspaces "
            "is deprecated. Users should instead pass in the bitstring matrix defining the subspaces.",
            DeprecationWarning,
            stacklevel=2,
        )
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


@deprecate_func(
    removal_timeline="no sooner than qiskit-addon-sqd 0.10.0",
    since="0.6.0",
    package_name="qiskit-addon-sqd",
    additional_msg="Use the bitstring_matrix_to_ci_strs function.",
)
def bitstring_matrix_to_sorted_addresses(
    bitstring_matrix: np.ndarray, open_shell: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a bitstring matrix into a sorted array of unique, unsigned integers.

    This function separates each bitstring in ``bitstring_matrix`` in half, flips the
    bits and translates them into integer representations, and finally appends them to
    their respective (spin-up or spin-down) lists. Those lists are sorted and output
    from this function.

    Args:
        bitstring_matrix: A 2D array of ``bool`` representations of bit
            values such that each row represents a single bitstring
        open_shell: A flag specifying whether unique addresses from the left and right
            halves of the bitstrings should be kept separate. If ``False``, addresses
            from the left and right halves of the bitstrings are combined into a single
            set of unique addresses. That combined set will be returned for both the left
            and right bitstrings.

    Returns:
        A length-2 tuple of sorted, unique determinants representing the left (spin-down) and
        right (spin-up) halves of the bitstrings, respectively.

    """
    norb = bitstring_matrix.shape[1] // 2
    num_configs = bitstring_matrix.shape[0]

    address_left = np.zeros(num_configs)
    address_right = np.zeros(num_configs)
    bts_matrix_left = bitstring_matrix[:, :norb]
    bts_matrix_right = bitstring_matrix[:, norb:]

    # For performance, we accumulate the left and right addresses together, column-wise,
    # across the two halves of the input bitstring matrix.
    for i in range(norb):
        address_left[:] += bts_matrix_left[:, i] * 2 ** (norb - 1 - i)
        address_right[:] += bts_matrix_right[:, i] * 2 ** (norb - 1 - i)

    addresses_right = np.unique(address_right.astype("longlong"))
    addresses_left = np.unique(address_left.astype("longlong"))

    if not open_shell:
        addresses_left = addresses_right = np.union1d(addresses_left, addresses_right)

    return addresses_left, addresses_right


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
