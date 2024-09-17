# This code is a Qiskit project.

# (C) Copyright IBM 2024.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Functions for the study of fermionic systems.

.. currentmodule:: qiskit_addon_sqd.fermion

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   bitstring_matrix_to_sorted_addresses
   enlarge_batch_from_transitions
   flip_orbital_occupancies
   solve_fermion
   optimize_orbitals
   rotate_integrals
"""

from __future__ import annotations

import numpy as np
from jax import Array, config, grad, jit, vmap
from jax import numpy as jnp
from jax.scipy.linalg import expm
from pyscf import fci
from scipy import linalg as LA

config.update("jax_enable_x64", True)  # To deal with large integers


def solve_fermion(
    addresses: tuple[np.ndarray, np.ndarray],
    hcore: np.ndarray,
    eri: np.ndarray,
    *,
    spin_sq: int | None = None,
    max_davidson: int = 100,
    verbose: int | None = None,
) -> tuple[float, np.ndarray, list[np.ndarray], float]:
    """
    Approximate the ground state given molecular integrals and Slater determinant addresses.

    .. note::
       The ``addresses`` are expected to be unique and sorted. While this will be handled
       for the user automatically, this function could become slower if the input
       addresses are not sorted or nearly-sorted.

    Args:
        addresses: A length-2 tuple of 1D arrays containing sorted, base-10
            representations of bitstrings. The first array represents configurations of the
            alpha particles, and the second array represents that of the beta particles.
        hcore: Core Hamiltonian matrix representing single-electron integrals
        eri: Electronic repulsion integrals representing two-electron integrals
        spin_sq: Target value for the total spin squared for the ground state.
            If ``None``, no spin will be imposed.
        max_davidson: The maximum number of cycles of Davidson's algorithm
        verbose: A verbosity level between 0 and 10

    Returns:
            A tuple containing:
                - Minimum energy from SCI calculation
                - SCI coefficients
                - Average orbital occupancy
                - Expectation value of spin-squared

    Raises:
            ValueError: The input determinant ``addresses`` must be non-empty, sorted arrays of integers.
    """
    addresses = _check_addresses(addresses)

    num_up = format(addresses[0][0], "b").count("1")
    num_dn = format(addresses[1][0], "b").count("1")

    # Number of molecular orbitals
    norb = hcore.shape[0]
    # Call the projection + eigenstate finder
    myci = fci.selected_ci.SelectedCI()
    if spin_sq is not None:
        myci = fci.addons.fix_spin_(myci, ss=spin_sq)
    e_sci, coeffs_sci = fci.selected_ci.kernel_fixed_space(
        myci,
        hcore,
        eri,
        norb,
        (num_up, num_dn),
        ci_strs=addresses,
        verbose=verbose,
        max_cycle=max_davidson,
    )
    # Calculate the avg occupancy of each orbital
    dm1 = myci.make_rdm1s(coeffs_sci, norb, (num_up, num_dn))
    avg_occupancy = [np.diagonal(dm1[0]), np.diagonal(dm1[1])]

    # Compute total spin
    spin_squared = myci.spin_square(coeffs_sci, norb, (num_up, num_dn))[0]

    return e_sci, coeffs_sci, avg_occupancy, spin_squared


def optimize_orbitals(
    addresses: tuple[np.ndarray, np.ndarray],
    hcore: np.ndarray,
    eri: np.ndarray,
    k_flat: np.ndarray,
    *,
    spin_sq: float = 0.0,
    num_iters: int = 10,
    num_steps_grad: int = 10_000,
    learning_rate: float = 0.01,
    max_davidson: int = 100,
) -> tuple[float, np.ndarray, list[np.ndarray]]:
    """
    Optimize orbitals to produce a minimal ground state.

    The process involves iterating over 3 steps:

    For ``num_iters`` iterations:
        - Rotate the integrals with respect to the parameters, ``k_flat``
        - Diagonalize and approximate the groundstate energy and wavefunction amplitudes
        - Optimize ``k_flat`` using gradient descent and the wavefunction
          amplitudes found in Step 2

    Refer to `Sec. II A 4 <https://arxiv.org/pdf/2405.05068>`_ for more detailed
    discussion on this orbital optimization technique.

    .. note::
       The input ``addresses`` are expected to be unique and sorted. While this will be
       handled for the user automatically, this function may become slower if the input
       addresses are not sorted or nearly-sorted.

    Args:
        addresses: A length-2 tuple of 1D arrays containing sorted, base-10
            representations of bitstrings. The first array represents configurations of the
            alpha particles, and the second array represents that of the beta particles.
        hcore: Core Hamiltonian matrix representing single-electron integrals
        eri: Electronic repulsion integrals representing two-electron integrals
        k_flat: 1D array defining the orbital transform. This array will be reshaped
            to be of shape (# orbitals, # orbitals) before being used as a
            similarity transform operator on the orbitals. Thus ``len(k_flat)=# orbitals**2``.
        spin_sq: Target value for the total spin squared for the ground state
        num_iters: The number of iterations of orbital optimization to perform
        max_davidson: The maximum number of cycles of Davidson's algorithm to
            perform during diagonalization.
        num_steps_grad: The number of steps of gradient descent to perform
            during each optimization iteration
        learning_rate: The learning rate to use during gradient descent

    Returns:
        A tuple containing:
            - The groundstate energy found during the last optimization iteration
            - An optimized 1D array defining the orbital transform
            - Average orbital occupancy
    """
    addresses = _check_addresses(addresses)

    num_up = format(addresses[0][0], "b").count("1")
    num_dn = format(addresses[1][0], "b").count("1")

    # TODO: Need metadata showing the optimization history
    ## hcore and eri in physicist ordering
    num_orbitals = hcore.shape[0]
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
            num_orbitals,
            (num_up, num_dn),
            ci_strs=addresses,
            max_cycle=max_davidson,
        )

        # Generate the one and two-body reduced density matrices from latest wavefunction amplitudes
        dm1, dm2_chem = myci.make_rdm12(amplitudes, num_orbitals, (num_up, num_up))
        dm2 = np.asarray(dm2_chem.transpose(0, 2, 3, 1), order="C")

        # TODO: Expose the momentum parameter as an input option
        # Optimize the basis rotations
        _optimize_orbitals_sci(
            k_flat, learning_rate, 0.9, num_steps_grad, dm1, dm2, hcore, eri_phys
        )

    return e_qsci, k_flat, [np.diagonal(dm1), np.diagonal(dm1)]


def rotate_integrals(
    hcore: np.ndarray, eri: np.ndarray, k_flat: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Perform a similarity transform on the integrals.

    The transformation is described as:

    .. math::

       \hat{\widetilde{H}} = \hat{U^{\dagger}}(k)\hat{H}\hat{U}(k)

    For more information on how :math:`\hat{U}` and :math:`\hat{U^{\dagger}}` are generated from ``k_flat``
    and applied to the one- and two-body integrals, refer to `Sec. II A 4 <https://arxiv.org/pdf/2405.05068>`_.

    Args:
        hcore: Core Hamiltonian matrix representing single-electron integrals
        eri: Electronic repulsion integrals representing two-electron integrals
        k_flat: 1D array defining the orbital transform. Refer to `Sec. II A 4 <https://arxiv.org/pdf/2405.05068>`_
            for more information on how these values are used to generate the transform operator.

    Returns:
        - The rotated core Hamiltonian matrix
        - The rotated ERI matrix
    """
    num_orbitals = hcore.shape[0]
    p = np.reshape(k_flat, (num_orbitals, num_orbitals))
    K = (p - np.transpose(p)) / 2.0
    U = LA.expm(K)
    hcore_rot = np.matmul(np.transpose(U), np.matmul(hcore, U))
    eri_rot = np.einsum("pqrs, pi, qj, rk, sl->ijkl", eri, U, U, U, U, optimize=True)

    return np.array(hcore_rot), np.array(eri_rot)


def flip_orbital_occupancies(occupancies: np.ndarray) -> np.ndarray:
    """
    Flip an orbital occupancy array to match the indexing of a bitstring.

    This function reformats a 1D array of spin-orbital occupancies formatted like:

        ``[occ_a_1, occ_a_2, ..., occ_a_N, occ_b_1, ..., occ_b_N]``

    To an array formatted like:

        ``[occ_a_N, ..., occ_a_1, occ_b_N, ..., occ_b_1]``

    where ``N`` is the number of spatial orbitals.
    """
    num_orbitals = occupancies.shape[0] // 2
    occ_up = occupancies[:num_orbitals]
    occ_dn = occupancies[num_orbitals:]
    occ_out = np.zeros(2 * num_orbitals)
    occ_out[:num_orbitals] = np.flip(occ_up)
    occ_out[num_orbitals:] = np.flip(occ_dn)

    return occ_out


def bitstring_matrix_to_sorted_addresses(
    bitstring_matrix: np.ndarray, open_shell: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a bitstring matrix into a sorted array of unique, unsigned base-10 representations.

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
        A length-2 tuple of sorted, unique base-10 determinant addresses representing the left
        and right halves of the bitstrings, respectively.
    """
    num_orbitals = bitstring_matrix.shape[1] // 2
    num_configs = bitstring_matrix.shape[0]

    address_left = np.zeros(num_configs)
    address_right = np.zeros(num_configs)
    bts_matrix_left = bitstring_matrix[:, :num_orbitals]
    bts_matrix_right = bitstring_matrix[:, num_orbitals:]

    # For performance, we accumulate the left and right addresses together, column-wise,
    # across the two halves of the input bitstring matrix.
    for i in range(num_orbitals):
        address_left[:] += bts_matrix_left[:, i] * 2 ** (num_orbitals - 1 - i)
        address_right[:] += bts_matrix_right[:, i] * 2 ** (num_orbitals - 1 - i)

    addresses_right = np.unique(address_right.astype("longlong"))
    addresses_left = np.unique(address_left.astype("longlong"))

    if not open_shell:
        addresses_left = addresses_right = np.union1d(addresses_left, addresses_right)

    return addresses_left, addresses_right


def enlarge_batch_from_transitions(
    bitstring_matrix: np.ndarray, transition_operators: np.ndarray
) -> np.ndarray:
    """
    Apply the set of transition operators to the configurations represented in ``bitstring_matrix``.

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


def _check_addresses(
    addresses: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Make sure the hamming weight is consistent in all determinants."""
    addr_up, addr_dn = addresses
    addr_up_ham = format(addr_up[0], "b").count("1")
    for i, addr in enumerate(addr_up):
        ham = format(addr, "b").count("1")
        if ham != addr_up_ham:
            raise ValueError(
                f"Spin-up address in index 0 has hamming weight {addr_up_ham}, but address in "
                f"index {i} has hamming weight {ham}."
            )
    addr_dn_ham = format(addr_dn[0], "b").count("1")
    for i, addr in enumerate(addr_dn):
        ham = format(addr, "b").count("1")
        if ham != addr_dn_ham:
            raise ValueError(
                f"Spin-down address in index 0 has hamming weight {addr_dn_ham}, but address in "
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
    """
    Optimize orbital rotation parameters in-place using gradient descent.

    This procedure is described in `Sec. II A 4 <https://arxiv.org/pdf/2405.05068>`_.
    """
    prev_update = np.zeros(len(k_flat))
    num_orbitals = dm1.shape[0]
    for _ in range(num_steps):
        grad = _SCISCF_Energy_contract_grad(dm1, dm2, hcore, eri, num_orbitals, k_flat)
        prev_update = learning_rate * grad + momentum * prev_update
        k_flat -= prev_update


def _SCISCF_Energy_contract(
    dm1: np.ndarray,
    dm2: np.ndarray,
    hcore: np.ndarray,
    eri: np.ndarray,
    num_orbitals: int,
    k_flat: np.ndarray,
) -> Array:
    """
    Calculate gradient.

    The gradient can be calculated by contracting the bare one and two-body
    reduced density matrices with the gradients of the of the one and two-body
    integrals with respect to the rotation parameters, ``k_flat``.
    """
    p = jnp.reshape(k_flat, (num_orbitals, num_orbitals))
    K = (p - jnp.transpose(p)) / 2.0
    U = expm(K)
    hcore_rot = jnp.matmul(jnp.transpose(U), jnp.matmul(hcore, U))
    eri_rot = jnp.einsum("pqrs, pi, qj, rk, sl->ijkl", eri, U, U, U, U)
    grad = jnp.sum(dm1 * hcore_rot) + jnp.sum(dm2 * eri_rot / 2.0)

    return grad


_SCISCF_Energy_contract_grad = jit(grad(_SCISCF_Energy_contract, argnums=5), static_argnums=4)


def _apply_excitation_single(
    single_bts: np.ndarray, diag: np.ndarray, create: np.ndarray, annihilate: np.ndarray
) -> tuple[jnp.ndarray, Array]:
    falses = jnp.array([False for _ in range(len(diag))])

    bts_ret = single_bts == diag
    create_crit = jnp.all(jnp.logical_or(diag, falses == jnp.logical_and(single_bts, create)))
    annihilate_crit = jnp.all(falses == jnp.logical_and(falses == single_bts, annihilate))

    include_crit = jnp.logical_and(create_crit, annihilate_crit)

    return bts_ret, include_crit


_apply_excitation = jit(vmap(_apply_excitation_single, (0, None, None, None), 0))

apply_excitations = jit(vmap(_apply_excitation, (None, 0, 0, 0), 0))


def _transition_str_to_bool(string_rep: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Transform string representations of a transition operator into bool representation.

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
