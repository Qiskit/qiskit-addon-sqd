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

"""Tests for the fermion module."""

import math
import pickle
import unittest

import numpy as np
import pyscf
import pyscf.mcscf
import pytest
from pyscf.fci import cistring, spin_square
from qiskit.primitives import BitArray
from qiskit_addon_sqd.counts import generate_bit_array_uniform
from qiskit_addon_sqd.fermion import (
    SCIResult,
    SCIState,
    StandardPolicy,
    SubspacePolicy,
    SubspaceRequest,
    batch_to_ci_strings,
    bitstring_matrix_to_ci_strs,
    diagonalize_fermionic_hamiltonian,
    solve_sci,
    solve_sci_batch,
)


def _sci_vec_to_fci_vec(
    coeffs: np.ndarray,
    strings_a: np.ndarray,
    strings_b: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
):
    """Convert SCI coefficients and strings to an FCI vector."""
    n_alpha, n_beta = nelec
    addresses_a = cistring.strs2addr(norb, n_alpha, strings_a)
    addresses_b = cistring.strs2addr(norb, n_beta, strings_b)
    dim_a = math.comb(norb, n_alpha)
    dim_b = math.comb(norb, n_beta)
    fci_vec = np.zeros((dim_a, dim_b), dtype=coeffs.dtype)
    fci_vec[np.ix_(addresses_a, addresses_b)] = coeffs
    return fci_vec.reshape(-1)


class TestFermion(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(190560294508743238113331500595174898458)

    def test_diagonalize_fermionic_hamiltonian_basic(self):
        """Test diagonalize_fermionic_hamiltonian basic usage."""
        # Build N2 molecule
        mol = pyscf.gto.Mole()
        mol.build(
            atom=[["N", (0, 0, 0)], ["N", (1.0, 0, 0)]],
            basis="sto-6g",
            symmetry="Dooh",
        )

        # Define active space
        n_frozen = 2
        active_space = range(n_frozen, mol.nao_nr())

        # Get molecular integrals
        scf = pyscf.scf.RHF(mol).run()
        norb = len(active_space)
        n_electrons = int(sum(scf.mo_occ[active_space]))
        n_alpha = (n_electrons + mol.spin) // 2
        n_beta = (n_electrons - mol.spin) // 2
        nelec = (n_alpha, n_beta)
        cas = pyscf.mcscf.CASCI(scf, norb, nelec)
        mo = cas.sort_mo(active_space, base=0)
        hcore, nuclear_repulsion_energy = cas.get_h1cas(mo)
        eri = pyscf.ao2mo.restore(1, cas.get_h2cas(mo), norb)
        dim_a = math.comb(norb, n_alpha)
        dim_b = math.comb(norb, n_beta)
        fci_dim = dim_a * dim_b

        # Compute exact energy
        _, _, fci_vec, _, _ = cas.kernel()
        exact_energy = cas.e_tot

        # Generate samples from ground state
        fci_vec = fci_vec.reshape(-1)
        probs = np.abs(fci_vec) ** 2
        addresses = self.rng.choice(fci_dim, size=10_000, p=probs)
        indices_a, indices_b = np.divmod(addresses, dim_b)
        strings_a = [int(s) for s in cistring.addrs2str(norb=norb, nelec=n_alpha, addrs=indices_a)]
        strings_b = [int(s) for s in cistring.addrs2str(norb=norb, nelec=n_beta, addrs=indices_b)]
        strings = [(sb << norb) + sa for sa, sb in zip(strings_a, strings_b)]
        bit_array_ground_state = BitArray.from_samples(strings, num_bits=2 * norb)

        # Generate random bitstrings
        bit_array_random = generate_bit_array_uniform(2_000, 2 * norb, rand_seed=self.rng)

        # Merge bitstrings
        bit_array = BitArray.concatenate_shots([bit_array_ground_state, bit_array_random])

        # Diagonalize
        result = diagonalize_fermionic_hamiltonian(
            hcore,
            eri,
            bit_array,
            samples_per_batch=10,
            norb=norb,
            nelec=nelec,
            max_iterations=5,
            symmetrize_spin=True,
            seed=self.rng,
        )
        sci_state = result.sci_state
        sci_dim = math.prod(sci_state.amplitudes.shape)
        expanded_vec = _sci_vec_to_fci_vec(
            sci_state.amplitudes, sci_state.ci_strs_a, sci_state.ci_strs_b, norb=norb, nelec=nelec
        )
        expected_spin_square, _ = spin_square(expanded_vec, norb, nelec)

        # Check
        self.assertLess(sci_dim, 0.5 * fci_dim)
        self.assertAlmostEqual(result.energy + nuclear_repulsion_energy, exact_energy, places=2)
        self.assertAlmostEqual(result.sci_state.spin_square(), expected_spin_square)

    def test_diagonalize_fermionic_hamiltonian_numpy_bitstrings(self):
        """Test diagonalization with bitstrings stored in a NumPy array."""
        mol = pyscf.gto.Mole()
        mol.build(
            atom=[["H", (0, 0, 0)], ["H", (0, 0, 0.735)]],
            basis="sto-3g",
        )

        scf = pyscf.scf.RHF(mol).run()
        norb = mol.nao_nr()
        nelec = (1, 1)
        cas = pyscf.mcscf.CASCI(scf, norb, nelec)
        hcore, nuclear_repulsion_energy = cas.get_h1cas()
        eri = pyscf.ao2mo.restore(1, cas.get_h2cas(), norb)
        cas.kernel()

        bitstrings = BitArray.from_samples(
            ["0101", "0110", "1001", "1010"], num_bits=2 * norb
        ).to_bool_array()

        result = diagonalize_fermionic_hamiltonian(
            hcore,
            eri,
            bitstrings,
            samples_per_batch=4,
            norb=norb,
            nelec=nelec,
            max_iterations=1,
            seed=self.rng,
        )

        self.assertAlmostEqual(result.energy + nuclear_repulsion_energy, cas.e_tot)

    def test_diagonalize_fermionic_hamiltonian_max_dim(self):
        """Test diagonalize_fermionic_hamiltonian with maximum dimension."""
        # Build N2 molecule
        mol = pyscf.gto.Mole()
        mol.build(
            atom=[["N", (0, 0, 0)], ["N", (1.0, 0, 0)]],
            basis="sto-6g",
            symmetry="Dooh",
        )

        # Define active space
        n_frozen = 2
        active_space = range(n_frozen, mol.nao_nr())

        # Get molecular integrals
        scf = pyscf.scf.RHF(mol).run()
        norb = len(active_space)
        n_electrons = int(sum(scf.mo_occ[active_space]))
        n_alpha = (n_electrons + mol.spin) // 2
        n_beta = (n_electrons - mol.spin) // 2
        nelec = (n_alpha, n_beta)
        cas = pyscf.mcscf.CASCI(scf, norb, nelec)
        mo = cas.sort_mo(active_space, base=0)
        hcore, _ = cas.get_h1cas(mo)
        eri = pyscf.ao2mo.restore(1, cas.get_h2cas(mo), norb)
        dim_a = math.comb(norb, n_alpha)
        dim_b = math.comb(norb, n_beta)
        fci_dim = dim_a * dim_b

        # Compute exact energy
        _, _, fci_vec, _, _ = cas.kernel()

        # Generate samples from ground state
        fci_vec = fci_vec.reshape(-1)
        probs = np.abs(fci_vec) ** 2
        addresses = self.rng.choice(fci_dim, size=10_000, p=probs)
        indices_a, indices_b = np.divmod(addresses, dim_b)
        strings_a = [int(s) for s in cistring.addrs2str(norb=norb, nelec=n_alpha, addrs=indices_a)]
        strings_b = [int(s) for s in cistring.addrs2str(norb=norb, nelec=n_beta, addrs=indices_b)]
        strings = [(sb << norb) + sa for sa, sb in zip(strings_a, strings_b)]
        bit_array_ground_state = BitArray.from_samples(strings, num_bits=2 * norb)

        # Generate random bitstrings
        bit_array_random = generate_bit_array_uniform(2_000, 2 * norb, rand_seed=self.rng)

        # Merge bitstrings
        bit_array = BitArray.concatenate_shots([bit_array_ground_state, bit_array_random])

        # Diagonalize
        result = diagonalize_fermionic_hamiltonian(
            hcore,
            eri,
            bit_array,
            samples_per_batch=10,
            norb=norb,
            nelec=nelec,
            max_iterations=5,
            symmetrize_spin=True,
            max_dim=10,
            seed=self.rng,
        )
        sci_state = result.sci_state
        sci_dim_a, sci_dim_b = sci_state.amplitudes.shape
        expanded_vec = _sci_vec_to_fci_vec(
            sci_state.amplitudes, sci_state.ci_strs_a, sci_state.ci_strs_b, norb=norb, nelec=nelec
        )
        expected_spin_square, _ = spin_square(expanded_vec, norb, nelec)

        # Check
        self.assertEqual(sci_dim_a, 10)
        self.assertEqual(sci_dim_b, 10)
        self.assertAlmostEqual(result.sci_state.spin_square(), expected_spin_square)

        # Diagonalize
        result = diagonalize_fermionic_hamiltonian(
            hcore,
            eri,
            bit_array,
            samples_per_batch=20,
            norb=norb,
            nelec=nelec,
            max_iterations=5,
            max_dim=(15, 10),
            seed=self.rng,
        )
        sci_state = result.sci_state
        sci_dim_a, sci_dim_b = sci_state.amplitudes.shape
        expanded_vec = _sci_vec_to_fci_vec(
            sci_state.amplitudes, sci_state.ci_strs_a, sci_state.ci_strs_b, norb=norb, nelec=nelec
        )
        expected_spin_square, _ = spin_square(expanded_vec, norb, nelec)

        # Check
        self.assertEqual(sci_dim_a, 15)
        self.assertEqual(sci_dim_b, 10)
        self.assertAlmostEqual(result.sci_state.spin_square(), expected_spin_square)

    def test_diagonalize_fermionic_hamiltonian_no_valid_bitstrings(self):
        """Test diagonalize_fermionic_hamiltonian when no valid bitstrings for subsampling."""
        # Build N2 molecule
        mol = pyscf.gto.Mole()
        mol.build(
            atom=[["N", (0, 0, 0)], ["N", (1.0, 0, 0)]],
            basis="sto-6g",
            symmetry="Dooh",
        )

        # Define active space
        n_frozen = 2
        active_space = range(n_frozen, mol.nao_nr())

        # Get molecular integrals
        scf = pyscf.scf.RHF(mol).run()
        norb = len(active_space)
        n_electrons = int(sum(scf.mo_occ[active_space]))
        n_alpha = (n_electrons + mol.spin) // 2
        n_beta = (n_electrons - mol.spin) // 2
        nelec = (n_alpha, n_beta)
        cas = pyscf.mcscf.CASCI(scf, norb, nelec)
        mo = cas.sort_mo(active_space, base=0)
        hcore, _ = cas.get_h1cas(mo)
        eri = pyscf.ao2mo.restore(1, cas.get_h2cas(mo), norb)

        # Generate invalid samples from ground state
        strings = ["00" * norb for _ in range(100)]

        bit_array = BitArray.from_samples(strings, num_bits=2 * norb)

        # Diagonalize
        # Check error with raise if no valid bitstrings
        with pytest.raises(ValueError, match="did not contain any valid bitstrings"):
            _ = diagonalize_fermionic_hamiltonian(
                hcore,
                eri,
                bit_array,
                samples_per_batch=10,
                norb=norb,
                nelec=nelec,
                max_iterations=5,
                symmetrize_spin=True,
                seed=self.rng,
            )

        # check when passing in initial_occupancies, no error will be raised
        average_occupancy = np.zeros(norb) + 0.1
        _ = diagonalize_fermionic_hamiltonian(
            hcore,
            eri,
            bit_array,
            samples_per_batch=1,
            norb=norb,
            nelec=nelec,
            max_iterations=5,
            symmetrize_spin=True,
            initial_occupancies=(average_occupancy, average_occupancy),
            seed=self.rng,
        )

    def test_diagonalize_fermionic_hamiltonian_reproducible_with_seed(self):
        """Test diagonalize_fermionic_hamiltonian result is reproducible with seed."""
        # Build N2 molecule
        mol = pyscf.gto.Mole()
        mol.build(
            atom=[["N", (0, 0, 0)], ["N", (1.0, 0, 0)]],
            basis="sto-6g",
            symmetry="Dooh",
        )

        # Define active space
        n_frozen = 2
        active_space = range(n_frozen, mol.nao_nr())

        # Get molecular integrals
        scf = pyscf.scf.RHF(mol).run()
        norb = len(active_space)
        n_electrons = int(sum(scf.mo_occ[active_space]))
        n_alpha = (n_electrons + mol.spin) // 2
        n_beta = (n_electrons - mol.spin) // 2
        nelec = (n_alpha, n_beta)
        cas = pyscf.mcscf.CASCI(scf, norb, nelec)
        mo = cas.sort_mo(active_space, base=0)
        hcore, _ = cas.get_h1cas(mo)
        eri = pyscf.ao2mo.restore(1, cas.get_h2cas(mo), norb)

        # Generate random bitstrings
        bit_array = generate_bit_array_uniform(2_000, 2 * norb, rand_seed=self.rng)

        # Diagonalize two times with the same seed
        result1 = diagonalize_fermionic_hamiltonian(
            hcore,
            eri,
            bit_array,
            samples_per_batch=10,
            norb=norb,
            nelec=nelec,
            max_iterations=3,
            max_dim=(10, 9),
            seed=12345,
        )
        result2 = diagonalize_fermionic_hamiltonian(
            hcore,
            eri,
            bit_array,
            samples_per_batch=10,
            norb=norb,
            nelec=nelec,
            max_iterations=3,
            max_dim=(10, 9),
            seed=12345,
        )

        # Check that the results match
        np.testing.assert_allclose(result1.energy, result2.energy)
        np.testing.assert_allclose(result1.sci_state.amplitudes, result2.sci_state.amplitudes)
        np.testing.assert_array_equal(result1.sci_state.ci_strs_a, result2.sci_state.ci_strs_a)
        np.testing.assert_allclose(result1.sci_state.ci_strs_b, result2.sci_state.ci_strs_b)

    def test_bitstring_matrix_to_ci_strs(self):
        norb = 57
        bitstring = "001111101111111110110001011101100001010000100101100001010"
        assert len(bitstring) == norb
        bitstrings = np.array([[b == "1" for b in bitstring + bitstring]])
        result = bitstring_matrix_to_ci_strs(bitstrings)
        result_string = format(result[0][0], f"0{norb}b")
        assert result_string == bitstring

    def test_bitstring_matrix_to_ci_strs_large(self):
        norb = 64
        bitstring = "0011111011111111101100010111011000010100001001011000010101111111"
        assert len(bitstring) == norb
        bitstrings = np.array([[b == "1" for b in bitstring + bitstring]])
        result = bitstring_matrix_to_ci_strs(bitstrings)
        result_string = format(result[0][0], f"0{norb}b")
        assert result_string == bitstring


def _hubbard_integrals(norb: int, u: float = 2.0):
    """One- and two-body tensors of a 1-D Hubbard chain, as a cheap test Hamiltonian."""
    one_body = np.zeros((norb, norb))
    for p in range(norb - 1):
        one_body[p, p + 1] = one_body[p + 1, p] = -1.0
    two_body = np.zeros((norb,) * 4)
    for p in range(norb):
        two_body[p, p, p, p] = u
    return one_body, two_body


def _trim_solver(record, *, trim_to=4, return_sci_state=True):
    """An sci_solver that selects its own carryover, as a distributed solver would.

    It diagonalizes with the default solver, then keeps only the ``trim_to``
    highest-weight strings per spin sector, reporting them via
    ``SCIResult.carryover``. ``record`` collects the carryover it returned on
    each call, so a test can check what the loop did with it.
    """

    def solver(ci_strings, one_body_tensor, two_body_tensor, norb, nelec):
        results = []
        for strs_a, strs_b in ci_strings:
            result = solve_sci(
                (strs_a, strs_b), one_body_tensor, two_body_tensor, norb=norb, nelec=nelec
            )
            state = result.sci_state
            weights_a = np.sum(np.abs(state.amplitudes) ** 2, axis=1)
            weights_b = np.sum(np.abs(state.amplitudes) ** 2, axis=0)
            keep_a = state.ci_strs_a[np.argsort(weights_a)[::-1][:trim_to]]
            keep_b = state.ci_strs_b[np.argsort(weights_b)[::-1][:trim_to]]
            record.append((keep_a, keep_b))
            results.append(
                SCIResult(
                    energy=result.energy,
                    # A solver that never materializes the eigenvector leaves this None.
                    sci_state=state if return_sci_state else None,
                    orbital_occupancies=result.orbital_occupancies,
                    carryover=(keep_a, keep_b),
                )
            )
        return results

    return solver


def test_sci_result_defaults_preserve_existing_behavior():
    """The new fields are optional and default to None."""
    state = SCIState(
        amplitudes=np.array([[1.0]]),
        ci_strs_a=np.array([0b011]),
        ci_strs_b=np.array([0b011]),
        norb=3,
        nelec=(2, 2),
    )
    result = SCIResult(energy=-1.0, sci_state=state, orbital_occupancies=(np.zeros(3), np.zeros(3)))
    assert result.carryover is None


@pytest.mark.parametrize("return_sci_state", [True, False])
def test_solver_supplied_carryover_is_used(return_sci_state):
    """A solver's own carryover strings seed the next iteration's subspace.

    Also covers ``sci_state=None``: a solver that does not materialize the
    eigenvector must still drive the loop, provided it supplies the carryover.
    """
    norb = 6
    nelec = (3, 3)
    one_body, two_body = _hubbard_integrals(norb)
    rng = np.random.default_rng(1234)
    bit_array = generate_bit_array_uniform(2_000, 2 * norb, rand_seed=rng)

    record: list[tuple[np.ndarray, np.ndarray]] = []
    trim_to = 4
    subspaces: list[tuple[np.ndarray, np.ndarray]] = []

    def solver(ci_strings, *args, **kwargs):
        subspaces.extend(ci_strings)
        return _trim_solver(record, trim_to=trim_to, return_sci_state=return_sci_state)(
            ci_strings, *args, **kwargs
        )

    result = diagonalize_fermionic_hamiltonian(
        one_body,
        two_body,
        bit_array,
        samples_per_batch=20,
        norb=norb,
        nelec=nelec,
        max_iterations=3,
        sci_solver=solver,
        carryover_threshold=1e-4,
        seed=rng,
    )

    assert len(record) >= 2, "the loop should have run more than one iteration"
    assert isinstance(result.energy, float)
    if return_sci_state:
        assert result.sci_state is not None
    else:
        assert result.sci_state is None

    # Every string the solver asked to carry over must appear in the next subspace.
    for iteration, (keep_a, keep_b) in enumerate(record[:-1]):
        next_a, next_b = subspaces[iteration + 1]
        assert set(int(s) for s in keep_a) <= set(int(s) for s in next_a)
        assert set(int(s) for s in keep_b) <= set(int(s) for s in next_b)


def test_solver_supplied_carryover_ignores_carryover_threshold():
    """carryover_threshold does not apply when the solver chose the carryover itself."""
    norb = 6
    nelec = (3, 3)
    one_body, two_body = _hubbard_integrals(norb)

    energies = []
    records = []
    for threshold in (1e-8, 0.5):
        rng = np.random.default_rng(99)
        bit_array = generate_bit_array_uniform(2_000, 2 * norb, rand_seed=rng)
        record: list[tuple[np.ndarray, np.ndarray]] = []
        records.append(record)
        result = diagonalize_fermionic_hamiltonian(
            one_body,
            two_body,
            bit_array,
            samples_per_batch=20,
            norb=norb,
            nelec=nelec,
            max_iterations=3,
            sci_solver=_trim_solver(record),
            carryover_threshold=threshold,
            seed=rng,
        )
        energies.append(result.energy)

    # A threshold spanning eight orders of magnitude changes nothing, because the
    # solver's carryover is used verbatim.
    assert energies[0] == energies[1]
    assert len(records[0]) == len(records[1])
    for (a1, b1), (a2, b2) in zip(records[0], records[1]):
        np.testing.assert_array_equal(a1, a2)
        np.testing.assert_array_equal(b1, b2)


def test_no_sci_state_and_no_carryover_raises():
    """A result with neither an SCI state nor carryover strings cannot drive the loop."""
    norb = 6
    nelec = (3, 3)
    one_body, two_body = _hubbard_integrals(norb)
    rng = np.random.default_rng(7)
    bit_array = generate_bit_array_uniform(500, 2 * norb, rand_seed=rng)

    # The integrals are unused here, as they are in the Fulqrum and SBD solvers, which
    # hold the Hamiltonian in their own format. See
    # https://github.com/Qiskit/qiskit-addon-sqd/issues/312.
    def solver(ci_strings, _one_body_tensor, _two_body_tensor, norb, _nelec):
        return [
            SCIResult(
                energy=-1.0 - index,
                sci_state=None,
                orbital_occupancies=(np.zeros(norb), np.zeros(norb)),
            )
            for index, _ in enumerate(ci_strings)
        ]

    with pytest.raises(ValueError, match=r"SCIResult\.carryover"):
        diagonalize_fermionic_hamiltonian(
            one_body,
            two_body,
            bit_array,
            samples_per_batch=20,
            norb=norb,
            nelec=nelec,
            max_iterations=3,
            sci_solver=solver,
            seed=rng,
        )


def test_solver_supplied_carryover_symmetrize_spin():
    """With symmetrize_spin, the solver's carryover is merged into one list."""
    norb = 6
    nelec = (3, 3)
    one_body, two_body = _hubbard_integrals(norb)
    rng = np.random.default_rng(5150)
    bit_array = generate_bit_array_uniform(2_000, 2 * norb, rand_seed=rng)

    subspaces: list[tuple[np.ndarray, np.ndarray]] = []

    def solver(ci_strings, *args, **kwargs):
        subspaces.extend(ci_strings)
        return _trim_solver([])(ci_strings, *args, **kwargs)

    diagonalize_fermionic_hamiltonian(
        one_body,
        two_body,
        bit_array,
        samples_per_batch=20,
        norb=norb,
        nelec=nelec,
        max_iterations=3,
        sci_solver=solver,
        symmetrize_spin=True,
        seed=rng,
    )

    # Spin symmetrization means both sectors span the same strings.
    for strs_a, strs_b in subspaces:
        np.testing.assert_array_equal(strs_a, strs_b)


def test_sci_state_save_load(tmp_path):
    """Test saving and loading SCIState."""
    norb = 5
    nelec = (3, 2)
    ci_strs_a = np.array([0b00111, 0b01011])
    ci_strs_b = np.array([0b00011, 0b00101])
    amplitudes = np.array([[0.5, 0.5], [0.5, 0.5]])

    sci_state = SCIState(
        amplitudes=amplitudes, ci_strs_a=ci_strs_a, ci_strs_b=ci_strs_b, norb=norb, nelec=nelec
    )
    filepath = tmp_path / "sci_state.npz"
    sci_state.save(filepath)
    loaded_state = SCIState.load(filepath)

    np.testing.assert_array_equal(loaded_state.amplitudes, sci_state.amplitudes)
    np.testing.assert_array_equal(loaded_state.ci_strs_a, sci_state.ci_strs_a)
    np.testing.assert_array_equal(loaded_state.ci_strs_b, sci_state.ci_strs_b)
    assert loaded_state.norb == sci_state.norb
    assert loaded_state.nelec == sci_state.nelec


def test_batch_to_ci_strings_ranks_by_sample_count():
    """Strings are returned by descending number of times sampled, carryover first."""
    norb = 3
    # Beta is the left half of a row, alpha the right half, and rows are MSB-first.
    a1, a2 = [True, True, False], [True, False, True]  # 0b110, 0b101
    b1, b2 = [False, True, True], [True, True, False]  # 0b011, 0b110
    batch = np.array([b1 + a1, b1 + a1, b2 + a2])
    # Alpha 0b110 sampled twice, 0b101 once. Beta 0b011 twice, 0b110 once.
    strs_a, strs_b = batch_to_ci_strings(batch, norb)
    assert list(strs_a) == [0b110, 0b101]
    assert list(strs_b) == [0b011, 0b110]

    # Carryover strings come first, in the order given, regardless of sample counts.
    strs_a, _ = batch_to_ci_strings(batch, norb, np.array([0b101, 0b011]), np.array([]))
    assert list(strs_a) == [0b101, 0b011, 0b110, 0b101]


def test_batch_to_ci_strings_symmetrize_spin_ranks_across_sectors():
    """With symmetrize_spin the ranking is over both sectors at once, not within each.

    This is why the merge cannot be deferred until after ranking: merging first can
    interleave the two sectors' strings differently than concatenating ranked lists
    would, which changes what a later truncation keeps.
    """
    norb = 3
    a1 = [True, True, False]  # 0b110
    b1, b2 = [True, False, True], [False, True, True]  # 0b101, 0b011
    batch = np.array([b1 + a1, b2 + a1])
    # Alpha 0b110 sampled twice; each beta string once. So it ranks first overall.
    strs_a, strs_b = batch_to_ci_strings(batch, norb, symmetrize_spin=True)
    assert strs_a is strs_b
    assert next(iter(strs_a)) == 0b110
    assert set(strs_a) == {0b110, 0b101, 0b011}


def test_default_policy_matches_omitting_it():
    """Passing StandardPolicy explicitly is the same as passing no policy at all."""
    norb, nelec = 6, (3, 3)
    one_body, two_body = _hubbard_integrals(norb)

    energies = []
    for kwargs in ({}, {"policy": StandardPolicy()}):
        rng = np.random.default_rng(4321)
        bit_array = generate_bit_array_uniform(2_000, 2 * norb, rand_seed=rng)
        result = diagonalize_fermionic_hamiltonian(
            one_body,
            two_body,
            bit_array,
            samples_per_batch=20,
            norb=norb,
            nelec=nelec,
            num_batches=3,
            max_iterations=3,
            seed=rng,
            **kwargs,
        )
        energies.append(result.energy)

    assert energies[0] == energies[1]


def test_policy_refine_drives_a_second_round():
    """A policy that refines gets a second solver call, then is asked again and stops."""
    norb, nelec = 6, (3, 3)
    one_body, two_body = _hubbard_integrals(norb)

    class TwoRound(StandardPolicy):
        """Merge the batches after the first round and diagonalize the union."""

        def refine(self, results, round_index):
            if round_index:
                return None
            states = [result.sci_state for result in results]
            return [
                (
                    np.unique(np.concatenate([state.ci_strs_a for state in states])),
                    np.unique(np.concatenate([state.ci_strs_b for state in states])),
                )
            ]

        def select_result(self, results):
            (result,) = results
            return result

    rounds: list[int] = []

    class Recording(TwoRound):
        def refine(self, results, round_index):
            rounds.append(round_index)
            return super().refine(results, round_index)

    call_sizes: list[int] = []

    def spy(ci_strings, *args, **kwargs):
        call_sizes.append(len(ci_strings))
        return solve_sci_batch(ci_strings, *args, **kwargs)

    rng = np.random.default_rng(7)
    bit_array = generate_bit_array_uniform(2_000, 2 * norb, rand_seed=rng)
    diagonalize_fermionic_hamiltonian(
        one_body,
        two_body,
        bit_array,
        samples_per_batch=20,
        norb=norb,
        nelec=nelec,
        num_batches=4,
        max_iterations=1,
        sci_solver=spy,
        policy=Recording(),
        seed=rng,
    )

    # refine is asked about round 0, returns a subspace, then asked about round 1 and
    # stops. The solver therefore sees the four batches, then the single merged subspace.
    assert rounds == [0, 1]
    assert call_sizes == [4, 1]


def test_policy_that_never_stops_raises():
    """A refine that always returns subspaces hits the round limit instead of hanging."""
    norb, nelec = 6, (3, 3)
    one_body, two_body = _hubbard_integrals(norb)

    class NeverStops(StandardPolicy):
        def refine(self, results, round_index):
            state = results[0].sci_state
            return [(state.ci_strs_a, state.ci_strs_b)]

    rng = np.random.default_rng(3)
    bit_array = generate_bit_array_uniform(2_000, 2 * norb, rand_seed=rng)
    with pytest.raises(ValueError, match="asked for another round"):
        diagonalize_fermionic_hamiltonian(
            one_body,
            two_body,
            bit_array,
            samples_per_batch=20,
            norb=norb,
            nelec=nelec,
            max_iterations=1,
            policy=NeverStops(),
            seed=rng,
        )


def test_carryover_threshold_with_a_policy_raises():
    """The threshold is a setting of the default policy, so pairing them is an error."""
    norb, nelec = 6, (3, 3)
    one_body, two_body = _hubbard_integrals(norb)
    rng = np.random.default_rng(5)
    bit_array = generate_bit_array_uniform(500, 2 * norb, rand_seed=rng)

    with pytest.raises(ValueError, match=r"carryover_threshold=1e-06"):
        diagonalize_fermionic_hamiltonian(
            one_body,
            two_body,
            bit_array,
            samples_per_batch=20,
            norb=norb,
            nelec=nelec,
            max_iterations=1,
            policy=StandardPolicy(),
            carryover_threshold=1e-6,
            seed=rng,
        )


def test_policy_select_carryover_is_honored():
    """The strings a policy carries over are the ones the next iteration builds from."""
    norb, nelec = 6, (3, 3)
    one_body, two_body = _hubbard_integrals(norb)

    chosen: list[tuple[np.ndarray, np.ndarray]] = []

    class FixedCarryover(StandardPolicy):
        """Carry over only the two highest-weight strings of each sector."""

        def select_carryover(self, result, *, symmetrize_spin=False):
            strs_a, strs_b = super().select_carryover(result, symmetrize_spin=symmetrize_spin)
            picked = (strs_a[:2], strs_b[:2])
            chosen.append(picked)
            return picked

    seen: list[tuple[np.ndarray, np.ndarray]] = []

    def spy(ci_strings, *args, **kwargs):
        seen.extend(ci_strings)
        return solve_sci_batch(ci_strings, *args, **kwargs)

    rng = np.random.default_rng(13)
    bit_array = generate_bit_array_uniform(2_000, 2 * norb, rand_seed=rng)
    diagonalize_fermionic_hamiltonian(
        one_body,
        two_body,
        bit_array,
        samples_per_batch=20,
        norb=norb,
        nelec=nelec,
        num_batches=2,
        max_iterations=2,
        sci_solver=spy,
        policy=FixedCarryover(),
        seed=rng,
    )

    assert chosen, "the policy should have been asked for a carryover"
    # The strings chosen after the first iteration appear in the subspaces built next.
    first_a, _ = chosen[0]
    later_a = {int(s) for strs_a, _ in seen[2:] for s in strs_a}
    assert {int(s) for s in first_a} <= later_a


def test_policies_are_picklable():
    """A policy and a request must survive the broadcast to the other processes."""
    request = SubspaceRequest(
        bitstrings=np.array([[True, False]]),
        probabilities=np.array([1.0]),
        carryover_strings_a=np.array([1]),
        carryover_strings_b=np.array([2]),
        norb=1,
        nelec=(1, 1),
        samples_per_batch=3,
        num_batches=4,
        rng=np.random.default_rng(0),
        iteration=0,
        symmetrize_spin=False,
    )
    restored = pickle.loads(pickle.dumps(request))
    assert restored.samples_per_batch == 3
    assert restored.num_batches == 4
    assert restored.iteration == 0

    policy = pickle.loads(pickle.dumps(StandardPolicy(carryover_threshold=1e-3)))
    assert policy.carryover_threshold == 1e-3


def test_standard_policy_satisfies_the_protocol():
    """The protocol is runtime-checkable, so a user can assert conformance."""
    assert isinstance(StandardPolicy(), SubspacePolicy)
