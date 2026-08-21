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

"""Tests for the fermion module."""

import math
import unittest

import numpy as np
import pyscf
import pyscf.mcscf
import pytest
from pyscf.fci import cistring, spin_square
from qiskit.primitives import BitArray
from qiskit_addon_sqd.counts import generate_bit_array_uniform
from qiskit_addon_sqd.fermion import (
    SCIState,
    bitstring_matrix_to_ci_strs,
    diagonalize_fermionic_hamiltonian,
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

    def test_diagonalize_fermionic_hamiltonian_resume(self):
        """Test diagonalize_fermionic_hamiltonian restart capability via initial_state."""
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

        # Define shared keyword arguments
        kwargs = dict(
            one_body_tensor=hcore,
            two_body_tensor=eri,
            bit_array=bit_array,
            samples_per_batch=10,
            norb=norb,
            nelec=nelec,
            symmetrize_spin=True,
            energy_tol=1e-12,
            occupancies_tol=1e-12,
            # Note: initial_occupancies is left as default None so the code
            # extracts it from the initial_state argument on the resumed run
        )

        # Create two independent but identical Random Number Generators
        rng_continuous = np.random.default_rng(12345)
        rng_interrupted = np.random.default_rng(12345)

        # Execute the continuous reference (2 iterations) using the first RNG
        result_continuous = diagonalize_fermionic_hamiltonian(
            max_iterations=2, seed=rng_continuous, **kwargs
        )

        # Execute the interrupted run (1 iteration) using the second RNG
        result_interrupted = diagonalize_fermionic_hamiltonian(
            max_iterations=1, seed=rng_interrupted, **kwargs
        )

        # Resume the interrupted run for 1 more iteration using the same RNG.
        # Its internal state was stored during the interrupted run, so it
        # can be resumed and its final state should match the continuous run.
        result_resumed = diagonalize_fermionic_hamiltonian(
            max_iterations=1,
            initial_state=result_interrupted.sci_state,
            seed=rng_interrupted,
            **kwargs,
        )

        # Check that the resumed run's results exactly match the continuous
        # run's results
        np.testing.assert_allclose(result_resumed.energy, result_continuous.energy)
        np.testing.assert_allclose(
            result_resumed.sci_state.amplitudes, result_continuous.sci_state.amplitudes
        )
        np.testing.assert_array_equal(
            result_resumed.sci_state.ci_strs_a, result_continuous.sci_state.ci_strs_a
        )
        np.testing.assert_array_equal(
            result_resumed.sci_state.ci_strs_b, result_continuous.sci_state.ci_strs_b
        )

    def test_extract_carryover_strings(self):
        """Test the extraction and sorting logic of carryover strings."""
        # Import the private helper directly for testing
        from qiskit_addon_sqd.fermion import _extract_carryover_strings

        # Construct a highly predictable 3x3 mock state
        ci_strs_a = np.array([10, 20, 30])
        ci_strs_b = np.array([1, 2, 3])

        # Amplitudes are chosen so we can easily calculate marginal weights:
        # Alpha weights (sum of squares across rows): [0.65, 0.36, 0.04]
        # Beta weights (sum of squares across cols):  [0.37, 0.64, 0.00]
        amplitudes = np.array([[0.1, 0.8, 0.0], [0.6, 0.0, 0.0], [0.0, 0.0, 0.2]])

        sci_state = SCIState(
            amplitudes=amplitudes, ci_strs_a=ci_strs_a, ci_strs_b=ci_strs_b, norb=3, nelec=(1, 1)
        )

        # Test without spin symmetrization
        # A threshold of 0.5 should only retain amplitudes 0.8 (0, 1)
        # and 0.6 (1, 0).
        str_a, str_b = _extract_carryover_strings(
            sci_state, carryover_threshold=0.5, symmetrize_spin=False
        )

        # Alpha keeps row indices 0 and 1 (strings 10 and 20).
        # Their weights are 0.65 and 0.36, so they should be sorted as [10, 20].
        np.testing.assert_array_equal(str_a, [10, 20])

        # Beta keeps column indices 1 and 0 (strings 2 and 1).
        # Their weights are 0.64 and 0.37, so they should be sorted as [2, 1].
        np.testing.assert_array_equal(str_b, [2, 1])

        # Test with spin symmetrization
        str_a_sym, str_b_sym = _extract_carryover_strings(
            sci_state, carryover_threshold=0.5, symmetrize_spin=True
        )

        # Symmetrization merges the kept strings into: [10, 20, 1, 2]
        # Their respective weights are: [0.65, 0.36, 0.37, 0.64]
        # Sorted by weight descending: 0.65 (10), 0.64 (2), 0.37 (1), 0.36 (20)
        expected_sym = [10, 2, 1, 20]

        np.testing.assert_array_equal(str_a_sym, expected_sym)
        np.testing.assert_array_equal(str_b_sym, expected_sym)

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
