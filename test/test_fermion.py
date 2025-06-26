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
from pyscf.fci import cistring, spin_square
from qiskit.primitives import BitArray
from qiskit_addon_sqd.counts import generate_bit_array_uniform
from qiskit_addon_sqd.fermion import bitstring_matrix_to_ci_strs, diagonalize_fermionic_hamiltonian


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

    def test_diagonalize_fermionic_hamiltonian(self):
        """Test diagonalize_fermionic_hamiltonian."""
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
