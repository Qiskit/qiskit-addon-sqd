// This code is a Qiskit project.
//
// (C) Copyright IBM 2026.
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

#include <array>
#include <cstdint>
#include <random>
#include <utility>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/pair.h>

#include "dynamic_bitset.hpp"
#include "qiskit/addon/sqd/configuration_recovery.hpp"

namespace nb = nanobind;

using qiskit_addon_sqd_accel::DynamicBitset;

namespace
{

// Accelerated configuration recovery.
//
// The bitstring matrix is expected in the *C++ bit convention* used by
// Qiskit::addon::sqd::recover_configurations: within each row, the first
// `partition_size` columns are the alpha (right/lower) orbitals and the next
// `partition_size` columns are the beta (left/upper) orbitals, with column j
// mapping to bit j.  The Python wrapper is responsible for translating the
// public numpy layout into this convention (and back), which keeps the
// orientation logic in Python where it is easy to test.
//
// - bitstring_matrix: (num_samples, 2 * partition_size) array of uint8 (0/1)
// - probabilities: (num_samples,) array of float64
// - occ_alpha, occ_beta: (partition_size,) arrays of float64
// - num_elec_a, num_elec_b: target Hamming weights for alpha and beta
// - seed: seed for the std::mt19937_64 used internally
//
// Returns (corrected_matrix, corrected_probs) with corrected_matrix in the same
// convention as the input.
std::pair<
    nb::ndarray<nb::numpy, std::uint8_t, nb::ndim<2>>,
    nb::ndarray<nb::numpy, double, nb::ndim<1>>>
recover_configurations(
    nb::ndarray<const std::uint8_t, nb::ndim<2>, nb::c_contig> bitstring_matrix,
    nb::ndarray<const double, nb::ndim<1>, nb::c_contig> probabilities,
    nb::ndarray<const double, nb::ndim<1>, nb::c_contig> occ_alpha,
    nb::ndarray<const double, nb::ndim<1>, nb::c_contig> occ_beta,
    std::uint64_t num_elec_a, std::uint64_t num_elec_b, std::uint64_t seed
)
{
    const std::size_t num_samples = bitstring_matrix.shape(0);
    const std::size_t num_bits = bitstring_matrix.shape(1);
    const std::size_t partition_size = occ_alpha.size();

    // Build the inputs for the C++ implementation.
    std::vector<DynamicBitset> bitstrings;
    bitstrings.reserve(num_samples);
    const std::uint8_t *bs_data = bitstring_matrix.data();
    for (std::size_t i = 0; i < num_samples; ++i) {
        DynamicBitset bs(num_bits);
        const std::uint8_t *row = bs_data + i * num_bits;
        for (std::size_t j = 0; j < num_bits; ++j) {
            if (row[j]) {
                bs.set(j, true);
            }
        }
        bitstrings.push_back(std::move(bs));
    }

    std::vector<double> probs(probabilities.data(), probabilities.data() + probabilities.size());

    std::array<std::vector<double>, 2> avg_occupancies;
    avg_occupancies[0].assign(occ_alpha.data(), occ_alpha.data() + occ_alpha.size());
    avg_occupancies[1].assign(occ_beta.data(), occ_beta.data() + occ_beta.size());

    std::mt19937_64 rng(seed);

    auto [out_bitstrings, out_probs] = Qiskit::addon::sqd::recover_configurations(
        bitstrings, probs, avg_occupancies, {num_elec_a, num_elec_b}, rng
    );

    // Marshal the corrected bitstrings back into a numpy uint8 matrix.
    const std::size_t out_rows = out_bitstrings.size();
    auto *out_mat = new std::uint8_t[out_rows * num_bits];
    for (std::size_t i = 0; i < out_rows; ++i) {
        const auto &bs = out_bitstrings[i];
        std::uint8_t *row = out_mat + i * num_bits;
        for (std::size_t j = 0; j < num_bits; ++j) {
            row[j] = bs[j] ? 1 : 0;
        }
    }
    nb::capsule mat_owner(out_mat, [](void *p) noexcept { delete[] static_cast<std::uint8_t *>(p); });

    auto *out_p = new double[out_rows];
    for (std::size_t i = 0; i < out_rows; ++i) {
        out_p[i] = out_probs[i];
    }
    nb::capsule probs_owner(out_p, [](void *p) noexcept { delete[] static_cast<double *>(p); });

    (void)partition_size;

    return {
        nb::ndarray<nb::numpy, std::uint8_t, nb::ndim<2>>(
            out_mat, {out_rows, num_bits}, mat_owner
        ),
        nb::ndarray<nb::numpy, double, nb::ndim<1>>(out_p, {out_rows}, probs_owner),
    };
}

} // namespace

NB_MODULE(_accel, m)
{
    m.def(
        "recover_configurations", &recover_configurations, nb::arg("bitstring_matrix"),
        nb::arg("probabilities"), nb::arg("occ_alpha"), nb::arg("occ_beta"),
        nb::arg("num_elec_a"), nb::arg("num_elec_b"), nb::arg("seed"),
        "Accelerated configuration recovery backed by qiskit-addon-sqd-hpc."
    );
}
