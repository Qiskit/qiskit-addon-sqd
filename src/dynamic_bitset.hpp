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

#ifndef QISKIT_ADDON_SQD_ACCEL_DYNAMIC_BITSET_HPP_
#define QISKIT_ADDON_SQD_ACCEL_DYNAMIC_BITSET_HPP_

// A minimal runtime-sized bitset supporting exactly the operations that
// Qiskit::addon::sqd::recover_configurations requires of its BitstringType:
// size(), count(), operator[], flip(idx), operator<<=, operator>>=,
// operator==, copy construction, and std::hash (for unordered_map dedup).
//
// This lets us drive the header-only C++ configuration-recovery implementation
// without vendoring boost::dynamic_bitset and its dependencies.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

namespace qiskit_addon_sqd_accel
{

class DynamicBitset
{
  public:
    using Block = std::uint64_t;
    static constexpr std::size_t bits_per_block = 64;

    DynamicBitset() = default;

    explicit DynamicBitset(std::size_t num_bits)
      : num_bits_(num_bits), blocks_((num_bits + bits_per_block - 1) / bits_per_block, 0)
    {
    }

    std::size_t size() const { return num_bits_; }

    // Read a single bit.  Returns bool so that `bitstring[j] == flip` (where
    // `flip` is a bool) compiles and behaves as expected.
    bool operator[](std::size_t idx) const
    {
        return (blocks_[idx / bits_per_block] >> (idx % bits_per_block)) & Block(1);
    }

    void set(std::size_t idx, bool value = true)
    {
        const auto b = idx / bits_per_block;
        const auto shift = idx % bits_per_block;
        if (value) {
            blocks_[b] |= (Block(1) << shift);
        } else {
            blocks_[b] &= ~(Block(1) << shift);
        }
    }

    void flip(std::size_t idx)
    {
        blocks_[idx / bits_per_block] ^= (Block(1) << (idx % bits_per_block));
    }

    std::size_t count() const
    {
        std::size_t total = 0;
        for (const Block block : blocks_) {
#if defined(__GNUC__) || defined(__clang__)
            total += static_cast<std::size_t>(__builtin_popcountll(block));
#else
            Block v = block;
            while (v) {
                v &= (v - 1);
                ++total;
            }
#endif
        }
        return total;
    }

    // Logical left shift by `shift` bit positions.
    DynamicBitset &operator<<=(std::size_t shift)
    {
        if (shift >= num_bits_) {
            for (Block &block : blocks_) {
                block = 0;
            }
            return *this;
        }
        const std::size_t block_shift = shift / bits_per_block;
        const std::size_t bit_shift = shift % bits_per_block;
        if (bit_shift == 0) {
            for (std::size_t i = blocks_.size(); i-- > block_shift;) {
                blocks_[i] = blocks_[i - block_shift];
            }
        } else {
            for (std::size_t i = blocks_.size(); i-- > block_shift;) {
                Block lo = blocks_[i - block_shift] << bit_shift;
                Block hi = (i - block_shift >= 1)
                               ? (blocks_[i - block_shift - 1] >> (bits_per_block - bit_shift))
                               : Block(0);
                blocks_[i] = lo | hi;
            }
        }
        for (std::size_t i = 0; i < block_shift; ++i) {
            blocks_[i] = 0;
        }
        clear_high_bits();
        return *this;
    }

    // Logical right shift by `shift` bit positions.
    DynamicBitset &operator>>=(std::size_t shift)
    {
        if (shift >= num_bits_) {
            for (Block &block : blocks_) {
                block = 0;
            }
            return *this;
        }
        const std::size_t block_shift = shift / bits_per_block;
        const std::size_t bit_shift = shift % bits_per_block;
        const std::size_t n = blocks_.size();
        if (bit_shift == 0) {
            for (std::size_t i = 0; i + block_shift < n; ++i) {
                blocks_[i] = blocks_[i + block_shift];
            }
        } else {
            for (std::size_t i = 0; i + block_shift < n; ++i) {
                Block lo = blocks_[i + block_shift] >> bit_shift;
                Block hi = (i + block_shift + 1 < n)
                               ? (blocks_[i + block_shift + 1] << (bits_per_block - bit_shift))
                               : Block(0);
                blocks_[i] = lo | hi;
            }
        }
        for (std::size_t i = (n >= block_shift) ? n - block_shift : 0; i < n; ++i) {
            blocks_[i] = 0;
        }
        // No need to clear high bits: a right shift never populates them.
        return *this;
    }

    bool operator==(const DynamicBitset &other) const
    {
        return num_bits_ == other.num_bits_ && blocks_ == other.blocks_;
    }

    const std::vector<Block> &blocks() const { return blocks_; }

  private:
    // After a left shift, bits at positions >= num_bits_ within the top block
    // must be cleared so that count() and equality stay correct.
    void clear_high_bits()
    {
        const std::size_t rem = num_bits_ % bits_per_block;
        if (rem != 0 && !blocks_.empty()) {
            const Block mask = (Block(1) << rem) - 1;
            blocks_.back() &= mask;
        }
    }

    std::size_t num_bits_ = 0;
    std::vector<Block> blocks_;
};

} // namespace qiskit_addon_sqd_accel

namespace std
{
template <>
struct hash<qiskit_addon_sqd_accel::DynamicBitset> {
    std::size_t operator()(const qiskit_addon_sqd_accel::DynamicBitset &bs) const
    {
        // Combine the block hashes (boost::hash_combine style).
        std::size_t seed = bs.size();
        for (const auto block : bs.blocks()) {
            seed ^= std::hash<std::uint64_t>{}(block) + 0x9e3779b97f4a7c15ULL + (seed << 6) +
                    (seed >> 2);
        }
        return seed;
    }
};
} // namespace std

#endif // QISKIT_ADDON_SQD_ACCEL_DYNAMIC_BITSET_HPP_
