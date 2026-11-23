#pragma once

#include "types.h"
#include <vector>
#include <mutex>
#include <memory>
#include <cstring>
#include <cassert>

namespace kvcache {

/// A physical memory block that stores KV data for `block_size` tokens
/// across all layers. Memory layout per layer:
///   [block_size * num_kv_heads * head_dim] for K
///   [block_size * num_kv_heads * head_dim] for V
///
/// The block holds data for ALL layers contiguously:
///   Layer 0 K | Layer 0 V | Layer 1 K | Layer 1 V | ...
struct PhysicalBlock {
    BlockId        id;
    int            ref_count   = 0;     ///< Reference count for Copy-on-Write
    int            used_slots  = 0;     ///< Number of token slots currently written
    std::vector<uint8_t> data;          ///< Raw byte buffer

    PhysicalBlock() = default;

    void init(BlockId block_id, size_t total_bytes) {
        id = block_id;
        ref_count = 0;
        used_slots = 0;
        data.resize(total_bytes, 0);
    }

    void reset() {
        ref_count = 0;
        used_slots = 0;
        std::memset(data.data(), 0, data.size());
    }
};

/// Pool-based block allocator implementing paged memory management.
/// All blocks are pre-allocated at construction time.
/// Allocation and deallocation are O(1) via a free list.
class BlockAllocator {
public:
    explicit BlockAllocator(const KVCacheConfig& config);

    /// Allocate a free block. Returns invalid BlockId if pool is exhausted.
    BlockId allocate();

    /// Free a block (decrements ref count; only truly freed when ref_count reaches 0).
    void free(BlockId id);

    /// Increment reference count (for Copy-on-Write sharing).
    void ref(BlockId id);

    /// Decrement reference count; frees if it drops to 0.
    void unref(BlockId id);

    /// Get the reference count for a block.
    int get_ref_count(BlockId id) const;

    /// Get a raw pointer to the start of a block's entire memory buffer.
    void* get_block_ptr(BlockId id);
    const void* get_block_ptr(BlockId id) const;

    /// Get a raw mutable pointer to the start of a specific layer's K data
    /// within a block. Returns nullptr on invalid input.
    void* get_key_ptr(BlockId id, int layer);

    /// Get a raw mutable pointer to the start of a specific layer's V data
    /// within a block. Returns nullptr on invalid input.
    void* get_value_ptr(BlockId id, int layer);

    /// Get const pointers.
    const void* get_key_ptr(BlockId id, int layer) const;
    const void* get_value_ptr(BlockId id, int layer) const;

    /// Get a mutable KV pair for a specific token slot within a block/layer.
    KVPairMut get_kv_slot_mut(BlockId id, int layer, int slot);

    /// Get a const KV pair for a specific token slot within a block/layer.
    KVPair get_kv_slot(BlockId id, int layer, int slot) const;

    /// Get/set used slots count on a block.
    int get_used_slots(BlockId id) const;
    void set_used_slots(BlockId id, int count);

    /// Number of free blocks remaining.
    int num_free() const;

    /// Number of used (allocated) blocks.
    int num_used() const;

    /// Total number of blocks.
    int num_total() const;

    /// Total pool memory in bytes.
    size_t total_memory_bytes() const;

    /// Per-block memory in bytes (all layers).
    size_t block_bytes() const;

    /// Get the config.
    const KVCacheConfig& config() const { return config_; }

    /// Make a deep copy of a block's data into a newly allocated block.
    /// Used for Copy-on-Write when ref_count > 1 and a write is needed.
    BlockId copy_on_write(BlockId src_id);

private:
    size_t compute_layer_offset(int layer) const;
    size_t compute_kv_offset_in_layer(bool is_value) const;
    size_t compute_slot_offset(int slot) const;

    bool valid_block(BlockId id) const {
        return id.valid() && id.value < static_cast<int32_t>(blocks_.size());
    }

    KVCacheConfig              config_;
    std::vector<PhysicalBlock> blocks_;
    std::vector<BlockId>       free_list_;
    mutable std::mutex         mutex_;

    // Statistics
    int64_t total_allocations_ = 0;
    int64_t total_frees_       = 0;

    // Precomputed sizes
    size_t kv_per_token_bytes_;  ///< bytes for K or V of one token: num_kv_heads * head_dim * dtype_size
    size_t layer_bytes_;         ///< bytes per layer per block: block_size * 2 * kv_per_token_bytes_
    size_t block_total_bytes_;   ///< bytes per block total: num_layers * layer_bytes_
};

} // namespace kvcache
