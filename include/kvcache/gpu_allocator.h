#pragma once

#include "types.h"
#include "config.h"
#include "logger.h"
#include <vector>
#include <mutex>
#include <cstring>

namespace kvcache {

/// GPU VRAM block allocator using CUDA device memory.
/// When CUDA is not available, this falls back to regular CPU allocation
/// (useful for development/testing without a GPU).
class GpuAllocator {
public:
    explicit GpuAllocator(const ProductionConfig& config);
    ~GpuAllocator();

    /// Allocate a free GPU block. Returns invalid BlockId if exhausted.
    BlockId allocate();

    /// Free a GPU block.
    void free(BlockId id);

    /// Get a mutable pointer to a block's data.
    void* get_block_ptr(BlockId id);
    const void* get_block_ptr(BlockId id) const;

    /// Get mutable KV slot pointers within a block.
    KVPairMut get_kv_slot_mut(BlockId id, int layer, int slot);
    KVPair get_kv_slot(BlockId id, int layer, int slot) const;

    /// Copy a block's data to CPU host memory.
    void copy_to_host(BlockId id, void* host_dst) const;

    /// Copy from CPU host memory into a GPU block.
    void copy_from_host(BlockId id, const void* host_src);

    /// Copy between GPU blocks.
    void copy_block(BlockId dst, BlockId src);

    int num_free() const;
    int num_used() const;
    int num_total() const { return max_blocks_; }
    size_t block_bytes() const { return block_total_bytes_; }
    size_t total_memory_bytes() const { return static_cast<size_t>(max_blocks_) * block_total_bytes_; }

    /// Whether this allocator is backed by real CUDA GPU memory.
    bool is_cuda_backed() const { return cuda_available_; }

    /// Get/set used slots.
    int get_used_slots(BlockId id) const;
    void set_used_slots(BlockId id, int count);

    /// Get/set reference count.
    int get_ref_count(BlockId id) const;
    void ref(BlockId id);
    void unref(BlockId id);

private:
    size_t compute_offset(BlockId id) const;
    bool valid_block(BlockId id) const {
        return id.valid() && id.value < max_blocks_;
    }

    ProductionConfig config_;
    int max_blocks_          = 0;
    size_t block_total_bytes_ = 0;
    size_t kv_per_token_bytes_ = 0;
    size_t layer_bytes_        = 0;
    bool cuda_available_       = false;

    void* device_pool_       = nullptr;   // CUDA device pointer (or CPU fallback)
    std::vector<int> ref_counts_;
    std::vector<int> used_slots_;
    std::vector<BlockId> free_list_;
    mutable std::mutex mutex_;
};

} // namespace kvcache
