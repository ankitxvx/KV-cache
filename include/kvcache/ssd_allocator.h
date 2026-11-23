#pragma once

#include "types.h"
#include "config.h"
#include "logger.h"
#include <vector>
#include <mutex>
#include <string>
#include <cstring>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace kvcache {

/// NVMe SSD-backed block allocator using memory-mapped I/O.
/// Creates a pre-allocated file on disk and maps it into virtual memory
/// for zero-copy block access.
class SsdAllocator {
public:
    explicit SsdAllocator(const ProductionConfig& config);
    ~SsdAllocator();

    /// Allocate a free SSD block. Returns invalid BlockId if exhausted.
    BlockId allocate();

    /// Free an SSD block.
    void free(BlockId id);

    /// Get a mutable pointer to a block's data in the memory-mapped region.
    void* get_block_ptr(BlockId id);
    const void* get_block_ptr(BlockId id) const;

    /// Get KV slot pointers within a block.
    KVPairMut get_kv_slot_mut(BlockId id, int layer, int slot);
    KVPair get_kv_slot(BlockId id, int layer, int slot) const;

    /// Copy a block's data to/from a CPU buffer.
    void copy_to_buffer(BlockId id, void* dst) const;
    void copy_from_buffer(BlockId id, const void* src);

    int num_free() const;
    int num_used() const;
    int num_total() const { return max_blocks_; }
    size_t block_bytes() const { return block_total_bytes_; }
    size_t total_memory_bytes() const { return static_cast<size_t>(max_blocks_) * block_total_bytes_; }

    /// Whether the SSD pool file was successfully mapped.
    bool is_mapped() const { return mapped_ptr_ != nullptr; }

    /// Get the file path.
    const std::string& file_path() const { return file_path_; }

    /// Get/set used slots.
    int get_used_slots(BlockId id) const;
    void set_used_slots(BlockId id, int count);

private:
    bool valid_block(BlockId id) const {
        return id.valid() && id.value < max_blocks_;
    }

    ProductionConfig config_;
    int max_blocks_           = 0;
    size_t block_total_bytes_ = 0;
    std::vector<bool> allocated_;
    size_t kv_per_token_bytes_ = 0;
    size_t layer_bytes_        = 0;
    size_t total_file_bytes_   = 0;
    std::string file_path_;

    uint8_t* mapped_ptr_      = nullptr;   // Memory-mapped file pointer

#ifdef _WIN32
    HANDLE file_handle_       = INVALID_HANDLE_VALUE;
    HANDLE mapping_handle_    = nullptr;
#else
    int fd_                   = -1;
#endif

    std::vector<int> used_slots_;
    std::vector<BlockId> free_list_;
    mutable std::mutex mutex_;
};

} // namespace kvcache
