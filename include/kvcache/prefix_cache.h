#pragma once

#include "types.h"
#include "block_allocator.h"
#include <vector>
#include <unordered_map>
#include <functional>

namespace kvcache {

/// Hash-based prefix cache for sharing KV blocks across sequences with common prefixes.
/// When multiple sequences share the same prompt tokens, their KV cache blocks can
/// be shared via Copy-on-Write to avoid redundant computation and memory usage.
class PrefixCache {
public:
    explicit PrefixCache(int block_size, BlockAllocator* allocator = nullptr, size_t max_entries = 256);
    ~PrefixCache();
    PrefixCache(const PrefixCache&) = delete;
    PrefixCache& operator=(const PrefixCache&) = delete;

    /// Try to find cached blocks matching a prefix of the given token IDs.
    /// Returns the number of tokens that were matched (always a multiple of block_size).
    /// Matched block IDs are written to out_blocks.
    int try_match(const int* token_ids, int length,
                  std::vector<BlockId>& out_blocks) const;

    /// Register a completed prefix's blocks for future sharing.
    /// token_ids should contain at least `length` token IDs.
    /// block_ids should contain the physical blocks for those tokens.
    /// Only full blocks (multiples of block_size) are registered.
    void register_prefix(const int* token_ids, int length,
                         const std::vector<BlockId>& block_ids);

    /// Evict a specific prefix entry by its hash.
    void evict(size_t hash);

    /// Evict all prefix entries.
    void clear();
    bool evict_one();

    /// Number of registered prefix entries.
    int size() const { return static_cast<int>(cache_.size()); }

    /// Get hit/miss statistics.
    int64_t hits() const { return hits_; }
    int64_t misses() const { return misses_; }

private:
    /// Compute a hash for a sequence of token IDs up to a block boundary.
    size_t hash_tokens(const int* token_ids, int count) const;

    struct PrefixEntry {
        std::vector<int>     token_ids;  ///< The token sequence for this prefix
        std::vector<BlockId> block_ids;  ///< Physical blocks storing this prefix's KV data
    };

    int block_size_;
    BlockAllocator* allocator_;
    size_t max_entries_;
    void release(const PrefixEntry& entry);
    std::unordered_map<size_t, PrefixEntry> cache_;  ///< hash → prefix entry

    mutable int64_t hits_   = 0;
    mutable int64_t misses_ = 0;
};

} // namespace kvcache
