#pragma once

#include "types.h"
#include "block_table.h"
#include "block_allocator.h"
#include <vector>
#include <cstring>

namespace kvcache {

/// Represents one inference request's KV cache state across all layers.
/// Each layer has its own BlockTable for independent block management.
class Sequence {
public:
    Sequence() = default;
    Sequence(SequenceId id, const KVCacheConfig& config, BlockAllocator& allocator);

    /// Append a new token's KV data for a specific layer.
    /// key_data and value_data should point to [num_kv_heads * head_dim] elements.
    Status append_token(int layer, const void* key_data, const void* value_data,
                        BlockAllocator& allocator);

    /// Append a token across ALL layers at once.
    /// key_data_per_layer[i] and value_data_per_layer[i] point to data for layer i.
    Status append_token_all_layers(const std::vector<const void*>& key_data_per_layer,
                                   const std::vector<const void*>& value_data_per_layer,
                                   BlockAllocator& allocator);

    /// Retrieve cached KV data for a specific layer and token index.
    KVPair get_kv(int layer, int token_idx, const BlockAllocator& allocator) const;

    /// Create a Copy-on-Write fork with a new sequence ID.
    Sequence fork(SequenceId new_id, BlockAllocator& allocator) const;

    /// Free all resources held by this sequence.
    void free_all(BlockAllocator& allocator);

    /// Truncate to a given number of tokens (frees trailing blocks).
    void truncate(int new_length, BlockAllocator& allocator);

    /// Number of tokens cached (based on layer 0's block table).
    int num_tokens() const;

    /// Number of tokens cached at a specific layer.
    int num_tokens_at_layer(int layer) const;

    /// Total number of blocks used across all layers.
    int total_blocks() const;

    /// Get the sequence ID.
    SequenceId id() const { return id_; }

    /// Check if initialized.
    bool valid() const { return id_.valid(); }

    /// Access block table for a specific layer (for advanced use / prefix cache).
    const BlockTable& block_table(int layer) const {
        if (layer < 0 || layer >= num_layers()) throw std::out_of_range("layer");
        return table_;
    }

    /// Number of layers.
    int num_layers() const { return static_cast<int>(layer_lengths_.size()); }
    bool complete() const;
    bool has_evictions() const;
    int cached_tokens() const;
    int evict_tokens(const std::vector<int>& tokens, BlockAllocator& allocator);
    Status attach_prefix(const std::vector<BlockId>& blocks, int tokens, BlockAllocator& allocator);

private:
    SequenceId id_;
    KVCacheConfig config_;
    Status prepare_write(int token, BlockAllocator& allocator, BlockId& block);
    BlockTable table_; // One all-layer physical block per logical page.
    std::vector<int> layer_lengths_;
    std::vector<bool> evicted_;
};

} // namespace kvcache
