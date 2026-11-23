#pragma once

#include "types.h"
#include "block_allocator.h"
#include "sequence.h"
#include "eviction_policy.h"
#include "prefix_cache.h"
#include <unordered_map>
#include <memory>
#include <mutex>

namespace kvcache {

/// Top-level KV Cache orchestrator. This is the main entry point for inference engines.
/// Manages sequences, block allocation, eviction, prefix sharing, and statistics.
class KVCacheManager {
public:
    explicit KVCacheManager(const KVCacheConfig& config);
    ~KVCacheManager();

    // -----------------------------------------------------------------------
    // Sequence lifecycle
    // -----------------------------------------------------------------------

    /// Create a new sequence. Returns ERROR_SEQUENCE_EXISTS if ID is already in use.
    Status create_sequence(SequenceId seq_id);

    /// Free a sequence and release all its blocks.
    Status free_sequence(SequenceId seq_id);

    /// Fork an existing sequence (Copy-on-Write) into a new sequence ID.
    /// Used for beam search and parallel sampling.
    Status fork_sequence(SequenceId src_id, SequenceId dst_id);

    // -----------------------------------------------------------------------
    // Token operations
    // -----------------------------------------------------------------------

    /// Append one token's KV data for a specific layer.
    /// key_data and value_data should each contain [num_kv_heads * head_dim] elements.
    Status append_kv(SequenceId seq_id, int layer,
                     const void* key_data, const void* value_data);

    /// Append one token's KV data for ALL layers at once.
    Status append_kv_all_layers(SequenceId seq_id,
                                const std::vector<const void*>& keys,
                                const std::vector<const void*>& values);

    /// Retrieve cached KV data for a specific token at a specific layer.
    KVPair get_kv(SequenceId seq_id, int layer, int token_idx) const;
    /// Copies under the manager lock; preferred for concurrent callers.
    Status copy_kv(SequenceId seq_id, int layer, int token_idx, void* key, void* value, size_t bytes) const;
    /// Export live tokens as contiguous FP32 K/V arrays and original positions.
    Status export_layer(SequenceId seq_id, int layer, std::vector<float>& keys,
                        std::vector<float>& values, std::vector<int>& positions) const;
    void clear_prefix_cache();

    /// Get the number of tokens cached for a sequence.
    int get_num_tokens(SequenceId seq_id) const;

    // -----------------------------------------------------------------------
    // Eviction
    // -----------------------------------------------------------------------

    /// Report attention scores for eviction policies (like H2O).
    /// scores[i] is the attention weight for token i.
    Status report_attention_scores(SequenceId seq_id, int layer,
                                   const float* scores, int num_tokens);

    /// Manually trigger eviction for a sequence. Returns the number of tokens evicted.
    int evict(SequenceId seq_id, int num_to_evict);

    // -----------------------------------------------------------------------
    // Prefix caching
    // -----------------------------------------------------------------------

    /// Try to reuse cached prefix blocks for a sequence.
    /// Returns the number of tokens matched from the prefix cache.
    int try_prefix_match(SequenceId seq_id, const int* token_ids, int length);

    /// Register a sequence's current blocks as a reusable prefix.
    void register_prefix(SequenceId seq_id, const int* token_ids, int length);

    // -----------------------------------------------------------------------
    // Queries & stats
    // -----------------------------------------------------------------------

    /// Get overall cache statistics.
    CacheStats get_stats() const;

    /// Get the configuration.
    const KVCacheConfig& config() const { return config_; }

    /// Check if a sequence exists.
    bool has_sequence(SequenceId seq_id) const;

    /// Get the number of active sequences.
    int num_sequences() const;

private:
    Sequence* find_sequence(SequenceId seq_id);
    const Sequence* find_sequence(SequenceId seq_id) const;
    int evict_locked(SequenceId seq_id, int count);

    KVCacheConfig                                                config_;
    std::unique_ptr<BlockAllocator>                              allocator_;
    std::unordered_map<int64_t, std::unique_ptr<Sequence>>       sequences_;
    std::unique_ptr<IEvictionPolicy>                             eviction_policy_;
    std::unique_ptr<PrefixCache>                                 prefix_cache_;
    mutable std::mutex                                           mutex_;

    // Global statistics
    int64_t total_evictions_ = 0;
};

} // namespace kvcache
