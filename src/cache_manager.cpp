#include "kvcache/cache_manager.h"

namespace kvcache {

KVCacheManager::KVCacheManager(const KVCacheConfig& config)
    : config_(config)
{
    allocator_ = std::make_unique<BlockAllocator>(config_);
    eviction_policy_ = create_eviction_policy(config_);
    prefix_cache_ = std::make_unique<PrefixCache>(config_.block_size);
}

KVCacheManager::~KVCacheManager() {
    // Free all sequences
    for (auto& [id, seq] : sequences_) {
        if (seq) {
            seq->free_all(*allocator_);
        }
    }
}

// ---------------------------------------------------------------------------
// Sequence lifecycle
// ---------------------------------------------------------------------------

Status KVCacheManager::create_sequence(SequenceId seq_id) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (sequences_.count(seq_id.value)) {
        return Status::ERROR_SEQUENCE_EXISTS;
    }

    auto seq = std::make_unique<Sequence>(seq_id, config_, *allocator_);
    sequences_[seq_id.value] = std::move(seq);

    if (eviction_policy_) {
        eviction_policy_->on_sequence_created(seq_id);
    }

    return Status::OK;
}

Status KVCacheManager::free_sequence(SequenceId seq_id) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = sequences_.find(seq_id.value);
    if (it == sequences_.end()) {
        return Status::ERROR_SEQUENCE_NOT_FOUND;
    }

    it->second->free_all(*allocator_);
    sequences_.erase(it);

    if (eviction_policy_) {
        eviction_policy_->on_sequence_freed(seq_id);
    }

    return Status::OK;
}

Status KVCacheManager::fork_sequence(SequenceId src_id, SequenceId dst_id) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (sequences_.count(dst_id.value)) {
        return Status::ERROR_SEQUENCE_EXISTS;
    }

    auto src_it = sequences_.find(src_id.value);
    if (src_it == sequences_.end()) {
        return Status::ERROR_SEQUENCE_NOT_FOUND;
    }

    auto forked = std::make_unique<Sequence>(src_it->second->fork(dst_id, *allocator_));
    sequences_[dst_id.value] = std::move(forked);

    if (eviction_policy_) {
        eviction_policy_->on_sequence_created(dst_id);
    }

    return Status::OK;
}

// ---------------------------------------------------------------------------
// Token operations
// ---------------------------------------------------------------------------

Status KVCacheManager::append_kv(SequenceId seq_id, int layer,
                                  const void* key_data, const void* value_data) {
    std::lock_guard<std::mutex> lock(mutex_);

    Sequence* seq = find_sequence(seq_id);
    if (!seq) return Status::ERROR_SEQUENCE_NOT_FOUND;

    Status s = seq->append_token(layer, key_data, value_data, *allocator_);

    if (s == Status::OK && eviction_policy_) {
        int token_idx = seq->num_tokens_at_layer(layer) - 1;
        eviction_policy_->on_token_accessed(seq_id, token_idx, 0.0f);
    }

    return s;
}

Status KVCacheManager::append_kv_all_layers(SequenceId seq_id,
                                             const std::vector<const void*>& keys,
                                             const std::vector<const void*>& values) {
    std::lock_guard<std::mutex> lock(mutex_);

    Sequence* seq = find_sequence(seq_id);
    if (!seq) return Status::ERROR_SEQUENCE_NOT_FOUND;

    Status s = seq->append_token_all_layers(keys, values, *allocator_);

    if (s == Status::OK && eviction_policy_) {
        int token_idx = seq->num_tokens() - 1;
        eviction_policy_->on_token_accessed(seq_id, token_idx, 0.0f);
    }

    return s;
}

KVPair KVCacheManager::get_kv(SequenceId seq_id, int layer, int token_idx) const {
    std::lock_guard<std::mutex> lock(mutex_);

    const Sequence* seq = find_sequence(seq_id);
    if (!seq) return KVPair{};

    return seq->get_kv(layer, token_idx, *allocator_);
}

int KVCacheManager::get_num_tokens(SequenceId seq_id) const {
    std::lock_guard<std::mutex> lock(mutex_);

    const Sequence* seq = find_sequence(seq_id);
    if (!seq) return 0;

    return seq->num_tokens();
}

// ---------------------------------------------------------------------------
// Eviction
// ---------------------------------------------------------------------------

Status KVCacheManager::report_attention_scores(SequenceId seq_id, int layer,
                                                const float* scores, int num_tokens) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!eviction_policy_) return Status::OK;  // No eviction configured

    if (!find_sequence(seq_id)) return Status::ERROR_SEQUENCE_NOT_FOUND;

    for (int i = 0; i < num_tokens; ++i) {
        eviction_policy_->on_token_accessed(seq_id, i, scores[i]);
    }

    return Status::OK;
}

int KVCacheManager::evict(SequenceId seq_id, int num_to_evict) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!eviction_policy_) return 0;

    Sequence* seq = find_sequence(seq_id);
    if (!seq) return 0;

    auto victims = eviction_policy_->select_victims(seq_id, num_to_evict);

    // Notify the policy about evictions
    for (int victim : victims) {
        eviction_policy_->on_token_evicted(seq_id, victim);
    }

    total_evictions_ += static_cast<int64_t>(victims.size());

    // Note: actual data removal from blocks is complex (tokens in the middle of blocks).
    // In production systems, eviction marks tokens as "dead" and they're skipped during attention.
    // For this implementation, we track evictions for metrics but the data remains in blocks
    // until the blocks are freed (which happens when sequences are freed or truncated).
    // A full implementation would compact blocks periodically.

    return static_cast<int>(victims.size());
}

// ---------------------------------------------------------------------------
// Prefix caching
// ---------------------------------------------------------------------------

int KVCacheManager::try_prefix_match(SequenceId seq_id, const int* token_ids, int length) {
    std::lock_guard<std::mutex> lock(mutex_);

    Sequence* seq = find_sequence(seq_id);
    if (!seq) return 0;

    std::vector<BlockId> matched_blocks;
    int matched_tokens = prefix_cache_->try_match(token_ids, length, matched_blocks);

    if (matched_tokens > 0) {
        // Ref the matched blocks and set them up in the sequence's block tables
        // For each layer, we reuse layer 0's block mapping
        // (prefix blocks store all layers, so the same block IDs apply)
        for (const auto& block_id : matched_blocks) {
            allocator_->ref(block_id);
        }
    }

    return matched_tokens;
}

void KVCacheManager::register_prefix(SequenceId seq_id, const int* token_ids, int length) {
    std::lock_guard<std::mutex> lock(mutex_);

    const Sequence* seq = find_sequence(seq_id);
    if (!seq || seq->num_layers() == 0) return;

    // Use layer 0's block table for the prefix registration
    const auto& blocks = seq->block_table(0).blocks();
    prefix_cache_->register_prefix(token_ids, length, blocks);
}

// ---------------------------------------------------------------------------
// Queries & stats
// ---------------------------------------------------------------------------

CacheStats KVCacheManager::get_stats() const {
    std::lock_guard<std::mutex> lock(mutex_);

    CacheStats stats;
    stats.total_blocks = allocator_->num_total();
    stats.free_blocks  = allocator_->num_free();
    stats.used_blocks  = allocator_->num_used();
    stats.active_sequences = static_cast<int>(sequences_.size());
    stats.total_memory_bytes = allocator_->total_memory_bytes();
    stats.used_memory_bytes  = static_cast<size_t>(stats.used_blocks) * allocator_->block_bytes();

    if (stats.total_blocks > 0) {
        stats.utilization_pct = 100.0 * stats.used_blocks / stats.total_blocks;
    }

    // Count total tokens across all sequences
    for (const auto& [id, seq] : sequences_) {
        stats.total_cached_tokens += seq->num_tokens();
    }

    stats.total_evictions = total_evictions_;
    stats.prefix_cache_hits = prefix_cache_->hits();

    return stats;
}

bool KVCacheManager::has_sequence(SequenceId seq_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return sequences_.count(seq_id.value) > 0;
}

int KVCacheManager::num_sequences() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return static_cast<int>(sequences_.size());
}

Sequence* KVCacheManager::find_sequence(SequenceId seq_id) {
    auto it = sequences_.find(seq_id.value);
    return (it != sequences_.end()) ? it->second.get() : nullptr;
}

const Sequence* KVCacheManager::find_sequence(SequenceId seq_id) const {
    auto it = sequences_.find(seq_id.value);
    return (it != sequences_.end()) ? it->second.get() : nullptr;
}

} // namespace kvcache
