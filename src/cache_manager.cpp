#include "kvcache/cache_manager.h"
#include "kvcache/quantization.h"
#include <cmath>

namespace kvcache {

KVCacheManager::KVCacheManager(const KVCacheConfig& config)
    : config_(config)
{
    allocator_ = std::make_unique<BlockAllocator>(config_);
    eviction_policy_ = create_eviction_policy(config_);
    prefix_cache_ = std::make_unique<PrefixCache>(config_.block_size, allocator_.get(), config_.max_prefix_entries);
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
    if (!seq_id.valid()) return Status::ERROR_INVALID_ARGUMENT;

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
    if (!dst_id.valid()) return Status::ERROR_INVALID_ARGUMENT;

    if (sequences_.count(dst_id.value)) {
        return Status::ERROR_SEQUENCE_EXISTS;
    }

    auto src_it = sequences_.find(src_id.value);
    if (src_it == sequences_.end()) {
        return Status::ERROR_SEQUENCE_NOT_FOUND;
    }
    if (!src_it->second->complete()) return Status::ERROR_INVALID_ARGUMENT;

    auto forked = std::make_unique<Sequence>(src_it->second->fork(dst_id, *allocator_));
    sequences_[dst_id.value] = std::move(forked);

    if (eviction_policy_) {
        eviction_policy_->on_sequence_created(dst_id);
        for (int i = 0; i < src_it->second->num_tokens(); ++i)
            if (src_it->second->get_kv(0, i, *allocator_).valid())
                eviction_policy_->on_token_accessed(dst_id, i, 0.0f);
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
    while (s == Status::ERROR_OUT_OF_MEMORY && prefix_cache_->evict_one())
        s = seq->append_token(layer, key_data, value_data, *allocator_);
    if (s == Status::ERROR_OUT_OF_MEMORY && seq->complete() && eviction_policy_) {
        evict_locked(seq_id, config_.block_size);
        s = seq->append_token(layer, key_data, value_data, *allocator_);
    }

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
    while (s == Status::ERROR_OUT_OF_MEMORY && prefix_cache_->evict_one())
        s = seq->append_token_all_layers(keys, values, *allocator_);
    if (s == Status::ERROR_OUT_OF_MEMORY && seq->complete() && eviction_policy_) {
        evict_locked(seq_id, config_.block_size);
        s = seq->append_token_all_layers(keys, values, *allocator_);
    }

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

    auto kv = seq->get_kv(layer, token_idx, *allocator_);
    if (kv.valid() && eviction_policy_) eviction_policy_->on_token_accessed(seq_id, token_idx, 0.0f);
    return kv;
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

    auto* seq = find_sequence(seq_id);
    if (!seq) return Status::ERROR_SEQUENCE_NOT_FOUND;
    if (!scores || layer < 0 || layer >= config_.num_layers || num_tokens < 0 || num_tokens > seq->num_tokens())
        return Status::ERROR_INVALID_ARGUMENT;
    for (int i = 0; i < num_tokens; ++i) if (!std::isfinite(scores[i])) return Status::ERROR_INVALID_ARGUMENT;

    for (int i = 0; i < num_tokens; ++i) {
        if (seq->get_kv(layer, i, *allocator_).valid()) eviction_policy_->on_token_accessed(seq_id, i, scores[i]);
    }

    return Status::OK;
}

int KVCacheManager::evict(SequenceId seq_id, int num_to_evict) {
    std::lock_guard<std::mutex> lock(mutex_);
    return evict_locked(seq_id, num_to_evict);
}
int KVCacheManager::evict_locked(SequenceId seq_id, int num_to_evict) {

    if (!eviction_policy_) return 0;

    Sequence* seq = find_sequence(seq_id);
    if (!seq || !seq->complete() || num_to_evict <= 0) return 0;

    auto victims = eviction_policy_->select_victims(seq_id, num_to_evict);
    int removed = seq->evict_tokens(victims, *allocator_);

    // Notify the policy about evictions
    for (int victim : victims) {
        eviction_policy_->on_token_evicted(seq_id, victim);
    }

    total_evictions_ += removed;

    // Logical positions remain stable. Fully dead pages release their ownership;
    // shared prefixes/forks may retain their own references to those pages.

    return removed;
}

// ---------------------------------------------------------------------------
// Prefix caching
// ---------------------------------------------------------------------------

int KVCacheManager::try_prefix_match(SequenceId seq_id, const int* token_ids, int length) {
    std::lock_guard<std::mutex> lock(mutex_);

    Sequence* seq = find_sequence(seq_id);
    if (!seq || !seq->complete() || seq->num_tokens() != 0) return 0;

    std::vector<BlockId> matched_blocks;
    int matched_tokens = prefix_cache_->try_match(token_ids, length, matched_blocks);

    if (matched_tokens > 0) {
        if (seq->attach_prefix(matched_blocks, matched_tokens, *allocator_) != Status::OK) return 0;
        if (eviction_policy_) for (int i = 0; i < matched_tokens; ++i)
            eviction_policy_->on_token_accessed(seq_id, i, 0.0f);
    }

    return matched_tokens;
}

void KVCacheManager::register_prefix(SequenceId seq_id, const int* token_ids, int length) {
    std::lock_guard<std::mutex> lock(mutex_);

    const Sequence* seq = find_sequence(seq_id);
    if (!seq || !seq->complete() || seq->has_evictions() || length > seq->num_tokens()) return;

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
        stats.total_cached_tokens += seq->cached_tokens();
    }

    stats.total_evictions = total_evictions_;
    stats.prefix_cache_hits = prefix_cache_->hits();

    return stats;
}

bool KVCacheManager::has_sequence(SequenceId seq_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return sequences_.count(seq_id.value) > 0;
}

void KVCacheManager::clear_prefix_cache() {
    std::lock_guard<std::mutex> lock(mutex_);
    prefix_cache_->clear();
}
Status KVCacheManager::copy_kv(SequenceId id, int layer, int token, void* key, void* value, size_t bytes) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!key || !value || bytes != config_.kv_pair_bytes_per_token() / 2) return Status::ERROR_INVALID_ARGUMENT;
    auto* seq = find_sequence(id);
    if (!seq) return Status::ERROR_SEQUENCE_NOT_FOUND;
    auto kv = seq->get_kv(layer, token, *allocator_);
    if (!kv.valid()) return Status::ERROR_TOKEN_OUT_OF_RANGE;
    std::memcpy(key, kv.key, bytes); std::memcpy(value, kv.value, bytes);
    if (eviction_policy_) eviction_policy_->on_token_accessed(id, token, 0.0f);
    return Status::OK;
}
Status KVCacheManager::export_layer(SequenceId id, int layer, std::vector<float>& keys,
                                    std::vector<float>& values, std::vector<int>& positions) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto* seq = find_sequence(id);
    if (!seq) return Status::ERROR_SEQUENCE_NOT_FOUND;
    if (layer < 0 || layer >= config_.num_layers || config_.dtype == DataType::INT8 || !seq->complete())
        return Status::ERROR_INVALID_ARGUMENT; // INT8 requires caller-owned scales.
    size_t elements = size_t(config_.num_kv_heads) * config_.head_dim;
    std::vector<float> k, v; std::vector<int> p;
    k.reserve(elements * seq->cached_tokens()); v.reserve(k.capacity()); p.reserve(seq->cached_tokens());
    for (int t = 0; t < seq->num_tokens(); ++t) {
        auto kv = seq->get_kv(layer, t, *allocator_);
        if (!kv.valid()) continue;
        size_t offset = k.size(); k.resize(offset + elements); v.resize(offset + elements);
        float scale = 1;
        Quantizer::convert(kv.key, k.data() + offset, config_.dtype, DataType::FP32, elements, scale);
        Quantizer::convert(kv.value, v.data() + offset, config_.dtype, DataType::FP32, elements, scale);
        p.push_back(t);
    }
    keys.swap(k); values.swap(v); positions.swap(p); return Status::OK;
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
