#include "kvcache/eviction_policy.h"

namespace kvcache {

// ===========================================================================
// LRU Eviction
// ===========================================================================

void LRUEviction::on_token_accessed(SequenceId seq_id, int token_idx, float /*attention_score*/) {
    auto& state = states_[seq_id.value];

    auto it = state.token_iter.find(token_idx);
    if (it != state.token_iter.end()) {
        // Move to back (MRU position)
        state.access_order.erase(it->second);
        state.access_order.push_back(token_idx);
        it->second = std::prev(state.access_order.end());
    } else {
        // New token — add to back
        state.access_order.push_back(token_idx);
        state.token_iter[token_idx] = std::prev(state.access_order.end());
    }
}

std::vector<int> LRUEviction::select_victims(SequenceId seq_id, int num_to_evict) {
    std::vector<int> victims;
    auto it = states_.find(seq_id.value);
    if (it == states_.end()) return victims;

    auto& state = it->second;
    auto list_it = state.access_order.begin();

    for (int i = 0; i < num_to_evict && list_it != state.access_order.end(); ++i) {
        victims.push_back(*list_it);
        ++list_it;
    }

    std::sort(victims.begin(), victims.end());
    return victims;
}

void LRUEviction::on_token_evicted(SequenceId seq_id, int token_idx) {
    auto it = states_.find(seq_id.value);
    if (it == states_.end()) return;

    auto& state = it->second;
    auto tok_it = state.token_iter.find(token_idx);
    if (tok_it != state.token_iter.end()) {
        state.access_order.erase(tok_it->second);
        state.token_iter.erase(tok_it);
    }
}

void LRUEviction::on_sequence_created(SequenceId seq_id) {
    states_[seq_id.value] = SeqState{};
}

void LRUEviction::on_sequence_freed(SequenceId seq_id) {
    states_.erase(seq_id.value);
}

void LRUEviction::reset() {
    states_.clear();
}

// ===========================================================================
// H2O (Heavy Hitter Oracle) Eviction
// ===========================================================================

H2OEviction::H2OEviction(float heavy_ratio, int sink_count)
    : heavy_ratio_(heavy_ratio)
    , sink_count_(sink_count)
{
}

void H2OEviction::on_token_accessed(SequenceId seq_id, int token_idx, float attention_score) {
    auto& state = states_[seq_id.value];
    state.cumulative_scores[token_idx] += attention_score;
    if (token_idx > state.max_token_idx) {
        state.max_token_idx = token_idx;
    }
}

std::vector<int> H2OEviction::select_victims(SequenceId seq_id, int num_to_evict) {
    std::vector<int> victims;
    auto it = states_.find(seq_id.value);
    if (it == states_.end()) return victims;

    auto& state = it->second;

    // Collect all tokens with their scores
    struct TokenScore {
        int idx;
        float score;
    };
    std::vector<TokenScore> candidates;

    for (const auto& [idx, score] : state.cumulative_scores) {
        // Protect sink tokens (first `sink_count_` tokens)
        if (idx < sink_count_) continue;

        candidates.push_back({idx, score});
    }

    // Sort by score ascending — lowest scores are eviction candidates
    std::sort(candidates.begin(), candidates.end(),
              [](const TokenScore& a, const TokenScore& b) {
                  return a.score < b.score;
              });

    // Calculate how many tokens to protect as heavy hitters
    int total_tokens = static_cast<int>(state.cumulative_scores.size());
    int num_heavy = static_cast<int>(std::ceil(total_tokens * heavy_ratio_));

    // The last `num_heavy` in the sorted list are protected
    int evictable = static_cast<int>(candidates.size()) - num_heavy;
    int actual_evict = std::min(num_to_evict, std::max(0, evictable));

    for (int i = 0; i < actual_evict; ++i) {
        victims.push_back(candidates[i].idx);
    }

    std::sort(victims.begin(), victims.end());
    return victims;
}

void H2OEviction::on_token_evicted(SequenceId seq_id, int token_idx) {
    auto it = states_.find(seq_id.value);
    if (it == states_.end()) return;
    it->second.cumulative_scores.erase(token_idx);
}

void H2OEviction::on_sequence_created(SequenceId seq_id) {
    states_[seq_id.value] = SeqState{};
}

void H2OEviction::on_sequence_freed(SequenceId seq_id) {
    states_.erase(seq_id.value);
}

void H2OEviction::reset() {
    states_.clear();
}

// ===========================================================================
// Sliding Window Eviction
// ===========================================================================

SlidingWindowEviction::SlidingWindowEviction(int window_size, int sink_count)
    : window_size_(window_size)
    , sink_count_(sink_count)
{
}

void SlidingWindowEviction::on_token_accessed(SequenceId seq_id, int token_idx, float /*score*/) {
    auto& state = states_[seq_id.value];
    if (token_idx > state.max_token_idx) {
        state.max_token_idx = token_idx;
        // Grow evicted tracking vector
        state.evicted.resize(token_idx + 1, false);
    }
}

std::vector<int> SlidingWindowEviction::select_victims(SequenceId seq_id, int num_to_evict) {
    std::vector<int> victims;
    auto it = states_.find(seq_id.value);
    if (it == states_.end()) return victims;

    auto& state = it->second;
    int total = state.max_token_idx + 1;

    // Window starts at max(sink_count, total - window_size)
    int window_start = std::max(sink_count_, total - window_size_);

    // Find tokens outside the protected zones (sinks and window)
    for (int i = sink_count_; i < window_start && static_cast<int>(victims.size()) < num_to_evict; ++i) {
        if (i < static_cast<int>(state.evicted.size()) && !state.evicted[i]) {
            victims.push_back(i);
        }
    }

    return victims;  // Already sorted (ascending)
}

void SlidingWindowEviction::on_token_evicted(SequenceId seq_id, int token_idx) {
    auto it = states_.find(seq_id.value);
    if (it == states_.end()) return;

    if (token_idx < static_cast<int>(it->second.evicted.size())) {
        it->second.evicted[token_idx] = true;
    }
}

void SlidingWindowEviction::on_sequence_created(SequenceId seq_id) {
    states_[seq_id.value] = SeqState{};
}

void SlidingWindowEviction::on_sequence_freed(SequenceId seq_id) {
    states_.erase(seq_id.value);
}

void SlidingWindowEviction::reset() {
    states_.clear();
}

// ===========================================================================
// Factory
// ===========================================================================

std::unique_ptr<IEvictionPolicy> create_eviction_policy(const KVCacheConfig& config) {
    switch (config.eviction_policy) {
        case EvictionType::NONE:
            return nullptr;
        case EvictionType::LRU:
            return std::make_unique<LRUEviction>();
        case EvictionType::H2O:
            return std::make_unique<H2OEviction>(config.h2o_heavy_ratio, config.sink_token_count);
        case EvictionType::SLIDING_WINDOW:
            return std::make_unique<SlidingWindowEviction>(config.sliding_window_size,
                                                           config.sink_token_count);
    }
    return nullptr;
}

} // namespace kvcache
