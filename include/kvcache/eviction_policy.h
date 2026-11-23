#pragma once

#include "types.h"
#include <vector>
#include <list>
#include <unordered_map>
#include <memory>
#include <algorithm>
#include <cmath>

namespace kvcache {

/// Abstract interface for KV cache eviction policies.
/// Eviction policies decide which tokens to discard when memory pressure requires it.
class IEvictionPolicy {
public:
    virtual ~IEvictionPolicy() = default;

    /// Called when a token is accessed (e.g., during attention computation).
    /// @param seq_id      Sequence that owns the token.
    /// @param token_idx   Logical index of the token.
    /// @param attention_score  The attention weight this token received (optional, 0.0 if unavailable).
    virtual void on_token_accessed(SequenceId seq_id, int token_idx, float attention_score = 0.0f) = 0;

    /// Select tokens to evict from a sequence.
    /// @param seq_id       Sequence to evict from.
    /// @param num_to_evict Number of tokens to select for eviction.
    /// @return             Indices of tokens to evict, sorted ascending.
    virtual std::vector<int> select_victims(SequenceId seq_id, int num_to_evict) = 0;

    /// Called after a token has been evicted (for bookkeeping).
    virtual void on_token_evicted(SequenceId seq_id, int token_idx) = 0;

    /// Called when a new sequence is registered.
    virtual void on_sequence_created(SequenceId seq_id) = 0;

    /// Called when a sequence is freed.
    virtual void on_sequence_freed(SequenceId seq_id) = 0;

    /// Reset all state.
    virtual void reset() = 0;
};

// ---------------------------------------------------------------------------
// LRU Eviction
// ---------------------------------------------------------------------------

/// Classic Least Recently Used eviction policy.
/// Maintains a per-sequence access order list. The least recently accessed
/// tokens are evicted first.
class LRUEviction : public IEvictionPolicy {
public:
    void on_token_accessed(SequenceId seq_id, int token_idx, float attention_score = 0.0f) override;
    std::vector<int> select_victims(SequenceId seq_id, int num_to_evict) override;
    void on_token_evicted(SequenceId seq_id, int token_idx) override;
    void on_sequence_created(SequenceId seq_id) override;
    void on_sequence_freed(SequenceId seq_id) override;
    void reset() override;

private:
    struct SeqState {
        std::list<int> access_order;  ///< Front = LRU, back = MRU
        std::unordered_map<int, std::list<int>::iterator> token_iter;
    };
    std::unordered_map<int64_t, SeqState> states_;
};

// ---------------------------------------------------------------------------
// H2O (Heavy Hitter Oracle) Eviction
// ---------------------------------------------------------------------------

/// Attention-score-based eviction inspired by the H2O paper.
/// Tracks cumulative attention scores per token and retains "heavy hitters"
/// (tokens with highest accumulated attention) plus a recent local window.
class H2OEviction : public IEvictionPolicy {
public:
    /// @param heavy_ratio  Fraction of total tokens to protect as heavy hitters (e.g., 0.2).
    /// @param sink_count   Number of initial tokens to always protect ("attention sinks").
    explicit H2OEviction(float heavy_ratio = 0.2f, int sink_count = 4);

    void on_token_accessed(SequenceId seq_id, int token_idx, float attention_score = 0.0f) override;
    std::vector<int> select_victims(SequenceId seq_id, int num_to_evict) override;
    void on_token_evicted(SequenceId seq_id, int token_idx) override;
    void on_sequence_created(SequenceId seq_id) override;
    void on_sequence_freed(SequenceId seq_id) override;
    void reset() override;

private:
    float heavy_ratio_;
    int   sink_count_;

    struct SeqState {
        std::unordered_map<int, float>   cumulative_scores; ///< token_idx → accumulated attention score
        int max_token_idx = -1;  ///< Highest token index seen
    };
    std::unordered_map<int64_t, SeqState> states_;
};

// ---------------------------------------------------------------------------
// Sliding Window Eviction
// ---------------------------------------------------------------------------

/// StreamingLLM-style eviction: keeps only the last `window_size` tokens
/// plus the first `sink_count` tokens (attention sinks).
/// All tokens outside this window are candidates for eviction.
class SlidingWindowEviction : public IEvictionPolicy {
public:
    /// @param window_size  Number of recent tokens to retain.
    /// @param sink_count   Number of initial tokens to always retain.
    explicit SlidingWindowEviction(int window_size = 256, int sink_count = 4);

    void on_token_accessed(SequenceId seq_id, int token_idx, float attention_score = 0.0f) override;
    std::vector<int> select_victims(SequenceId seq_id, int num_to_evict) override;
    void on_token_evicted(SequenceId seq_id, int token_idx) override;
    void on_sequence_created(SequenceId seq_id) override;
    void on_sequence_freed(SequenceId seq_id) override;
    void reset() override;

private:
    int window_size_;
    int sink_count_;

    struct SeqState {
        int max_token_idx = -1;
        std::vector<bool> evicted;  ///< Tracks which tokens have been evicted
    };
    std::unordered_map<int64_t, SeqState> states_;
};

/// Factory function to create an eviction policy from config.
std::unique_ptr<IEvictionPolicy> create_eviction_policy(const KVCacheConfig& config);

} // namespace kvcache
