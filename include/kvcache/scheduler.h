#pragma once

#include "config.h"
#include "logger.h"
#include <vector>
#include <queue>
#include <mutex>
#include <atomic>
#include <chrono>
#include <unordered_map>
#include <functional>
#include <optional>

namespace kvcache {

/// Unique ID for a scheduled request.
struct RequestId {
    int64_t value = -1;
    bool valid() const { return value >= 0; }
    bool operator==(const RequestId& o) const { return value == o.value; }
};

/// An inference request to the scheduler.
struct InferenceRequest {
    RequestId     id;
    SequenceId    seq_id;
    int           prompt_length    = 0;    ///< Number of input tokens
    int           max_new_tokens   = 256;  ///< Max tokens to generate
    int           generated_tokens = 0;    ///< Tokens generated so far
    RequestState  state            = RequestState::WAITING;
    int           priority         = 0;    ///< Higher = more urgent
    std::chrono::steady_clock::time_point arrival_time;
    std::chrono::steady_clock::time_point start_time;
};

/// Output of a scheduling step — which sequences to run.
struct SchedulerOutput {
    /// Sequences to run prefill on this step.
    std::vector<RequestId> prefill_requests;
    /// Sequences to run decode on this step.
    std::vector<RequestId> decode_requests;
    /// Sequences that were preempted (swapped out).
    std::vector<RequestId> preempted_requests;
    /// Total tokens in the batch.
    int total_batch_tokens = 0;
    /// Batch size.
    int batch_size = 0;
};

/// Continuous batching scheduler for LLM inference.
///
/// Dynamically manages which requests to prefill, decode, or preempt
/// based on available memory and batch size constraints.
class Scheduler {
public:
    explicit Scheduler(const ProductionConfig& config);

    /// Add a new request to the waiting queue.
    RequestId add_request(SequenceId seq_id, int prompt_length, int max_new_tokens,
                          int priority = 0);

    /// Run one scheduling step: decide which requests to process.
    SchedulerOutput schedule();

    /// Update a request's state after processing.
    void update(RequestId id, RequestState new_state);

    /// Mark a request as having generated one more token.
    void on_token_generated(RequestId id);

    /// Abort a request.
    void abort(RequestId id);

    /// Get request info.
    std::optional<InferenceRequest> get_request(RequestId id) const;

    /// Queue sizes.
    int waiting_count() const;
    int running_count() const;
    int completed_count() const { return completed_count_.load(); }
    int total_count() const { return total_requests_.load(); }

    /// Available GPU/CPU block budget (set externally by the cache manager).
    void set_available_blocks(int gpu_blocks, int cpu_blocks);

private:
    /// Try to admit waiting requests into the running batch.
    void admit_requests(SchedulerOutput& output);

    /// Check if a new request can fit in the batch.
    bool can_admit(const InferenceRequest& req) const;

    ProductionConfig config_;

    /// Request storage.
    std::unordered_map<int64_t, InferenceRequest> requests_;

    /// Waiting queue (priority + FIFO).
    struct WaitingComparator {
        bool operator()(const RequestId& a, const RequestId& b) const {
            // Lower value = lower priority (min-heap), so invert for max-priority
            return a.value > b.value;
        }
    };
    std::vector<RequestId> waiting_queue_;
    std::vector<RequestId> running_queue_;

    int available_gpu_blocks_ = 0;
    int available_cpu_blocks_ = 0;
    bool budget_set_ = false;

    std::atomic<int64_t> next_id_{0};
    std::atomic<int>     completed_count_{0};
    std::atomic<int>     total_requests_{0};

    mutable std::mutex mutex_;
};

} // namespace kvcache
