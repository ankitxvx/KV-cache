#pragma once

#include "config.h"
#include "tier_manager.h"
#include "logger.h"
#include "metrics.h"
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <atomic>
#include <future>
#include <chrono>

namespace kvcache {

/// Handle for tracking an async transfer.
struct TransferHandle {
    int64_t id = -1;
    bool valid() const { return id >= 0; }
};

/// A request to move data between tiers.
struct TransferRequest {
    TieredBlockId    src;
    TieredBlockId    dst;
    TransferPriority priority = TransferPriority::NORMAL;
    std::function<void(bool success)> callback;  ///< Optional completion callback
};

/// Async transfer engine for moving KV blocks between storage tiers.
/// Uses a thread pool for CPU↔SSD and CUDA streams for GPU↔CPU.
class TransferEngine {
public:
    explicit TransferEngine(TierManager& tier_manager, const ProductionConfig& config);
    ~TransferEngine();

    /// Submit a transfer request. Returns immediately.
    TransferHandle submit(TransferRequest request);

    /// Submit multiple transfers in batch.
    std::vector<TransferHandle> batch_submit(std::vector<TransferRequest> requests);

    /// Wait for a specific transfer to complete.
    void wait(TransferHandle handle);

    /// Check if a transfer is complete.
    bool poll(TransferHandle handle) const;

    /// Wait for all pending transfers to complete.
    void drain();

    /// Get transfer statistics.
    TransferStats get_stats() const;

    /// Number of pending transfers.
    int pending_count() const;

    /// Shutdown the engine (waits for pending transfers).
    void shutdown();

private:
    /// Worker thread function.
    void worker_loop();

    /// Execute a single transfer.
    bool execute_transfer(const TransferRequest& req);

    struct InternalTransfer {
        TransferRequest request;
        TransferHandle  handle;
        std::promise<bool> promise;
    };

    TierManager& tier_manager_;
    ProductionConfig config_;

    // Thread pool
    std::vector<std::thread> workers_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::priority_queue<
        std::pair<int, std::shared_ptr<InternalTransfer>>,
        std::vector<std::pair<int, std::shared_ptr<InternalTransfer>>>,
        std::greater<>
    > transfer_queue_;

    // Transfer tracking
    std::unordered_map<int64_t, std::shared_future<bool>> transfer_futures_;
    mutable std::mutex futures_mutex_;
    std::atomic<int64_t> next_handle_id_{0};

    // Statistics
    std::atomic<int64_t> completed_transfers_{0};
    std::atomic<int64_t> failed_transfers_{0};
    std::atomic<int64_t> gpu_cpu_count_{0};
    std::atomic<int64_t> cpu_gpu_count_{0};
    std::atomic<int64_t> cpu_ssd_count_{0};
    std::atomic<int64_t> ssd_cpu_count_{0};
    std::atomic<double>  total_bytes_transferred_{0.0};
    std::atomic<double>  total_transfer_time_us_{0.0};

    // Staging buffer (for GPU ↔ SSD via CPU)
    std::vector<uint8_t> staging_buffer_;

    std::atomic<bool> running_{false};
    std::atomic<int>  pending_count_{0};
};

} // namespace kvcache
