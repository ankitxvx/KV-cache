#include "kvcache/transfer_engine.h"
#include <sstream>

namespace kvcache {

TransferEngine::TransferEngine(TierManager& tier_manager, const ProductionConfig& config)
    : tier_manager_(tier_manager)
    , config_(config)
{
    running_ = true;

    staging_buffer_.resize(config.block_total_bytes());

    int num_threads = std::max(1, config.num_transfer_threads);
    for (int i = 0; i < num_threads; ++i) {
        workers_.emplace_back(&TransferEngine::worker_loop, this);
    }

    std::ostringstream msg;
    msg << "Transfer engine started with " << num_threads << " worker threads";
    KVCACHE_LOG_INFO("transfer", msg.str());
}

TransferEngine::~TransferEngine() {
    shutdown();
}

void TransferEngine::shutdown() {
    if (!running_.exchange(false)) return;

    queue_cv_.notify_all();
    for (auto& t : workers_) {
        if (t.joinable()) t.join();
    }
    workers_.clear();

    KVCACHE_LOG_INFO("transfer", "Transfer engine shut down");
}

TransferHandle TransferEngine::submit(TransferRequest request) {
    TransferHandle handle;
    handle.id = next_handle_id_.fetch_add(1);

    auto internal = std::make_shared<InternalTransfer>();
    internal->request = std::move(request);
    internal->handle = handle;

    auto future = internal->promise.get_future().share();

    {
        std::lock_guard<std::mutex> lock(futures_mutex_);
        transfer_futures_[handle.id] = future;
    }

    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        int priority = static_cast<int>(internal->request.priority);
        transfer_queue_.push({priority, internal});
        pending_count_.fetch_add(1);
    }

    queue_cv_.notify_one();
    return handle;
}

std::vector<TransferHandle> TransferEngine::batch_submit(std::vector<TransferRequest> requests) {
    std::vector<TransferHandle> handles;
    handles.reserve(requests.size());

    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        for (auto& req : requests) {
            TransferHandle handle;
            handle.id = next_handle_id_.fetch_add(1);

            auto internal = std::make_shared<InternalTransfer>();
            internal->request = std::move(req);
            internal->handle = handle;

            auto future = internal->promise.get_future().share();

            {
                std::lock_guard<std::mutex> flock(futures_mutex_);
                transfer_futures_[handle.id] = future;
            }

            int priority = static_cast<int>(internal->request.priority);
            transfer_queue_.push({priority, internal});
            pending_count_.fetch_add(1);

            handles.push_back(handle);
        }
    }

    queue_cv_.notify_all();
    return handles;
}

void TransferEngine::wait(TransferHandle handle) {
    std::shared_future<bool> future;
    {
        std::lock_guard<std::mutex> lock(futures_mutex_);
        auto it = transfer_futures_.find(handle.id);
        if (it == transfer_futures_.end()) return;
        future = it->second;
    }
    future.wait();
}

bool TransferEngine::poll(TransferHandle handle) const {
    std::lock_guard<std::mutex> lock(futures_mutex_);
    auto it = transfer_futures_.find(handle.id);
    if (it == transfer_futures_.end()) return true;  // Already cleaned up = done
    return it->second.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
}

void TransferEngine::drain() {
    while (pending_count_.load() > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

TransferStats TransferEngine::get_stats() const {
    TransferStats stats;
    stats.total_transfers = completed_transfers_.load() + failed_transfers_.load();
    stats.gpu_to_cpu_transfers = gpu_cpu_count_.load();
    stats.cpu_to_gpu_transfers = cpu_gpu_count_.load();
    stats.cpu_to_ssd_transfers = cpu_ssd_count_.load();
    stats.ssd_to_cpu_transfers = ssd_cpu_count_.load();
    stats.total_bytes_transferred = total_bytes_transferred_.load();

    double total_time = total_transfer_time_us_.load();
    if (stats.total_transfers > 0) {
        stats.avg_transfer_us = total_time / stats.total_transfers;
    }

    return stats;
}

int TransferEngine::pending_count() const {
    return pending_count_.load();
}

void TransferEngine::worker_loop() {
    while (running_) {
        std::shared_ptr<InternalTransfer> transfer;

        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            queue_cv_.wait(lock, [this]() {
                return !running_ || !transfer_queue_.empty();
            });

            if (!running_ && transfer_queue_.empty()) return;
            if (transfer_queue_.empty()) continue;

            transfer = transfer_queue_.top().second;
            transfer_queue_.pop();
        }

        auto start = std::chrono::high_resolution_clock::now();
        bool success = execute_transfer(transfer->request);
        auto end = std::chrono::high_resolution_clock::now();

        double elapsed_us = std::chrono::duration<double, std::micro>(end - start).count();
        atomic_add(total_transfer_time_us_, elapsed_us);

        MetricsRegistry::instance().transfer_latency_us.observe(elapsed_us);
        MetricsRegistry::instance().total_transfers.increment();

        if (success) {
            completed_transfers_.fetch_add(1);
            atomic_add(total_bytes_transferred_, static_cast<double>(config_.block_total_bytes()));
        } else {
            failed_transfers_.fetch_add(1);
        }

        // Fulfill the promise
        transfer->promise.set_value(success);

        // Call user callback
        if (transfer->request.callback) {
            transfer->request.callback(success);
        }

        pending_count_.fetch_sub(1);
    }
}

bool TransferEngine::execute_transfer(const TransferRequest& req) {
    if (!req.src.valid() || !req.dst.valid()) return false;

    try {
        tier_manager_.copy_block_data(req.dst, req.src);

        // Track directional stats
        if (req.src.tier == TierType::GPU_VRAM && req.dst.tier == TierType::CPU_DRAM) {
            gpu_cpu_count_.fetch_add(1);
        } else if (req.src.tier == TierType::CPU_DRAM && req.dst.tier == TierType::GPU_VRAM) {
            cpu_gpu_count_.fetch_add(1);
        } else if (req.src.tier == TierType::CPU_DRAM && req.dst.tier == TierType::NVME_SSD) {
            cpu_ssd_count_.fetch_add(1);
        } else if (req.src.tier == TierType::NVME_SSD && req.dst.tier == TierType::CPU_DRAM) {
            ssd_cpu_count_.fetch_add(1);
        }

        return true;
    } catch (const std::exception& e) {
        KVCACHE_LOG_ERROR("transfer", std::string("Transfer failed: ") + e.what());
        return false;
    }
}

} // namespace kvcache
