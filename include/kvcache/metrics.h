#pragma once

#include "types.h"
#include <atomic>
#include <mutex>
#include <string>
#include <sstream>
#include <vector>
#include <chrono>
#include <cmath>
#include <cstring>

namespace kvcache {

/// Atomic add helper for std::atomic<double> in C++17
inline void atomic_add(std::atomic<double>& target, double val) {
    double current = target.load(std::memory_order_relaxed);
    while (!target.compare_exchange_weak(current, current + val,
                                         std::memory_order_relaxed,
                                         std::memory_order_relaxed));
}

/// Atomic counter — monotonically increasing.
class Counter {
public:
    explicit Counter(const std::string& name, const std::string& help = "")
        : name_(name), help_(help) {}

    void increment(int64_t delta = 1) { value_.fetch_add(delta, std::memory_order_relaxed); }
    int64_t value() const { return value_.load(std::memory_order_relaxed); }
    void reset() { value_.store(0, std::memory_order_relaxed); }

    const std::string& name() const { return name_; }
    const std::string& help() const { return help_; }

private:
    std::string name_;
    std::string help_;
    std::atomic<int64_t> value_{0};
};

/// Atomic gauge — can go up and down.
class Gauge {
public:
    explicit Gauge(const std::string& name, const std::string& help = "")
        : name_(name), help_(help) {}

    void set(double val) {
        int64_t bits;
        std::memcpy(&bits, &val, sizeof(bits));
        value_.store(bits, std::memory_order_relaxed);
    }
    void increment(double delta = 1.0) { set(value() + delta); }
    void decrement(double delta = 1.0) { set(value() - delta); }

    double value() const {
        int64_t bits = value_.load(std::memory_order_relaxed);
        double val;
        std::memcpy(&val, &bits, sizeof(val));
        return val;
    }

    const std::string& name() const { return name_; }
    const std::string& help() const { return help_; }

private:
    std::string name_;
    std::string help_;
    std::atomic<int64_t> value_{0};
};

/// Simple histogram with fixed buckets.
class Histogram {
public:
    Histogram(const std::string& name, const std::vector<double>& buckets,
              const std::string& help = "")
        : name_(name), help_(help), bucket_bounds_(buckets)
        , bucket_counts_(buckets.size() + 1) // +1 for +Inf
    {
        for (auto& c : bucket_counts_) c.store(0, std::memory_order_relaxed);
    }

    void observe(double value) {
        atomic_add(total_sum_, value);
        total_count_.fetch_add(1, std::memory_order_relaxed);
        for (size_t i = 0; i < bucket_bounds_.size(); ++i) {
            if (value <= bucket_bounds_[i]) {
                bucket_counts_[i].fetch_add(1, std::memory_order_relaxed);
                return;
            }
        }
        // +Inf bucket
        bucket_counts_.back().fetch_add(1, std::memory_order_relaxed);
    }

    int64_t count() const { return total_count_.load(std::memory_order_relaxed); }

    const std::string& name() const { return name_; }
    const std::string& help() const { return help_; }

    /// Export in Prometheus text format.
    std::string to_prometheus() const;

private:
    std::string name_;
    std::string help_;
    std::vector<double> bucket_bounds_;
    std::vector<std::atomic<int64_t>> bucket_counts_;
    std::atomic<int64_t> total_count_{0};
    std::atomic<double>  total_sum_{0.0};
};

/// High-resolution timer for measuring latencies.
class ScopedTimer {
public:
    explicit ScopedTimer(Histogram& histogram)
        : histogram_(histogram)
        , start_(std::chrono::high_resolution_clock::now()) {}

    ~ScopedTimer() {
        auto end = std::chrono::high_resolution_clock::now();
        double us = std::chrono::duration<double, std::micro>(end - start_).count();
        histogram_.observe(us);
    }

private:
    Histogram& histogram_;
    std::chrono::high_resolution_clock::time_point start_;
};

/// Centralized metrics registry for the KV Cache system.
class MetricsRegistry {
public:
    static MetricsRegistry& instance();

    // --- Counters ---
    Counter total_requests{"pageforge_total_requests", "Total inference requests received"};
    Counter total_tokens_generated{"pageforge_total_tokens", "Total tokens generated"};
    Counter total_evictions{"pageforge_total_evictions", "Total KV blocks evicted"};
    Counter cache_hits{"pageforge_cache_hits", "Prefix cache hits"};
    Counter cache_misses{"pageforge_cache_misses", "Prefix cache misses"};
    Counter total_promotions{"pageforge_total_promotions", "Blocks promoted to faster tier"};
    Counter total_demotions{"pageforge_total_demotions", "Blocks demoted to slower tier"};
    Counter total_transfers{"pageforge_total_transfers", "Total inter-tier transfers"};

    // --- Gauges ---
    Gauge gpu_blocks_used{"pageforge_gpu_blocks_used", "GPU VRAM blocks in use"};
    Gauge cpu_blocks_used{"pageforge_cpu_blocks_used", "CPU DRAM blocks in use"};
    Gauge ssd_blocks_used{"pageforge_ssd_blocks_used", "NVMe SSD blocks in use"};
    Gauge gpu_utilization_pct{"pageforge_gpu_util_pct", "GPU tier utilization %"};
    Gauge cpu_utilization_pct{"pageforge_cpu_util_pct", "CPU tier utilization %"};
    Gauge ssd_utilization_pct{"pageforge_ssd_util_pct", "SSD tier utilization %"};
    Gauge active_sequences{"pageforge_active_sequences", "Currently active sequences"};
    Gauge active_batch_size{"pageforge_batch_size", "Current batch size"};

    // --- Histograms ---
    Histogram request_latency_ms{
        "pageforge_request_latency_ms",
        {1, 5, 10, 25, 50, 100, 250, 500, 1000, 5000},
        "End-to-end request latency in ms"
    };
    Histogram token_latency_us{
        "pageforge_token_latency_us",
        {10, 50, 100, 250, 500, 1000, 2500, 5000, 10000},
        "Per-token generation latency in us"
    };
    Histogram transfer_latency_us{
        "pageforge_transfer_latency_us",
        {1, 5, 10, 50, 100, 500, 1000, 5000},
        "Inter-tier transfer latency in us"
    };

    /// Export all metrics in Prometheus text exposition format.
    std::string export_prometheus() const;

    /// Reset all metrics.
    void reset();

private:
    MetricsRegistry() = default;
};

} // namespace kvcache
