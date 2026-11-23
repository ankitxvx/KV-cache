#include "kvcache/metrics.h"
#include <sstream>
#include <iomanip>

namespace kvcache {

MetricsRegistry& MetricsRegistry::instance() {
    static MetricsRegistry registry;
    return registry;
}

std::string Histogram::to_prometheus() const {
    std::ostringstream oss;
    if (!help_.empty()) {
        oss << "# HELP " << name_ << " " << help_ << "\n";
    }
    oss << "# TYPE " << name_ << " histogram\n";

    int64_t cumulative = 0;
    for (size_t i = 0; i < bucket_bounds_.size(); ++i) {
        cumulative += bucket_counts_[i].load(std::memory_order_relaxed);
        oss << name_ << "_bucket{le=\"" << bucket_bounds_[i] << "\"} "
            << cumulative << "\n";
    }
    cumulative += bucket_counts_.back().load(std::memory_order_relaxed);
    oss << name_ << "_bucket{le=\"+Inf\"} " << cumulative << "\n";
    oss << name_ << "_sum " << total_sum_.load(std::memory_order_relaxed) << "\n";
    oss << name_ << "_count " << total_count_.load(std::memory_order_relaxed) << "\n";

    return oss.str();
}

std::string MetricsRegistry::export_prometheus() const {
    std::ostringstream oss;

    // Helper lambda for counter
    auto emit_counter = [&](const Counter& c) {
        if (!c.help().empty()) oss << "# HELP " << c.name() << " " << c.help() << "\n";
        oss << "# TYPE " << c.name() << " counter\n";
        oss << c.name() << " " << c.value() << "\n";
    };

    // Helper lambda for gauge
    auto emit_gauge = [&](const Gauge& g) {
        if (!g.help().empty()) oss << "# HELP " << g.name() << " " << g.help() << "\n";
        oss << "# TYPE " << g.name() << " gauge\n";
        oss << g.name() << " " << std::fixed << std::setprecision(2) << g.value() << "\n";
    };

    // Counters
    emit_counter(total_requests);
    emit_counter(total_tokens_generated);
    emit_counter(total_evictions);
    emit_counter(cache_hits);
    emit_counter(cache_misses);
    emit_counter(total_promotions);
    emit_counter(total_demotions);
    emit_counter(total_transfers);

    // Gauges
    emit_gauge(gpu_blocks_used);
    emit_gauge(cpu_blocks_used);
    emit_gauge(ssd_blocks_used);
    emit_gauge(gpu_utilization_pct);
    emit_gauge(cpu_utilization_pct);
    emit_gauge(ssd_utilization_pct);
    emit_gauge(active_sequences);
    emit_gauge(active_batch_size);

    // Histograms
    oss << request_latency_ms.to_prometheus();
    oss << token_latency_us.to_prometheus();
    oss << transfer_latency_us.to_prometheus();

    return oss.str();
}

void MetricsRegistry::reset() {
    total_requests.reset();
    total_tokens_generated.reset();
    total_evictions.reset();
    cache_hits.reset();
    cache_misses.reset();
    total_promotions.reset();
    total_demotions.reset();
    total_transfers.reset();

    gpu_blocks_used.set(0);
    cpu_blocks_used.set(0);
    ssd_blocks_used.set(0);
    gpu_utilization_pct.set(0);
    cpu_utilization_pct.set(0);
    ssd_utilization_pct.set(0);
    active_sequences.set(0);
    active_batch_size.set(0);
}

} // namespace kvcache
