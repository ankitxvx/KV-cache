#pragma once

#include "types.h"
#include "logger.h"
#include <string>

namespace kvcache {

/// Storage tier types for the 3-tier memory hierarchy.
enum class TierType : uint8_t {
    GPU_VRAM  = 0,   ///< NVIDIA GPU High Bandwidth Memory (fastest, smallest)
    CPU_DRAM  = 1,   ///< Host DDR5 RAM (medium speed, medium capacity)
    NVME_SSD  = 2,   ///< NVMe SSD storage (slowest, largest)
};

inline const char* tier_name(TierType t) {
    switch (t) {
        case TierType::GPU_VRAM: return "GPU_VRAM";
        case TierType::CPU_DRAM: return "CPU_DRAM";
        case TierType::NVME_SSD: return "NVME_SSD";
    }
    return "UNKNOWN";
}

/// Request state for the scheduler.
enum class RequestState : uint8_t {
    WAITING    = 0,
    PREFILLING = 1,
    DECODING   = 2,
    COMPLETED  = 3,
    PREEMPTED  = 4,
    ABORTED    = 5,
};

/// Transfer priority.
enum class TransferPriority : uint8_t {
    LOW      = 0,
    NORMAL   = 1,
    HIGH     = 2,
    CRITICAL = 3,
};

/// A block identifier that includes its storage tier.
struct TieredBlockId {
    BlockId   block_id;
    TierType  tier = TierType::CPU_DRAM;
    int64_t global_id = -1; // Stable identity; location fields are a snapshot.
    bool valid() const { return block_id.valid(); }
};

/// Statistics for a single storage tier.
struct TierStats {
    TierType tier;
    int      total_blocks   = 0;
    int      used_blocks    = 0;
    int      free_blocks    = 0;
    size_t   total_bytes    = 0;
    size_t   used_bytes     = 0;
    double   utilization_pct = 0.0;
};

/// Transfer statistics.
struct TransferStats {
    int64_t total_transfers       = 0;
    int64_t gpu_to_cpu_transfers  = 0;
    int64_t cpu_to_gpu_transfers  = 0;
    int64_t cpu_to_ssd_transfers  = 0;
    int64_t ssd_to_cpu_transfers  = 0;
    double  avg_transfer_us       = 0.0;
    double  total_bytes_transferred = 0.0;
};

/// Production configuration for the entire PageForge system.
struct ProductionConfig {
    // ── Model parameters ──
    int num_layers    = 32;
    int num_kv_heads  = 8;
    int head_dim      = 128;
    int block_size    = 16;
    DataType dtype    = DataType::FP16;
    EvictionType eviction_policy = EvictionType::LRU;

    // ── Tier capacities ──
    size_t gpu_memory_bytes   = 3ULL * 1024 * 1024 * 1024;   // 3GB of 4GB VRAM
    size_t cpu_memory_bytes   = 16ULL * 1024 * 1024 * 1024;  // 16GB of 24GB RAM
    size_t ssd_capacity_bytes = 100ULL * 1024 * 1024 * 1024;  // 100GB of 512GB SSD
    std::string ssd_cache_path = "pageforge_ssd_pool.dat";
    bool enable_gpu           = true;   // Auto-disabled if CUDA not available
    bool allow_gpu_emulation = false; // Tests only; never silently claim GPU capacity.
    bool enable_ssd           = true;

    // ── Scheduler ──
    int max_batch_size      = 32;
    int max_prefill_tokens  = 2048;
    int max_waiting_requests = 256;

    // ── Transfer engine ──
    int  num_transfer_threads = 4;
    int max_pending_transfers = 1024;
    bool use_pinned_memory    = true;
    size_t staging_buffer_bytes = 256 * 1024 * 1024;  // 256MB staging

    // ── Eviction thresholds ──
    float gpu_eviction_threshold = 0.90f;
    float cpu_eviction_threshold = 0.85f;
    float gpu_promotion_threshold = 0.70f; // Promote to GPU when below 70%

    // ── Eviction-specific ──
    int sliding_window_size = 256;
    int sink_token_count    = 4;
    float h2o_heavy_ratio   = 0.2f;

    // ── Logging ──
    std::string log_level = "INFO";
    std::string log_file  = "";

    // ── Server ──
    int server_port = 8080;
    std::string server_host = "127.0.0.1";

    void validate() const {
        KVCacheConfig c; c.num_layers=num_layers; c.num_kv_heads=num_kv_heads;
        c.head_dim=head_dim; c.block_size=block_size; c.max_blocks=0; c.dtype=dtype; c.validate();
        if (max_batch_size <= 0 || max_prefill_tokens <= 0 || max_waiting_requests <= 0 ||
            num_transfer_threads <= 0 || num_transfer_threads > 128 || max_pending_transfers <= 0 ||
            !std::isfinite(gpu_eviction_threshold) || gpu_eviction_threshold <= 0 || gpu_eviction_threshold > 1 ||
            !std::isfinite(cpu_eviction_threshold) || cpu_eviction_threshold <= 0 || cpu_eviction_threshold > 1)
            throw std::invalid_argument("Invalid scheduler/transfer configuration");
        for (auto budget : {gpu_memory_bytes, cpu_memory_bytes, ssd_capacity_bytes})
            if (budget / block_total_bytes() > static_cast<size_t>(std::numeric_limits<int>::max()) / block_size)
                throw std::overflow_error("Tier capacity exceeds supported block count");
    }

    // ── Computed helpers ──

    /// Bytes per KV pair (K+V) for one token in one layer.
    size_t kv_pair_bytes_per_token() const {
        return size_t(2) * num_kv_heads * head_dim * dtype_size(dtype);
    }

    /// Bytes per block for one layer.
    size_t block_bytes_per_layer() const {
        return block_size * kv_pair_bytes_per_token();
    }

    /// Total bytes per block across all layers.
    size_t block_total_bytes() const {
        return static_cast<size_t>(num_layers) * block_bytes_per_layer();
    }

    /// Max blocks that fit in GPU VRAM.
    int gpu_max_blocks() const {
        if (!enable_gpu || gpu_memory_bytes == 0) return 0;
        return static_cast<int>(gpu_memory_bytes / block_total_bytes());
    }

    /// Max blocks that fit in CPU DRAM.
    int cpu_max_blocks() const {
        return static_cast<int>(cpu_memory_bytes / block_total_bytes());
    }

    /// Max blocks that fit on SSD.
    int ssd_max_blocks() const {
        if (!enable_ssd || ssd_capacity_bytes == 0) return 0;
        return static_cast<int>(ssd_capacity_bytes / block_total_bytes());
    }

    /// Total blocks across all tiers.
    int total_max_blocks() const {
        return gpu_max_blocks() + cpu_max_blocks() + ssd_max_blocks();
    }

    /// Convert to the basic KVCacheConfig for backward compatibility.
    KVCacheConfig to_kvcache_config() const {
        KVCacheConfig cfg;
        cfg.num_layers = num_layers;
        cfg.num_kv_heads = num_kv_heads;
        cfg.head_dim = head_dim;
        cfg.block_size = block_size;
        cfg.max_blocks = cpu_max_blocks();  // CPU tier for the base allocator
        cfg.dtype = dtype;
        cfg.eviction_policy = eviction_policy;
        cfg.sliding_window_size = sliding_window_size;
        cfg.sink_token_count = sink_token_count;
        cfg.h2o_heavy_ratio = h2o_heavy_ratio;
        return cfg;
    }

    /// Initialize logging from this config.
    void apply_logging() const {
        auto& logger = Logger::instance();
        logger.set_level(log_level_from_string(log_level));
        if (!log_file.empty()) {
            logger.set_file(log_file);
        }
    }
};

} // namespace kvcache
