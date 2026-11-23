#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <limits>
#include <ostream>
#include <stdexcept>
#include <cmath>
#include <algorithm>

namespace kvcache {

// ---------------------------------------------------------------------------
// Strongly-typed IDs
// ---------------------------------------------------------------------------

struct BlockId {
    int32_t value = -1;
    bool valid() const { return value >= 0; }
    bool operator==(const BlockId& o) const { return value == o.value; }
    bool operator!=(const BlockId& o) const { return value != o.value; }
    bool operator<(const BlockId& o) const  { return value < o.value; }
};

struct SequenceId {
    int64_t value = -1;
    bool valid() const { return value >= 0; }
    bool operator==(const SequenceId& o) const { return value == o.value; }
    bool operator!=(const SequenceId& o) const { return value != o.value; }
    bool operator<(const SequenceId& o) const  { return value < o.value; }
};

// Hash support for use in unordered containers
struct SequenceIdHash {
    size_t operator()(const SequenceId& id) const {
        return std::hash<int64_t>{}(id.value);
    }
};

struct BlockIdHash {
    size_t operator()(const BlockId& id) const {
        return std::hash<int32_t>{}(id.value);
    }
};

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------

/// Data type for KV cache storage.
enum class DataType : uint8_t {
    FP32 = 0,   ///< 32-bit floating point (4 bytes)
    FP16 = 1,   ///< 16-bit floating point (2 bytes)
    INT8 = 2,   ///< 8-bit integer with per-channel scale (1 byte + scale)
};

/// Returns the size in bytes for a single element of the given data type.
inline size_t dtype_size(DataType dt) {
    switch (dt) {
        case DataType::FP32: return 4;
        case DataType::FP16: return 2;
        case DataType::INT8: return 1;
    }
    return 4; // default
}

/// Returns a human-readable name for the data type.
inline const char* dtype_name(DataType dt) {
    switch (dt) {
        case DataType::FP32: return "FP32";
        case DataType::FP16: return "FP16";
        case DataType::INT8: return "INT8";
    }
    return "UNKNOWN";
}

/// Eviction policy type.
enum class EvictionType : uint8_t {
    NONE           = 0,   ///< No eviction — allocation fails when full
    LRU            = 1,   ///< Least Recently Used
    H2O            = 2,   ///< Heavy Hitter Oracle (attention-score based)
    SLIDING_WINDOW = 3,   ///< Keep last N tokens + initial sink tokens
};

inline const char* eviction_name(EvictionType et) {
    switch (et) {
        case EvictionType::NONE:           return "NONE";
        case EvictionType::LRU:            return "LRU";
        case EvictionType::H2O:            return "H2O";
        case EvictionType::SLIDING_WINDOW: return "SLIDING_WINDOW";
    }
    return "UNKNOWN";
}

// ---------------------------------------------------------------------------
// Status codes
// ---------------------------------------------------------------------------

enum class Status : uint8_t {
    OK = 0,
    ERROR_OUT_OF_MEMORY,
    ERROR_SEQUENCE_NOT_FOUND,
    ERROR_SEQUENCE_EXISTS,
    ERROR_INVALID_ARGUMENT,
    ERROR_BLOCK_NOT_FOUND,
    ERROR_TOKEN_OUT_OF_RANGE,
    ERROR_INTERNAL,
};

inline const char* status_name(Status s) {
    switch (s) {
        case Status::OK:                      return "OK";
        case Status::ERROR_OUT_OF_MEMORY:     return "ERROR_OUT_OF_MEMORY";
        case Status::ERROR_SEQUENCE_NOT_FOUND:return "ERROR_SEQUENCE_NOT_FOUND";
        case Status::ERROR_SEQUENCE_EXISTS:   return "ERROR_SEQUENCE_EXISTS";
        case Status::ERROR_INVALID_ARGUMENT:  return "ERROR_INVALID_ARGUMENT";
        case Status::ERROR_BLOCK_NOT_FOUND:   return "ERROR_BLOCK_NOT_FOUND";
        case Status::ERROR_TOKEN_OUT_OF_RANGE:return "ERROR_TOKEN_OUT_OF_RANGE";
        case Status::ERROR_INTERNAL:          return "ERROR_INTERNAL";
    }
    return "UNKNOWN";
}

inline std::ostream& operator<<(std::ostream& os, Status s) {
    return os << status_name(s);
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Master configuration for the KV Cache system.
struct KVCacheConfig {
    int num_layers    = 32;      ///< Number of transformer layers
    int num_kv_heads  = 32;      ///< Number of K/V attention heads (may differ from Q heads for GQA)
    int head_dim      = 128;     ///< Dimension per attention head
    int block_size    = 16;      ///< Number of tokens per physical block
    int max_blocks    = 1024;    ///< Total physical blocks in the pool
    DataType dtype    = DataType::FP32;  ///< Storage data type
    EvictionType eviction_policy = EvictionType::NONE;

    // Eviction-specific parameters
    int sliding_window_size = 256;   ///< For SLIDING_WINDOW: number of recent tokens to keep
    int sink_token_count    = 4;     ///< For SLIDING_WINDOW/H2O: initial "sink" tokens to always keep
    float h2o_heavy_ratio   = 0.2f;  ///< For H2O: fraction of tokens to keep as heavy hitters
    int max_sequence_tokens = 1048576;
    int max_prefix_entries = 256;

    void validate() const {
        if (num_layers <= 0 || num_kv_heads <= 0 || head_dim <= 0 || block_size <= 0 ||
            max_blocks < 0 || max_sequence_tokens <= 0 || max_prefix_entries < 0 ||
            sliding_window_size < 0 || sink_token_count < 0 ||
            !std::isfinite(h2o_heavy_ratio) || h2o_heavy_ratio < 0 || h2o_heavy_ratio > 1 ||
            static_cast<unsigned>(dtype) > static_cast<unsigned>(DataType::INT8))
            throw std::invalid_argument("Invalid KV cache configuration");
        size_t bytes = dtype_size(dtype);
        for (size_t factor : {size_t(2), size_t(num_layers), size_t(num_kv_heads),
                              size_t(head_dim), size_t(block_size), size_t(std::max(1, max_blocks))}) {
            if (bytes > std::numeric_limits<size_t>::max() / factor)
                throw std::overflow_error("KV cache capacity overflow");
            bytes *= factor;
        }
        if (static_cast<int64_t>(max_blocks) * block_size > std::numeric_limits<int>::max())
            throw std::overflow_error("KV cache token capacity overflow");
    }

    /// Computed: bytes per KV pair (K + V) for a single token in one layer.
    size_t kv_pair_bytes_per_token() const {
        // K and V each have shape [num_kv_heads, head_dim]
        return size_t(2) * num_kv_heads * head_dim * dtype_size(dtype);
    }

    /// Computed: bytes per block for one layer.
    size_t block_bytes_per_layer() const {
        return block_size * kv_pair_bytes_per_token();
    }

    /// Computed: total bytes for the entire block pool (all layers).
    size_t total_pool_bytes() const {
        return static_cast<size_t>(max_blocks) * num_layers * block_bytes_per_layer();
    }

    /// Computed: maximum number of tokens that can be stored.
    int max_tokens() const {
        return max_blocks * block_size;
    }
};

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

/// Cache utilization and performance statistics.
struct CacheStats {
    int total_blocks       = 0;
    int used_blocks        = 0;
    int free_blocks        = 0;
    int active_sequences   = 0;
    int total_cached_tokens = 0;
    size_t total_memory_bytes = 0;
    size_t used_memory_bytes  = 0;
    double utilization_pct    = 0.0;  ///< used_blocks / total_blocks * 100
    int64_t total_allocations = 0;
    int64_t total_frees       = 0;
    int64_t total_evictions   = 0;
    int64_t prefix_cache_hits = 0;
};

// ---------------------------------------------------------------------------
// KV data pointer pair
// ---------------------------------------------------------------------------

/// A pair of pointers to key and value data for a single token at one layer.
struct KVPair {
    const void* key   = nullptr;
    const void* value = nullptr;
    bool valid() const { return key != nullptr && value != nullptr; }
};

/// Mutable version for writing.
struct KVPairMut {
    void* key   = nullptr;
    void* value = nullptr;
    bool valid() const { return key != nullptr && value != nullptr; }
};

} // namespace kvcache
