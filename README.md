# KV Cache System for LLM Inference Optimization

A high-performance C++17 library implementing a **paged KV (Key-Value) cache** for Transformer-based LLM inference. Eliminates memory fragmentation via PagedAttention, supports multiple eviction policies, quantization, and prefix sharing.

## Features

| Feature | Description |
|---------|-------------|
| **PagedAttention** | OS-inspired paged memory management — near-zero fragmentation (<4%) |
| **Copy-on-Write** | Efficient sequence forking for beam search and parallel sampling |
| **Eviction Policies** | LRU, H2O (Heavy Hitter Oracle), Sliding Window (StreamingLLM-style) |
| **Quantization** | FP32 → FP16 / INT8 with accuracy metrics (MAE, SQNR) |
| **Prefix Caching** | Hash-based prefix sharing across sequences with common prompts |
| **Zero Dependencies** | Pure C++17 — no external libraries required |

## Architecture

```
┌──────────────────────────────────────────────┐
│              KVCacheManager                  │
│  (top-level orchestrator / public API)       │
├──────────┬───────────┬───────────┬───────────┤
│  Block   │ Sequence  │ Eviction  │  Prefix   │
│ Allocator│  Manager  │  Engine   │  Cache    │
├──────────┴───────────┴───────────┴───────────┤
│              Physical Block Pool             │
│     [block_size tokens × num_layers × KV]    │
└──────────────────────────────────────────────┘
```

## Quick Start

### Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build .
```

### Run Tests

```bash
cd build
ctest --output-on-failure
```

### Run Benchmark

```bash
./bench_throughput          # Linux/macOS
.\Release\bench_throughput  # Windows
```

### Run Interactive CLI

```bash
./kvcache_cli               # Linux/macOS
.\Release\kvcache_cli       # Windows
```

#### CLI Commands

```
kvcache> create 1              # Create sequence with ID 1
kvcache> append_n 1 32         # Append 32 tokens (all layers)
kvcache> get 1 0 5             # Get KV data for seq 1, layer 0, token 5
kvcache> fork 1 2              # Fork seq 1 → seq 2 (Copy-on-Write)
kvcache> stats                 # Print cache utilization stats
kvcache> bench 4 64            # Benchmark: 4 sequences × 64 tokens
kvcache> free 1                # Free sequence 1
kvcache> quit
```

## Usage (as a library)

```cpp
#include "kvcache/kvcache.h"

using namespace kvcache;

// Configure for a small model
KVCacheConfig config;
config.num_layers   = 32;
config.num_kv_heads = 8;     // GQA: fewer KV heads than Q heads
config.head_dim     = 128;
config.block_size   = 16;
config.max_blocks   = 1024;
config.dtype        = DataType::FP32;
config.eviction_policy = EvictionType::LRU;

// Create manager
KVCacheManager manager(config);

// Create a sequence for a new request
manager.create_sequence(SequenceId{1});

// Append KV data for each token, each layer
std::vector<float> key_data(config.num_kv_heads * config.head_dim);
std::vector<float> val_data(config.num_kv_heads * config.head_dim);
// ... fill key_data, val_data from attention computation ...

for (int layer = 0; layer < config.num_layers; ++layer) {
    manager.append_kv(SequenceId{1}, layer, key_data.data(), val_data.data());
}

// Retrieve cached KV for a specific token
KVPair kv = manager.get_kv(SequenceId{1}, /*layer=*/0, /*token_idx=*/0);
const float* k = static_cast<const float*>(kv.key);
const float* v = static_cast<const float*>(kv.value);

// Fork for beam search (Copy-on-Write — no data copied)
manager.fork_sequence(SequenceId{1}, SequenceId{2});

// Check stats
CacheStats stats = manager.get_stats();
printf("Utilization: %.1f%%\n", stats.utilization_pct);

// Cleanup
manager.free_sequence(SequenceId{1});
manager.free_sequence(SequenceId{2});
```

## Project Structure

```
KVcacheproject/
├── CMakeLists.txt              # Root build system
├── README.md
├── include/kvcache/
│   ├── kvcache.h               # Public API (single include)
│   ├── types.h                 # Core types, enums, config
│   ├── block_allocator.h       # Paged block memory allocator
│   ├── block_table.h           # Logical → physical mapping
│   ├── sequence.h              # Per-request KV state
│   ├── eviction_policy.h       # Eviction interface + implementations
│   ├── quantization.h          # FP16/INT8 conversion utilities
│   ├── prefix_cache.h          # Prefix sharing / deduplication
│   └── cache_manager.h         # Top-level orchestrator
├── src/                        # Implementation files
├── cli/main.cpp                # Interactive CLI + benchmark
├── tests/                      # Comprehensive test suite
└── benchmarks/                 # Throughput benchmarks
```

## Configuration Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_layers` | 32 | Transformer layer count |
| `num_kv_heads` | 32 | K/V attention heads (supports GQA) |
| `head_dim` | 128 | Dimension per head |
| `block_size` | 16 | Tokens per physical block |
| `max_blocks` | 1024 | Total blocks in pool |
| `dtype` | FP32 | Storage type: FP32, FP16, INT8 |
| `eviction_policy` | NONE | NONE, LRU, H2O, SLIDING_WINDOW |
| `sliding_window_size` | 256 | Recent tokens to keep (Sliding Window) |
| `sink_token_count` | 4 | Initial "sink" tokens to protect |
| `h2o_heavy_ratio` | 0.2 | Fraction of heavy hitters to protect (H2O) |

## License

MIT
