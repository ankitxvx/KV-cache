#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include "kvcache/kvcache.h"

using namespace kvcache;

// ---------------------------------------------------------------------------
// Benchmark configurations
// ---------------------------------------------------------------------------

struct BenchConfig {
    std::string name;
    int num_layers;
    int num_kv_heads;
    int head_dim;
    int block_size;
    int max_blocks;
    int num_sequences;
    int seq_length;
    DataType dtype;
};

static void run_bench(const BenchConfig& bcfg) {
    KVCacheConfig config;
    config.num_layers   = bcfg.num_layers;
    config.num_kv_heads = bcfg.num_kv_heads;
    config.head_dim     = bcfg.head_dim;
    config.block_size   = bcfg.block_size;
    config.max_blocks   = bcfg.max_blocks;
    config.dtype        = bcfg.dtype;

    KVCacheManager manager(config);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    int kv_elements = config.num_kv_heads * config.head_dim;

    std::cout << "  [" << bcfg.name << "] "
              << bcfg.num_sequences << " seqs x " << bcfg.seq_length << " tokens, "
              << bcfg.num_layers << "L x " << bcfg.num_kv_heads << "H x " << bcfg.head_dim << "D, "
              << dtype_name(bcfg.dtype) << ", bs=" << bcfg.block_size << "\n";

    // --- Append ---
    auto start = std::chrono::high_resolution_clock::now();
    int total_token_appends = 0;

    for (int s = 0; s < bcfg.num_sequences; ++s) {
        SequenceId sid{s};
        manager.create_sequence(sid);

        for (int t = 0; t < bcfg.seq_length; ++t) {
            bool ok = true;
            for (int layer = 0; layer < bcfg.num_layers; ++layer) {
                std::vector<float> key(kv_elements);
                std::vector<float> val(kv_elements);
                for (auto& v : key) v = dist(rng);
                for (auto& v : val) v = dist(rng);

                Status status = manager.append_kv(sid, layer, key.data(), val.data());
                if (status != Status::OK) {
                    ok = false;
                    break;
                }
            }
            if (!ok) break;
            total_token_appends++;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    double append_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double tokens_per_sec = (total_token_appends / append_ms) * 1000.0;
    double append_latency_ns = (append_ms * 1e6) / (total_token_appends * bcfg.num_layers);

    // --- Retrieval ---
    start = std::chrono::high_resolution_clock::now();
    int total_gets = 0;

    for (int s = 0; s < bcfg.num_sequences; ++s) {
        SequenceId sid{s};
        int seq_len = manager.get_num_tokens(sid);
        for (int t = 0; t < seq_len; ++t) {
            for (int layer = 0; layer < bcfg.num_layers; ++layer) {
                KVPair kv = manager.get_kv(sid, layer, t);
                if (kv.valid()) total_gets++;
            }
        }
    }

    end = std::chrono::high_resolution_clock::now();
    double get_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double gets_per_sec = (total_gets / get_ms) * 1000.0;
    double get_latency_ns = (get_ms * 1e6) / total_gets;

    // --- Fork ---
    start = std::chrono::high_resolution_clock::now();
    int fork_count = 0;

    for (int s = 0; s < bcfg.num_sequences; ++s) {
        SequenceId src{s};
        SequenceId dst{1000 + s};
        Status status = manager.fork_sequence(src, dst);
        if (status == Status::OK) fork_count++;
    }

    end = std::chrono::high_resolution_clock::now();
    double fork_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double fork_latency_us = (fork_count > 0) ? (fork_ms * 1000.0 / fork_count) : 0;

    // --- Stats ---
    CacheStats stats = manager.get_stats();

    std::cout << std::fixed << std::setprecision(1);
    std::cout << "    Append:  " << total_token_appends << " tokens in "
              << append_ms << " ms → " << std::setprecision(0) << tokens_per_sec
              << " tok/s, " << std::setprecision(0) << append_latency_ns << " ns/layer-append\n";
    std::cout << "    Get:     " << total_gets << " gets in "
              << std::setprecision(1) << get_ms << " ms → " << std::setprecision(0) << gets_per_sec
              << " gets/s, " << get_latency_ns << " ns/get\n";
    std::cout << "    Fork:    " << fork_count << " forks in "
              << std::setprecision(2) << fork_ms << " ms → " << fork_latency_us << " µs/fork\n";
    std::cout << "    Memory:  " << (stats.used_memory_bytes / 1024) << " KB / "
              << (stats.total_memory_bytes / 1024) << " KB ("
              << std::setprecision(1) << stats.utilization_pct << "% used)\n";
    std::cout << "\n";

    // Cleanup
    for (int s = 0; s < bcfg.num_sequences; ++s) {
        manager.free_sequence(SequenceId{s});
        manager.free_sequence(SequenceId{1000 + s});
    }
}

int main() {
    std::cout << R"(
╔═══════════════════════════════════════════════════════════════╗
║            KV Cache Throughput Benchmark Suite                ║
╚═══════════════════════════════════════════════════════════════╝
)" << "\n";

    std::vector<BenchConfig> configs = {
        // Small model — stress test
        {"Small-StressTest", 4, 4, 64, 16, 512, 16, 128, DataType::FP32},

        // Medium model — typical inference
        {"Medium-Typical", 12, 8, 64, 16, 1024, 4, 256, DataType::FP32},

        // Large model — realistic config (reduced dimensions for memory)
        {"Large-Realistic", 32, 8, 128, 16, 2048, 2, 128, DataType::FP32},

        // Small model — block size comparison
        {"SmallBlock-BS4", 4, 4, 64, 4, 2048, 8, 128, DataType::FP32},
        {"SmallBlock-BS32", 4, 4, 64, 32, 256, 8, 128, DataType::FP32},

        // Single long sequence
        {"LongSequence", 4, 4, 64, 16, 1024, 1, 1024, DataType::FP32},
    };

    for (const auto& cfg : configs) {
        run_bench(cfg);
    }

    std::cout << "  All benchmarks complete.\n\n";
    return 0;
}
