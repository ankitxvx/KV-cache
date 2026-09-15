#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include <cstring>

// Include from the project's include directory
#include "kvcache/kvcache.h"

using namespace kvcache;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static void print_banner() {
    std::cout << R"(
 ██╗  ██╗██╗   ██╗  ██████╗ █████╗  ██████╗██╗  ██╗███████╗
 ██║ ██╔╝██║   ██║ ██╔════╝██╔══██╗██╔════╝██║  ██║██╔════╝
 █████╔╝ ██║   ██║ ██║     ███████║██║     ███████║█████╗
 ██╔═██╗ ╚██╗ ██╔╝ ██║     ██╔══██║██║     ██╔══██║██╔══╝
 ██║  ██╗ ╚████╔╝  ╚██████╗██║  ██║╚██████╗██║  ██║███████╗
 ╚═╝  ╚═╝  ╚═══╝    ╚═════╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝╚══════╝
    )" << "\n";
    std::cout << "  KV Cache System for LLM Inference — Interactive CLI\n";
    std::cout << "  Type 'help' for available commands.\n\n";
}

static void print_help() {
    std::cout << "\n  Commands:\n";
    std::cout << "    create <seq_id>                   Create a new sequence\n";
    std::cout << "    append <seq_id> <layer>            Append random KV data to a sequence\n";
    std::cout << "    append_n <seq_id> <n>              Append N tokens (all layers)\n";
    std::cout << "    get <seq_id> <layer> <token_idx>   Retrieve cached KV data\n";
    std::cout << "    fork <src_id> <dst_id>             Fork a sequence (Copy-on-Write)\n";
    std::cout << "    free <seq_id>                      Free a sequence\n";
    std::cout << "    tokens <seq_id>                    Show token count\n";
    std::cout << "    stats                              Print cache statistics\n";
    std::cout << "    bench <num_seqs> <seq_length>      Run throughput benchmark\n";
    std::cout << "    config                             Print current configuration\n";
    std::cout << "    help                               Show this help message\n";
    std::cout << "    quit / exit                        Exit the CLI\n";
    std::cout << "\n";
}

static void print_config(const KVCacheConfig& config) {
    std::cout << "\n  Configuration:\n";
    std::cout << "    Layers:          " << config.num_layers << "\n";
    std::cout << "    KV Heads:        " << config.num_kv_heads << "\n";
    std::cout << "    Head Dim:        " << config.head_dim << "\n";
    std::cout << "    Block Size:      " << config.block_size << " tokens\n";
    std::cout << "    Max Blocks:      " << config.max_blocks << "\n";
    std::cout << "    Data Type:       " << dtype_name(config.dtype) << "\n";
    std::cout << "    Eviction:        " << eviction_name(config.eviction_policy) << "\n";
    std::cout << "    KV/token/layer:  " << config.kv_pair_bytes_per_token() << " bytes\n";
    std::cout << "    Total Pool:      " << (config.total_pool_bytes() / (1024 * 1024)) << " MB\n";
    std::cout << "    Max Tokens:      " << config.max_tokens() << "\n";
    std::cout << "\n";
}

static void print_stats(const CacheStats& stats) {
    std::cout << "\n  Cache Statistics:\n";
    std::cout << "    Blocks:          " << stats.used_blocks << " / " << stats.total_blocks
              << " (" << std::fixed << std::setprecision(1) << stats.utilization_pct << "% used)\n";
    std::cout << "    Free Blocks:     " << stats.free_blocks << "\n";
    std::cout << "    Active Seqs:     " << stats.active_sequences << "\n";
    std::cout << "    Cached Tokens:   " << stats.total_cached_tokens << "\n";
    std::cout << "    Memory Used:     " << (stats.used_memory_bytes / 1024) << " KB / "
              << (stats.total_memory_bytes / 1024) << " KB\n";
    std::cout << "    Evictions:       " << stats.total_evictions << "\n";
    std::cout << "    Prefix Hits:     " << stats.prefix_cache_hits << "\n";
    std::cout << "\n";
}

static std::vector<float> generate_random_kv(int size, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> data(size);
    for (auto& v : data) v = dist(rng);
    return data;
}

// ---------------------------------------------------------------------------
// Benchmark
// ---------------------------------------------------------------------------

static void run_benchmark(KVCacheManager& manager, const KVCacheConfig& config,
                           int num_seqs, int seq_length) {
    std::cout << "\n  Running benchmark: " << num_seqs << " sequences x "
              << seq_length << " tokens...\n";

    std::mt19937 rng(42);
    int kv_elements = config.num_kv_heads * config.head_dim;

    // --- Append Benchmark ---
    auto start = std::chrono::high_resolution_clock::now();
    int total_appends = 0;

    for (int s = 0; s < num_seqs; ++s) {
        SequenceId sid{1000 + s};
        Status status = manager.create_sequence(sid);
        if (status != Status::OK) {
            std::cout << "  [ERROR] Failed to create seq " << sid.value << ": "
                      << status_name(status) << "\n";
            break;
        }

        for (int t = 0; t < seq_length; ++t) {
            for (int layer = 0; layer < config.num_layers; ++layer) {
                auto key_data = generate_random_kv(kv_elements, rng);
                auto val_data = generate_random_kv(kv_elements, rng);
                status = manager.append_kv(sid, layer, key_data.data(), val_data.data());
                if (status != Status::OK) {
                    std::cout << "  [ERROR] Append failed at seq=" << s << " token=" << t
                              << " layer=" << layer << ": " << status_name(status) << "\n";
                    goto bench_done;
                }
                total_appends++;
            }
        }
    }
bench_done:
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double appends_per_sec = (total_appends / elapsed_ms) * 1000.0;

    std::cout << "  Append Results:\n";
    std::cout << "    Total appends:   " << total_appends << " (layer-level)\n";
    std::cout << "    Time:            " << std::fixed << std::setprecision(2) << elapsed_ms << " ms\n";
    std::cout << "    Throughput:      " << std::fixed << std::setprecision(0) << appends_per_sec << " appends/sec\n";

    // --- Retrieval Benchmark ---
    start = std::chrono::high_resolution_clock::now();
    int total_gets = 0;

    for (int s = 0; s < num_seqs; ++s) {
        SequenceId sid{1000 + s};
        for (int t = 0; t < seq_length; ++t) {
            for (int layer = 0; layer < config.num_layers; ++layer) {
                KVPair kv = manager.get_kv(sid, layer, t);
                if (kv.valid()) total_gets++;
            }
        }
    }

    end = std::chrono::high_resolution_clock::now();
    elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double gets_per_sec = (total_gets / elapsed_ms) * 1000.0;

    std::cout << "  Retrieval Results:\n";
    std::cout << "    Total gets:      " << total_gets << "\n";
    std::cout << "    Time:            " << std::fixed << std::setprecision(2) << elapsed_ms << " ms\n";
    std::cout << "    Throughput:      " << std::fixed << std::setprecision(0) << gets_per_sec << " gets/sec\n";

    // Print stats
    print_stats(manager.get_stats());

    // Cleanup
    for (int s = 0; s < num_seqs; ++s) {
        manager.free_sequence(SequenceId{1000 + s});
    }

    std::cout << "  Benchmark complete. Sequences freed.\n\n";
}

// ---------------------------------------------------------------------------
// Main REPL
// ---------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    // Default config — small model for demo
    KVCacheConfig config;
    config.num_layers   = 4;
    config.num_kv_heads = 4;
    config.head_dim     = 64;
    config.block_size   = 16;
    config.max_blocks   = 256;
    config.dtype        = DataType::FP32;
    config.eviction_policy = EvictionType::NONE;

    // Parse optional command-line overrides
    for (int i = 1; i < argc - 1; i += 2) {
        std::string flag = argv[i];
        int val = std::atoi(argv[i + 1]);
        if (flag == "--layers")     config.num_layers = val;
        else if (flag == "--heads") config.num_kv_heads = val;
        else if (flag == "--dim")   config.head_dim = val;
        else if (flag == "--block") config.block_size = val;
        else if (flag == "--blocks") config.max_blocks = val;
    }

    KVCacheManager manager(config);
    std::mt19937 rng(12345);

    print_banner();
    print_config(config);

    std::string line;
    while (true) {
        std::cout << "kvcache> ";
        if (!std::getline(std::cin, line)) break;

        std::istringstream iss(line);
        std::string cmd;
        if (!(iss >> cmd)) continue;

        if (cmd == "quit" || cmd == "exit") {
            std::cout << "  Goodbye.\n";
            break;
        } else if (cmd == "help") {
            print_help();
        } else if (cmd == "config") {
            print_config(config);
        } else if (cmd == "stats") {
            print_stats(manager.get_stats());
        } else if (cmd == "create") {
            int64_t seq_id;
            if (!(iss >> seq_id)) {
                std::cout << "  Usage: create <seq_id>\n";
                continue;
            }
            Status s = manager.create_sequence(SequenceId{seq_id});
            std::cout << "  create seq " << seq_id << ": " << status_name(s) << "\n";
        } else if (cmd == "free") {
            int64_t seq_id;
            if (!(iss >> seq_id)) {
                std::cout << "  Usage: free <seq_id>\n";
                continue;
            }
            Status s = manager.free_sequence(SequenceId{seq_id});
            std::cout << "  free seq " << seq_id << ": " << status_name(s) << "\n";
        } else if (cmd == "append") {
            int64_t seq_id;
            int layer;
            if (!(iss >> seq_id >> layer)) {
                std::cout << "  Usage: append <seq_id> <layer>\n";
                continue;
            }
            int kv_elements = config.num_kv_heads * config.head_dim;
            auto key_data = generate_random_kv(kv_elements, rng);
            auto val_data = generate_random_kv(kv_elements, rng);
            Status s = manager.append_kv(SequenceId{seq_id}, layer,
                                          key_data.data(), val_data.data());
            std::cout << "  append seq " << seq_id << " layer " << layer
                      << ": " << status_name(s) << "\n";
        } else if (cmd == "append_n") {
            int64_t seq_id;
            int n;
            if (!(iss >> seq_id >> n)) {
                std::cout << "  Usage: append_n <seq_id> <n>\n";
                continue;
            }
            int kv_elements = config.num_kv_heads * config.head_dim;
            int success = 0;
            for (int t = 0; t < n; ++t) {
                bool ok = true;
                for (int layer = 0; layer < config.num_layers; ++layer) {
                    auto key_data = generate_random_kv(kv_elements, rng);
                    auto val_data = generate_random_kv(kv_elements, rng);
                    Status s = manager.append_kv(SequenceId{seq_id}, layer,
                                                  key_data.data(), val_data.data());
                    if (s != Status::OK) {
                        std::cout << "  [ERROR] at token " << t << " layer " << layer
                                  << ": " << status_name(s) << "\n";
                        ok = false;
                        break;
                    }
                }
                if (!ok) break;
                success++;
            }
            std::cout << "  Appended " << success << " / " << n << " tokens to seq "
                      << seq_id << "\n";
        } else if (cmd == "get") {
            int64_t seq_id;
            int layer, token_idx;
            if (!(iss >> seq_id >> layer >> token_idx)) {
                std::cout << "  Usage: get <seq_id> <layer> <token_idx>\n";
                continue;
            }
            KVPair kv = manager.get_kv(SequenceId{seq_id}, layer, token_idx);
            if (kv.valid()) {
                const float* k = static_cast<const float*>(kv.key);
                const float* v = static_cast<const float*>(kv.value);
                std::cout << "  K[0..3]: [" << k[0] << ", " << k[1] << ", "
                          << k[2] << ", " << k[3] << ", ...]\n";
                std::cout << "  V[0..3]: [" << v[0] << ", " << v[1] << ", "
                          << v[2] << ", " << v[3] << ", ...]\n";
            } else {
                std::cout << "  [NOT FOUND]\n";
            }
        } else if (cmd == "fork") {
            int64_t src, dst;
            if (!(iss >> src >> dst)) {
                std::cout << "  Usage: fork <src_id> <dst_id>\n";
                continue;
            }
            Status s = manager.fork_sequence(SequenceId{src}, SequenceId{dst});
            std::cout << "  fork " << src << " -> " << dst << ": " << status_name(s) << "\n";
        } else if (cmd == "tokens") {
            int64_t seq_id;
            if (!(iss >> seq_id)) {
                std::cout << "  Usage: tokens <seq_id>\n";
                continue;
            }
            int n = manager.get_num_tokens(SequenceId{seq_id});
            std::cout << "  seq " << seq_id << ": " << n << " tokens\n";
        } else if (cmd == "bench") {
            int num_seqs = 4, seq_length = 64;
            iss >> num_seqs >> seq_length;
            run_benchmark(manager, config, num_seqs, seq_length);
        } else {
            std::cout << "  Unknown command: '" << cmd << "'. Type 'help' for usage.\n";
        }
    }

    return 0;
}
