/// @file server/main.cpp
/// @brief Lightweight HTTP API server for the PageForge KV Cache system.
/// Zero external dependencies — uses raw Winsock2/POSIX sockets.

#include "kvcache/kvcache.h"
#include "kvcache/config.h"
#include "kvcache/tier_manager.h"
#include "kvcache/transfer_engine.h"
#include "kvcache/scheduler.h"
#include "kvcache/attention.h"
#include "kvcache/metrics.h"
#include "kvcache/logger.h"

#ifdef _WIN32
#pragma comment(lib, "ws2_32.lib")
#include <winsock2.h>
#include <ws2tcpip.h>
typedef SOCKET socket_t;
#define CLOSE_SOCKET closesocket
#define SOCKET_ERROR_VAL INVALID_SOCKET
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
typedef int socket_t;
#define CLOSE_SOCKET close
#define SOCKET_ERROR_VAL -1
#endif

#include <iostream>
#include <sstream>
#include <string>
#include <cstring>
#include <thread>
#include <atomic>
#include <memory>
#include <vector>
#include <csignal>

namespace {

std::atomic<bool> g_running{true};

void signal_handler(int) {
    g_running = false;
}

// ── Simple HTTP helpers ─────────────────────────────────────────────────

struct HttpRequest {
    std::string method;
    std::string path;
    std::string body;
};

HttpRequest parse_http_request(const std::string& raw) {
    HttpRequest req;
    std::istringstream iss(raw);
    iss >> req.method >> req.path;

    // Find body (after \r\n\r\n)
    auto pos = raw.find("\r\n\r\n");
    if (pos != std::string::npos) {
        req.body = raw.substr(pos + 4);
    }
    return req;
}

std::string http_response(int status, const std::string& content_type,
                           const std::string& body) {
    std::ostringstream oss;
    oss << "HTTP/1.1 " << status;
    switch (status) {
        case 200: oss << " OK"; break;
        case 201: oss << " Created"; break;
        case 400: oss << " Bad Request"; break;
        case 404: oss << " Not Found"; break;
        case 500: oss << " Internal Server Error"; break;
        default:  oss << " Unknown"; break;
    }
    oss << "\r\n";
    oss << "Content-Type: " << content_type << "\r\n";
    oss << "Content-Length: " << body.size() << "\r\n";
    oss << "Access-Control-Allow-Origin: *\r\n";
    oss << "Connection: close\r\n";
    oss << "\r\n";
    oss << body;
    return oss.str();
}

std::string json_ok(const std::string& data = "{}") {
    return http_response(200, "application/json", data);
}

std::string json_error(int status, const std::string& msg) {
    return http_response(status, "application/json",
        "{\"error\":\"" + msg + "\"}");
}

// ── Route handlers ──────────────────────────────────────────────────────

std::string handle_health() {
    return json_ok("{\"status\":\"healthy\",\"service\":\"pageforge\"}");
}

std::string handle_metrics() {
    auto& metrics = kvcache::MetricsRegistry::instance();
    std::string body = metrics.export_prometheus();
    return http_response(200, "text/plain; charset=utf-8", body);
}

std::string handle_stats(kvcache::TierManager& tier_mgr, kvcache::Scheduler& scheduler) {
    auto gpu_s = tier_mgr.get_tier_stats(kvcache::TierType::GPU_VRAM);
    auto cpu_s = tier_mgr.get_tier_stats(kvcache::TierType::CPU_DRAM);
    auto ssd_s = tier_mgr.get_tier_stats(kvcache::TierType::NVME_SSD);
    auto total = tier_mgr.get_total_stats();

    std::ostringstream oss;
    oss << "{\n";
    oss << "  \"tiers\": {\n";
    oss << "    \"gpu\": {\"total\":" << gpu_s.total_blocks
        << ",\"used\":" << gpu_s.used_blocks
        << ",\"free\":" << gpu_s.free_blocks
        << ",\"util_pct\":" << std::fixed << std::setprecision(1) << gpu_s.utilization_pct
        << ",\"bytes\":" << gpu_s.total_bytes << "},\n";
    oss << "    \"cpu\": {\"total\":" << cpu_s.total_blocks
        << ",\"used\":" << cpu_s.used_blocks
        << ",\"free\":" << cpu_s.free_blocks
        << ",\"util_pct\":" << std::fixed << std::setprecision(1) << cpu_s.utilization_pct
        << ",\"bytes\":" << cpu_s.total_bytes << "},\n";
    oss << "    \"ssd\": {\"total\":" << ssd_s.total_blocks
        << ",\"used\":" << ssd_s.used_blocks
        << ",\"free\":" << ssd_s.free_blocks
        << ",\"util_pct\":" << std::fixed << std::setprecision(1) << ssd_s.utilization_pct
        << ",\"bytes\":" << ssd_s.total_bytes << "}\n";
    oss << "  },\n";
    oss << "  \"total\": {\"blocks\":" << total.total_blocks
        << ",\"used\":" << total.used_blocks
        << ",\"memory_gb\":" << std::fixed << std::setprecision(2)
        << (total.total_memory_bytes / (1024.0*1024.0*1024.0)) << "},\n";
    oss << "  \"scheduler\": {\"waiting\":" << scheduler.waiting_count()
        << ",\"running\":" << scheduler.running_count()
        << ",\"completed\":" << scheduler.completed_count()
        << ",\"total\":" << scheduler.total_count() << "}\n";
    oss << "}";

    return json_ok(oss.str());
}

std::string handle_config(const kvcache::ProductionConfig& config) {
    std::ostringstream oss;
    oss << "{\n";
    oss << "  \"model\": {\"layers\":" << config.num_layers
        << ",\"kv_heads\":" << config.num_kv_heads
        << ",\"head_dim\":" << config.head_dim
        << ",\"block_size\":" << config.block_size
        << ",\"dtype\":\"" << kvcache::dtype_name(config.dtype) << "\"},\n";
    oss << "  \"tiers\": {\"gpu_mb\":" << (config.gpu_memory_bytes / (1024*1024))
        << ",\"cpu_mb\":" << (config.cpu_memory_bytes / (1024*1024))
        << ",\"ssd_mb\":" << (config.ssd_capacity_bytes / (1024*1024)) << "},\n";
    oss << "  \"scheduler\": {\"max_batch\":" << config.max_batch_size
        << ",\"max_prefill_tokens\":" << config.max_prefill_tokens << "},\n";
    oss << "  \"block_bytes\":" << config.block_total_bytes()
        << ",\"total_blocks\":" << config.total_max_blocks() << "\n";
    oss << "}";
    return json_ok(oss.str());
}

// ── Request router ──────────────────────────────────────────────────────

std::string route_request(const HttpRequest& req,
                          kvcache::TierManager& tier_mgr,
                          kvcache::Scheduler& scheduler,
                          const kvcache::ProductionConfig& config) {
    if (req.path == "/health" || req.path == "/v1/health") {
        return handle_health();
    }
    if (req.path == "/metrics") {
        return handle_metrics();
    }
    if (req.path == "/v1/stats") {
        return handle_stats(tier_mgr, scheduler);
    }
    if (req.path == "/v1/config") {
        return handle_config(config);
    }

    return json_error(404, "Not found: " + req.path);
}

// ── Client handler ──────────────────────────────────────────────────────

void handle_client(socket_t client,
                   kvcache::TierManager& tier_mgr,
                   kvcache::Scheduler& scheduler,
                   const kvcache::ProductionConfig& config) {
    char buffer[4096];
    int bytes = recv(client, buffer, sizeof(buffer) - 1, 0);
    if (bytes <= 0) {
        CLOSE_SOCKET(client);
        return;
    }
    buffer[bytes] = '\0';

    HttpRequest req = parse_http_request(std::string(buffer, bytes));
    std::string response = route_request(req, tier_mgr, scheduler, config);

    send(client, response.c_str(), static_cast<int>(response.size()), 0);
    CLOSE_SOCKET(client);
}

} // anonymous namespace

// ── Main ────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    using namespace kvcache;

    // Default production config optimized for RTX 3050A + 24GB DDR5 + 512GB NVMe
    ProductionConfig config;

    // Use smaller sizes for the server demo to avoid huge allocations
    config.gpu_memory_bytes  = 512ULL * 1024 * 1024;   // 512MB GPU
    config.cpu_memory_bytes  = 2ULL * 1024 * 1024 * 1024; // 2GB CPU
    config.ssd_capacity_bytes = 0;  // Disable SSD for server demo
    config.enable_ssd = false;
    config.num_layers = 4;
    config.num_kv_heads = 4;
    config.head_dim = 64;

    // Parse port from args
    if (argc > 1) {
        config.server_port = std::atoi(argv[1]);
    }

    config.apply_logging();

    KVCACHE_LOG_INFO("server", "=== PageForge KV Cache Server ===");
    KVCACHE_LOG_INFO("server", "Initializing tier manager...");

    TierManager tier_mgr(config);
    Scheduler scheduler(config);
    TransferEngine transfer(tier_mgr, config);

    // Start HTTP server
#ifdef _WIN32
    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) {
        KVCACHE_LOG_FATAL("server", "WSAStartup failed");
        return 1;
    }
#endif

    socket_t server_fd = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (server_fd == SOCKET_ERROR_VAL) {
        KVCACHE_LOG_FATAL("server", "Failed to create socket");
        return 1;
    }

    // Allow address reuse
    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR,
               reinterpret_cast<const char*>(&opt), sizeof(opt));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(static_cast<uint16_t>(config.server_port));

    if (bind(server_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        KVCACHE_LOG_FATAL("server", "Failed to bind to port " +
                          std::to_string(config.server_port));
        CLOSE_SOCKET(server_fd);
        return 1;
    }

    if (listen(server_fd, 64) < 0) {
        KVCACHE_LOG_FATAL("server", "Failed to listen");
        CLOSE_SOCKET(server_fd);
        return 1;
    }

    std::signal(SIGINT, signal_handler);

    std::ostringstream msg;
    msg << "Server listening on http://" << config.server_host
        << ":" << config.server_port;
    KVCACHE_LOG_INFO("server", msg.str());
    KVCACHE_LOG_INFO("server", "Endpoints: /health /metrics /v1/stats /v1/config");

    std::cout << "\n"
              << "  ╔═══════════════════════════════════════╗\n"
              << "  ║     PageForge KV Cache Server         ║\n"
              << "  ║  http://localhost:" << config.server_port
              << std::string(24 - std::to_string(config.server_port).size(), ' ') << "║\n"
              << "  ╚═══════════════════════════════════════╝\n"
              << "\n"
              << "  GET /health      - Health check\n"
              << "  GET /metrics     - Prometheus metrics\n"
              << "  GET /v1/stats    - Tier & scheduler stats\n"
              << "  GET /v1/config   - Current configuration\n"
              << "\n"
              << "  Press Ctrl+C to stop.\n\n";

    while (g_running) {
        // Use select() for timeout-based accept so we can check g_running
        fd_set readfds;
        FD_ZERO(&readfds);
        FD_SET(server_fd, &readfds);

        timeval tv;
        tv.tv_sec = 1;
        tv.tv_usec = 0;

        int ready = select(static_cast<int>(server_fd) + 1, &readfds, nullptr, nullptr, &tv);
        if (ready <= 0) continue;

        sockaddr_in client_addr{};
        int client_len = sizeof(client_addr);
        socket_t client = accept(server_fd, reinterpret_cast<sockaddr*>(&client_addr),
#ifdef _WIN32
                                  &client_len
#else
                                  reinterpret_cast<socklen_t*>(&client_len)
#endif
        );
        if (client == SOCKET_ERROR_VAL) continue;

        // Handle in a detached thread
        std::thread([client, &tier_mgr, &scheduler, &config]() {
            handle_client(client, tier_mgr, scheduler, config);
        }).detach();
    }

    KVCACHE_LOG_INFO("server", "Shutting down...");
    CLOSE_SOCKET(server_fd);
    transfer.shutdown();

#ifdef _WIN32
    WSACleanup();
#endif

    KVCACHE_LOG_INFO("server", "Server stopped.");
    return 0;
}
