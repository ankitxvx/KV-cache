#include "kvcache/attention.h"
#include "kvcache/cuda/paged_attention.cuh"

#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check_cuda(cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(const std::vector<T>& host) {
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&data), host.size() * sizeof(T)));
        cudaError_t status = cudaMemcpy(data, host.data(), host.size() * sizeof(T),
                                       cudaMemcpyHostToDevice);
        if (status != cudaSuccess) {
            cudaFree(data);
            check_cuda(status);
        }
    }
    ~DeviceBuffer() { cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    T* data = nullptr;
};

void compare_with_cpu(int head_dim, bool later_warps_only, float scale_multiplier) {
    constexpr int batch = 2;
    constexpr int q_heads = 4;
    constexpr int kv_heads = 2;
    constexpr int block_size = 4;
    constexpr int max_blocks = 3;
    constexpr int physical_blocks = 6;
    // Different sequence lengths, partial final pages, and permuted physical pages.
    std::vector<int> tables = {4, 1, 5, 2, 0, 3};
    std::vector<int> lengths = {9, 5};
    size_t slot_size = static_cast<size_t>(kv_heads) * head_dim;
    size_t half = block_size * slot_size;
    std::vector<float> query(batch * q_heads * head_dim);
    std::vector<float> cache(physical_blocks * 2 * half);
    std::mt19937 rng(1234 + head_dim);
    std::uniform_real_distribution<float> values(-1.0f, 1.0f);
    for (auto& value : query) value = values(rng);
    for (auto& value : cache) value = values(rng);

    if (later_warps_only) {
        // The first warp's dot product is zero. Only later warps determine
        // relevance, but every output dimension must use their combined score.
        for (size_t i = 0; i < query.size(); ++i) {
            query[i] = (i % head_dim < 32) ? 0.0f : 1.0f;
        }
        for (int b = 0; b < batch; ++b) {
            for (int t = 0; t < lengths[b]; ++t) {
                int page = tables[b * max_blocks + t / block_size];
                for (int h = 0; h < kv_heads; ++h) {
                    for (int d = 0; d < head_dim; ++d) {
                        size_t offset = page * 2 * half + (t % block_size) * slot_size
                                      + h * head_dim + d;
                        cache[offset] = (t == 0) ? 1.0f : -1.0f;
                        cache[offset + half] = (t == 0) ? 1.0f : 0.0f;
                    }
                }
            }
        }
    }

    float scale = scale_multiplier / std::sqrt(static_cast<float>(head_dim));
    std::vector<float> expected(query.size(), 0.0f);
    std::vector<float> actual(query.size(), std::numeric_limits<float>::quiet_NaN());
    kvcache::PagedAttentionCPU::compute_batch(
        expected.data(), query.data(), cache.data(), tables.data(), lengths.data(),
        batch, max_blocks, q_heads, kv_heads, head_dim, block_size, scale);

    DeviceBuffer<float> d_query(query), d_cache(cache), d_output(actual);
    DeviceBuffer<int> d_tables(tables), d_lengths(lengths);
    kvcache::cuda::launch_paged_attention(
        d_output.data, d_query.data, d_cache.data, d_tables.data, d_lengths.data,
        batch, q_heads, kv_heads, head_dim, block_size, max_blocks, scale);
    check_cuda(cudaGetLastError());
    check_cuda(cudaDeviceSynchronize());
    check_cuda(cudaMemcpy(actual.data(), d_output.data, actual.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < actual.size(); ++i) {
        float tolerance = 2e-4f + 2e-4f * std::fabs(expected[i]);
        if (!std::isfinite(actual[i]) || std::fabs(actual[i] - expected[i]) > tolerance) {
            throw std::runtime_error("head_dim=" + std::to_string(head_dim)
                + " index=" + std::to_string(i)
                + " expected=" + std::to_string(expected[i])
                + " actual=" + std::to_string(actual[i]));
        }
    }
    std::cout << "PASS head_dim=" << head_dim
              << " later_warps_only=" << later_warps_only
              << " scale_multiplier=" << scale_multiplier << '\n';
}

} // namespace

int main() {
    int devices = 0;
    cudaError_t status = cudaGetDeviceCount(&devices);
    if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver
        || (status == cudaSuccess && devices == 0)) {
        std::cout << "SKIP: no usable CUDA device/driver\n";
        return 77;
    }
    try {
        check_cuda(status);
        compare_with_cpu(128, true, 1.0f);
        for (int dim : {1, 16, 31, 32, 33, 64, 96, 128, 160, 256, 320}) {
            compare_with_cpu(dim, false, 1.0f);
        }
        // Exercise online-softmax rescaling with large scores.
        compare_with_cpu(128, false, 100.0f);
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
