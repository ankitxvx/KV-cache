/// @file paged_attention.cu
/// @brief CUDA PagedAttention kernel implementation.
///
/// One thread block per (batch, query_head) pair.
/// Warps iterate over sequence tokens in blocks, computing Q·K^T scores,
/// performing online softmax, and accumulating weighted V.

#ifdef KVCACHE_CUDA_ENABLED

#include "kvcache/cuda/paged_attention.cuh"
#include <cstdio>
#include <cfloat>

namespace kvcache {
namespace cuda {

// Warp-level reduction for sum
__inline__ __device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return __shfl_sync(0xffffffff, val, 0);
}

/// Main paged attention kernel.
/// Grid:  (batch_size, num_q_heads, 1)
/// Block: up to 256 threads, rounded up to a whole number of warps.
__global__ void paged_attention_kernel(
    float* __restrict__ output,
    const float* __restrict__ query,
    const float* __restrict__ kv_cache,
    const int* __restrict__ block_tables,
    const int* __restrict__ seq_lengths,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int max_blocks_per_seq,
    float scale)
{
    int batch_idx = blockIdx.x;
    int q_head    = blockIdx.y;
    int tid       = threadIdx.x;

    int seq_len = seq_lengths[batch_idx];
    if (seq_len <= 0) return;

    int gqa_ratio = num_q_heads / num_kv_heads;
    int kv_head   = q_head / gqa_ratio;

    // Offsets into the flat arrays
    int q_offset = (batch_idx * num_q_heads + q_head) * head_dim;
    int bt_offset = batch_idx * max_blocks_per_seq;

    // KV cache layout per block:
    //   [block_id][K: block_size * num_kv_heads * head_dim][V: same]
    size_t slot_kv_size = (size_t)num_kv_heads * head_dim;
    size_t block_half   = (size_t)block_size * slot_kv_size;
    size_t block_stride = 2 * block_half;

    // Each dimension's accumulator has one owning thread. Striding also
    // handles heads larger than the thread block.
    extern __shared__ float accumulators[];
    __shared__ float warp_sums[8];
    __shared__ float block_score;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        accumulators[d] = 0.0f;
    }
    int lane = tid % 32;
    int warp = tid / 32;
    int num_warps = blockDim.x / 32;

    // Online softmax: track running max and sum
    float global_max = -FLT_MAX;
    float global_sum = 0.0f;

    // Iterate over all tokens
    for (int t = 0; t < seq_len; ++t) {
        int block_idx = t / block_size;
        int slot_idx  = t % block_size;
        int block_id  = block_tables[bt_offset + block_idx];

        size_t k_offset = (size_t)block_id * block_stride
                        + (size_t)slot_idx * slot_kv_size
                        + (size_t)kv_head * head_dim;
        float dot = 0.0f;
        for (int d = tid; d < head_dim; d += blockDim.x) {
            dot += query[q_offset + d] * kv_cache[k_offset + d];
        }

        // Reduce within each warp, then combine every warp's partial sum.
        // All launched warps are full; threads beyond head_dim contribute zero.
        dot = warp_reduce_sum(dot);
        if (lane == 0) warp_sums[warp] = dot;
        __syncthreads();

        if (warp == 0) {
            float total = warp_reduce_sum(lane < num_warps ? warp_sums[lane] : 0.0f);
            if (lane == 0) block_score = total * scale;
        }
        __syncthreads();
        float score = block_score;
        // Ensure all warps have consumed this score before the next token
        // can overwrite shared reduction storage.
        __syncthreads();

        // Online softmax update
        float new_max = fmaxf(global_max, score);
        float exp_score = expf(score - new_max);
        float correction = expf(global_max - new_max);

        // Rescale accumulator and sum
        for (int d = tid; d < head_dim; d += blockDim.x) {
            accumulators[d] = accumulators[d] * correction
                            + exp_score * kv_cache[k_offset + block_half + d];
        }
        global_sum = global_sum * correction + exp_score;
        global_max = new_max;
    }

    // Normalize
    if (global_sum > 0.0f) {
        for (int d = tid; d < head_dim; d += blockDim.x) {
            output[q_offset + d] = accumulators[d] / global_sum;
        }
    }
}

void launch_paged_attention(
    float* output,
    const float* query,
    const float* kv_cache,
    const int* block_tables,
    const int* seq_lengths,
    int batch_size,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int max_blocks_per_seq,
    float scale,
    cudaStream_t stream)
{
    dim3 grid(batch_size, num_q_heads, 1);
    int block_dim = (head_dim < 256) ? ((head_dim + 31) / 32) * 32 : 256;
    dim3 block(block_dim, 1, 1);

    size_t smem_size = head_dim * sizeof(float);

    paged_attention_kernel<<<grid, block, smem_size, stream>>>(
        output, query, kv_cache, block_tables, seq_lengths,
        num_q_heads, num_kv_heads, head_dim, block_size,
        max_blocks_per_seq, scale
    );
}

static char device_info_buf[256] = {0};

bool cuda_is_available() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    return (err == cudaSuccess && count > 0);
}

const char* cuda_device_info() {
    if (device_info_buf[0] != 0) return device_info_buf;

    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        snprintf(device_info_buf, sizeof(device_info_buf), "No CUDA devices found");
        return device_info_buf;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    snprintf(device_info_buf, sizeof(device_info_buf),
             "%s (SM %d.%d, %zu MB VRAM, %d SMs)",
             prop.name, prop.major, prop.minor,
             prop.totalGlobalMem / (1024*1024),
             prop.multiProcessorCount);
    return device_info_buf;
}

} // namespace cuda
} // namespace kvcache

#endif // KVCACHE_CUDA_ENABLED
