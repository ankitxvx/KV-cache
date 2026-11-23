#pragma once

/// @file paged_attention.cuh
/// @brief CUDA PagedAttention kernel for GPU-accelerated attention computation.
///
/// This kernel reads K and V from non-contiguous paged blocks using a block
/// table, supporting variable-length sequences within a batch.
///
/// Optimized for NVIDIA Ada Lovelace (compute capability 8.9, RTX 3050A).
/// Supports FP32 computation. FP16 Tensor Core support requires sm_80+.

#ifdef KVCACHE_CUDA_ENABLED

#include <cuda_runtime.h>

namespace kvcache {
namespace cuda {

/// Launch the paged attention kernel.
///
/// @param output       Device output [batch * num_q_heads * head_dim]
/// @param query        Device query  [batch * num_q_heads * head_dim]
/// @param kv_cache     Device KV pool [num_blocks * 2 * block_size * num_kv_heads * head_dim]
/// @param block_tables Device block tables [batch * max_blocks_per_seq]
/// @param seq_lengths  Device sequence lengths [batch]
/// @param batch_size   Number of sequences
/// @param num_q_heads  Query attention heads
/// @param num_kv_heads KV attention heads (for GQA)
/// @param head_dim     Dimension per head
/// @param block_size   Tokens per block
/// @param max_blocks_per_seq Maximum blocks per sequence
/// @param scale        Attention scale factor (1/sqrt(head_dim))
/// @param stream       CUDA stream for async execution
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
    cudaStream_t stream = 0
);

/// Query whether CUDA is available and functional.
bool cuda_is_available();

/// Get CUDA device properties string.
const char* cuda_device_info();

} // namespace cuda
} // namespace kvcache

#endif // KVCACHE_CUDA_ENABLED
