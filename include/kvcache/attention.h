#pragma once

#include "config.h"
#include <vector>
#include <cmath>
#include <cstring>
#include <algorithm>

namespace kvcache {

/// CPU reference implementation of scaled dot-product attention
/// with paged KV cache support.
///
/// This serves as:
/// 1. A functional fallback when CUDA is not available
/// 2. A correctness reference for validating the CUDA kernel
class PagedAttentionCPU {
public:
    /// Compute paged attention for a single query.
    ///
    /// @param output       Output buffer [num_q_heads * head_dim]
    /// @param query        Query vector  [num_q_heads * head_dim]
    /// @param kv_cache     Flat KV pool  [num_blocks * 2 * block_size * num_kv_heads * head_dim]
    /// @param block_table  Block IDs for this sequence [max_blocks_per_seq]
    /// @param seq_len      Actual sequence length (tokens cached)
    /// @param num_q_heads  Number of query heads
    /// @param num_kv_heads Number of KV heads (for GQA)
    /// @param head_dim     Dimension per head
    /// @param block_size   Tokens per block
    /// @param scale        Attention scale factor (typically 1/sqrt(head_dim))
    static void compute(
        float* output,
        const float* query,
        const float* kv_cache,
        const int* block_table,
        int seq_len,
        int num_q_heads,
        int num_kv_heads,
        int head_dim,
        int block_size,
        float scale
    );

    /// Batched paged attention — processes multiple sequences.
    ///
    /// @param output       [batch_size * num_q_heads * head_dim]
    /// @param queries      [batch_size * num_q_heads * head_dim]
    /// @param kv_cache     Shared KV block pool
    /// @param block_tables [batch_size * max_blocks_per_seq]
    /// @param seq_lengths  [batch_size]
    /// @param batch_size   Number of sequences in the batch
    static void compute_batch(
        float* output,
        const float* queries,
        const float* kv_cache,
        const int* block_tables,
        const int* seq_lengths,
        int batch_size,
        int max_blocks_per_seq,
        int num_q_heads,
        int num_kv_heads,
        int head_dim,
        int block_size,
        float scale
    );
};

} // namespace kvcache
