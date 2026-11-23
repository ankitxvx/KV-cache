#include "kvcache/attention.h"
#include <limits>

namespace kvcache {

void PagedAttentionCPU::compute(
    float* output,
    const float* query,
    const float* kv_cache,
    const int* block_table,
    int seq_len,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int block_size,
    float scale)
{
    if (!output || !query || !kv_cache || !block_table || seq_len <= 0) return;

    // GQA ratio: how many Q heads share one KV head
    int gqa_ratio = num_q_heads / num_kv_heads;

    // Per-block layout in kv_cache:
    //   [block_id, 0, slot, kv_head, dim] = key
    //   [block_id, 1, slot, kv_head, dim] = value
    size_t slot_kv_size = static_cast<size_t>(num_kv_heads) * head_dim;
    size_t block_half   = static_cast<size_t>(block_size) * slot_kv_size;
    size_t block_stride = 2 * block_half;  // K and V halves

    for (int qh = 0; qh < num_q_heads; ++qh) {
        int kv_head = qh / gqa_ratio;

        const float* q = query + static_cast<size_t>(qh) * head_dim;
        float* out = output + static_cast<size_t>(qh) * head_dim;

        // Phase 1: Compute attention scores (Q·K^T)
        std::vector<float> scores(seq_len);
        float max_score = -std::numeric_limits<float>::infinity();

        for (int t = 0; t < seq_len; ++t) {
            int block_idx = t / block_size;
            int slot_idx  = t % block_size;
            int block_id  = block_table[block_idx];

            // Pointer to K[block_id][slot_idx][kv_head]
            const float* k_ptr = kv_cache
                + static_cast<size_t>(block_id) * block_stride
                + static_cast<size_t>(slot_idx) * slot_kv_size
                + static_cast<size_t>(kv_head) * head_dim;

            // Dot product: Q · K
            float dot = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                dot += q[d] * k_ptr[d];
            }
            scores[t] = dot * scale;
            max_score = std::max(max_score, scores[t]);
        }

        // Phase 2: Softmax
        float sum_exp = 0.0f;
        for (int t = 0; t < seq_len; ++t) {
            scores[t] = std::exp(scores[t] - max_score);
            sum_exp += scores[t];
        }
        if (sum_exp > 0.0f) {
            for (int t = 0; t < seq_len; ++t) {
                scores[t] /= sum_exp;
            }
        }

        // Phase 3: Weighted sum of values
        std::memset(out, 0, static_cast<size_t>(head_dim) * sizeof(float));

        for (int t = 0; t < seq_len; ++t) {
            if (scores[t] < 1e-9f) continue;  // Skip near-zero weights

            int block_idx = t / block_size;
            int slot_idx  = t % block_size;
            int block_id  = block_table[block_idx];

            // Pointer to V[block_id][slot_idx][kv_head]
            const float* v_ptr = kv_cache
                + static_cast<size_t>(block_id) * block_stride
                + block_half  // V half starts after K half
                + static_cast<size_t>(slot_idx) * slot_kv_size
                + static_cast<size_t>(kv_head) * head_dim;

            for (int d = 0; d < head_dim; ++d) {
                out[d] += scores[t] * v_ptr[d];
            }
        }
    }
}

void PagedAttentionCPU::compute_batch(
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
    float scale)
{
    if (!output || !queries || !kv_cache || !block_tables || !seq_lengths) return;

    size_t head_stride = static_cast<size_t>(num_q_heads) * head_dim;

    for (int b = 0; b < batch_size; ++b) {
        compute(
            output + b * head_stride,
            queries + b * head_stride,
            kv_cache,
            block_tables + b * max_blocks_per_seq,
            seq_lengths[b],
            num_q_heads,
            num_kv_heads,
            head_dim,
            block_size,
            scale
        );
    }
}

} // namespace kvcache
