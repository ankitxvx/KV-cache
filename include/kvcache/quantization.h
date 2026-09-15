#pragma once

#include "types.h"
#include <vector>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <algorithm>

namespace kvcache {

/// Quantization utilities for reducing KV cache memory footprint.
/// Supports FP32 ↔ FP16 and FP32 ↔ INT8 conversions.
class Quantizer {
public:
    // -----------------------------------------------------------------------
    // FP32 ↔ FP16 (IEEE 754 half-precision)
    // -----------------------------------------------------------------------

    /// Convert a single FP32 value to FP16 (stored as uint16_t).
    static uint16_t fp32_to_fp16(float value);

    /// Convert a single FP16 value (uint16_t) back to FP32.
    static float fp16_to_fp32(uint16_t value);

    /// Bulk convert FP32 array to FP16 array.
    static void fp32_to_fp16_bulk(const float* src, uint16_t* dst, size_t count);

    /// Bulk convert FP16 array to FP32 array.
    static void fp16_to_fp32_bulk(const uint16_t* src, float* dst, size_t count);

    // -----------------------------------------------------------------------
    // FP32 ↔ INT8 (per-tensor symmetric quantization)
    // -----------------------------------------------------------------------

    /// Quantize FP32 array to INT8 with symmetric per-tensor quantization.
    /// Computes scale = max(|src|) / 127.
    /// @param src     Input FP32 array.
    /// @param dst     Output INT8 array.
    /// @param count   Number of elements.
    /// @param scale   Output: computed scale factor.
    static void quantize_int8(const float* src, int8_t* dst, size_t count, float& scale);

    /// Dequantize INT8 array back to FP32.
    /// result[i] = src[i] * scale
    static void dequantize_int8(const int8_t* src, float* dst, size_t count, float scale);

    // -----------------------------------------------------------------------
    // Accuracy metrics
    // -----------------------------------------------------------------------

    /// Compute mean absolute error between two FP32 arrays.
    static float mean_absolute_error(const float* a, const float* b, size_t count);

    /// Compute max absolute error between two FP32 arrays.
    static float max_absolute_error(const float* a, const float* b, size_t count);

    /// Compute signal-to-quantization-noise ratio (SQNR) in dB.
    static float sqnr_db(const float* original, const float* reconstructed, size_t count);

    // -----------------------------------------------------------------------
    // In-place block conversion
    // -----------------------------------------------------------------------

    /// Convert a raw data buffer from one dtype to another.
    /// The caller must ensure dst has enough space.
    /// @param src          Source data pointer.
    /// @param dst          Destination data pointer.
    /// @param src_dtype    Source data type.
    /// @param dst_dtype    Destination data type.
    /// @param num_elements Number of logical float elements.
    /// @param scale        For INT8 conversions, the scale factor (output if quantizing, input if dequantizing).
    static void convert(const void* src, void* dst,
                        DataType src_dtype, DataType dst_dtype,
                        size_t num_elements, float& scale);
};

} // namespace kvcache
