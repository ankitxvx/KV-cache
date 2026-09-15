#include "kvcache/quantization.h"
#include <limits>

namespace kvcache {

// ===========================================================================
// FP32 ↔ FP16 (IEEE 754 half-precision)
// ===========================================================================

uint16_t Quantizer::fp32_to_fp16(float value) {
    // IEEE 754 single precision: 1 sign + 8 exponent + 23 mantissa
    // IEEE 754 half precision:   1 sign + 5 exponent + 10 mantissa
    uint32_t f32;
    std::memcpy(&f32, &value, sizeof(f32));

    uint16_t sign = (f32 >> 16) & 0x8000;
    int32_t exponent = ((f32 >> 23) & 0xFF) - 127 + 15;  // Rebias from 127 to 15
    uint32_t mantissa = f32 & 0x7FFFFF;

    if (exponent <= 0) {
        // Underflow → zero (or subnormal, simplified to zero)
        return sign;
    } else if (exponent >= 31) {
        // Overflow → infinity
        if (((f32 >> 23) & 0xFF) == 0xFF && mantissa != 0) {
            // NaN
            return sign | 0x7C00 | (mantissa >> 13);
        }
        return sign | 0x7C00;  // Infinity
    }

    return sign | (static_cast<uint16_t>(exponent) << 10) | (mantissa >> 13);
}

float Quantizer::fp16_to_fp32(uint16_t value) {
    uint32_t sign = (value & 0x8000) << 16;
    uint32_t exponent = (value >> 10) & 0x1F;
    uint32_t mantissa = value & 0x3FF;

    uint32_t f32;
    if (exponent == 0) {
        if (mantissa == 0) {
            // Zero
            f32 = sign;
        } else {
            // Subnormal
            exponent = 1;
            while (!(mantissa & 0x400)) {
                mantissa <<= 1;
                exponent--;
            }
            mantissa &= 0x3FF;
            f32 = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
        }
    } else if (exponent == 31) {
        // Inf or NaN
        f32 = sign | 0x7F800000 | (mantissa << 13);
    } else {
        // Normal
        f32 = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
    }

    float result;
    std::memcpy(&result, &f32, sizeof(result));
    return result;
}

void Quantizer::fp32_to_fp16_bulk(const float* src, uint16_t* dst, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        dst[i] = fp32_to_fp16(src[i]);
    }
}

void Quantizer::fp16_to_fp32_bulk(const uint16_t* src, float* dst, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        dst[i] = fp16_to_fp32(src[i]);
    }
}

// ===========================================================================
// FP32 ↔ INT8 (per-tensor symmetric quantization)
// ===========================================================================

void Quantizer::quantize_int8(const float* src, int8_t* dst, size_t count, float& scale) {
    // Find max absolute value
    float max_abs = 0.0f;
    for (size_t i = 0; i < count; ++i) {
        float abs_val = std::fabs(src[i]);
        if (abs_val > max_abs) max_abs = abs_val;
    }

    // Compute scale
    if (max_abs == 0.0f) {
        scale = 1.0f;
        std::memset(dst, 0, count);
        return;
    }

    scale = max_abs / 127.0f;
    float inv_scale = 127.0f / max_abs;

    for (size_t i = 0; i < count; ++i) {
        float scaled = src[i] * inv_scale;
        // Clamp to [-127, 127] and round
        scaled = std::max(-127.0f, std::min(127.0f, scaled));
        dst[i] = static_cast<int8_t>(std::round(scaled));
    }
}

void Quantizer::dequantize_int8(const int8_t* src, float* dst, size_t count, float scale) {
    for (size_t i = 0; i < count; ++i) {
        dst[i] = static_cast<float>(src[i]) * scale;
    }
}

// ===========================================================================
// Accuracy metrics
// ===========================================================================

float Quantizer::mean_absolute_error(const float* a, const float* b, size_t count) {
    if (count == 0) return 0.0f;
    double sum = 0.0;
    for (size_t i = 0; i < count; ++i) {
        sum += std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
    }
    return static_cast<float>(sum / count);
}

float Quantizer::max_absolute_error(const float* a, const float* b, size_t count) {
    float max_err = 0.0f;
    for (size_t i = 0; i < count; ++i) {
        float err = std::fabs(a[i] - b[i]);
        if (err > max_err) max_err = err;
    }
    return max_err;
}

float Quantizer::sqnr_db(const float* original, const float* reconstructed, size_t count) {
    if (count == 0) return 0.0f;

    double signal_power = 0.0;
    double noise_power = 0.0;

    for (size_t i = 0; i < count; ++i) {
        double sig = static_cast<double>(original[i]);
        double diff = sig - static_cast<double>(reconstructed[i]);
        signal_power += sig * sig;
        noise_power += diff * diff;
    }

    if (noise_power == 0.0) return std::numeric_limits<float>::infinity();
    if (signal_power == 0.0) return 0.0f;

    return static_cast<float>(10.0 * std::log10(signal_power / noise_power));
}

// ===========================================================================
// Generic conversion
// ===========================================================================

void Quantizer::convert(const void* src, void* dst,
                        DataType src_dtype, DataType dst_dtype,
                        size_t num_elements, float& scale) {
    if (src_dtype == dst_dtype) {
        std::memcpy(dst, src, num_elements * dtype_size(src_dtype));
        return;
    }

    if (src_dtype == DataType::FP32 && dst_dtype == DataType::FP16) {
        fp32_to_fp16_bulk(static_cast<const float*>(src),
                          static_cast<uint16_t*>(dst), num_elements);
    } else if (src_dtype == DataType::FP16 && dst_dtype == DataType::FP32) {
        fp16_to_fp32_bulk(static_cast<const uint16_t*>(src),
                          static_cast<float*>(dst), num_elements);
    } else if (src_dtype == DataType::FP32 && dst_dtype == DataType::INT8) {
        quantize_int8(static_cast<const float*>(src),
                      static_cast<int8_t*>(dst), num_elements, scale);
    } else if (src_dtype == DataType::INT8 && dst_dtype == DataType::FP32) {
        dequantize_int8(static_cast<const int8_t*>(src),
                        static_cast<float*>(dst), num_elements, scale);
    } else if (src_dtype == DataType::FP16 && dst_dtype == DataType::INT8) {
        // FP16 → FP32 → INT8
        std::vector<float> tmp(num_elements);
        fp16_to_fp32_bulk(static_cast<const uint16_t*>(src), tmp.data(), num_elements);
        quantize_int8(tmp.data(), static_cast<int8_t*>(dst), num_elements, scale);
    } else if (src_dtype == DataType::INT8 && dst_dtype == DataType::FP16) {
        // INT8 → FP32 → FP16
        std::vector<float> tmp(num_elements);
        dequantize_int8(static_cast<const int8_t*>(src), tmp.data(), num_elements, scale);
        fp32_to_fp16_bulk(tmp.data(), static_cast<uint16_t*>(dst), num_elements);
    }
}

} // namespace kvcache
