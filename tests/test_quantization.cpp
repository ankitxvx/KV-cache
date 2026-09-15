#include "test_common.h"
#include "kvcache/quantization.h"
#include <vector>
#include <random>

using namespace kvcache;

void test_fp16_roundtrip() {
    // Test common values
    float values[] = {0.0f, 1.0f, -1.0f, 0.5f, -0.5f, 3.14159f, 100.0f, -100.0f};
    for (float v : values) {
        uint16_t fp16 = Quantizer::fp32_to_fp16(v);
        float back = Quantizer::fp16_to_fp32(fp16);
        // FP16 has ~3 decimal digits of precision
        TEST_ASSERT_NEAR(back, v, std::fabs(v) * 0.01f + 0.001f);
    }
}

void test_fp16_bulk() {
    std::vector<float> src = {1.0f, 2.0f, 3.0f, 4.0f, -5.0f, 0.0f, 0.125f, -0.125f};
    std::vector<uint16_t> fp16(src.size());
    std::vector<float> dst(src.size());

    Quantizer::fp32_to_fp16_bulk(src.data(), fp16.data(), src.size());
    Quantizer::fp16_to_fp32_bulk(fp16.data(), dst.data(), src.size());

    for (size_t i = 0; i < src.size(); ++i) {
        TEST_ASSERT_NEAR(dst[i], src[i], std::fabs(src[i]) * 0.01f + 0.001f);
    }
}

void test_fp16_special() {
    // Zero
    uint16_t z = Quantizer::fp32_to_fp16(0.0f);
    TEST_ASSERT_NEAR(Quantizer::fp16_to_fp32(z), 0.0f, 1e-10f);

    // Negative zero
    z = Quantizer::fp32_to_fp16(-0.0f);
    TEST_ASSERT_NEAR(Quantizer::fp16_to_fp32(z), 0.0f, 1e-10f);
}

void test_int8_roundtrip() {
    std::vector<float> src = {0.0f, 1.0f, -1.0f, 0.5f, -0.5f, 0.25f, 2.0f, -2.0f};
    std::vector<int8_t> quantized(src.size());
    std::vector<float> dst(src.size());
    float scale;

    Quantizer::quantize_int8(src.data(), quantized.data(), src.size(), scale);
    Quantizer::dequantize_int8(quantized.data(), dst.data(), src.size(), scale);

    // INT8 has ~1% error for well-distributed values
    for (size_t i = 0; i < src.size(); ++i) {
        TEST_ASSERT_NEAR(dst[i], src[i], std::fabs(src[i]) * 0.05f + 0.02f);
    }
}

void test_int8_zeros() {
    std::vector<float> src(10, 0.0f);
    std::vector<int8_t> quantized(10);
    std::vector<float> dst(10);
    float scale;

    Quantizer::quantize_int8(src.data(), quantized.data(), src.size(), scale);
    Quantizer::dequantize_int8(quantized.data(), dst.data(), src.size(), scale);

    for (size_t i = 0; i < src.size(); ++i) {
        TEST_ASSERT_NEAR(dst[i], 0.0f, 1e-6f);
    }
}

void test_int8_scale_computation() {
    std::vector<float> src = {-3.0f, 0.0f, 3.0f};
    std::vector<int8_t> quantized(3);
    float scale;

    Quantizer::quantize_int8(src.data(), quantized.data(), src.size(), scale);

    // scale should be max_abs / 127 = 3.0 / 127
    TEST_ASSERT_NEAR(scale, 3.0f / 127.0f, 1e-6f);

    // 3.0 should map to 127
    TEST_ASSERT_EQ(quantized[2], 127);
    // -3.0 should map to -127
    TEST_ASSERT_EQ(quantized[0], -127);
}

void test_accuracy_metrics() {
    std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> b = {1.1f, 1.9f, 3.2f, 3.8f};

    float mae = Quantizer::mean_absolute_error(a.data(), b.data(), a.size());
    TEST_ASSERT_NEAR(mae, 0.15f, 0.01f);  // mean of {0.1, 0.1, 0.2, 0.2} = 0.15

    float max_err = Quantizer::max_absolute_error(a.data(), b.data(), a.size());
    TEST_ASSERT_NEAR(max_err, 0.2f, 0.01f);
}

void test_sqnr() {
    // High SQNR = low quantization noise
    std::vector<float> original = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    std::vector<float> reconstructed = {1.001f, 2.001f, 3.001f, 4.001f, 5.001f};

    float sqnr = Quantizer::sqnr_db(original.data(), reconstructed.data(), original.size());
    TEST_ASSERT(sqnr > 60.0f);  // Very high quality reconstruction
}

void test_convert_fp32_to_fp16() {
    std::vector<float> src = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<uint16_t> dst(4);
    float scale = 0.0f;

    Quantizer::convert(src.data(), dst.data(), DataType::FP32, DataType::FP16, 4, scale);

    // Convert back to verify
    std::vector<float> back(4);
    Quantizer::convert(dst.data(), back.data(), DataType::FP16, DataType::FP32, 4, scale);

    for (int i = 0; i < 4; ++i) {
        TEST_ASSERT_NEAR(back[i], src[i], 0.01f);
    }
}

void test_large_random_int8() {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    const int N = 1024;
    std::vector<float> src(N);
    for (auto& v : src) v = dist(rng);

    std::vector<int8_t> quantized(N);
    std::vector<float> reconstructed(N);
    float scale;

    Quantizer::quantize_int8(src.data(), quantized.data(), N, scale);
    Quantizer::dequantize_int8(quantized.data(), reconstructed.data(), N, scale);

    float mae = Quantizer::mean_absolute_error(src.data(), reconstructed.data(), N);
    float sqnr = Quantizer::sqnr_db(src.data(), reconstructed.data(), N);

    // INT8 should give reasonable quality
    TEST_ASSERT(mae < 0.1f);
    TEST_ASSERT(sqnr > 35.0f);
}

TEST_MAIN_BEGIN("Quantization Tests")
    RUN_TEST(test_fp16_roundtrip);
    prev_failed = g_tests_failed;
    RUN_TEST(test_fp16_bulk);
    prev_failed = g_tests_failed;
    RUN_TEST(test_fp16_special);
    prev_failed = g_tests_failed;
    RUN_TEST(test_int8_roundtrip);
    prev_failed = g_tests_failed;
    RUN_TEST(test_int8_zeros);
    prev_failed = g_tests_failed;
    RUN_TEST(test_int8_scale_computation);
    prev_failed = g_tests_failed;
    RUN_TEST(test_accuracy_metrics);
    prev_failed = g_tests_failed;
    RUN_TEST(test_sqnr);
    prev_failed = g_tests_failed;
    RUN_TEST(test_convert_fp32_to_fp16);
    prev_failed = g_tests_failed;
    RUN_TEST(test_large_random_int8);
    prev_failed = g_tests_failed;
TEST_MAIN_END()
