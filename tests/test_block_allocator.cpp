#include "test_common.h"
#include "kvcache/block_allocator.h"

using namespace kvcache;

static KVCacheConfig small_config() {
    KVCacheConfig cfg;
    cfg.num_layers = 2;
    cfg.num_kv_heads = 2;
    cfg.head_dim = 4;
    cfg.block_size = 4;
    cfg.max_blocks = 8;
    cfg.dtype = DataType::FP32;
    return cfg;
}

void test_allocate_and_free() {
    auto cfg = small_config();
    BlockAllocator alloc(cfg);

    TEST_ASSERT_EQ(alloc.num_total(), 8);
    TEST_ASSERT_EQ(alloc.num_free(), 8);
    TEST_ASSERT_EQ(alloc.num_used(), 0);

    BlockId b1 = alloc.allocate();
    TEST_ASSERT(b1.valid());
    TEST_ASSERT_EQ(alloc.num_free(), 7);
    TEST_ASSERT_EQ(alloc.num_used(), 1);

    BlockId b2 = alloc.allocate();
    TEST_ASSERT(b2.valid());
    TEST_ASSERT(b1 != b2);

    alloc.free(b1);
    TEST_ASSERT_EQ(alloc.num_free(), 7);

    alloc.free(b2);
    TEST_ASSERT_EQ(alloc.num_free(), 8);
}

void test_pool_exhaustion() {
    auto cfg = small_config();
    cfg.max_blocks = 3;
    BlockAllocator alloc(cfg);

    BlockId b1 = alloc.allocate();
    BlockId b2 = alloc.allocate();
    BlockId b3 = alloc.allocate();
    TEST_ASSERT(b1.valid());
    TEST_ASSERT(b2.valid());
    TEST_ASSERT(b3.valid());

    // 4th allocation should fail
    BlockId b4 = alloc.allocate();
    TEST_ASSERT(!b4.valid());

    // Free one and retry
    alloc.free(b2);
    b4 = alloc.allocate();
    TEST_ASSERT(b4.valid());
}

void test_ref_counting() {
    auto cfg = small_config();
    BlockAllocator alloc(cfg);

    BlockId b1 = alloc.allocate();
    TEST_ASSERT_EQ(alloc.get_ref_count(b1), 1);

    alloc.ref(b1);
    TEST_ASSERT_EQ(alloc.get_ref_count(b1), 2);

    alloc.unref(b1);
    TEST_ASSERT_EQ(alloc.get_ref_count(b1), 1);
    TEST_ASSERT_EQ(alloc.num_used(), 1);  // Still alive

    alloc.unref(b1);
    TEST_ASSERT_EQ(alloc.num_used(), 0);  // Now freed
}

void test_kv_slot_read_write() {
    auto cfg = small_config();
    BlockAllocator alloc(cfg);

    BlockId b = alloc.allocate();
    TEST_ASSERT(b.valid());

    // Write data to slot 0, layer 0
    int kv_elements = cfg.num_kv_heads * cfg.head_dim;  // 2*4 = 8 floats
    std::vector<float> key_data(kv_elements, 1.5f);
    std::vector<float> val_data(kv_elements, -2.5f);

    KVPairMut kv = alloc.get_kv_slot_mut(b, 0, 0);
    TEST_ASSERT(kv.valid());

    std::memcpy(kv.key, key_data.data(), kv_elements * sizeof(float));
    std::memcpy(kv.value, val_data.data(), kv_elements * sizeof(float));

    // Read it back
    KVPair kv_read = alloc.get_kv_slot(b, 0, 0);
    TEST_ASSERT(kv_read.valid());

    const float* k = static_cast<const float*>(kv_read.key);
    const float* v = static_cast<const float*>(kv_read.value);

    for (int i = 0; i < kv_elements; ++i) {
        TEST_ASSERT_NEAR(k[i], 1.5f, 1e-6f);
        TEST_ASSERT_NEAR(v[i], -2.5f, 1e-6f);
    }
}

void test_copy_on_write() {
    auto cfg = small_config();
    BlockAllocator alloc(cfg);

    BlockId b1 = alloc.allocate();
    int kv_elements = cfg.num_kv_heads * cfg.head_dim;
    std::vector<float> key_data(kv_elements, 3.14f);
    KVPairMut kv = alloc.get_kv_slot_mut(b1, 0, 0);
    std::memcpy(kv.key, key_data.data(), kv_elements * sizeof(float));

    // CoW copy
    BlockId b2 = alloc.copy_on_write(b1);
    TEST_ASSERT(b2.valid());
    TEST_ASSERT(b1 != b2);

    // Verify copy has same data
    KVPair kv1 = alloc.get_kv_slot(b1, 0, 0);
    KVPair kv2 = alloc.get_kv_slot(b2, 0, 0);
    const float* k1 = static_cast<const float*>(kv1.key);
    const float* k2 = static_cast<const float*>(kv2.key);

    for (int i = 0; i < kv_elements; ++i) {
        TEST_ASSERT_NEAR(k1[i], k2[i], 1e-6f);
    }

    // Modify the copy — original should be unchanged
    KVPairMut kv2_mut = alloc.get_kv_slot_mut(b2, 0, 0);
    static_cast<float*>(kv2_mut.key)[0] = 999.0f;

    TEST_ASSERT_NEAR(static_cast<const float*>(alloc.get_kv_slot(b1, 0, 0).key)[0], 3.14f, 1e-6f);
    TEST_ASSERT_NEAR(static_cast<const float*>(alloc.get_kv_slot(b2, 0, 0).key)[0], 999.0f, 1e-6f);
}

void test_memory_sizes() {
    auto cfg = small_config();
    BlockAllocator alloc(cfg);

    // Per token KV pair: 2 * num_kv_heads * head_dim * 4 bytes = 2 * 2 * 4 * 4 = 64 bytes
    size_t expected_kv_per_token = 2 * 2 * 4 * 4;  // 64 bytes

    // Per block per layer: block_size * kv_per_token = 4 * 64 = 256
    // Per block total: num_layers * 256 = 2 * 256 = 512
    size_t expected_block_bytes = cfg.num_layers * cfg.block_size * expected_kv_per_token;
    TEST_ASSERT_EQ(alloc.block_bytes(), expected_block_bytes);

    // Total: max_blocks * block_bytes = 8 * 512 = 4096
    TEST_ASSERT_EQ(alloc.total_memory_bytes(), cfg.max_blocks * expected_block_bytes);
}

TEST_MAIN_BEGIN("BlockAllocator Tests")
    RUN_TEST(test_allocate_and_free);
    prev_failed = g_tests_failed;
    RUN_TEST(test_pool_exhaustion);
    prev_failed = g_tests_failed;
    RUN_TEST(test_ref_counting);
    prev_failed = g_tests_failed;
    RUN_TEST(test_kv_slot_read_write);
    prev_failed = g_tests_failed;
    RUN_TEST(test_copy_on_write);
    prev_failed = g_tests_failed;
    RUN_TEST(test_memory_sizes);
    prev_failed = g_tests_failed;
TEST_MAIN_END()
