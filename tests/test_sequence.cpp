#include "test_common.h"
#include "kvcache/sequence.h"
#include <vector>

using namespace kvcache;

static KVCacheConfig small_config() {
    KVCacheConfig cfg;
    cfg.num_layers = 2;
    cfg.num_kv_heads = 2;
    cfg.head_dim = 4;
    cfg.block_size = 4;
    cfg.max_blocks = 32;
    cfg.dtype = DataType::FP32;
    return cfg;
}

void test_sequence_create() {
    auto cfg = small_config();
    BlockAllocator alloc(cfg);
    Sequence seq(SequenceId{1}, cfg, alloc);

    TEST_ASSERT(seq.valid());
    TEST_ASSERT_EQ(seq.id().value, 1);
    TEST_ASSERT_EQ(seq.num_tokens(), 0);
    TEST_ASSERT_EQ(seq.num_layers(), 2);
}

void test_append_and_retrieve() {
    auto cfg = small_config();
    BlockAllocator alloc(cfg);
    Sequence seq(SequenceId{1}, cfg, alloc);

    int kv_elements = cfg.num_kv_heads * cfg.head_dim;  // 8
    std::vector<float> key(kv_elements, 1.0f);
    std::vector<float> val(kv_elements, 2.0f);

    // Append to layer 0
    Status s = seq.append_token(0, key.data(), val.data(), alloc);
    TEST_ASSERT_EQ(s, Status::OK);
    TEST_ASSERT_EQ(seq.num_tokens_at_layer(0), 1);

    // Append to layer 1
    std::vector<float> key2(kv_elements, 3.0f);
    std::vector<float> val2(kv_elements, 4.0f);
    s = seq.append_token(1, key2.data(), val2.data(), alloc);
    TEST_ASSERT_EQ(s, Status::OK);
    TEST_ASSERT_EQ(seq.num_tokens_at_layer(1), 1);

    // Retrieve layer 0
    KVPair kv = seq.get_kv(0, 0, alloc);
    TEST_ASSERT(kv.valid());
    TEST_ASSERT_NEAR(static_cast<const float*>(kv.key)[0], 1.0f, 1e-6f);
    TEST_ASSERT_NEAR(static_cast<const float*>(kv.value)[0], 2.0f, 1e-6f);

    // Retrieve layer 1
    kv = seq.get_kv(1, 0, alloc);
    TEST_ASSERT(kv.valid());
    TEST_ASSERT_NEAR(static_cast<const float*>(kv.key)[0], 3.0f, 1e-6f);
    TEST_ASSERT_NEAR(static_cast<const float*>(kv.value)[0], 4.0f, 1e-6f);
}

void test_append_all_layers() {
    auto cfg = small_config();
    BlockAllocator alloc(cfg);
    Sequence seq(SequenceId{1}, cfg, alloc);

    int kv_elements = cfg.num_kv_heads * cfg.head_dim;
    std::vector<float> k0(kv_elements, 10.0f), v0(kv_elements, 20.0f);
    std::vector<float> k1(kv_elements, 30.0f), v1(kv_elements, 40.0f);

    std::vector<const void*> keys = {k0.data(), k1.data()};
    std::vector<const void*> vals = {v0.data(), v1.data()};

    Status s = seq.append_token_all_layers(keys, vals, alloc);
    TEST_ASSERT_EQ(s, Status::OK);
    TEST_ASSERT_EQ(seq.num_tokens(), 1);

    KVPair kv0 = seq.get_kv(0, 0, alloc);
    KVPair kv1 = seq.get_kv(1, 0, alloc);
    TEST_ASSERT_NEAR(static_cast<const float*>(kv0.key)[0], 10.0f, 1e-6f);
    TEST_ASSERT_NEAR(static_cast<const float*>(kv1.key)[0], 30.0f, 1e-6f);
}

void test_fork() {
    auto cfg = small_config();
    BlockAllocator alloc(cfg);
    Sequence seq(SequenceId{1}, cfg, alloc);

    int kv_elements = cfg.num_kv_heads * cfg.head_dim;
    for (int t = 0; t < 5; ++t) {
        std::vector<float> k(kv_elements, static_cast<float>(t));
        std::vector<float> v(kv_elements, static_cast<float>(t + 100));
        for (int layer = 0; layer < cfg.num_layers; ++layer) {
            seq.append_token(layer, k.data(), v.data(), alloc);
        }
    }

    int used_before = alloc.num_used();
    Sequence forked = seq.fork(SequenceId{2}, alloc);

    // Fork should share blocks (no new allocation)
    TEST_ASSERT_EQ(alloc.num_used(), used_before);
    TEST_ASSERT_EQ(forked.num_tokens(), 5);
    TEST_ASSERT_EQ(forked.id().value, 2);

    // Verify data is accessible from fork
    KVPair kv = forked.get_kv(0, 3, alloc);
    TEST_ASSERT(kv.valid());
    TEST_ASSERT_NEAR(static_cast<const float*>(kv.key)[0], 3.0f, 1e-6f);

    forked.free_all(alloc);
    seq.free_all(alloc);
}

void test_truncate() {
    auto cfg = small_config();
    BlockAllocator alloc(cfg);
    Sequence seq(SequenceId{1}, cfg, alloc);

    int kv_elements = cfg.num_kv_heads * cfg.head_dim;
    for (int t = 0; t < 10; ++t) {
        std::vector<float> k(kv_elements, static_cast<float>(t));
        std::vector<float> v(kv_elements, static_cast<float>(t));
        for (int layer = 0; layer < cfg.num_layers; ++layer) {
            seq.append_token(layer, k.data(), v.data(), alloc);
        }
    }

    TEST_ASSERT_EQ(seq.num_tokens(), 10);

    seq.truncate(3, alloc);
    TEST_ASSERT_EQ(seq.num_tokens(), 3);

    // Token 2 should still be accessible
    KVPair kv = seq.get_kv(0, 2, alloc);
    TEST_ASSERT(kv.valid());

    // Token 3 should NOT be accessible
    kv = seq.get_kv(0, 3, alloc);
    TEST_ASSERT(!kv.valid());

    seq.free_all(alloc);
}

TEST_MAIN_BEGIN("Sequence Tests")
    RUN_TEST(test_sequence_create);
    prev_failed = g_tests_failed;
    RUN_TEST(test_append_and_retrieve);
    prev_failed = g_tests_failed;
    RUN_TEST(test_append_all_layers);
    prev_failed = g_tests_failed;
    RUN_TEST(test_fork);
    prev_failed = g_tests_failed;
    RUN_TEST(test_truncate);
    prev_failed = g_tests_failed;
TEST_MAIN_END()
