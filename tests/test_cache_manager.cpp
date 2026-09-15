#include "test_common.h"
#include "kvcache/cache_manager.h"
#include <vector>

using namespace kvcache;

static KVCacheConfig small_config() {
    KVCacheConfig cfg;
    cfg.num_layers = 2;
    cfg.num_kv_heads = 2;
    cfg.head_dim = 4;
    cfg.block_size = 4;
    cfg.max_blocks = 64;
    cfg.dtype = DataType::FP32;
    cfg.eviction_policy = EvictionType::NONE;
    return cfg;
}

void test_create_and_free_sequence() {
    auto cfg = small_config();
    KVCacheManager manager(cfg);

    Status s = manager.create_sequence(SequenceId{1});
    TEST_ASSERT_EQ(s, Status::OK);
    TEST_ASSERT(manager.has_sequence(SequenceId{1}));
    TEST_ASSERT_EQ(manager.num_sequences(), 1);

    // Duplicate should fail
    s = manager.create_sequence(SequenceId{1});
    TEST_ASSERT_EQ(s, Status::ERROR_SEQUENCE_EXISTS);

    // Free
    s = manager.free_sequence(SequenceId{1});
    TEST_ASSERT_EQ(s, Status::OK);
    TEST_ASSERT(!manager.has_sequence(SequenceId{1}));

    // Free non-existent
    s = manager.free_sequence(SequenceId{999});
    TEST_ASSERT_EQ(s, Status::ERROR_SEQUENCE_NOT_FOUND);
}

void test_append_and_get() {
    auto cfg = small_config();
    KVCacheManager manager(cfg);

    manager.create_sequence(SequenceId{1});

    int kv_elements = cfg.num_kv_heads * cfg.head_dim;
    std::vector<float> key(kv_elements, 5.0f);
    std::vector<float> val(kv_elements, 10.0f);

    Status s = manager.append_kv(SequenceId{1}, 0, key.data(), val.data());
    TEST_ASSERT_EQ(s, Status::OK);

    TEST_ASSERT_EQ(manager.get_num_tokens(SequenceId{1}), 1);

    KVPair kv = manager.get_kv(SequenceId{1}, 0, 0);
    TEST_ASSERT(kv.valid());
    TEST_ASSERT_NEAR(static_cast<const float*>(kv.key)[0], 5.0f, 1e-6f);
    TEST_ASSERT_NEAR(static_cast<const float*>(kv.value)[0], 10.0f, 1e-6f);

    // Non-existent sequence
    s = manager.append_kv(SequenceId{999}, 0, key.data(), val.data());
    TEST_ASSERT_EQ(s, Status::ERROR_SEQUENCE_NOT_FOUND);
}

void test_multi_token_multi_layer() {
    auto cfg = small_config();
    KVCacheManager manager(cfg);
    manager.create_sequence(SequenceId{1});

    int kv_elements = cfg.num_kv_heads * cfg.head_dim;

    // Append 8 tokens across both layers
    for (int t = 0; t < 8; ++t) {
        for (int layer = 0; layer < cfg.num_layers; ++layer) {
            float fill = static_cast<float>(t * 100 + layer);
            std::vector<float> key(kv_elements, fill);
            std::vector<float> val(kv_elements, fill + 0.5f);
            manager.append_kv(SequenceId{1}, layer, key.data(), val.data());
        }
    }

    // Verify each token at each layer
    for (int t = 0; t < 8; ++t) {
        for (int layer = 0; layer < cfg.num_layers; ++layer) {
            float expected_k = static_cast<float>(t * 100 + layer);
            float expected_v = expected_k + 0.5f;
            KVPair kv = manager.get_kv(SequenceId{1}, layer, t);
            TEST_ASSERT(kv.valid());
            TEST_ASSERT_NEAR(static_cast<const float*>(kv.key)[0], expected_k, 1e-6f);
            TEST_ASSERT_NEAR(static_cast<const float*>(kv.value)[0], expected_v, 1e-6f);
        }
    }
}

void test_fork_sequence() {
    auto cfg = small_config();
    KVCacheManager manager(cfg);
    manager.create_sequence(SequenceId{1});

    int kv_elements = cfg.num_kv_heads * cfg.head_dim;
    for (int t = 0; t < 4; ++t) {
        for (int layer = 0; layer < cfg.num_layers; ++layer) {
            std::vector<float> key(kv_elements, static_cast<float>(t));
            std::vector<float> val(kv_elements, static_cast<float>(t + 50));
            manager.append_kv(SequenceId{1}, layer, key.data(), val.data());
        }
    }

    // Fork
    Status s = manager.fork_sequence(SequenceId{1}, SequenceId{2});
    TEST_ASSERT_EQ(s, Status::OK);
    TEST_ASSERT(manager.has_sequence(SequenceId{2}));
    TEST_ASSERT_EQ(manager.get_num_tokens(SequenceId{2}), 4);

    // Verify forked data
    KVPair kv = manager.get_kv(SequenceId{2}, 0, 2);
    TEST_ASSERT(kv.valid());
    TEST_ASSERT_NEAR(static_cast<const float*>(kv.key)[0], 2.0f, 1e-6f);

    // Free source — fork should still work
    manager.free_sequence(SequenceId{1});
    kv = manager.get_kv(SequenceId{2}, 0, 2);
    TEST_ASSERT(kv.valid());

    manager.free_sequence(SequenceId{2});
}

void test_stats() {
    auto cfg = small_config();
    KVCacheManager manager(cfg);

    CacheStats stats = manager.get_stats();
    TEST_ASSERT_EQ(stats.total_blocks, cfg.max_blocks);
    TEST_ASSERT_EQ(stats.used_blocks, 0);
    TEST_ASSERT_EQ(stats.active_sequences, 0);

    manager.create_sequence(SequenceId{1});
    int kv_elements = cfg.num_kv_heads * cfg.head_dim;
    std::vector<float> data(kv_elements, 1.0f);
    manager.append_kv(SequenceId{1}, 0, data.data(), data.data());

    stats = manager.get_stats();
    TEST_ASSERT_EQ(stats.active_sequences, 1);
    TEST_ASSERT(stats.used_blocks > 0);
    TEST_ASSERT(stats.utilization_pct > 0.0);
}

TEST_MAIN_BEGIN("CacheManager Tests")
    RUN_TEST(test_create_and_free_sequence);
    prev_failed = g_tests_failed;
    RUN_TEST(test_append_and_get);
    prev_failed = g_tests_failed;
    RUN_TEST(test_multi_token_multi_layer);
    prev_failed = g_tests_failed;
    RUN_TEST(test_fork_sequence);
    prev_failed = g_tests_failed;
    RUN_TEST(test_stats);
    prev_failed = g_tests_failed;
TEST_MAIN_END()
