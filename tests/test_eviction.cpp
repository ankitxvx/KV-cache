#include "test_common.h"
#include "kvcache/eviction_policy.h"
#include <algorithm>

using namespace kvcache;

// ===========================================================================
// LRU Tests
// ===========================================================================

void test_lru_basic_eviction() {
    LRUEviction lru;
    SequenceId sid{1};
    lru.on_sequence_created(sid);

    // Access tokens 0,1,2,3,4 in order
    for (int i = 0; i < 5; ++i) {
        lru.on_token_accessed(sid, i);
    }

    // Evict 2 — should evict tokens 0 and 1 (least recently used)
    auto victims = lru.select_victims(sid, 2);
    TEST_ASSERT_EQ(static_cast<int>(victims.size()), 2);
    TEST_ASSERT_EQ(victims[0], 0);
    TEST_ASSERT_EQ(victims[1], 1);
}

void test_lru_reaccess_promotes() {
    LRUEviction lru;
    SequenceId sid{1};
    lru.on_sequence_created(sid);

    for (int i = 0; i < 5; ++i) {
        lru.on_token_accessed(sid, i);
    }

    // Re-access token 0 — promotes it to MRU
    lru.on_token_accessed(sid, 0);

    // Evict 1 — should evict token 1 (now the LRU)
    auto victims = lru.select_victims(sid, 1);
    TEST_ASSERT_EQ(static_cast<int>(victims.size()), 1);
    TEST_ASSERT_EQ(victims[0], 1);
}

void test_lru_evict_notify() {
    LRUEviction lru;
    SequenceId sid{1};
    lru.on_sequence_created(sid);

    for (int i = 0; i < 5; ++i) {
        lru.on_token_accessed(sid, i);
    }

    // Evict token 0
    lru.on_token_evicted(sid, 0);

    // Now evict 1 — should evict token 1 (0 is gone)
    auto victims = lru.select_victims(sid, 1);
    TEST_ASSERT_EQ(static_cast<int>(victims.size()), 1);
    TEST_ASSERT_EQ(victims[0], 1);
}

// ===========================================================================
// H2O Tests
// ===========================================================================

void test_h2o_heavy_hitters_protected() {
    // heavy_ratio=0.4 means 40% of tokens are heavy hitters (protected)
    H2OEviction h2o(0.4f, 2);  // 2 sink tokens
    SequenceId sid{1};
    h2o.on_sequence_created(sid);

    // 10 tokens: 0-1 are sinks (protected), 2-9 are candidates
    // Give high attention scores to tokens 5,6,7,8,9
    for (int i = 0; i < 10; ++i) {
        float score = (i >= 5) ? 10.0f : 0.1f;
        h2o.on_token_accessed(sid, i, score);
    }

    // Try to evict 3. Tokens 0,1 are sinks (protected).
    // Heavy hitters: 40% of 10 = 4 tokens (the top 4 by score: 6,7,8,9)
    // Evictable: 2,3,4,5 (from lowest score up, excluding sinks)
    // Wait — tokens 2,3,4 have score 0.1, token 5 has score 10.0
    // Sorted by score ascending: 2(0.1), 3(0.1), 4(0.1), 5(10), 6(10), 7(10), 8(10), 9(10)
    // Top 4 heavy hitters: 5,6,7,8,9 (well, top 4 would be 6,7,8,9)
    // So evictable: 2,3,4,5 — first 3 victims: 2,3,4
    auto victims = h2o.select_victims(sid, 3);
    TEST_ASSERT_EQ(static_cast<int>(victims.size()), 3);
    // Victims should be the lowest-scored non-sink tokens
    TEST_ASSERT_EQ(victims[0], 2);
    TEST_ASSERT_EQ(victims[1], 3);
    TEST_ASSERT_EQ(victims[2], 4);
}

void test_h2o_sinks_never_evicted() {
    H2OEviction h2o(0.0f, 3);  // 3 sink tokens, no heavy hitters
    SequenceId sid{1};
    h2o.on_sequence_created(sid);

    for (int i = 0; i < 6; ++i) {
        h2o.on_token_accessed(sid, i, 1.0f);
    }

    // Try to evict all 6 — only 3 should be evictable (sinks 0,1,2 protected)
    auto victims = h2o.select_victims(sid, 6);
    TEST_ASSERT_EQ(static_cast<int>(victims.size()), 3);
    for (int v : victims) {
        TEST_ASSERT(v >= 3);  // No sink tokens evicted
    }
}

// ===========================================================================
// Sliding Window Tests
// ===========================================================================

void test_sliding_window_basic() {
    SlidingWindowEviction sw(4, 2);  // window=4, sinks=2
    SequenceId sid{1};
    sw.on_sequence_created(sid);

    // 10 tokens: 0-9
    for (int i = 0; i < 10; ++i) {
        sw.on_token_accessed(sid, i);
    }

    // Protected: sinks 0,1 + window 6,7,8,9 = 6 protected
    // Evictable: 2,3,4,5
    auto victims = sw.select_victims(sid, 10);
    TEST_ASSERT_EQ(static_cast<int>(victims.size()), 4);
    TEST_ASSERT_EQ(victims[0], 2);
    TEST_ASSERT_EQ(victims[1], 3);
    TEST_ASSERT_EQ(victims[2], 4);
    TEST_ASSERT_EQ(victims[3], 5);
}

void test_sliding_window_already_evicted() {
    SlidingWindowEviction sw(4, 2);
    SequenceId sid{1};
    sw.on_sequence_created(sid);

    for (int i = 0; i < 10; ++i) {
        sw.on_token_accessed(sid, i);
    }

    // Evict token 2
    sw.on_token_evicted(sid, 2);

    // Now select victims — 2 should not be in the list
    auto victims = sw.select_victims(sid, 10);
    for (int v : victims) {
        TEST_ASSERT(v != 2);
    }
    TEST_ASSERT_EQ(static_cast<int>(victims.size()), 3);  // 3,4,5
}

// ===========================================================================
// Factory Test
// ===========================================================================

void test_factory() {
    KVCacheConfig cfg;

    cfg.eviction_policy = EvictionType::NONE;
    auto p = create_eviction_policy(cfg);
    TEST_ASSERT(p == nullptr);

    cfg.eviction_policy = EvictionType::LRU;
    p = create_eviction_policy(cfg);
    TEST_ASSERT(p != nullptr);

    cfg.eviction_policy = EvictionType::H2O;
    p = create_eviction_policy(cfg);
    TEST_ASSERT(p != nullptr);

    cfg.eviction_policy = EvictionType::SLIDING_WINDOW;
    p = create_eviction_policy(cfg);
    TEST_ASSERT(p != nullptr);
}

TEST_MAIN_BEGIN("Eviction Policy Tests")
    RUN_TEST(test_lru_basic_eviction);
    prev_failed = g_tests_failed;
    RUN_TEST(test_lru_reaccess_promotes);
    prev_failed = g_tests_failed;
    RUN_TEST(test_lru_evict_notify);
    prev_failed = g_tests_failed;
    RUN_TEST(test_h2o_heavy_hitters_protected);
    prev_failed = g_tests_failed;
    RUN_TEST(test_h2o_sinks_never_evicted);
    prev_failed = g_tests_failed;
    RUN_TEST(test_sliding_window_basic);
    prev_failed = g_tests_failed;
    RUN_TEST(test_sliding_window_already_evicted);
    prev_failed = g_tests_failed;
    RUN_TEST(test_factory);
    prev_failed = g_tests_failed;
TEST_MAIN_END()
