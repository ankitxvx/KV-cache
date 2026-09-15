#include "test_common.h"
#include "kvcache/prefix_cache.h"

using namespace kvcache;

void test_register_and_match() {
    PrefixCache cache(4);  // block_size = 4

    // Register a prefix of 8 tokens (2 blocks)
    int tokens[] = {10, 20, 30, 40, 50, 60, 70, 80};
    std::vector<BlockId> blocks = {BlockId{0}, BlockId{1}};

    cache.register_prefix(tokens, 8, blocks);
    TEST_ASSERT_EQ(cache.size(), 1);

    // Match the same prefix
    std::vector<BlockId> matched;
    int matched_tokens = cache.try_match(tokens, 8, matched);
    TEST_ASSERT_EQ(matched_tokens, 8);
    TEST_ASSERT_EQ(static_cast<int>(matched.size()), 2);
    TEST_ASSERT(matched[0] == BlockId{0});
    TEST_ASSERT(matched[1] == BlockId{1});
    TEST_ASSERT_EQ(cache.hits(), 1);
}

void test_partial_match() {
    PrefixCache cache(4);

    int tokens[] = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<BlockId> blocks = {BlockId{0}, BlockId{1}};
    cache.register_prefix(tokens, 8, blocks);

    // Query with only the first 4 tokens
    std::vector<BlockId> matched;
    int matched_tokens = cache.try_match(tokens, 4, matched);
    TEST_ASSERT_EQ(matched_tokens, 4);
    TEST_ASSERT_EQ(static_cast<int>(matched.size()), 1);
}

void test_no_match() {
    PrefixCache cache(4);

    int tokens[] = {1, 2, 3, 4};
    std::vector<BlockId> blocks = {BlockId{0}};
    cache.register_prefix(tokens, 4, blocks);

    // Different tokens — should not match
    int diff_tokens[] = {5, 6, 7, 8};
    std::vector<BlockId> matched;
    int matched_tokens = cache.try_match(diff_tokens, 4, matched);
    TEST_ASSERT_EQ(matched_tokens, 0);
    TEST_ASSERT_EQ(cache.misses(), 1);
}

void test_partial_block_not_registered() {
    PrefixCache cache(4);

    // 6 tokens — only 4 (1 full block) should be registered
    int tokens[] = {1, 2, 3, 4, 5, 6};
    std::vector<BlockId> blocks = {BlockId{0}};
    cache.register_prefix(tokens, 6, blocks);

    // Query all 6 — should only match 4
    std::vector<BlockId> matched;
    int matched_tokens = cache.try_match(tokens, 6, matched);
    TEST_ASSERT_EQ(matched_tokens, 4);
}

void test_evict_prefix() {
    PrefixCache cache(4);

    int tokens[] = {1, 2, 3, 4};
    std::vector<BlockId> blocks = {BlockId{0}};
    cache.register_prefix(tokens, 4, blocks);
    TEST_ASSERT_EQ(cache.size(), 1);

    // Clear all
    cache.clear();
    TEST_ASSERT_EQ(cache.size(), 0);

    std::vector<BlockId> matched;
    int matched_tokens = cache.try_match(tokens, 4, matched);
    TEST_ASSERT_EQ(matched_tokens, 0);
}

void test_empty_input() {
    PrefixCache cache(4);

    std::vector<BlockId> matched;
    int n = cache.try_match(nullptr, 0, matched);
    TEST_ASSERT_EQ(n, 0);

    cache.register_prefix(nullptr, 0, {});
    TEST_ASSERT_EQ(cache.size(), 0);
}

TEST_MAIN_BEGIN("PrefixCache Tests")
    RUN_TEST(test_register_and_match);
    prev_failed = g_tests_failed;
    RUN_TEST(test_partial_match);
    prev_failed = g_tests_failed;
    RUN_TEST(test_no_match);
    prev_failed = g_tests_failed;
    RUN_TEST(test_partial_block_not_registered);
    prev_failed = g_tests_failed;
    RUN_TEST(test_evict_prefix);
    prev_failed = g_tests_failed;
    RUN_TEST(test_empty_input);
    prev_failed = g_tests_failed;
TEST_MAIN_END()
