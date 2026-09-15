#include "test_common.h"
#include "kvcache/block_table.h"
#include "kvcache/block_allocator.h"

using namespace kvcache;

static KVCacheConfig small_config() {
    KVCacheConfig cfg;
    cfg.num_layers = 2;
    cfg.num_kv_heads = 2;
    cfg.head_dim = 4;
    cfg.block_size = 4;
    cfg.max_blocks = 16;
    cfg.dtype = DataType::FP32;
    return cfg;
}

void test_basic_append() {
    auto cfg = small_config();
    BlockAllocator alloc(cfg);
    BlockTable table(cfg.block_size);

    TEST_ASSERT_EQ(table.num_tokens(), 0);
    TEST_ASSERT_EQ(table.num_blocks(), 0);

    // Append first token — should allocate a new block
    BlockId block;
    int slot;
    Status s = table.append_slot(alloc, block, slot);
    TEST_ASSERT_EQ(s, Status::OK);
    TEST_ASSERT(block.valid());
    TEST_ASSERT_EQ(slot, 0);
    TEST_ASSERT_EQ(table.num_tokens(), 1);
    TEST_ASSERT_EQ(table.num_blocks(), 1);

    // Append 3 more tokens — should fill the first block
    for (int i = 1; i < 4; ++i) {
        s = table.append_slot(alloc, block, slot);
        TEST_ASSERT_EQ(s, Status::OK);
        TEST_ASSERT_EQ(slot, i);
    }
    TEST_ASSERT_EQ(table.num_blocks(), 1);

    // 5th token — should allocate a second block
    s = table.append_slot(alloc, block, slot);
    TEST_ASSERT_EQ(s, Status::OK);
    TEST_ASSERT_EQ(slot, 0);
    TEST_ASSERT_EQ(table.num_blocks(), 2);
}

void test_logical_mapping() {
    auto cfg = small_config();
    BlockAllocator alloc(cfg);
    BlockTable table(cfg.block_size);

    // Append 10 tokens (block_size=4, so 3 blocks: 4+4+2)
    for (int i = 0; i < 10; ++i) {
        BlockId b; int s;
        table.append_slot(alloc, b, s);
    }

    TEST_ASSERT_EQ(table.num_blocks(), 3);
    TEST_ASSERT_EQ(table.num_tokens(), 10);

    // Check logical mapping
    TEST_ASSERT_EQ(table.get_block_index(0), 0);
    TEST_ASSERT_EQ(table.get_slot_in_block(0), 0);
    TEST_ASSERT_EQ(table.get_block_index(3), 0);
    TEST_ASSERT_EQ(table.get_slot_in_block(3), 3);
    TEST_ASSERT_EQ(table.get_block_index(4), 1);
    TEST_ASSERT_EQ(table.get_slot_in_block(4), 0);
    TEST_ASSERT_EQ(table.get_block_index(9), 2);
    TEST_ASSERT_EQ(table.get_slot_in_block(9), 1);
}

void test_fork_cow() {
    auto cfg = small_config();
    BlockAllocator alloc(cfg);
    BlockTable table(cfg.block_size);

    for (int i = 0; i < 8; ++i) {
        BlockId b; int s;
        table.append_slot(alloc, b, s);
    }

    int used_before = alloc.num_used();
    BlockTable forked = table.fork(alloc);

    // Fork should NOT allocate new blocks — just increment ref counts
    TEST_ASSERT_EQ(alloc.num_used(), used_before);
    TEST_ASSERT_EQ(forked.num_tokens(), 8);
    TEST_ASSERT_EQ(forked.num_blocks(), 2);

    // Both tables should map to the same physical blocks
    for (int i = 0; i < 2; ++i) {
        TEST_ASSERT(table.blocks()[i] == forked.blocks()[i]);
        TEST_ASSERT_EQ(alloc.get_ref_count(table.blocks()[i]), 2);
    }

    // Free the fork — original should still work
    forked.free_all(alloc);
    for (int i = 0; i < 2; ++i) {
        TEST_ASSERT_EQ(alloc.get_ref_count(table.blocks()[i]), 1);
    }

    table.free_all(alloc);
}

void test_truncate() {
    auto cfg = small_config();
    BlockAllocator alloc(cfg);
    BlockTable table(cfg.block_size);

    for (int i = 0; i < 10; ++i) {
        BlockId b; int s;
        table.append_slot(alloc, b, s);
    }
    TEST_ASSERT_EQ(table.num_blocks(), 3);

    // Truncate to 5 tokens — should keep 2 blocks (5/4 = ceil to 2)
    table.truncate(5, alloc);
    TEST_ASSERT_EQ(table.num_tokens(), 5);
    TEST_ASSERT_EQ(table.num_blocks(), 2);

    // Truncate to 0 — should free all
    table.truncate(0, alloc);
    TEST_ASSERT_EQ(table.num_tokens(), 0);
    TEST_ASSERT_EQ(table.num_blocks(), 0);
}

void test_out_of_range() {
    auto cfg = small_config();
    BlockAllocator alloc(cfg);
    BlockTable table(cfg.block_size);

    TEST_ASSERT(!table.get_physical_block(0).valid());
    TEST_ASSERT_EQ(table.get_slot_in_block(0), -1);
    TEST_ASSERT_EQ(table.get_block_index(-1), -1);
}

TEST_MAIN_BEGIN("BlockTable Tests")
    RUN_TEST(test_basic_append);
    prev_failed = g_tests_failed;
    RUN_TEST(test_logical_mapping);
    prev_failed = g_tests_failed;
    RUN_TEST(test_fork_cow);
    prev_failed = g_tests_failed;
    RUN_TEST(test_truncate);
    prev_failed = g_tests_failed;
    RUN_TEST(test_out_of_range);
    prev_failed = g_tests_failed;
TEST_MAIN_END()
