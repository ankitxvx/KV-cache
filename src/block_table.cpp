#include "kvcache/block_table.h"
#include "kvcache/block_allocator.h"

namespace kvcache {

BlockTable::BlockTable(int block_size)
    : block_size_(block_size)
    , num_tokens_(0)
{
}

Status BlockTable::append_slot(BlockAllocator& allocator, BlockId& out_block, int& out_slot) {
    int slot_in_block = num_tokens_ % block_size_;

    // Need a new block?
    if (slot_in_block == 0) {
        BlockId new_block = allocator.allocate();
        if (!new_block.valid()) {
            return Status::ERROR_OUT_OF_MEMORY;
        }
        blocks_.push_back(new_block);
    }

    out_block = blocks_.back();
    out_slot  = slot_in_block;
    num_tokens_++;

    // Update used slots count on the block
    allocator.set_used_slots(out_block, out_slot + 1);

    return Status::OK;
}

BlockId BlockTable::get_physical_block(int token_idx) const {
    if (token_idx < 0 || token_idx >= num_tokens_) {
        return BlockId{-1};
    }
    int block_index = token_idx / block_size_;
    return blocks_[block_index];
}

int BlockTable::get_slot_in_block(int token_idx) const {
    if (token_idx < 0 || token_idx >= num_tokens_) {
        return -1;
    }
    return token_idx % block_size_;
}

int BlockTable::get_block_index(int token_idx) const {
    if (token_idx < 0 || token_idx >= num_tokens_) {
        return -1;
    }
    return token_idx / block_size_;
}

BlockTable BlockTable::fork(BlockAllocator& allocator) const {
    BlockTable copy(block_size_);
    copy.num_tokens_ = num_tokens_;
    copy.blocks_ = blocks_;  // Copy the block ID list

    // Increment ref counts on all shared blocks
    for (const auto& block_id : copy.blocks_) {
        allocator.ref(block_id);
    }

    return copy;
}

void BlockTable::free_all(BlockAllocator& allocator) {
    for (const auto& block_id : blocks_) {
        allocator.unref(block_id);
    }
    blocks_.clear();
    num_tokens_ = 0;
}

void BlockTable::truncate(int new_num_tokens, BlockAllocator& allocator) {
    if (new_num_tokens >= num_tokens_) return;
    if (new_num_tokens < 0) new_num_tokens = 0;

    // Calculate how many blocks we need
    int new_num_blocks = (new_num_tokens > 0)
                       ? ((new_num_tokens + block_size_ - 1) / block_size_)
                       : 0;

    // Free trailing blocks
    for (int i = static_cast<int>(blocks_.size()) - 1; i >= new_num_blocks; --i) {
        allocator.unref(blocks_[i]);
    }
    blocks_.resize(new_num_blocks);
    num_tokens_ = new_num_tokens;

    // Update used_slots on the last block if any
    if (!blocks_.empty() && new_num_tokens > 0) {
        int last_used = new_num_tokens % block_size_;
        if (last_used == 0) last_used = block_size_;
        allocator.set_used_slots(blocks_.back(), last_used);
    }
}

} // namespace kvcache
