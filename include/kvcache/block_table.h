#pragma once

#include "types.h"
#include <vector>

namespace kvcache {

class BlockAllocator;

/// Maps a sequence's logical token positions to physical blocks for a single layer.
/// Tokens are addressed as (block_index, slot_within_block).
class BlockTable {
public:
    BlockTable() = default;
    explicit BlockTable(int block_size);

    /// Allocate a new slot for the next token. Allocates a new physical block
    /// from the allocator if the current block is full.
    /// Returns the BlockId and slot index for the new token.
    Status append_slot(BlockAllocator& allocator, BlockId& out_block, int& out_slot);

    /// Get the physical block ID for a given logical token index.
    BlockId get_physical_block(int token_idx) const;

    /// Get the slot index within a block for a given logical token index.
    int get_slot_in_block(int token_idx) const;

    /// Get the block index (index into blocks_ vector) for a given token index.
    int get_block_index(int token_idx) const;

    /// Create a Copy-on-Write fork. All block ref counts are incremented.
    BlockTable fork(BlockAllocator& allocator) const;

    /// Release all blocks back to the allocator.
    void free_all(BlockAllocator& allocator);

    /// Truncate to a given number of tokens, freeing trailing blocks.
    void truncate(int new_num_tokens, BlockAllocator& allocator);

    /// Number of tokens currently mapped.
    int num_tokens() const { return num_tokens_; }

    /// Number of physical blocks used.
    int num_blocks() const { return static_cast<int>(blocks_.size()); }

    /// Direct access to block IDs (for prefix cache hashing etc.)
    const std::vector<BlockId>& blocks() const { return blocks_; }

    /// Replace the last block ID (used during Copy-on-Write).
    void block_table_mut_last(BlockId new_id) {
        if (!blocks_.empty()) {
            blocks_.back() = new_id;
        }
    }

    /// Replace block at a given index.
    void set_block(int index, BlockId new_id) {
        if (index >= 0 && index < static_cast<int>(blocks_.size())) {
            blocks_[index] = new_id;
        }
    }

    /// Get the block size.
    int block_size() const { return block_size_; }

private:
    friend class Sequence;
    int block_size_  = 16;
    int num_tokens_  = 0;
    std::vector<BlockId> blocks_;   ///< Ordered physical block IDs
};

} // namespace kvcache
