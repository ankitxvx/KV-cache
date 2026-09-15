#include "kvcache/sequence.h"

namespace kvcache {

Sequence::Sequence(SequenceId id, const KVCacheConfig& config, BlockAllocator& allocator)
    : id_(id)
    , config_(config)
{
    layer_tables_.reserve(config.num_layers);
    for (int i = 0; i < config.num_layers; ++i) {
        layer_tables_.emplace_back(config.block_size);
    }
}

Status Sequence::append_token(int layer, const void* key_data, const void* value_data,
                               BlockAllocator& allocator) {
    if (layer < 0 || layer >= static_cast<int>(layer_tables_.size())) {
        return Status::ERROR_INVALID_ARGUMENT;
    }
    if (!key_data || !value_data) {
        return Status::ERROR_INVALID_ARGUMENT;
    }

    BlockId block_id;
    int slot;
    Status status = layer_tables_[layer].append_slot(allocator, block_id, slot);
    if (status != Status::OK) {
        return status;
    }

    // If this block is shared (ref_count > 1), we need Copy-on-Write
    if (allocator.get_ref_count(block_id) > 1) {
        BlockId new_block = allocator.copy_on_write(block_id);
        if (!new_block.valid()) {
            return Status::ERROR_OUT_OF_MEMORY;
        }
        allocator.unref(block_id);
        // Update the block table to point to the new block
        // The last block in the table is the one we just appended to
        layer_tables_[layer].block_table_mut_last(new_block);
        block_id = new_block;
    }

    // Copy KV data into the block
    size_t kv_bytes = static_cast<size_t>(config_.num_kv_heads) * config_.head_dim
                    * dtype_size(config_.dtype);

    KVPairMut kv = allocator.get_kv_slot_mut(block_id, layer, slot);
    if (!kv.valid()) {
        return Status::ERROR_INTERNAL;
    }

    std::memcpy(kv.key, key_data, kv_bytes);
    std::memcpy(kv.value, value_data, kv_bytes);

    return Status::OK;
}

Status Sequence::append_token_all_layers(const std::vector<const void*>& key_data_per_layer,
                                          const std::vector<const void*>& value_data_per_layer,
                                          BlockAllocator& allocator) {
    if (static_cast<int>(key_data_per_layer.size()) != config_.num_layers ||
        static_cast<int>(value_data_per_layer.size()) != config_.num_layers) {
        return Status::ERROR_INVALID_ARGUMENT;
    }

    for (int layer = 0; layer < config_.num_layers; ++layer) {
        Status s = append_token(layer, key_data_per_layer[layer],
                                value_data_per_layer[layer], allocator);
        if (s != Status::OK) {
            return s;
        }
    }
    return Status::OK;
}

KVPair Sequence::get_kv(int layer, int token_idx, const BlockAllocator& allocator) const {
    KVPair result;
    if (layer < 0 || layer >= static_cast<int>(layer_tables_.size())) {
        return result;
    }

    BlockId block_id = layer_tables_[layer].get_physical_block(token_idx);
    if (!block_id.valid()) {
        return result;
    }

    int slot = layer_tables_[layer].get_slot_in_block(token_idx);
    return allocator.get_kv_slot(block_id, layer, slot);
}

Sequence Sequence::fork(SequenceId new_id, BlockAllocator& allocator) const {
    Sequence copy;
    copy.id_ = new_id;
    copy.config_ = config_;
    copy.layer_tables_.reserve(layer_tables_.size());

    for (const auto& table : layer_tables_) {
        copy.layer_tables_.push_back(table.fork(allocator));
    }

    return copy;
}

void Sequence::free_all(BlockAllocator& allocator) {
    for (auto& table : layer_tables_) {
        table.free_all(allocator);
    }
    layer_tables_.clear();
    id_ = SequenceId{-1};
}

void Sequence::truncate(int new_length, BlockAllocator& allocator) {
    for (auto& table : layer_tables_) {
        table.truncate(new_length, allocator);
    }
}

int Sequence::num_tokens() const {
    if (layer_tables_.empty()) return 0;
    return layer_tables_[0].num_tokens();
}

int Sequence::num_tokens_at_layer(int layer) const {
    if (layer < 0 || layer >= static_cast<int>(layer_tables_.size())) return 0;
    return layer_tables_[layer].num_tokens();
}

int Sequence::total_blocks() const {
    int total = 0;
    for (const auto& table : layer_tables_) {
        total += table.num_blocks();
    }
    return total;
}

} // namespace kvcache
