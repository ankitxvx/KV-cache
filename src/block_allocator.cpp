#include "kvcache/block_allocator.h"
#include <stdexcept>

namespace kvcache {

BlockAllocator::BlockAllocator(const KVCacheConfig& config)
    : config_(config)
{
    config_.validate();
    // Precompute sizes
    // K or V for one token = num_kv_heads * head_dim * dtype_size
    kv_per_token_bytes_ = static_cast<size_t>(config_.num_kv_heads)
                        * config_.head_dim
                        * dtype_size(config_.dtype);

    // One layer per block = block_size tokens * (K + V)
    layer_bytes_ = static_cast<size_t>(config_.block_size) * 2 * kv_per_token_bytes_;

    // Total bytes per block = all layers
    block_total_bytes_ = static_cast<size_t>(config_.num_layers) * layer_bytes_;

    // Pre-allocate all physical blocks
    blocks_.resize(config_.max_blocks);
    free_list_.reserve(config_.max_blocks);

    for (int i = 0; i < config_.max_blocks; ++i) {
        blocks_[i].init(BlockId{i}, block_total_bytes_);
        free_list_.push_back(BlockId{i});
    }
}

BlockId BlockAllocator::allocate() {
    std::lock_guard<std::mutex> lock(mutex_);

    if (free_list_.empty()) {
        return BlockId{-1};  // Out of memory
    }

    BlockId id = free_list_.back();
    free_list_.pop_back();

    blocks_[id.value].ref_count = 1;
    blocks_[id.value].used_slots = 0;
    total_allocations_++;

    return id;
}

void BlockAllocator::free(BlockId id) {
    if (!valid_block(id)) return;

    std::lock_guard<std::mutex> lock(mutex_);

    auto& block = blocks_[id.value];
    if (block.ref_count == 0) return;
    if (--block.ref_count > 0) return;
    block.used_slots = 0;
    // Don't zero the data — it will be overwritten on next use
    free_list_.push_back(id);
    total_frees_++;
}

void BlockAllocator::ref(BlockId id) {
    if (!valid_block(id)) return;

    std::lock_guard<std::mutex> lock(mutex_);
    if (blocks_[id.value].ref_count > 0) blocks_[id.value].ref_count++;
}

void BlockAllocator::unref(BlockId id) {
    if (!valid_block(id)) return;

    std::lock_guard<std::mutex> lock(mutex_);
    auto& block = blocks_[id.value];
    if (block.ref_count == 0) return;
    block.ref_count--;

    if (block.ref_count <= 0) {
        block.ref_count = 0;
        block.used_slots = 0;
        free_list_.push_back(id);
        total_frees_++;
    }
}

int BlockAllocator::get_ref_count(BlockId id) const {
    if (!valid_block(id)) return 0;
    std::lock_guard<std::mutex> lock(mutex_);
    return blocks_[id.value].ref_count;
}

void* BlockAllocator::get_block_ptr(BlockId id) {
    if (!valid_block(id)) return nullptr;
    return blocks_[id.value].data.data();
}

const void* BlockAllocator::get_block_ptr(BlockId id) const {
    if (!valid_block(id)) return nullptr;
    return blocks_[id.value].data.data();
}

size_t BlockAllocator::compute_layer_offset(int layer) const {
    return static_cast<size_t>(layer) * layer_bytes_;
}

size_t BlockAllocator::compute_kv_offset_in_layer(bool is_value) const {
    // Within a layer: first half is K, second half is V
    // K starts at offset 0, V starts at block_size * kv_per_token_bytes_
    if (!is_value) return 0;
    return static_cast<size_t>(config_.block_size) * kv_per_token_bytes_;
}

size_t BlockAllocator::compute_slot_offset(int slot) const {
    return static_cast<size_t>(slot) * kv_per_token_bytes_;
}

void* BlockAllocator::get_key_ptr(BlockId id, int layer) {
    if (!valid_block(id) || layer < 0 || layer >= config_.num_layers) return nullptr;
    size_t offset = compute_layer_offset(layer) + compute_kv_offset_in_layer(false);
    return blocks_[id.value].data.data() + offset;
}

void* BlockAllocator::get_value_ptr(BlockId id, int layer) {
    if (!valid_block(id) || layer < 0 || layer >= config_.num_layers) return nullptr;
    size_t offset = compute_layer_offset(layer) + compute_kv_offset_in_layer(true);
    return blocks_[id.value].data.data() + offset;
}

const void* BlockAllocator::get_key_ptr(BlockId id, int layer) const {
    if (!valid_block(id) || layer < 0 || layer >= config_.num_layers) return nullptr;
    size_t offset = compute_layer_offset(layer) + compute_kv_offset_in_layer(false);
    return blocks_[id.value].data.data() + offset;
}

const void* BlockAllocator::get_value_ptr(BlockId id, int layer) const {
    if (!valid_block(id) || layer < 0 || layer >= config_.num_layers) return nullptr;
    size_t offset = compute_layer_offset(layer) + compute_kv_offset_in_layer(true);
    return blocks_[id.value].data.data() + offset;
}

KVPairMut BlockAllocator::get_kv_slot_mut(BlockId id, int layer, int slot) {
    KVPairMut result;
    if (!valid_block(id) || layer < 0 || layer >= config_.num_layers
        || slot < 0 || slot >= config_.block_size) {
        return result;
    }

    size_t layer_off = compute_layer_offset(layer);
    size_t slot_off  = compute_slot_offset(slot);

    uint8_t* base = blocks_[id.value].data.data();
    result.key   = base + layer_off + compute_kv_offset_in_layer(false) + slot_off;
    result.value = base + layer_off + compute_kv_offset_in_layer(true)  + slot_off;
    return result;
}

KVPair BlockAllocator::get_kv_slot(BlockId id, int layer, int slot) const {
    KVPair result;
    if (!valid_block(id) || layer < 0 || layer >= config_.num_layers
        || slot < 0 || slot >= config_.block_size) {
        return result;
    }

    size_t layer_off = compute_layer_offset(layer);
    size_t slot_off  = compute_slot_offset(slot);

    const uint8_t* base = blocks_[id.value].data.data();
    result.key   = base + layer_off + compute_kv_offset_in_layer(false) + slot_off;
    result.value = base + layer_off + compute_kv_offset_in_layer(true)  + slot_off;
    return result;
}

int BlockAllocator::get_used_slots(BlockId id) const {
    if (!valid_block(id)) return 0;
    std::lock_guard<std::mutex> lock(mutex_);
    return blocks_[id.value].used_slots;
}

void BlockAllocator::set_used_slots(BlockId id, int count) {
    if (!valid_block(id)) return;
    std::lock_guard<std::mutex> lock(mutex_);
    blocks_[id.value].used_slots = count;
}

int BlockAllocator::num_free() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return static_cast<int>(free_list_.size());
}

int BlockAllocator::num_used() const {
    return config_.max_blocks - num_free();
}

int BlockAllocator::num_total() const {
    return config_.max_blocks;
}

size_t BlockAllocator::total_memory_bytes() const {
    return static_cast<size_t>(config_.max_blocks) * block_total_bytes_;
}

size_t BlockAllocator::block_bytes() const {
    return block_total_bytes_;
}

BlockId BlockAllocator::copy_on_write(BlockId src_id) {
    if (!valid_block(src_id)) return BlockId{-1};

    BlockId new_id = allocate();
    if (!new_id.valid()) return new_id;

    // Deep copy the data
    std::memcpy(blocks_[new_id.value].data.data(),
                blocks_[src_id.value].data.data(),
                block_total_bytes_);

    blocks_[new_id.value].used_slots = blocks_[src_id.value].used_slots;

    return new_id;
}

} // namespace kvcache
