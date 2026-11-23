#include "kvcache/sequence.h"
#include <algorithm>

namespace kvcache {
Sequence::Sequence(SequenceId id, const KVCacheConfig& config, BlockAllocator&)
    : id_(id), config_(config), table_(config.block_size), layer_lengths_(config.num_layers, 0) {}

Status Sequence::prepare_write(int token, BlockAllocator& allocator, BlockId& block) {
    if (token >= config_.max_sequence_tokens) return Status::ERROR_OUT_OF_MEMORY;
    if (token == table_.num_tokens()) {
        evicted_.reserve(static_cast<size_t>(token) + 1);
        int slot;
        auto status = table_.append_slot(allocator, block, slot);
        if (status != Status::OK) return status;
        evicted_.push_back(false);
    } else {
        block = table_.get_physical_block(token);
        if (!block.valid() || evicted_[token]) return Status::ERROR_INVALID_ARGUMENT;
        if (allocator.get_ref_count(block) > 1) {
            auto replacement = allocator.copy_on_write(block);
            if (!replacement.valid()) return Status::ERROR_OUT_OF_MEMORY;
            table_.set_block(token / config_.block_size, replacement);
            allocator.unref(block); block = replacement;
        }
    }
    return Status::OK;
}
Status Sequence::append_token(int layer, const void* key, const void* value, BlockAllocator& allocator) {
    if (layer < 0 || layer >= num_layers() || !key || !value) return Status::ERROR_INVALID_ARGUMENT;
    int token = layer_lengths_[layer];
    if (token > *std::min_element(layer_lengths_.begin(), layer_lengths_.end()))
        return Status::ERROR_INVALID_ARGUMENT;
    BlockId block;
    auto status = prepare_write(token, allocator, block);
    if (status != Status::OK) return status;
    auto kv = allocator.get_kv_slot_mut(block, layer, token % config_.block_size);
    size_t bytes = config_.kv_pair_bytes_per_token() / 2;
    std::memcpy(kv.key, key, bytes); std::memcpy(kv.value, value, bytes);
    ++layer_lengths_[layer];
    return Status::OK;
}
Status Sequence::append_token_all_layers(const std::vector<const void*>& keys,
                                        const std::vector<const void*>& values, BlockAllocator& allocator) {
    if (keys.size() != layer_lengths_.size() || values.size() != keys.size() || !complete())
        return Status::ERROR_INVALID_ARGUMENT;
    for (size_t i = 0; i < keys.size(); ++i)
        if (!keys[i] || !values[i]) return Status::ERROR_INVALID_ARGUMENT;
    int token = num_tokens();
    BlockId block;
    auto status = prepare_write(token, allocator, block);
    if (status != Status::OK) return status;
    size_t bytes = config_.kv_pair_bytes_per_token() / 2;
    for (int layer = 0; layer < num_layers(); ++layer) {
        auto kv = allocator.get_kv_slot_mut(block, layer, token % config_.block_size);
        std::memcpy(kv.key, keys[layer], bytes); std::memcpy(kv.value, values[layer], bytes);
        ++layer_lengths_[layer];
    }
    return Status::OK;
}
KVPair Sequence::get_kv(int layer, int token, const BlockAllocator& allocator) const {
    if (layer < 0 || layer >= num_layers() || token < 0 || token >= layer_lengths_[layer] || evicted_[token])
        return {};
    return allocator.get_kv_slot(table_.get_physical_block(token), layer, token % config_.block_size);
}
Sequence Sequence::fork(SequenceId id, BlockAllocator& allocator) const {
    Sequence result;
    result.id_ = id; result.config_ = config_;
    result.layer_lengths_ = layer_lengths_; result.evicted_ = evicted_;
    result.table_ = table_.fork(allocator);
    return result;
}
void Sequence::free_all(BlockAllocator& allocator) {
    table_.free_all(allocator); layer_lengths_.clear(); evicted_.clear(); id_ = {-1};
}
void Sequence::truncate(int length, BlockAllocator& allocator) {
    length = std::max(0, std::min(length, table_.num_tokens()));
    table_.truncate(length, allocator); evicted_.resize(length);
    for (auto& n : layer_lengths_) n = std::min(n, length);
}
int Sequence::num_tokens() const { return layer_lengths_.empty() ? 0 : layer_lengths_[0]; }
int Sequence::num_tokens_at_layer(int layer) const {
    return layer < 0 || layer >= num_layers() ? 0 : layer_lengths_[layer];
}
int Sequence::total_blocks() const {
    return static_cast<int>(std::count_if(table_.blocks_.begin(), table_.blocks_.end(), [](BlockId b) {return b.valid();}));
}
bool Sequence::complete() const {
    return !layer_lengths_.empty() && std::all_of(layer_lengths_.begin(), layer_lengths_.end(),
        [this](int n) { return n == num_tokens(); });
}
bool Sequence::has_evictions() const { return std::find(evicted_.begin(), evicted_.end(), true) != evicted_.end(); }
int Sequence::cached_tokens() const {
    return num_tokens() - static_cast<int>(std::count(evicted_.begin(), evicted_.begin() + num_tokens(), true));
}
int Sequence::evict_tokens(const std::vector<int>& tokens, BlockAllocator& allocator) {
    if (!complete()) return 0;
    int count = 0;
    for (int token : tokens) {
        if (token >= 0 && token < num_tokens() && !evicted_[token]) { evicted_[token] = true; ++count; }
    }
    for (size_t i = 0; i < table_.blocks_.size(); ++i) {
        if (!table_.blocks_[i].valid()) continue;
        int first = static_cast<int>(i) * config_.block_size;
        int end = std::min(first + config_.block_size, num_tokens());
        // Retain a partial tail because future tokens still append there.
        if (end - first == config_.block_size &&
            std::all_of(evicted_.begin() + first, evicted_.begin() + end, [](bool v) {return v;})) {
            allocator.unref(table_.blocks_[i]); table_.blocks_[i] = {-1};
        }
    }
    return count;
}
Status Sequence::attach_prefix(const std::vector<BlockId>& blocks, int tokens, BlockAllocator& allocator) {
    if (!complete() || num_tokens() != 0 || tokens <= 0 || tokens > config_.max_sequence_tokens ||
        tokens % config_.block_size || blocks.size() != static_cast<size_t>(tokens / config_.block_size))
        return Status::ERROR_INVALID_ARGUMENT;
    for (auto block : blocks) if (allocator.get_ref_count(block) == 0) return Status::ERROR_BLOCK_NOT_FOUND;
    auto copy = blocks;
    std::vector<bool> live(tokens, false);
    for (auto block : copy) allocator.ref(block);
    table_.blocks_ = std::move(copy); table_.num_tokens_ = tokens; evicted_ = std::move(live);
    std::fill(layer_lengths_.begin(), layer_lengths_.end(), tokens);
    return Status::OK;
}
} // namespace kvcache
