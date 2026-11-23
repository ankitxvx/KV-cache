#include "kvcache/prefix_cache.h"
#include <functional>

namespace kvcache {

PrefixCache::PrefixCache(int block_size, BlockAllocator* allocator, size_t max_entries)
    : block_size_(block_size), allocator_(allocator), max_entries_(max_entries)
{
    if (block_size <= 0) throw std::invalid_argument("block_size must be positive");
}
PrefixCache::~PrefixCache() { clear(); }
void PrefixCache::release(const PrefixEntry& entry) {
    if (allocator_) for (auto block : entry.block_ids) allocator_->unref(block);
}
bool PrefixCache::evict_one() {
    if (cache_.empty()) return false;
    auto it = cache_.begin(); release(it->second); cache_.erase(it); return true;
}

size_t PrefixCache::hash_tokens(const int* token_ids, int count) const {
    // FNV-1a hash over token IDs
    size_t hash = 14695981039346656037ULL;  // FNV offset basis
    for (int i = 0; i < count; ++i) {
        hash ^= static_cast<size_t>(token_ids[i]);
        hash *= 1099511628211ULL;  // FNV prime
    }
    return hash;
}

int PrefixCache::try_match(const int* token_ids, int length,
                            std::vector<BlockId>& out_blocks) const {
    out_blocks.clear();

    if (!token_ids || length <= 0) {
        misses_++;
        return 0;
    }

    // Try matching progressively longer prefixes at block boundaries
    int best_match = 0;

    for (int prefix_len = block_size_; prefix_len <= length; prefix_len += block_size_) {
        size_t h = hash_tokens(token_ids, prefix_len);
        auto it = cache_.find(h);

        const PrefixEntry* matched_entry = nullptr;
        if (it != cache_.end()) {
            matched_entry = &it->second;
        } else {
            // Check if any longer cached entry has this prefix
            for (const auto& pair : cache_) {
                if (static_cast<int>(pair.second.token_ids.size()) >= prefix_len) {
                    bool match = true;
                    for (int i = 0; i < prefix_len; ++i) {
                        if (pair.second.token_ids[i] != token_ids[i]) {
                            match = false;
                            break;
                        }
                    }
                    if (match) {
                        matched_entry = &pair.second;
                        break;
                    }
                }
            }
        }

        if (matched_entry) {
            // Verify the tokens actually match (hash collision guard)
            if (static_cast<int>(matched_entry->token_ids.size()) >= prefix_len) {
                bool match = true;
                for (int i = 0; i < prefix_len; ++i) {
                    if (matched_entry->token_ids[i] != token_ids[i]) {
                        match = false;
                        break;
                    }
                }
                if (match) {
                    int num_blocks_needed = prefix_len / block_size_;
                    if (num_blocks_needed <= static_cast<int>(matched_entry->block_ids.size())) {
                        out_blocks.assign(matched_entry->block_ids.begin(),
                                          matched_entry->block_ids.begin() + num_blocks_needed);
                        best_match = prefix_len;
                    }
                }
            }
        }
    }

    if (best_match > 0) {
        hits_++;
    } else {
        misses_++;
    }

    return best_match;
}

void PrefixCache::register_prefix(const int* token_ids, int length,
                                   const std::vector<BlockId>& block_ids) {
    if (!token_ids || length <= 0 || max_entries_ == 0) return;

    // Only register complete blocks
    int registerable_tokens = (length / block_size_) * block_size_;
    if (registerable_tokens <= 0) return;

    int num_blocks = registerable_tokens / block_size_;
    if (num_blocks > static_cast<int>(block_ids.size())) {
        num_blocks = static_cast<int>(block_ids.size());
        registerable_tokens = num_blocks * block_size_;
    }

    size_t h = hash_tokens(token_ids, registerable_tokens);

    PrefixEntry entry;
    entry.token_ids.assign(token_ids, token_ids + registerable_tokens);
    entry.block_ids.assign(block_ids.begin(), block_ids.begin() + num_blocks);

    if (num_blocks == 0) return;
    if (allocator_) for (auto b : entry.block_ids)
        if (allocator_->get_ref_count(b) == 0) return;
    auto old = cache_.find(h);
    // Hold references before replacement, including overlapping prefixes.
    if (allocator_) for (auto b : entry.block_ids) allocator_->ref(b);
    try {
        if (old != cache_.end()) { release(old->second); old->second = std::move(entry); }
        else {
            while (cache_.size() >= max_entries_) evict_one();
            cache_.emplace(h, std::move(entry));
        }
    } catch (...) { release(entry); throw; }
}

void PrefixCache::evict(size_t hash) {
    auto it = cache_.find(hash);
    if (it != cache_.end()) { release(it->second); cache_.erase(it); }
}

void PrefixCache::clear() {
    for (const auto& pair : cache_) release(pair.second);
    cache_.clear();
    hits_ = 0;
    misses_ = 0;
}

} // namespace kvcache
