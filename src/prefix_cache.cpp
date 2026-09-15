#include "kvcache/prefix_cache.h"
#include <functional>

namespace kvcache {

PrefixCache::PrefixCache(int block_size)
    : block_size_(block_size)
{
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

        if (it != cache_.end()) {
            // Verify the tokens actually match (hash collision guard)
            const auto& entry = it->second;
            if (static_cast<int>(entry.token_ids.size()) >= prefix_len) {
                bool match = true;
                for (int i = 0; i < prefix_len; ++i) {
                    if (entry.token_ids[i] != token_ids[i]) {
                        match = false;
                        break;
                    }
                }
                if (match) {
                    int num_blocks_needed = prefix_len / block_size_;
                    if (num_blocks_needed <= static_cast<int>(entry.block_ids.size())) {
                        out_blocks.assign(entry.block_ids.begin(),
                                          entry.block_ids.begin() + num_blocks_needed);
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
    if (!token_ids || length <= 0) return;

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

    cache_[h] = std::move(entry);
}

void PrefixCache::evict(size_t hash) {
    cache_.erase(hash);
}

void PrefixCache::clear() {
    cache_.clear();
    hits_ = 0;
    misses_ = 0;
}

} // namespace kvcache
