#pragma once
#include "config.h"
#include "block_allocator.h"
#include "gpu_allocator.h"
#include "ssd_allocator.h"
#include "metrics.h"
#include <memory>
#include <unordered_map>
#include <mutex>

namespace kvcache {
/// Stable handles survive migration. Raw pointer views require external exclusion
/// of free/migration; use read_block/write_block for concurrent access.
class TierManager {
public:
    explicit TierManager(const ProductionConfig& config);
    TieredBlockId allocate_block();
    TieredBlockId allocate_in_tier(TierType tier);
    void free_block(TieredBlockId id);
    TieredBlockId promote(TieredBlockId id, TierType target);
    TieredBlockId demote(TieredBlockId id, TierType target);
    void touch(TieredBlockId id);
    int auto_rebalance();
    void* get_block_data(TieredBlockId id);
    const void* get_block_data(TieredBlockId id) const;
    KVPairMut get_kv_slot_mut(TieredBlockId id, int layer, int slot);
    KVPair get_kv_slot(TieredBlockId id, int layer, int slot) const;
    bool copy_block_data(TieredBlockId dst, TieredBlockId src);
    bool read_block(TieredBlockId id, void* dst, size_t bytes) const;
    bool write_block(TieredBlockId id, const void* src, size_t bytes);
    bool pin_pair(TieredBlockId src, TieredBlockId dst);
    void unpin_pair(TieredBlockId src, TieredBlockId dst);
    TierStats get_tier_stats(TierType tier) const;
    CacheStats get_total_stats() const;
    TierType get_block_tier(int64_t global_id) const;
    TieredBlockId resolve(TieredBlockId id) const;
    GpuAllocator* gpu_allocator() {return gpu_.get();}
    BlockAllocator* cpu_allocator() {return cpu_.get();}
    SsdAllocator* ssd_allocator() {return ssd_.get();}
    const ProductionConfig& config() const {return config_;}
private:
    struct Location { TieredBlockId id; uint64_t accessed=0; int pins=0; bool released=false; };
    BlockId allocate_physical(TierType tier);
    void free_physical(TieredBlockId id);
    TieredBlockId move(TieredBlockId id, TierType target);
    bool evict_one(TierType tier);
    void unpin(TieredBlockId id);
    ProductionConfig config_;
    std::unique_ptr<BlockAllocator> cpu_;
    std::unique_ptr<GpuAllocator> gpu_;
    std::unique_ptr<SsdAllocator> ssd_;
    std::unordered_map<int64_t, Location> locations_;
    int64_t next_id_=0;
    uint64_t clock_=0;
    mutable std::recursive_mutex mutex_;
};
}
