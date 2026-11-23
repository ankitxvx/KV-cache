#include "kvcache/tier_manager.h"
#include <cstring>
#include <stdexcept>

namespace kvcache {
TierManager::TierManager(const ProductionConfig& config):config_(config) {
    config_.validate();
    cpu_=std::make_unique<BlockAllocator>(config.to_kvcache_config());
    if(config.enable_gpu && config.gpu_max_blocks()) gpu_=std::make_unique<GpuAllocator>(config);
    if(config.enable_ssd && config.ssd_max_blocks()) {
        ssd_=std::make_unique<SsdAllocator>(config);
        if(!ssd_->is_mapped()) throw std::runtime_error("SSD pool mapping failed");
    }
}
BlockId TierManager::allocate_physical(TierType tier) {
    switch(tier) {
        case TierType::GPU_VRAM:return gpu_?gpu_->allocate():BlockId{};
        case TierType::CPU_DRAM:return cpu_->allocate();
        case TierType::NVME_SSD:return ssd_?ssd_->allocate():BlockId{};
    }
    return {};
}
void TierManager::free_physical(TieredBlockId id) {
    switch(id.tier) {
        case TierType::GPU_VRAM:gpu_->free(id.block_id);break;
        case TierType::CPU_DRAM:cpu_->free(id.block_id);break;
        case TierType::NVME_SSD:ssd_->free(id.block_id);break;
    }
}
TieredBlockId TierManager::allocate_in_tier(TierType tier) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    auto block=allocate_physical(tier); if(!block.valid()) return {};
    TieredBlockId id{block,tier,next_id_++};
    try {locations_.emplace(id.global_id,Location{id,++clock_,0,false});}
    catch(...) {free_physical(id);throw;}
    return id;
}
TieredBlockId TierManager::allocate_block() {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    for(auto tier:{TierType::GPU_VRAM,TierType::CPU_DRAM,TierType::NVME_SSD}) {
        auto id=allocate_in_tier(tier); if(id.valid()) return id;
    }
    return {};
}
TieredBlockId TierManager::resolve(TieredBlockId id) const {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    auto it=locations_.find(id.global_id);
    return it==locations_.end()?TieredBlockId{}:it->second.id;
}
void TierManager::free_block(TieredBlockId id) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    auto it=locations_.find(id.global_id);if(it==locations_.end()) return;
    if(it->second.pins) {it->second.released=true;return;}
    free_physical(it->second.id);locations_.erase(it);
}
void TierManager::touch(TieredBlockId id) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    auto it=locations_.find(id.global_id);if(it!=locations_.end()) it->second.accessed=++clock_;
}
TieredBlockId TierManager::move(TieredBlockId id,TierType target) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    auto it=locations_.find(id.global_id);if(it==locations_.end()) return {};
    id=it->second.id;
    if(id.tier==target || it->second.pins || it->second.released) return id;
    auto dst=allocate_in_tier(target);if(!dst.valid()) return id;
    try {if(!copy_block_data(dst,id)) {free_block(dst);return id;}}
    catch(...) {free_block(dst);throw;}
    // allocate_in_tier may rehash locations_; reacquire the iterator.
    locations_.erase(dst.global_id); dst.global_id=id.global_id;
    locations_.at(id.global_id).id=dst; free_physical(id);
    return dst;
}
TieredBlockId TierManager::promote(TieredBlockId id,TierType target) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    auto original=resolve(id);if(!original.valid()) return {};
    if(static_cast<int>(target)>=static_cast<int>(original.tier)) return original;
    auto result=move(id,target);
    if(result.tier==original.tier && evict_one(target)) result=move(id,target);
    if(result.tier!=original.tier) MetricsRegistry::instance().total_promotions.increment();
    return result;
}
TieredBlockId TierManager::demote(TieredBlockId id,TierType target) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    auto original=resolve(id);if(!original.valid()) return {};
    if(static_cast<int>(target)<=static_cast<int>(original.tier)) return original;
    auto result=move(id,target);
    if(result.tier!=original.tier) MetricsRegistry::instance().total_demotions.increment();
    return result;
}
bool TierManager::evict_one(TierType tier) {
    if(tier==TierType::NVME_SSD) return false;
    TieredBlockId victim;uint64_t oldest=UINT64_MAX;
    for(const auto& pair:locations_) {
        const auto& l=pair.second;
        if(l.id.tier==tier && !l.pins && !l.released && l.accessed<oldest) {victim=l.id;oldest=l.accessed;}
    }
    if(!victim.valid()) return false;
    auto target=tier==TierType::GPU_VRAM?TierType::CPU_DRAM:TierType::NVME_SSD;
    auto result=demote(victim,target);
    if(result.tier==tier && target==TierType::CPU_DRAM && evict_one(target)) result=demote(victim,target);
    if(result.tier==tier) return false;
    MetricsRegistry::instance().total_evictions.increment();return true;
}
int TierManager::auto_rebalance() {
    std::lock_guard<std::recursive_mutex> lock(mutex_);int moved=0;
    for(auto tier:{TierType::CPU_DRAM,TierType::GPU_VRAM}) {
        double threshold=tier==TierType::GPU_VRAM?config_.gpu_eviction_threshold:config_.cpu_eviction_threshold;
        while(get_tier_stats(tier).utilization_pct>threshold*100 && evict_one(tier)) ++moved;
    }
    return moved;
}
void* TierManager::get_block_data(TieredBlockId id) {
    return const_cast<void*>(static_cast<const TierManager*>(this)->get_block_data(id));
}
const void* TierManager::get_block_data(TieredBlockId id) const {
    std::lock_guard<std::recursive_mutex> lock(mutex_);id=resolve(id);if(!id.valid()) return nullptr;
    switch(id.tier) {
        case TierType::GPU_VRAM:return gpu_->get_block_ptr(id.block_id);
        case TierType::CPU_DRAM:return cpu_->get_block_ptr(id.block_id);
        case TierType::NVME_SSD:return ssd_->get_block_ptr(id.block_id);
    }
    return nullptr;
}
KVPair TierManager::get_kv_slot(TieredBlockId id,int layer,int slot) const {
    std::lock_guard<std::recursive_mutex> lock(mutex_);id=resolve(id);if(!id.valid()) return {};
    switch(id.tier) {
        case TierType::GPU_VRAM:return gpu_->get_kv_slot(id.block_id,layer,slot);
        case TierType::CPU_DRAM:return cpu_->get_kv_slot(id.block_id,layer,slot);
        case TierType::NVME_SSD:return ssd_->get_kv_slot(id.block_id,layer,slot);
    }
    return {};
}
KVPairMut TierManager::get_kv_slot_mut(TieredBlockId id,int layer,int slot) {
    auto p=get_kv_slot(id,layer,slot);return {const_cast<void*>(p.key),const_cast<void*>(p.value)};
}
bool TierManager::read_block(TieredBlockId id,void* dst,size_t bytes) const {
    std::lock_guard<std::recursive_mutex> lock(mutex_);id=resolve(id);
    if(!id.valid() || !dst || bytes!=config_.block_total_bytes()) return false;
    if(id.tier==TierType::GPU_VRAM) gpu_->copy_to_host(id.block_id,dst);
    else std::memcpy(dst,get_block_data(id),bytes);
    return true;
}
bool TierManager::write_block(TieredBlockId id,const void* src,size_t bytes) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);id=resolve(id);
    if(!id.valid() || !src || bytes!=config_.block_total_bytes()) return false;
    if(id.tier==TierType::GPU_VRAM) gpu_->copy_from_host(id.block_id,src);
    else std::memcpy(get_block_data(id),src,bytes);
    touch(id);return true;
}
bool TierManager::copy_block_data(TieredBlockId dst,TieredBlockId src) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);dst=resolve(dst);src=resolve(src);
    if(!dst.valid() || !src.valid()) return false;
    if(dst.global_id==src.global_id) return true;
    if(dst.tier==TierType::GPU_VRAM && src.tier==TierType::GPU_VRAM) gpu_->copy_block(dst.block_id,src.block_id);
    else if(src.tier==TierType::GPU_VRAM) gpu_->copy_to_host(src.block_id,get_block_data(dst));
    else if(dst.tier==TierType::GPU_VRAM) gpu_->copy_from_host(dst.block_id,get_block_data(src));
    else std::memcpy(get_block_data(dst),get_block_data(src),config_.block_total_bytes());
    return true;
}
bool TierManager::pin_pair(TieredBlockId src,TieredBlockId dst) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    auto s=locations_.find(src.global_id),d=locations_.find(dst.global_id);
    if(s==locations_.end() || d==locations_.end() || s->second.released || d->second.released) return false;
    ++s->second.pins;++d->second.pins;return true;
}
void TierManager::unpin(TieredBlockId id) {
    auto it=locations_.find(id.global_id);if(it==locations_.end() || !it->second.pins) return;
    if(--it->second.pins==0 && it->second.released) free_block(id);
}
void TierManager::unpin_pair(TieredBlockId src,TieredBlockId dst) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);unpin(src);unpin(dst);
}
TierType TierManager::get_block_tier(int64_t id) const {
    std::lock_guard<std::recursive_mutex> lock(mutex_);return locations_.at(id).id.tier;
}
TierStats TierManager::get_tier_stats(TierType tier) const {
    std::lock_guard<std::recursive_mutex> lock(mutex_);TierStats s{};s.tier=tier;
    switch(tier) {
        case TierType::GPU_VRAM:if(gpu_) {s.total_blocks=gpu_->num_total();s.free_blocks=gpu_->num_free();}break;
        case TierType::CPU_DRAM:s.total_blocks=cpu_->num_total();s.free_blocks=cpu_->num_free();break;
        case TierType::NVME_SSD:if(ssd_) {s.total_blocks=ssd_->num_total();s.free_blocks=ssd_->num_free();}break;
    }
    s.used_blocks=s.total_blocks-s.free_blocks;s.total_bytes=size_t(s.total_blocks)*config_.block_total_bytes();
    s.used_bytes=size_t(s.used_blocks)*config_.block_total_bytes();
    s.utilization_pct=s.total_blocks?100.0*s.used_blocks/s.total_blocks:0;
    return s;
}
CacheStats TierManager::get_total_stats() const {
    std::lock_guard<std::recursive_mutex> lock(mutex_);CacheStats s;
    for(auto tier:{TierType::GPU_VRAM,TierType::CPU_DRAM,TierType::NVME_SSD}) {
        auto t=get_tier_stats(tier);s.total_blocks+=t.total_blocks;s.used_blocks+=t.used_blocks;s.free_blocks+=t.free_blocks;
        s.total_memory_bytes+=t.total_bytes;s.used_memory_bytes+=t.used_bytes;
    }
    s.utilization_pct=s.total_blocks?100.0*s.used_blocks/s.total_blocks:0;return s;
}
}
