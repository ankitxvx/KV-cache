#include "kvcache/gpu_allocator.h"

#ifdef KVCACHE_CUDA_ENABLED
#include <cuda_runtime.h>
#endif

#include <cstdlib>
#include <cstring>
#include <sstream>
#include <stdexcept>

namespace kvcache {

GpuAllocator::GpuAllocator(const ProductionConfig& config)
    : config_(config)
{
    config_.validate();
    kv_per_token_bytes_ = static_cast<size_t>(config_.num_kv_heads)
                        * config_.head_dim * dtype_size(config_.dtype);
    layer_bytes_ = static_cast<size_t>(config_.block_size) * 2 * kv_per_token_bytes_;
    block_total_bytes_ = static_cast<size_t>(config_.num_layers) * layer_bytes_;

    max_blocks_ = config_.gpu_max_blocks();
    if (max_blocks_ <= 0) {
        KVCACHE_LOG_WARN("gpu_alloc", "GPU tier disabled (0 blocks)");
        return;
    }

    size_t total_bytes = static_cast<size_t>(max_blocks_) * block_total_bytes_;

#ifdef KVCACHE_CUDA_ENABLED
    cudaError_t err = cudaMalloc(&device_pool_, total_bytes);
    if (err == cudaSuccess) {
        cuda_available_ = true;
        cudaMemset(device_pool_, 0, total_bytes);
        std::ostringstream msg;
        msg << "CUDA GPU pool allocated: " << max_blocks_ << " blocks, "
            << (total_bytes / (1024*1024)) << " MB";
        KVCACHE_LOG_INFO("gpu_alloc", msg.str());
    } else {
        if (!config.allow_gpu_emulation) throw std::runtime_error(cudaGetErrorString(err));
        KVCACHE_LOG_WARN("gpu_alloc", "cudaMalloc failed, falling back to CPU emulation");
        device_pool_ = std::calloc(1, total_bytes);
        cuda_available_ = false;
    }
#else
    if (!config.allow_gpu_emulation) throw std::runtime_error("GPU enabled but library was built without CUDA");
    // CPU fallback — allocate aligned memory to emulate GPU pool
    device_pool_ = std::calloc(1, total_bytes);
    cuda_available_ = false;
    std::ostringstream msg;
    msg << "CPU-emulated GPU pool: " << max_blocks_ << " blocks, "
        << (total_bytes / (1024*1024)) << " MB (CUDA not available)";
    KVCACHE_LOG_INFO("gpu_alloc", msg.str());
#endif

    if (!device_pool_) throw std::bad_alloc();

    ref_counts_.resize(max_blocks_, 0);
    used_slots_.resize(max_blocks_, 0);
    free_list_.reserve(max_blocks_);
    for (int i = 0; i < max_blocks_; ++i) {
        free_list_.push_back(BlockId{i});
    }
}

GpuAllocator::~GpuAllocator() {
    if (device_pool_) {
#ifdef KVCACHE_CUDA_ENABLED
        if (cuda_available_) {
            cudaFree(device_pool_);
        } else {
            std::free(device_pool_);
        }
#else
        std::free(device_pool_);
#endif
        device_pool_ = nullptr;
    }
}

BlockId GpuAllocator::allocate() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (free_list_.empty()) return BlockId{-1};

    BlockId id = free_list_.back();
    free_list_.pop_back();
    ref_counts_[id.value] = 1;
    used_slots_[id.value] = 0;
    return id;
}

void GpuAllocator::free(BlockId id) {
    if (!valid_block(id)) return;
    std::lock_guard<std::mutex> lock(mutex_);
    if (ref_counts_[id.value] == 0) return;
    if (--ref_counts_[id.value] > 0) return;
    used_slots_[id.value] = 0;
    free_list_.push_back(id);
}

size_t GpuAllocator::compute_offset(BlockId id) const {
    return static_cast<size_t>(id.value) * block_total_bytes_;
}

void* GpuAllocator::get_block_ptr(BlockId id) {
    if (!valid_block(id) || !device_pool_) return nullptr;
    return static_cast<uint8_t*>(device_pool_) + compute_offset(id);
}

const void* GpuAllocator::get_block_ptr(BlockId id) const {
    if (!valid_block(id) || !device_pool_) return nullptr;
    return static_cast<const uint8_t*>(device_pool_) + compute_offset(id);
}

KVPairMut GpuAllocator::get_kv_slot_mut(BlockId id, int layer, int slot) {
    KVPairMut result;
    if (!valid_block(id) || layer < 0 || layer >= config_.num_layers
        || slot < 0 || slot >= config_.block_size || !device_pool_) {
        return result;
    }

    size_t base = compute_offset(id);
    size_t layer_off = static_cast<size_t>(layer) * layer_bytes_;
    size_t k_off = static_cast<size_t>(slot) * kv_per_token_bytes_;
    size_t v_off = static_cast<size_t>(config_.block_size) * kv_per_token_bytes_
                 + static_cast<size_t>(slot) * kv_per_token_bytes_;

    uint8_t* ptr = static_cast<uint8_t*>(device_pool_);
    result.key   = ptr + base + layer_off + k_off;
    result.value = ptr + base + layer_off + v_off;
    return result;
}

KVPair GpuAllocator::get_kv_slot(BlockId id, int layer, int slot) const {
    KVPair result;
    if (!valid_block(id) || layer < 0 || layer >= config_.num_layers
        || slot < 0 || slot >= config_.block_size || !device_pool_) {
        return result;
    }

    size_t base = compute_offset(id);
    size_t layer_off = static_cast<size_t>(layer) * layer_bytes_;
    size_t k_off = static_cast<size_t>(slot) * kv_per_token_bytes_;
    size_t v_off = static_cast<size_t>(config_.block_size) * kv_per_token_bytes_
                 + static_cast<size_t>(slot) * kv_per_token_bytes_;

    const uint8_t* ptr = static_cast<const uint8_t*>(device_pool_);
    result.key   = ptr + base + layer_off + k_off;
    result.value = ptr + base + layer_off + v_off;
    return result;
}

void GpuAllocator::copy_to_host(BlockId id, void* host_dst) const {
    if (!valid_block(id) || !device_pool_ || !host_dst) return;
    const void* src = static_cast<const uint8_t*>(device_pool_) + compute_offset(id);
#ifdef KVCACHE_CUDA_ENABLED
    if (cuda_available_) {
        auto error = cudaMemcpy(host_dst, src, block_total_bytes_, cudaMemcpyDeviceToHost);
        if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
        return;
    }
#endif
    std::memcpy(host_dst, src, block_total_bytes_);
}

void GpuAllocator::copy_from_host(BlockId id, const void* host_src) {
    if (!valid_block(id) || !device_pool_ || !host_src) return;
    void* dst = static_cast<uint8_t*>(device_pool_) + compute_offset(id);
#ifdef KVCACHE_CUDA_ENABLED
    if (cuda_available_) {
        auto error = cudaMemcpy(dst, host_src, block_total_bytes_, cudaMemcpyHostToDevice);
        if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
        return;
    }
#endif
    std::memcpy(dst, host_src, block_total_bytes_);
}

void GpuAllocator::copy_block(BlockId dst, BlockId src) {
    if (!valid_block(dst) || !valid_block(src) || !device_pool_) return;
    void* d = static_cast<uint8_t*>(device_pool_) + compute_offset(dst);
    const void* s = static_cast<const uint8_t*>(device_pool_) + compute_offset(src);
#ifdef KVCACHE_CUDA_ENABLED
    if (cuda_available_) {
        auto error = cudaMemcpy(d, s, block_total_bytes_, cudaMemcpyDeviceToDevice);
        if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
        return;
    }
#endif
    std::memcpy(d, s, block_total_bytes_);
}

int GpuAllocator::num_free() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return static_cast<int>(free_list_.size());
}

int GpuAllocator::num_used() const {
    return max_blocks_ - num_free();
}

int GpuAllocator::get_used_slots(BlockId id) const {
    if (!valid_block(id)) return 0;
    return used_slots_[id.value];
}

void GpuAllocator::set_used_slots(BlockId id, int count) {
    if (!valid_block(id)) return;
    used_slots_[id.value] = count;
}

int GpuAllocator::get_ref_count(BlockId id) const {
    if (!valid_block(id)) return 0;
    return ref_counts_[id.value];
}

void GpuAllocator::ref(BlockId id) {
    if (!valid_block(id)) return;
    std::lock_guard<std::mutex> lock(mutex_);
    if (ref_counts_[id.value] > 0) ref_counts_[id.value]++;
}

void GpuAllocator::unref(BlockId id) {
    if (!valid_block(id)) return;
    std::lock_guard<std::mutex> lock(mutex_);
    if (ref_counts_[id.value] == 0) return;
    ref_counts_[id.value]--;
    if (ref_counts_[id.value] <= 0) {
        ref_counts_[id.value] = 0;
        used_slots_[id.value] = 0;
        free_list_.push_back(id);
    }
}

} // namespace kvcache
