#include "kvcache/ssd_allocator.h"
#include <sstream>
#include <cstring>

namespace kvcache {

SsdAllocator::SsdAllocator(const ProductionConfig& config)
    : config_(config)
    , file_path_(config.ssd_cache_path)
{
    config_.validate();
    kv_per_token_bytes_ = static_cast<size_t>(config_.num_kv_heads)
                        * config_.head_dim * dtype_size(config_.dtype);
    layer_bytes_ = static_cast<size_t>(config_.block_size) * 2 * kv_per_token_bytes_;
    block_total_bytes_ = static_cast<size_t>(config_.num_layers) * layer_bytes_;

    max_blocks_ = config_.ssd_max_blocks();
    if (max_blocks_ <= 0) {
        KVCACHE_LOG_WARN("ssd_alloc", "SSD tier disabled (0 blocks)");
        return;
    }

    total_file_bytes_ = static_cast<size_t>(max_blocks_) * block_total_bytes_;

#ifdef _WIN32
    // Windows: CreateFile + CreateFileMapping + MapViewOfFile
    file_handle_ = CreateFileA(
        file_path_.c_str(),
        GENERIC_READ | GENERIC_WRITE,
        FILE_SHARE_READ,
        nullptr,
        OPEN_ALWAYS,
        FILE_ATTRIBUTE_NORMAL | FILE_FLAG_RANDOM_ACCESS,
        nullptr
    );

    if (file_handle_ == INVALID_HANDLE_VALUE) {
        KVCACHE_LOG_ERROR("ssd_alloc", "Failed to create SSD pool file: " + file_path_);
        max_blocks_ = 0;
        return;
    }

    // Set file size
    LARGE_INTEGER file_size;
    file_size.QuadPart = static_cast<LONGLONG>(total_file_bytes_);
    SetFilePointerEx(file_handle_, file_size, nullptr, FILE_BEGIN);
    SetEndOfFile(file_handle_);

    // Create file mapping
    DWORD high = static_cast<DWORD>(total_file_bytes_ >> 32);
    DWORD low  = static_cast<DWORD>(total_file_bytes_ & 0xFFFFFFFF);
    mapping_handle_ = CreateFileMappingA(
        file_handle_, nullptr, PAGE_READWRITE, high, low, nullptr
    );

    if (!mapping_handle_) {
        KVCACHE_LOG_ERROR("ssd_alloc", "Failed to create file mapping");
        CloseHandle(file_handle_);
        file_handle_ = INVALID_HANDLE_VALUE;
        max_blocks_ = 0;
        return;
    }

    mapped_ptr_ = static_cast<uint8_t*>(
        MapViewOfFile(mapping_handle_, FILE_MAP_ALL_ACCESS, 0, 0, total_file_bytes_)
    );

    if (!mapped_ptr_) {
        KVCACHE_LOG_ERROR("ssd_alloc", "Failed to map view of file");
        CloseHandle(mapping_handle_);
        CloseHandle(file_handle_);
        mapping_handle_ = nullptr;
        file_handle_ = INVALID_HANDLE_VALUE;
        max_blocks_ = 0;
        return;
    }
#else
    // POSIX: open + ftruncate + mmap
    fd_ = open(file_path_.c_str(), O_RDWR | O_CREAT, 0644);
    if (fd_ < 0) {
        KVCACHE_LOG_ERROR("ssd_alloc", "Failed to create SSD pool file: " + file_path_);
        max_blocks_ = 0;
        return;
    }

    if (ftruncate(fd_, static_cast<off_t>(total_file_bytes_)) != 0) {
        KVCACHE_LOG_ERROR("ssd_alloc", "Failed to set SSD pool file size");
        close(fd_);
        fd_ = -1;
        max_blocks_ = 0;
        return;
    }

    mapped_ptr_ = static_cast<uint8_t*>(
        mmap(nullptr, total_file_bytes_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0)
    );

    if (mapped_ptr_ == MAP_FAILED) {
        mapped_ptr_ = nullptr;
        KVCACHE_LOG_ERROR("ssd_alloc", "Failed to mmap SSD pool file");
        close(fd_);
        fd_ = -1;
        max_blocks_ = 0;
        return;
    }
#endif

    used_slots_.resize(max_blocks_, 0);
    allocated_.resize(max_blocks_, false);
    free_list_.reserve(max_blocks_);
    for (int i = 0; i < max_blocks_; ++i) {
        free_list_.push_back(BlockId{i});
    }

    std::ostringstream msg;
    msg << "SSD pool mapped: " << max_blocks_ << " blocks, "
        << (total_file_bytes_ / (1024*1024)) << " MB at " << file_path_;
    KVCACHE_LOG_INFO("ssd_alloc", msg.str());
}

SsdAllocator::~SsdAllocator() {
#ifdef _WIN32
    if (mapped_ptr_) {
        UnmapViewOfFile(mapped_ptr_);
        mapped_ptr_ = nullptr;
    }
    if (mapping_handle_) {
        CloseHandle(mapping_handle_);
        mapping_handle_ = nullptr;
    }
    if (file_handle_ != INVALID_HANDLE_VALUE) {
        CloseHandle(file_handle_);
        file_handle_ = INVALID_HANDLE_VALUE;
    }
#else
    if (mapped_ptr_) {
        munmap(mapped_ptr_, total_file_bytes_);
        mapped_ptr_ = nullptr;
    }
    if (fd_ >= 0) {
        close(fd_);
        fd_ = -1;
    }
#endif
}

BlockId SsdAllocator::allocate() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (free_list_.empty()) return BlockId{-1};

    BlockId id = free_list_.back();
    free_list_.pop_back();
    allocated_[id.value] = true;
    used_slots_[id.value] = 0;
    return id;
}

void SsdAllocator::free(BlockId id) {
    if (!valid_block(id)) return;
    std::lock_guard<std::mutex> lock(mutex_);
    if (!allocated_[id.value]) return;
    allocated_[id.value] = false;
    used_slots_[id.value] = 0;
    free_list_.push_back(id);
}

void* SsdAllocator::get_block_ptr(BlockId id) {
    if (!valid_block(id) || !mapped_ptr_) return nullptr;
    return mapped_ptr_ + static_cast<size_t>(id.value) * block_total_bytes_;
}

const void* SsdAllocator::get_block_ptr(BlockId id) const {
    if (!valid_block(id) || !mapped_ptr_) return nullptr;
    return mapped_ptr_ + static_cast<size_t>(id.value) * block_total_bytes_;
}

KVPairMut SsdAllocator::get_kv_slot_mut(BlockId id, int layer, int slot) {
    KVPairMut result;
    if (!valid_block(id) || layer < 0 || layer >= config_.num_layers
        || slot < 0 || slot >= config_.block_size || !mapped_ptr_) {
        return result;
    }

    size_t base = static_cast<size_t>(id.value) * block_total_bytes_;
    size_t layer_off = static_cast<size_t>(layer) * layer_bytes_;
    size_t k_off = static_cast<size_t>(slot) * kv_per_token_bytes_;
    size_t v_off = static_cast<size_t>(config_.block_size) * kv_per_token_bytes_
                 + static_cast<size_t>(slot) * kv_per_token_bytes_;

    result.key   = mapped_ptr_ + base + layer_off + k_off;
    result.value = mapped_ptr_ + base + layer_off + v_off;
    return result;
}

KVPair SsdAllocator::get_kv_slot(BlockId id, int layer, int slot) const {
    KVPair result;
    if (!valid_block(id) || layer < 0 || layer >= config_.num_layers
        || slot < 0 || slot >= config_.block_size || !mapped_ptr_) {
        return result;
    }

    size_t base = static_cast<size_t>(id.value) * block_total_bytes_;
    size_t layer_off = static_cast<size_t>(layer) * layer_bytes_;
    size_t k_off = static_cast<size_t>(slot) * kv_per_token_bytes_;
    size_t v_off = static_cast<size_t>(config_.block_size) * kv_per_token_bytes_
                 + static_cast<size_t>(slot) * kv_per_token_bytes_;

    result.key   = mapped_ptr_ + base + layer_off + k_off;
    result.value = mapped_ptr_ + base + layer_off + v_off;
    return result;
}

void SsdAllocator::copy_to_buffer(BlockId id, void* dst) const {
    if (!valid_block(id) || !mapped_ptr_ || !dst) return;
    const void* src = mapped_ptr_ + static_cast<size_t>(id.value) * block_total_bytes_;
    std::memcpy(dst, src, block_total_bytes_);
}

void SsdAllocator::copy_from_buffer(BlockId id, const void* src) {
    if (!valid_block(id) || !mapped_ptr_ || !src) return;
    void* dst = mapped_ptr_ + static_cast<size_t>(id.value) * block_total_bytes_;
    std::memcpy(dst, src, block_total_bytes_);
}

int SsdAllocator::num_free() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return static_cast<int>(free_list_.size());
}

int SsdAllocator::num_used() const {
    return max_blocks_ - num_free();
}

int SsdAllocator::get_used_slots(BlockId id) const {
    if (!valid_block(id)) return 0;
    return used_slots_[id.value];
}

void SsdAllocator::set_used_slots(BlockId id, int count) {
    if (!valid_block(id)) return;
    used_slots_[id.value] = count;
}

} // namespace kvcache
