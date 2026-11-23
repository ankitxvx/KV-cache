cmake_minimum_required(VERSION 3.16)
project(PageForge VERSION 2.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── CUDA detection (optional) ────────────────────────────────────────────
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(KVCACHE_CUDA_ENABLED ON)
    add_compile_definitions(KVCACHE_CUDA_ENABLED)
    message(STATUS "CUDA found — GPU acceleration enabled")
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
else()
    message(STATUS "CUDA not found — building CPU-only mode")
    set(KVCACHE_CUDA_ENABLED OFF)
endif()

# ── Compiler warnings ────────────────────────────────────────────────────
if(MSVC)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/W4>)
else()
    # These are C++ compiler flags, not nvcc driver flags.
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>")
endif()

# ── Release optimization ─────────────────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2")
else()
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
endif()

# ── Core Library (v1) ────────────────────────────────────────────────────
set(KVCACHE_CORE_SOURCES
    src/block_allocator.cpp
    src/block_table.cpp
    src/sequence.cpp
    src/eviction_policy.cpp
    src/quantization.cpp
    src/prefix_cache.cpp
    src/cache_manager.cpp
)

# ── Production Layer (v2) ────────────────────────────────────────────────
set(KVCACHE_PROD_SOURCES
    src/logger.cpp
    src/metrics.cpp
    src/gpu_allocator.cpp
    src/ssd_allocator.cpp
    src/tier_manager.cpp
    src/transfer_engine.cpp
    src/attention.cpp
    src/scheduler.cpp
)

# ── CUDA sources ─────────────────────────────────────────────────────────
set(KVCACHE_CUDA_SOURCES "")
if(KVCACHE_CUDA_ENABLED)
    set(KVCACHE_CUDA_SOURCES
        src/cuda/paged_attention.cu
    )
endif()

# ── Build the library ────────────────────────────────────────────────────
add_library(pageforge STATIC
    ${KVCACHE_CORE_SOURCES}
    ${KVCACHE_PROD_SOURCES}
    ${KVCACHE_CUDA_SOURCES}
)
target_include_directories(pageforge PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

if(KVCACHE_CUDA_ENABLED)
    set_target_properties(pageforge PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_include_directories(pageforge PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()

# ── Threading ─────────────────────────────────────────────────────────────
find_package(Threads REQUIRED)
target_link_libraries(pageforge PUBLIC Threads::Threads)

# ── Platform-specific linking ─────────────────────────────────────────────
if(WIN32)
    target_link_libraries(pageforge PUBLIC ws2_32)
endif()

# ── CLI Tool ──────────────────────────────────────────────────────────────
add_executable(pageforge_cli cli/main.cpp)
target_link_libraries(pageforge_cli PRIVATE pageforge)

# ── API Server ────────────────────────────────────────────────────────────
add_executable(pageforge_server server/main.cpp)
target_link_libraries(pageforge_server PRIVATE pageforge)

# ── Tests ─────────────────────────────────────────────────────────────────
enable_testing()
add_subdirectory(tests)

# ── Benchmarks ────────────────────────────────────────────────────────────
add_subdirectory(benchmarks)

# ── Install ───────────────────────────────────────────────────────────────
install(TARGETS pageforge ARCHIVE DESTINATION lib)
install(DIRECTORY include/kvcache DESTINATION include)
install(TARGETS pageforge_cli pageforge_server RUNTIME DESTINATION bin)
