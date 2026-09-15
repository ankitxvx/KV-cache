cmake_minimum_required(VERSION 3.16)
project(KVCache VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler warnings
if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Release optimization
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2")
else()
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
endif()

# ── Core Library ──────────────────────────────────────────────────────────
set(KVCACHE_SOURCES
    src/block_allocator.cpp
    src/block_table.cpp
    src/sequence.cpp
    src/eviction_policy.cpp
    src/quantization.cpp
    src/prefix_cache.cpp
    src/cache_manager.cpp
)

add_library(kvcache STATIC ${KVCACHE_SOURCES})
target_include_directories(kvcache PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# ── CLI Tool ──────────────────────────────────────────────────────────────
add_executable(kvcache_cli cli/main.cpp)
target_link_libraries(kvcache_cli PRIVATE kvcache)

# ── Tests ─────────────────────────────────────────────────────────────────
enable_testing()
add_subdirectory(tests)

# ── Benchmarks ────────────────────────────────────────────────────────────
add_subdirectory(benchmarks)

# ── Install ───────────────────────────────────────────────────────────────
install(TARGETS kvcache ARCHIVE DESTINATION lib)
install(DIRECTORY include/kvcache DESTINATION include)
install(TARGETS kvcache_cli RUNTIME DESTINATION bin)
