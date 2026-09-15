# Lightweight test framework — no external dependencies
# Each test file is a standalone executable

set(TEST_SOURCES
    test_block_allocator.cpp
    test_block_table.cpp
    test_sequence.cpp
    test_cache_manager.cpp
    test_eviction.cpp
    test_quantization.cpp
    test_prefix_cache.cpp
)

foreach(test_source ${TEST_SOURCES})
    get_filename_component(test_name ${test_source} NAME_WE)
    add_executable(${test_name} ${test_source})
    target_link_libraries(${test_name} PRIVATE kvcache)
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()
