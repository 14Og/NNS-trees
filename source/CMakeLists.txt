set(COMMON_DEFS
    DATA_DIR="${PROJECT_SOURCE_DIR}/data"
    OUT_DIR="${PROJECT_SOURCE_DIR}/assets/output"
)

add_executable(benchmark
    benchmark/benchmark.cc
)
target_include_directories(benchmark PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(benchmark PRIVATE ${COMMON_DEFS})
set_target_properties(benchmark PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})

# ENABLE_LOGGING is injected by the `logging` CMake preset.
# When set, Logger::log() emits events to stdout alongside normal stats output.
if(ENABLE_LOGGING)
    target_compile_definitions(benchmark PRIVATE ENABLE_LOGGING)
endif()
