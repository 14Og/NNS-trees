add_executable(benchmark
    benchmark/benchmark.cc
    # quadtree/quadtree.cc  — header-only, no .cc needed
    # kdtree/kdtree.cc      — header-only, no .cc needed
)


target_include_directories(benchmark PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(benchmark PRIVATE DATA_DIR="${PROJECT_SOURCE_DIR}/data")
target_compile_definitions(benchmark PRIVATE OUT_DIR="${PROJECT_SOURCE_DIR}/assets/output")