add_executable(benchmark
    benchmark/benchmark.cc
    exhaustive_knn/exhaustive_knn.cc
    # quadtree/quadtree.cc
    # kdtree/kdtree.cc
)
target_include_directories(benchmark PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(benchmark PRIVATE cxx_std_20)
