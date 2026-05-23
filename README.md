# NNS-Trees

Benchmarking nearest-neighbour search algorithms (Exhaustive KNN, KD-Tree, QuadTree)
with Python-based animation of build and query traversal.

---

## Current Status

### Algorithms

| Algorithm | Build | Query | Notes |
|---|---|---|---|
| `ExhaustiveKNN` | ✅ | ✅ | Linear scan, max-heap; correctness reference |
| `KDTree` | ✅ | ✅ | Median split via `nth_element`, pruning by hyperplane distance |
| `QuadTree` | 🔲 | 🔲 | Not yet started |

### Infrastructure

| Component | Status | Notes |
|---|---|---|
| `NNSIndex` interface | ✅ | `build(PointsPtr)` + `query(Point, k) → QueryResult` |
| `Benchmark` harness | ✅ | Timed build/query, CSV stats + neighbours output |
| Logger with compile-time switch | ✅ | Zero overhead in `release`; enabled via `logging` CMake preset |
| CMake presets | ✅ | `release`, `debug`, `logging` — separate `build/<preset>/` dirs |
| Data generator | ✅ | Uniform / Clustered / Skewed, arbitrary `n`/`d`/seed |
| Git hooks | ✅ | `clang-format` + `black` + `isort` on commit |

### Visualiser ([source/visualize.py](source/visualize.py))

| Feature | Status | Notes |
|---|---|---|
| Dataset scatter plots | ✅ | `scatter` subcommand, multi-file side-by-side |
| KD-Tree build animation | ✅ | Partition lines coloured by split axis (red = x, blue = y) |
| KD-Tree query animation | ✅ | Visited / accepted / evicted / result states; acceptance radius circle; pruned cells shaded |
| Exhaustive KNN query animation | ✅ | Same event model as KD-Tree |
| Save to `assets/figures/` | ✅ | `--save` flag; filenames derived from log stem + stage |
| QuadTree animation | 🔲 | Pending QuadTree implementation |

#### KD-Tree Build — uniform n=200 d=2

<img src="assets/figures/kdtree_uniform_n200_d2_s42_build.gif" width="380"/>

#### KD-Tree Query — uniform n=200 d=2

<img src="assets/figures/kdtree_uniform_n200_d2_s42_query.gif" width="380"/>

#### Exhaustive KNN Query — uniform n=200 d=2

<img src="assets/figures/exhaustive_uniform_n200_d2_s42_query.gif" width="380"/>

---

## Quick Start

### Prerequisites

| Tool | Purpose |
|---|---|
| CMake ≥ 3.20, Make | Build system |
| GCC / Clang with C++20 | Compiler |
| Python ≥ 3.12 | Data generation + visualisation |
| `uv` | Python environment and tool management |

### Python environment

```bash
uv sync                              # install runtime deps (numpy, matplotlib, scienceplots)
uv tool install black isort          # formatting tools used by the git hook
```

### Build

```bash
# Optimised build (no logging overhead)
cmake --preset release && cmake --build build/release

# Debug build
cmake --preset debug && cmake --build build/debug

# Logging build — same optimisation as release but emits event logs
cmake --preset logging && cmake --build build/logging
```

### Generate data

```bash
uv run source/data_gen.py --dist uniform   --n 2000 --dims 2 --seed 42
uv run source/data_gen.py --dist clustered --n 2000 --dims 2 --seed 42
uv run source/data_gen.py --dist skewed    --n 2000 --dims 2 --seed 42
# Output: data/<dist>_n<n>_d<dims>_s<seed>.csv
```

### Run a benchmark

```bash
./build/release/benchmark \
    --data clustered_n2000_d2_s42.csv \
    --algo kdtree \
    --k 5 --iters 100
# Output: assets/output/kdtree_clustered_n2000_d2_s42_stats.csv
```

### Produce animations

```bash
# 1. Build with logging enabled
cmake --preset logging && cmake --build build/logging

# 2. Run benchmark to emit the event log (1 iteration is enough for animation)
./build/logging/benchmark \
    --data uniform_n200_d2_s42.csv \
    --algo kdtree --k 5 --iters 1
# Log: assets/output/logs/kdtree_uniform_n200_d2_s42.log

# 3. Animate interactively (both build + query)
uv run source/visualize.py animate \
    --log  assets/output/logs/kdtree_uniform_n200_d2_s42.log \
    --data data/uniform_n200_d2_s42.csv

# 4. Save GIFs to assets/figures/
uv run source/visualize.py animate --save \
    --log  assets/output/logs/kdtree_uniform_n200_d2_s42.log \
    --data data/uniform_n200_d2_s42.csv
```

---

## Architecture

### Source layout

```
source/
├── generic/
│   ├── types.hh          # Point, Points, Neighbour, QueryResult
│   ├── index.hh          # NNSIndex abstract base class
│   ├── functions.hh      # L2norm (squared, avoids sqrt on every comparison)
│   └── logger.hh         # Logger (zero cost without ENABLE_LOGGING)
├── exhaustive_knn/
│   └── exhaustive_knn.hh # Header-only ExhaustiveKNN
├── kd_tree/
│   ├── kd_node.hh        # KDNode struct + KDNodePtr
│   └── kd_tree.hh        # Header-only KDTree
├── benchmark/
│   └── benchmark.cc      # Single TU; includes all algorithm headers
└── visualize.py          # Scatter plots + build/query animations
```

### NNSIndex interface

Every algorithm inherits from `NNSIndex` ([source/generic/index.hh](source/generic/index.hh))
and implements two methods:

```cpp
class NNSIndex {
public:
    virtual void        build(PointsPtr aPoints) = 0;
    virtual QueryResult query(const Point &aQ, size_t aK) const = 0;
protected:
    PointsPtr points;   // shared ownership of the dataset
};

struct QueryResult {
    Neighbours neighbours;   // k nearest neighbours, sorted ascending by L2
    size_t     nodesVisited; // reported in benchmark stats
};
```

Core types ([source/generic/types.hh](source/generic/types.hh)):

```cpp
using Point      = std::vector<float>;
using Points     = std::vector<Point>;
using PointsPtr  = std::shared_ptr<Points>;

struct Neighbour {
    float  dist; // squared L2 during search; true L2 in the final result
    size_t idx;  // index into the original Points array
    bool operator<(const Neighbour &) const; // max-heap order (largest dist on top)
};
```

### Adding a new algorithm (e.g. QuadTree)

1. Create `source/quad_tree/quad_tree.hh` — header-only, include `"generic/index.hh"`.
2. Inherit from `NNSIndex`, implement `build()` and `query()`.
3. Call `Logger` methods at the right points (see [source/kd_tree/kd_tree.hh](source/kd_tree/kd_tree.hh) as reference).
4. Add `#include "quad_tree/quad_tree.hh"` and a `"quadtree"` branch in
   [source/benchmark/benchmark.cc](source/benchmark/benchmark.cc).
5. Add `"quadtree"` to `_ALGO_TITLES` in [source/visualize.py](source/visualize.py).

### Logger

`Logger` ([source/generic/logger.hh](source/generic/logger.hh)) is a compile-time
switch — all calls compile to nothing unless the `logging` CMake preset is used (zero overhead in `release`):

```cpp
Logger::split(nodeIdx, axis, depth, loBound, hiBound); // build: node created
Logger::visit(idx);                                     // query: node examined
Logger::accept(idx, heapWorstDistSq);                   // query: added to heap
Logger::evict(idx, heapWorstDistSq);                    // query: evicted from heap
Logger::prune(subtreeRootIdx, hyperplaneDistSq);        // query: subtree pruned
Logger::result(idx, dist);                              // query: final neighbour
```

Call `Logger::init(path)` once before `build()`/`query()` to open the log file.
The log is consumed by `visualize.py animate` to produce animations.

### Datasets

All datasets are generated in $[0,1]^d$ by [source/data_gen.py](source/data_gen.py):

| Distribution | Description | Stress target |
|---|---|---|
| **Uniform** | i.i.d. from $U([0,1]^d)$ | Clean asymptotic baseline |
| **Clustered** | Gaussian blobs, σ=0.05, clipped to unit box | Spatial-midpoint splits (QuadTree) |
| **Skewed** | All points in $[0, 0.1]^d$ | Worst-case for midpoint partitioning |

### Benchmark pipeline

```
data/<stem>.csv  →  benchmark binary  →  assets/output/<algo>_<stem>_stats.csv
                                     →  assets/output/logs/<algo>_<stem>.log  (logging preset only)
```

```bash
./build/release/benchmark --data <stem>.csv --algo <name> [--k 5] [--iters 100]
                          [--noise 0.01] [--seed 42] [--save-neighbours]
```

Query points are drawn by sampling a random dataset point and adding
$\mathcal{N}(0, \texttt{noise})$ per dimension, clamped to $[0,1]^d$.

---

## Development Setup

### Git hooks (one-time, per clone)

Pre-commit hooks for auto-formatting are tracked in [`hooks/`](hooks/).
After cloning, activate them with:

```bash
git config core.hooksPath hooks
```

The hook runs on every `git commit`:
- **C/C++** (`.cc .cpp .cxx .hh .h .hpp`) — `clang-format` using [`.clang-format`](.clang-format)
- **Python** (`.py`) — `isort` then `black`

Required tools: `clang-format` (system), `black` and `isort` (see [Python environment](#python-environment) above).
