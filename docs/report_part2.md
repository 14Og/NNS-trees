# Part II: KD-Tree & QuadTree for k-NN Search

## Implementation

**KD-Tree** (`source/kd_tree/kd_tree.hh`) — recursive binary tree, header-only C++20.
Each node stores the split axis (highest variance dimension), split value (median), and
bounding box lo/hi. Build selects the pivot by `nth_element` at the median index.
Query uses a max-heap of size k; a subtree is pruned when the distance from the query
point to the node's bounding box exceeds the current heap worst.

**QuadTree** (`source/quad_tree/quad_tree.hh`) — 2D-only recursive structure, same interface.
Each node is split into four equal quadrants at the geometric midpoint; cells with ≤ 8 points
or depth ≥ 32 become leaves. Query pruning is the same box-distance rule applied to each
quadrant's bounding box; children are visited in order of increasing box distance.

Both implement `NNSIndex` (`source/generic/index.hh`):

```cpp
virtual void build(PointsPtr aPoints) = 0;
virtual QueryResult query(const Point &aQ, size_t aK) const = 0;
```

**Logger** (`source/generic/logger.hh`) — compile-time–switched event emitter
(`ENABLE_LOGGING` CMake flag). Emits structured tokens (`SPLIT`, `QUAD`, `PRUNE`,
`PRUNEBOX`, `VISIT`, `ACCEPT`, `EVICT`, `RESULT`) consumed by the visualiser.
Zero overhead in release builds (`if constexpr`).

**Benchmark** (`source/benchmark/`) — measures build time, mean/std query latency, and
mean/std nodes visited over configurable iterations. Stats accumulate per `(algo, distribution)`
in CSV files. `sweep.sh` automates data generation, correctness check (neighbour index
comparison via `diff`), and two experiments.

**Visualiser** (`source/visualize.py`) — `matplotlib.FuncAnimation` driven by event logs.
Renders KD-Tree axis-split lines, QuadTree midpoint crosses (depth-coloured), and query
animations (visited/accepted/pruned points, expanding search radius).

---

## Visualisation

Final frames of build and query animations on uniform n=200, d=2.

| KD-Tree build | KD-Tree query |
|:---:|:---:|
| ![](../assets/figures/kdtree_uniform_n200_d2_s42_build_last.png) | ![](../assets/figures/kdtree_uniform_n200_d2_s42_query_last.png) |

| QuadTree build | QuadTree query |
|:---:|:---:|
| ![](../assets/figures/quadtree_uniform_n200_d2_s42_build_last.png) | ![](../assets/figures/quadtree_uniform_n200_d2_s42_query_last.png) |

---

## Benchmark Results

All runs: k=10, 200 iterations, seed=42.

### Query latency vs n — d=2

![](../assets/figures/bench_query_vs_n.png)

| n | Exhaustive | KD-Tree | QuadTree | KD speedup | Quad speedup |
|--:|--:|--:|--:|--:|--:|
| 1 000 | ~13 μs | ~4 μs | ~4 μs | 3× | 3× |
| 10 000 | ~52 μs | ~5 μs | ~5 μs | 10× | 10× |
| 100 000 | ~390 μs | ~7 μs | ~7 μs | 56× | 56× |
| 200 000 | ~648 μs | ~7.5 μs | ~7.0 μs | **86×** | **93×** |

### Pruning efficiency & curse of dimensionality

![](../assets/figures/bench_nodes_and_dims.png)

Nodes visited at n=200 000, d=2, uniform:

| Algorithm | Nodes visited | % of dataset |
|:--|--:|--:|
| Exhaustive | 200 000 | 100% |
| KD-Tree | ~53 | 0.03% |
| QuadTree | ~31 | 0.02% |

### Effect of distribution — n=200 000, d=2

| Distribution | Exhaustive | KD-Tree | QuadTree |
|:--|--:|--:|--:|
| Uniform | ~645 μs | ~7.5 μs | ~7.0 μs |
| Clustered | ~643 μs | ~7.3 μs | ~6.7 μs |
| Skewed | ~652 μs | ~9.1 μs | ~6.1 μs |

KD-Tree degrades slightly on skewed data (unbalanced median splits).
QuadTree is unaffected — the dense corner gets isolated quickly regardless.

### KD-Tree: curse of dimensionality — uniform, n=50 000

| d | Exhaustive | KD-Tree | KD speedup |
|--:|--:|--:|--:|
| 2 | ~220 μs | ~7 μs | 31× |
| 5 | ~275 μs | ~30 μs | 9× |
| 10 | ~298 μs | ~620 μs | 0.5× |
| 20 | ~535 μs | ~1 750 μs | 0.3× |
| 30 | ~824 μs | ~2 060 μs | 0.4× |

KD-Tree breaks even with exhaustive around d=8–10; at d=20+ it visits
essentially the entire dataset and becomes strictly slower.
