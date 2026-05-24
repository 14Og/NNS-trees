#!/usr/bin/env bash
# sweep.sh — run all benchmark experiments and collect stats
#
# Experiments
# -----------
# 0. viz logs    : logging build, n=200 d=2, k=10, iters=1; all algos × all dists
# 1. n-scaling   : d=2, k=10, 200 iters; algos × dists × n values
# 2. d-scaling   : uniform, n=50000, k=10, 200 iters; exhaustive + kdtree
#
# Flags
# -----
#   --skip-build   skip cmake configure + build (both presets)
#   --skip-data    skip dataset generation (error if a required file is missing)
#
# Usage
# -----
#   ./sweep.sh
#   ./sweep.sh --skip-build
#   ./sweep.sh --skip-build --skip-data

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="$ROOT/build/release/benchmark"
BIN_LOG="$ROOT/build/logging/benchmark"
DATA="$ROOT/data"
STATS="$ROOT/assets/output/stats"
LOGS="$ROOT/assets/output/logs"
TMP="$ROOT/assets/output/stats/.correctness"

SKIP_BUILD=0
SKIP_DATA=0
for arg in "$@"; do
    case "$arg" in
        --skip-build) SKIP_BUILD=1 ;;
        --skip-data)  SKIP_DATA=1  ;;
        *) echo "Unknown flag: $arg" >&2; exit 1 ;;
    esac
done

# ── 1. Build ──────────────────────────────────────────────────────────────────
if [[ $SKIP_BUILD -eq 0 ]]; then
    echo "==> Configuring (release preset)..."
    cmake --preset release -S "$ROOT" -B "$ROOT/build/release"
    echo "==> Building (release)..."
    cmake --build "$ROOT/build/release" -j"$(nproc)" --target benchmark

    echo "==> Configuring (logging preset)..."
    cmake --preset logging -S "$ROOT" -B "$ROOT/build/logging"
    echo "==> Building (logging)..."
    cmake --build "$ROOT/build/logging" -j"$(nproc)" --target benchmark
fi

[[ -x "$BIN"     ]] || { echo "Binary not found: $BIN"     >&2; exit 1; }
[[ -x "$BIN_LOG" ]] || { echo "Binary not found: $BIN_LOG" >&2; exit 1; }
mkdir -p "$STATS" "$LOGS" "$TMP"

# ── 2. Dataset generation helper ─────────────────────────────────────────────
gen() {
    local dist="$1" n="$2" d="$3" seed="${4:-42}"
    local file="$DATA/${dist}_n${n}_d${d}_s${seed}.csv"
    if [[ ! -f "$file" ]]; then
        if [[ $SKIP_DATA -eq 1 ]]; then
            echo "Missing dataset (--skip-data set): $file" >&2
            exit 1
        fi
        echo "  gen $dist n=$n d=$d seed=$seed"
        uv run --project "$ROOT" python "$ROOT/source/data_gen.py" \
            "$dist" "$n" --dims "$d" --seed "$seed"
    fi
}

# ── 3. Benchmark helper ───────────────────────────────────────────────────────
bench() {
    local algo="$1" file="$2"
    shift 2
    echo "  bench $algo $file $*"
    "$BIN" --data "$file" --algo "$algo" "$@"
}

# ── 4. Viz logs — logging build, small 2D dataset, 1 query each ──────────────
# Produces assets/output/logs/<algo>_<dist>_n200_d2_s42.log
# Feed directly to visualize.py for build/query animations.
gen_viz_logs() {
    local dists=(uniform clustered skewed)
    local algos=(exhaustive kdtree quadtree)
    local n=200 d=2 k=10 seed=42

    echo "==> Viz logs (n=$n d=$d k=$k, logging build)..."

    for dist in "${dists[@]}"; do
        gen "$dist" $n $d $seed
    done

    for algo in "${algos[@]}"; do
        for dist in "${dists[@]}"; do
            local stem="${dist}_n${n}_d${d}_s${seed}"
            echo "  log $algo $stem"
            "$BIN_LOG" --data "${stem}.csv" --algo "$algo" \
                       --k $k --iters 1 --seed $seed
            # benchmark writes the log to LOGS/<algo>_<stem>.log automatically
        done
    done
}

# ── 5. Correctness check ──────────────────────────────────────────────────────
# Run all three algos on a tiny dataset with --save-neighbours, then compare
# the idx column (skip the -1 query row).  Uses cut+sort+diff — no Python.
check_correctness() {
    local dist="uniform" n=500 d=2 seed=42
    local stem="${dist}_n${n}_d${d}_s${seed}"
    gen "$dist" $n $d $seed

    echo "==> Correctness check ($stem, k=10, iters=1)..."

    # Run each algo and extract sorted neighbour indices to a tmp file.
    for algo in exhaustive kdtree quadtree; do
        "$BIN" --data "${stem}.csv" --algo "$algo" \
               --k 10 --iters 1 --seed $seed --save-neighbours
        # neighbours file: STATS/<algo>_<stem>_neighbours.csv
        local neigh="$STATS/${algo}_${stem}_neighbours.csv"
        [[ -f "$neigh" ]] || { echo "Missing neighbours file: $neigh" >&2; exit 1; }
        # col 1 is idx; skip header (NR>1) and query row (idx=-1), sort numerically
        awk -F',' 'NR>1 && $1!=-1 {print $1}' "$neigh" | sort -n > "$TMP/${algo}.ids"
    done

    # Compare kdtree and quadtree against exhaustive baseline.
    local ok=1
    for algo in kdtree quadtree; do
        if ! diff -q "$TMP/exhaustive.ids" "$TMP/${algo}.ids" > /dev/null; then
            echo "  FAIL: $algo neighbours differ from exhaustive" >&2
            echo "  exhaustive: $(cat "$TMP/exhaustive.ids" | tr '\n' ' ')" >&2
            echo "  $algo:      $(cat "$TMP/${algo}.ids"   | tr '\n' ' ')" >&2
            ok=0
        fi
    done
    [[ $ok -eq 1 ]] && echo "  OK — exhaustive / kdtree / quadtree agree"
    rm -rf "$TMP"
    [[ $ok -eq 1 ]] || exit 1
}

# ── 5. Experiment 1: n-scaling (d=2) ─────────────────────────────────────────
exp1_n_scaling() {
    local dists=(uniform clustered skewed)
    local algos=(exhaustive kdtree quadtree)
    local ns=(500 1000 2000 5000 10000 50000 100000 200000)
    local d=2 k=10 iters=200 seed=42

    echo "==> Experiment 1: n-scaling (d=$d, k=$k, iters=$iters)"

    for dist in "${dists[@]}"; do
        for n in "${ns[@]}"; do
            gen "$dist" "$n" "$d" "$seed"
        done
    done

    for algo in "${algos[@]}"; do
        for dist in "${dists[@]}"; do
            echo "  [$algo / $dist]"
            for n in "${ns[@]}"; do
                bench "$algo" "${dist}_n${n}_d${d}_s${seed}.csv" \
                    --k "$k" --iters "$iters" --seed "$seed"
            done
        done
    done
}

# ── 6. Experiment 2: d-scaling (uniform, n=50000) ────────────────────────────
exp2_d_scaling() {
    local algos=(exhaustive kdtree)
    local ds=(2 5 10 20 30)
    local dist=uniform n=50000 k=10 iters=200 seed=42

    echo "==> Experiment 2: d-scaling (n=$n, $dist, k=$k, iters=$iters)"

    for d in "${ds[@]}"; do
        gen "$dist" "$n" "$d" "$seed"
    done

    for algo in "${algos[@]}"; do
        echo "  [$algo / $dist]"
        for d in "${ds[@]}"; do
            bench "$algo" "${dist}_n${n}_d${d}_s${seed}.csv" \
                --k "$k" --iters "$iters" --seed "$seed"
        done
    done
}

# ── Run everything ─────────────────────────────────────────────────────────────
gen_viz_logs
check_correctness
exp1_n_scaling
exp2_d_scaling

echo ""
echo "Done. Stats written to $STATS/"
echo "==> Generating plots..."
uv run --project "$ROOT" python "$ROOT/source/benchmark_plot.py" --save
echo "Plots written to $ROOT/assets/figures/"
