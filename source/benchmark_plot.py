"""
benchmark_plot.py — visualise sweep results from assets/output/stats/

Figures
-------
1. Query time vs n  (1×3 grid, log-log, all algos, one column per distribution)
2. Side-by-side:
   left  — Nodes visited vs n  (uniform, log-log, all algos)
   right — Query time vs d     (uniform n=50000, exhaustive vs kdtree)

Usage
-----
    python source/benchmark_plot.py            # show live
    python source/benchmark_plot.py --save     # save to assets/figures/
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import scienceplots  # noqa: F401 — registers the style

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
STATS_DIR = ROOT / "assets" / "output" / "stats"
FIGURES_DIR = ROOT / "assets" / "figures"

plt.style.use(["science", "grid"])
plt.rcParams["figure.dpi"] = 200

# ── style ──────────────────────────────────────────────────────────────────────
ALGO_STYLE = {
    "exhaustive": dict(color="#e06c75", marker="o", label="Exhaustive"),
    "kdtree": dict(color="#61afef", marker="s", label="KD-Tree"),
    "quadtree": dict(color="#55a868", marker="^", label="QuadTree"),
}
DIST_TITLES = {
    "uniform": "Uniform",
    "clustered": "Clustered",
    "skewed": "Skewed",
}


# ── data loading ───────────────────────────────────────────────────────────────
def load_all() -> pd.DataFrame:
    frames = []
    for path in STATS_DIR.glob("*_stats.csv"):
        stem = path.stem  # e.g. kdtree_uniform_stats  →  kdtree_uniform
        parts = stem.replace("_stats", "").split("_", 1)
        if len(parts) != 2:
            continue
        algo, dist = parts
        df = pd.read_csv(path)
        df["algo"] = algo
        df["dist"] = dist
        frames.append(df)
    if not frames:
        print(f"No stats CSVs found in {STATS_DIR}", file=sys.stderr)
        sys.exit(1)
    return pd.concat(frames, ignore_index=True)


# ── figure 1: query time vs n ─────────────────────────────────────────────────
def fig_query_vs_n(df: pd.DataFrame) -> plt.Figure:
    # n-scaling rows: d=2, iters=200
    data = df[(df["d"] == 2) & (df["iters"] == 200)].copy()
    dists = ["uniform", "clustered", "skewed"]
    algos = ["exhaustive", "kdtree", "quadtree"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)
    fig.suptitle(r"Query time vs dataset size  ($d$=2, $k$=10, 200 iters)", fontsize=12)

    for ax, dist in zip(axes, dists):
        for algo in algos:
            sub = data[(data["dist"] == dist) & (data["algo"] == algo)].sort_values("n")
            if sub.empty:
                continue
            st = ALGO_STYLE[algo]
            ax.plot(
                sub["n"],
                sub["query_mean_us"],
                color=st["color"],
                marker=st["marker"],
                label=st["label"],
                linewidth=1.5,
                markersize=4,
            )
            ax.fill_between(
                sub["n"],
                sub["query_mean_us"] - sub["query_std_us"],
                sub["query_mean_us"] + sub["query_std_us"],
                color=st["color"],
                alpha=0.15,
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(DIST_TITLES[dist])
        ax.set_xlabel("$n$")
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        if ax is axes[0]:
            ax.set_ylabel(r"Query time ($\mu$s)")

    handles, labels = axes[0].get_legend_handles_labels()
    if not handles:
        handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.05),
        frameon=False,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    return fig


# ── figure 2: nodes visited + d-scaling ───────────────────────────────────────
def fig_nodes_and_dims(df: pd.DataFrame) -> plt.Figure:
    fig, (ax_nodes, ax_d) = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle(r"Pruning efficiency \& curse of dimensionality", fontsize=12)

    # left: nodes visited vs n (uniform, d=2, iters=200)
    data_n = df[(df["dist"] == "uniform") & (df["d"] == 2) & (df["iters"] == 200)]
    for algo in ["exhaustive", "kdtree", "quadtree"]:
        sub = data_n[data_n["algo"] == algo].sort_values("n")
        if sub.empty:
            continue
        st = ALGO_STYLE[algo]
        ax_nodes.plot(
            sub["n"],
            sub["nodes_mean"],
            color=st["color"],
            marker=st["marker"],
            label=st["label"],
            linewidth=1.5,
            markersize=4,
        )
        ax_nodes.fill_between(
            sub["n"],
            sub["nodes_mean"] - sub["nodes_std"],
            sub["nodes_mean"] + sub["nodes_std"],
            color=st["color"],
            alpha=0.15,
        )
    ax_nodes.set_xscale("log")
    ax_nodes.set_yscale("log")
    ax_nodes.set_xlabel("$n$")
    ax_nodes.set_ylabel("Nodes visited per query")
    ax_nodes.set_title(r"Nodes visited vs $n$  (uniform, $d$=2)")
    ax_nodes.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax_nodes.legend(frameon=False)

    # right: query time vs d (uniform, n=50000, iters=200)
    data_d = df[
        (df["dist"] == "uniform") & (df["n"] == 50000) & (df["iters"] == 200)
    ].copy()
    for algo in ["exhaustive", "kdtree"]:
        sub = data_d[data_d["algo"] == algo].sort_values("d")
        if sub.empty:
            continue
        st = ALGO_STYLE[algo]
        ax_d.plot(
            sub["d"],
            sub["query_mean_us"],
            color=st["color"],
            marker=st["marker"],
            label=st["label"],
            linewidth=1.5,
            markersize=4,
        )
        ax_d.fill_between(
            sub["d"],
            sub["query_mean_us"] - sub["query_std_us"],
            sub["query_mean_us"] + sub["query_std_us"],
            color=st["color"],
            alpha=0.15,
        )
    ax_d.set_xlabel("Dimensions ($d$)")
    ax_d.set_ylabel(r"Query time ($\mu$s)")
    ax_d.set_title(r"Query time vs $d$  (uniform, $n$=50\,000)")
    ax_d.set_xticks([2, 5, 10, 20, 30])
    ax_d.legend(frameon=False)

    fig.tight_layout()
    return fig


# ── main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save figures to assets/figures/ instead of showing live.",
    )
    args = parser.parse_args()

    df = load_all()

    fig1 = fig_query_vs_n(df)
    fig2 = fig_nodes_and_dims(df)

    if args.save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        p1 = FIGURES_DIR / "bench_query_vs_n.png"
        p2 = FIGURES_DIR / "bench_nodes_and_dims.png"
        fig1.savefig(p1, bbox_inches="tight")
        fig2.savefig(p2, bbox_inches="tight")
        print(f"saved → {p1}")
        print(f"saved → {p2}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
