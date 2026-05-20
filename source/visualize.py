"""
visualize.py — dataset and benchmark result visualizer for NNS project

Current scope: scatter-plot generated point clouds (one subplot per file).
Future scope:  benchmark result plots (query time vs n, vs dims, etc.)
               and per-query traversal animations — added as new free functions.

Usage
-----
#Plot one or more datasets
python visualize.py data/uniform_n1000_d2_s42.csv data/clustered_n1000_d2_s42.csv

#Save figure instead of interactive show
python visualize.py data/*.csv --output figures/distributions.png

# Override subplot titles
python visualize.py data/uniform_n1000_d2_s42.csv --titles "Uniform 1k"
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_points(path: Path) -> np.ndarray:
    """Load a headerless CSV of floats into a (n, dims) float32 array."""
    return np.loadtxt(path, delimiter=",", dtype=np.float32)


# One colour per dataset when multiple files are plotted together
_PALETTE = [
    "#4c72b0",  # blue
    "#dd8452",  # orange
    "#55a868",  # green
    "#c44e52",  # red
    "#8172b3",  # purple
    "#937860",  # brown
]


def scatter_2d(
    ax: plt.Axes,
    pts: np.ndarray,
    *,
    title: str = "",
    color: str = "#4c72b0",
    alpha: float = 0.8,
    point_size: float = 4.0,
) -> None:
    """
    Scatter-plot pts on ax using the first two dimensions.

    If pts has more than 2 dimensions the plot uses dims 0 and 1 and
    notes this in the axis label so it's not silently misleading.
    """
    x, y = pts[:, 0], pts[:, 1]

    ax.scatter(x, y, s=point_size, alpha=alpha, color=color, linewidths=0)
    ax.set_aspect("equal")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    projected = pts.shape[1] > 2
    ax.set_xlabel("dim 0" + ("  (projection)" if projected else ""), fontsize=9)
    ax.set_ylabel("dim 1" + ("  (projection)" if projected else ""), fontsize=9)

    if title:
        subtitle = f"n={len(pts):,}  d={pts.shape[1]}"
        ax.set_title(f"{title}\n{subtitle}", fontsize=10)


def plot_datasets(
    datasets: list[tuple[str, np.ndarray]],
    *,
    suptitle: str = "",
    output: Path | None = None,
) -> None:
    """
    Plot each (title, pts) pair as a subplot in a single row.

    Parameters
    ----------
    datasets : list of (title, array) pairs
    suptitle : optional figure-level title
    output   : save path; if None the figure is shown interactively
    """
    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.5), squeeze=False)
    axes = axes[0]  # unwrap the single row → shape (n,)

    for i, (title, pts) in enumerate(datasets):
        scatter_2d(axes[i], pts, title=title, color=_PALETTE[i % len(_PALETTE)])

    if suptitle:
        fig.suptitle(suptitle, fontsize=12, y=1.01)

    fig.tight_layout()

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=150, bbox_inches="tight")
        print(f"Saved → {output}")
    else:
        plt.show()

    plt.close(fig)


def _stem_title(path: Path) -> str:
    """Turn 'uniform_n1000_d2_s42' into a readable subplot title."""
    parts = path.stem.split("_")
    dist = parts[0].capitalize()
    tags = "  ".join(parts[1:])
    return f"{dist}\n{tags}" if tags else dist


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Visualize NNS point-cloud datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "files",
        nargs="+",
        type=Path,
        metavar="CSV",
        help="One or more dataset CSV files to plot.",
    )
    p.add_argument(
        "--titles",
        nargs="*",
        metavar="TITLE",
        default=None,
        help="Subplot titles, one per file. Defaults to the filename stem.",
    )
    p.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        metavar="PATH",
        help="Save figure to this path instead of showing interactively.",
    )
    p.add_argument(
        "--suptitle",
        type=str,
        default="",
        metavar="TEXT",
        help="Optional figure-level title.",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()

    if args.titles is not None and len(args.titles) != len(args.files):
        print(
            f"error: --titles has {len(args.titles)} entries "
            f"but {len(args.files)} files were given.",
            file=sys.stderr,
        )
        sys.exit(1)

    datasets: list[tuple[str, np.ndarray]] = []
    for i, path in enumerate(args.files):
        if not path.exists():
            print(f"error: file not found: {path}", file=sys.stderr)
            sys.exit(1)
        pts = load_points(path)
        if pts.ndim == 1:
            pts = pts.reshape(-1, 1)
        title = args.titles[i] if args.titles else _stem_title(path)
        datasets.append((title, pts))

    plot_datasets(datasets, suptitle=args.suptitle, output=args.output)


if __name__ == "__main__":
    main()
