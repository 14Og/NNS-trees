"""
visualize.py — NNS dataset scatter plots and algorithm animations.

Subcommands
-----------
scatter  Scatter-plot one or more point-cloud CSV files.
animate  Animate KD-tree build or KNN query from a C++ event log.

Usage
-----
# Plot datasets
python visualize.py scatter data/uniform_n200_d2_s42.csv data/clustered_n200_d2_s42.csv

# Animate KD-tree build + query (both modes, interactive)
python visualize.py animate \\
    --log  assets/output/logs/kdtree_clustered_n200_d2_s42.log \\
    --data data/clustered_n200_d2_s42.csv

# Save both animations to assets/figures/ (filenames auto-derived from log stem):
#   assets/figures/kdtree_clustered_n200_d2_s42_build.gif
#   assets/figures/kdtree_clustered_n200_d2_s42_query.gif
python visualize.py animate --save \\
    --log  assets/output/logs/kdtree_clustered_n200_d2_s42.log \\
    --data data/clustered_n200_d2_s42.csv
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import scienceplots
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle

plt.style.use("science")

# ---------------------------------------------------------------------------
# Point cloud utilities
# ---------------------------------------------------------------------------

_PALETTE = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3", "#937860"]


def load_points(path: Path) -> np.ndarray:
    """Load a headerless CSV of floats into an (n, d) float32 array."""
    return np.loadtxt(path, delimiter=",", dtype=np.float32)


def scatter_2d(
    ax: plt.Axes,
    pts: np.ndarray,
    *,
    title: str = "",
    color: str = "#4c72b0",
    alpha: float = 0.8,
    point_size: float = 4.0,
) -> None:
    ax.scatter(
        pts[:, 0], pts[:, 1], s=point_size, alpha=alpha, color=color, linewidths=0
    )
    ax.set_aspect("equal")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    projected = pts.shape[1] > 2
    ax.set_xlabel("dim 0" + ("  (projection)" if projected else ""), fontsize=9)
    ax.set_ylabel("dim 1" + ("  (projection)" if projected else ""), fontsize=9)
    if title:
        ax.set_title(f"{title}\nn={len(pts):,}  d={pts.shape[1]}", fontsize=10)


def plot_datasets(
    datasets: list[tuple[str, np.ndarray]],
    *,
    suptitle: str = "",
    output: Path | None = None,
) -> None:
    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.5), squeeze=False)
    for i, (title, pts) in enumerate(datasets):
        scatter_2d(axes[0][i], pts, title=title, color=_PALETTE[i % len(_PALETTE)])
    if suptitle:
        fig.suptitle(suptitle, fontsize=12, y=1.01)
    fig.tight_layout()
    _save_or_show(fig, output)


def _stem_title(path: Path) -> str:
    parts = path.stem.split("_")
    dist = parts[0].capitalize()
    tags = "  ".join(parts[1:])
    return f"{dist}\n{tags}" if tags else dist


# ---------------------------------------------------------------------------
# Log event types
# ---------------------------------------------------------------------------


@dataclass
class SplitEvent:
    idx: int
    axis: int
    depth: int
    lo: np.ndarray  # shape (d,)
    hi: np.ndarray  # shape (d,)


@dataclass
class QueryPointEvent:
    coords: np.ndarray  # shape (d,)


@dataclass
class VisitEvent:
    idx: int


@dataclass
class AcceptEvent:
    idx: int
    dist_sq: float  # squared L2 (what C++ logs before sqrt)


@dataclass
class EvictEvent:
    idx: int
    dist_sq: float


@dataclass
class PruneEvent:
    idx: int  # root of pruned subtree (KD-Tree)
    diff_sq: float  # squared distance to splitting hyperplane


@dataclass
class PruneBoxEvent:
    dist_sq: float  # squared distance to cell box (QuadTree)
    lo: np.ndarray  # shape (2,)
    hi: np.ndarray  # shape (2,)


@dataclass
class QuadEvent:
    depth: int
    lo: np.ndarray  # shape (2,)
    hi: np.ndarray  # shape (2,)


@dataclass
class ResultEvent:
    idx: int
    dist: float  # final L2 distance (after sqrt)


# ---------------------------------------------------------------------------
# Log parser
# ---------------------------------------------------------------------------


def parse_log(
    path: Path,
) -> tuple[list[SplitEvent], list]:
    """
    Returns (build_events, query_events).

    query_events is a flat list of QueryPointEvent / VisitEvent / AcceptEvent
    / EvictEvent / PruneEvent / ResultEvent in log order.
    """
    build_events: list[SplitEvent] = []
    query_events: list = []

    with open(path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            kind = parts[0]

            if kind == "SPLIT":
                idx, axis, depth = int(parts[1]), int(parts[2]), int(parts[3])
                d = (len(parts) - 4) // 2
                lo = np.array(parts[4 : 4 + d], dtype=np.float32)
                hi = np.array(parts[4 + d :], dtype=np.float32)
                build_events.append(SplitEvent(idx, axis, depth, lo, hi))

            elif kind == "QUERY":
                coords = np.array(parts[1:], dtype=np.float32)
                query_events.append(QueryPointEvent(coords))

            elif kind == "VISIT":
                query_events.append(VisitEvent(int(parts[1])))
            elif kind == "ACCEPT":
                query_events.append(AcceptEvent(int(parts[1]), float(parts[2])))
            elif kind == "EVICT":
                query_events.append(EvictEvent(int(parts[1]), float(parts[2])))
            elif kind == "PRUNE":
                query_events.append(PruneEvent(int(parts[1]), float(parts[2])))
            elif kind == "PRUNEBOX":
                # QuadTree: PRUNEBOX distSq lo0 lo1 hi0 hi1
                lo = np.array(parts[2:4], dtype=np.float32)
                hi = np.array(parts[4:6], dtype=np.float32)
                query_events.append(PruneBoxEvent(float(parts[1]), lo, hi))
            elif kind == "QUAD":
                # QuadTree: QUAD depth lo0 lo1 hi0 hi1
                lo = np.array(parts[2:4], dtype=np.float32)
                hi = np.array(parts[4:6], dtype=np.float32)
                build_events.append(QuadEvent(int(parts[1]), lo, hi))
            elif kind == "RESULT":
                query_events.append(ResultEvent(int(parts[1]), float(parts[2])))

    return build_events, query_events


# ---------------------------------------------------------------------------
# Shared animation helpers
# ---------------------------------------------------------------------------


def _setup_ax(ax: plt.Axes) -> None:
    ax.set_aspect("equal")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("x0", fontsize=9)
    ax.set_ylabel("x1", fontsize=9)


def _draw_split_line(
    ax: plt.Axes,
    pts: np.ndarray,
    e: SplitEvent,
    *,
    color,
    alpha: float = 0.8,
    lw: float = 1.0,
):
    """Draw partition line for a SPLIT event clipped to its cell bounds."""
    x, y = pts[e.idx, 0], pts[e.idx, 1]
    if e.axis == 0:  # vertical line at x, clipped to cell's y range
        return ax.plot([x, x], [e.lo[1], e.hi[1]], color=color, lw=lw, alpha=alpha)[0]
    else:  # horizontal line at y, clipped to cell's x range
        return ax.plot([e.lo[0], e.hi[0]], [y, y], color=color, lw=lw, alpha=alpha)[0]


def _save_or_show(fig_or_anim, output: Path | None, *, fps: int = 15) -> None:
    """Save animation/figure to file, or show interactively."""
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(fig_or_anim, FuncAnimation):
            fig_or_anim.save(str(output), fps=fps, dpi=150)
        else:
            fig_or_anim.savefig(str(output), dpi=150, bbox_inches="tight")
        print(f"Saved → {output}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Build animation
# ---------------------------------------------------------------------------

_AXIS_COLORS = ["#e06c75", "#61afef"]  # axis 0 → red, axis 1 → blue
_DEPTH_COLORS = ["#e06c75", "#61afef", "#55a868", "#c44e52", "#8172b3", "#937860"]


def _draw_quad_split(
    ax: plt.Axes,
    e: QuadEvent,
    *,
    color,
    alpha: float = 0.8,
    lw: float = 1.0,
) -> tuple:
    """Draw the two midpoint lines (vertical + horizontal) for a QUAD event."""
    mid_x = (e.lo[0] + e.hi[0]) * 0.5
    mid_y = (e.lo[1] + e.hi[1]) * 0.5
    lv = ax.plot([mid_x, mid_x], [e.lo[1], e.hi[1]], color=color, lw=lw, alpha=alpha)[0]
    lh = ax.plot([e.lo[0], e.hi[0]], [mid_y, mid_y], color=color, lw=lw, alpha=alpha)[0]
    return lv, lh


def animate_build(
    pts: np.ndarray,
    events: list,
    *,
    title: str = "Build",
    output: Path | None = None,
    interval: int = 150,
) -> None:
    """
    Animate tree build — handles both SplitEvent (KD-Tree) and QuadEvent (QuadTree).

    KD-Tree: one line per split, coloured by axis.
    QuadTree: two midpoint lines per subdivision, coloured by depth.
    """
    if not events:
        print("No build events in log — nothing to animate.")
        return

    is_quad = isinstance(events[0], QuadEvent)

    fig, ax = plt.subplots(figsize=(6, 6))
    _setup_ax(ax)
    ax.scatter(pts[:, 0], pts[:, 1], s=10, alpha=0.75, color="#555555", linewidths=0)
    title_artist = ax.set_title(title)

    if is_quad:
        # Depth-coloured legend for QuadTree
        for i, label in enumerate(["depth 0", "depth 1", "depth 2+"]):
            ax.plot([], [], color=_DEPTH_COLORS[i], lw=1.5, label=label)
    else:
        for i, label in enumerate(["axis 0 (vertical)", "axis 1 (horizontal)"]):
            ax.plot([], [], color=_AXIS_COLORS[i], lw=1.5, label=label)
    ax.legend(loc="upper right", fontsize=8)

    artists: list = []
    total = len(events)

    def update(frame: int):
        e = events[frame]
        if isinstance(e, QuadEvent):
            color = _DEPTH_COLORS[e.depth % len(_DEPTH_COLORS)]
            lv, lh = _draw_quad_split(ax, e, color=color)
            artists.extend([lv, lh])
        else:
            color = _AXIS_COLORS[e.axis % len(_AXIS_COLORS)]
            line = _draw_split_line(ax, pts, e, color=color)
            dot = ax.plot(
                pts[e.idx, 0], pts[e.idx, 1], "o", color=color, markersize=5, zorder=5
            )[0]
            artists.extend([line, dot])
        title_artist.set_text(f"{title}  [{frame + 1}/{total}]")
        return artists

    anim = FuncAnimation(
        fig, update, frames=total, interval=interval, blit=False, repeat=False
    )
    _save_or_show(anim, output, fps=max(1, 1000 // interval))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Query animation
# ---------------------------------------------------------------------------

_C_UNVISITED = np.array([0.80, 0.80, 0.80, 0.70])
_C_VISITED = np.array([0.45, 0.65, 0.85, 0.80])
_C_ACCEPTED = np.array([0.20, 0.65, 0.30, 0.90])
_C_EVICTED = np.array([0.75, 0.40, 0.35, 0.60])
_C_RESULT = np.array([0.05, 0.45, 0.10, 1.00])


def animate_query(
    pts: np.ndarray,
    build_events: list[SplitEvent],
    query_events: list,
    *,
    title: str = "KNN Query",
    output: Path | None = None,
    interval: int = 80,
) -> None:
    """
    Animate KNN query: visited nodes light up, accepted nodes go green,
    evicted nodes dim, pruned subtree cells are shaded orange, and a dashed
    circle shows the current acceptance radius.
    Partition lines from the build are drawn as a faint background.
    """
    # Extract query point; remaining events drive the animation
    query_pt = None
    events = []
    for e in query_events:
        if isinstance(e, QueryPointEvent):
            query_pt = e.coords
        else:
            events.append(e)

    if query_pt is None:
        print("No QUERY event in log — cannot determine query point.")
        return

    # Index KD-Tree split events by point id for O(1) prune lookup.
    split_by_idx = {e.idx: e for e in build_events if isinstance(e, SplitEvent)}

    fig, ax = plt.subplots(figsize=(6, 6))
    _setup_ax(ax)
    title_artist = ax.set_title(title)

    # Faint partition lines from build phase
    for e in build_events:
        if isinstance(e, QuadEvent):
            color = _DEPTH_COLORS[e.depth % len(_DEPTH_COLORS)]
            _draw_quad_split(ax, e, color=color, alpha=0.18, lw=0.7)
        else:
            _draw_split_line(
                ax,
                pts,
                e,
                color=_AXIS_COLORS[e.axis % len(_AXIS_COLORS)],
                alpha=0.18,
                lw=0.7,
            )

    # Point scatter — facecolors updated each frame
    colors = np.tile(_C_UNVISITED, (len(pts), 1))
    scat = ax.scatter(pts[:, 0], pts[:, 1], c=colors, s=10, zorder=3)

    # Query point marker
    ax.plot(query_pt[0], query_pt[1], "r*", markersize=14, zorder=10, label="query")

    # Acceptance radius circle (starts with radius 0)
    circle = Circle(
        query_pt[:2], 0.0, fill=False, color="crimson", lw=1.5, ls="--", zorder=8
    )
    ax.add_patch(circle)

    ax.legend(loc="upper right", fontsize=8)

    total = len(events)

    def update(frame: int):
        e = events[frame]
        title_artist.set_text(f"{title}  [{frame + 1}/{total}]")

        if isinstance(e, VisitEvent):
            if not np.array_equal(colors[e.idx], _C_ACCEPTED):
                colors[e.idx] = _C_VISITED
        elif isinstance(e, AcceptEvent):
            colors[e.idx] = _C_ACCEPTED
            circle.set_radius(np.sqrt(e.dist_sq))
        elif isinstance(e, EvictEvent):
            colors[e.idx] = _C_EVICTED
        elif isinstance(e, PruneEvent):
            # KD-Tree: look up the pruned subtree root's cell bounds
            split = split_by_idx.get(e.idx)
            if split is not None and split.lo.shape[0] >= 2:
                from matplotlib.patches import Rectangle

                ax.add_patch(
                    Rectangle(
                        (split.lo[0], split.lo[1]),
                        split.hi[0] - split.lo[0],
                        split.hi[1] - split.lo[1],
                        color="orange",
                        alpha=0.18,
                        zorder=2,
                    )
                )
        elif isinstance(e, PruneBoxEvent):
            # QuadTree: cell bounds come directly from the event
            from matplotlib.patches import Rectangle

            ax.add_patch(
                Rectangle(
                    (e.lo[0], e.lo[1]),
                    e.hi[0] - e.lo[0],
                    e.hi[1] - e.lo[1],
                    color="orange",
                    alpha=0.18,
                    zorder=2,
                )
            )
        elif isinstance(e, ResultEvent):
            colors[e.idx] = _C_RESULT

        scat.set_facecolors(colors)
        return [scat, circle]

    anim = FuncAnimation(
        fig, update, frames=total, interval=interval, blit=False, repeat=False
    )
    _save_or_show(anim, output, fps=max(1, 1000 // interval))
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _scatter_main(args: argparse.Namespace) -> None:
    if args.titles is not None and len(args.titles) != len(args.files):
        print(
            f"error: --titles has {len(args.titles)} entries "
            f"but {len(args.files)} files given.",
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


def _animate_main(args: argparse.Namespace) -> None:
    for p in (args.log, args.data):
        if not p.exists():
            print(f"error: file not found: {p}", file=sys.stderr)
            sys.exit(1)

    pts = load_points(args.data)
    build_events, query_events = parse_log(args.log)

    _ALGO_TITLES = {
        "kdtree": "KD-Tree",
        "exhaustive": "Exhaustive KNN",
        "quadtree": "QuadTree",
    }
    stem_parts = args.log.stem.split("_")
    algo_key = stem_parts[0]
    algo_label = _ALGO_TITLES.get(algo_key, algo_key)

    # Dataset subtitle from the rest of the log stem, e.g.
    # "kdtree_clustered_n200_d2_s42" → "clustered  n=200  d=2  s=42"
    def _fmt_part(p: str) -> str:
        """'n200' → 'n=200', 'clustered' → 'clustered'."""
        return (
            p[0] + "=" + p[1:]
            if len(p) > 1 and p[0].isalpha() and p[1:].isdigit()
            else p
        )

    data_info = "  ".join(_fmt_part(p) for p in stem_parts[1:])

    def _title(stage: str) -> str:
        return (
            f"{algo_label} {stage}\n{data_info}"
            if data_info
            else f"{algo_label} {stage}"
        )

    _OUT_DIR = Path("assets/figures")

    def out_for(tag: str) -> Path | None:
        """Build assets/figures/<log_stem>_<tag>.gif, or None for live mode."""
        if not args.save:
            return None
        return _OUT_DIR / f"{args.log.stem}_{tag}.gif"

    if args.mode in ("build", "both"):
        animate_build(
            pts,
            build_events,
            title=_title("Build"),
            output=out_for("build"),
            interval=args.interval,
        )

    if args.mode in ("query", "both"):
        animate_query(
            pts,
            build_events,
            query_events,
            title=_title("Query"),
            output=out_for("query"),
            interval=args.interval,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize NNS datasets and algorithm animations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command")

    # --- scatter subcommand ---
    sc = sub.add_parser("scatter", help="Scatter-plot point-cloud datasets.")
    sc.add_argument("files", nargs="+", type=Path, metavar="CSV")
    sc.add_argument("--titles", nargs="*", metavar="TITLE", default=None)
    sc.add_argument("--output", "-o", type=Path, default=None)
    sc.add_argument("--suptitle", type=str, default="")

    # --- animate subcommand ---
    an = sub.add_parser("animate", help="Animate build or query from an event log.")
    an.add_argument(
        "--log",
        type=Path,
        required=True,
        metavar="LOG",
        help="Event log from the `logging` preset.",
    )
    an.add_argument(
        "--data",
        type=Path,
        required=True,
        metavar="CSV",
        help="Dataset CSV used during benchmarking.",
    )
    an.add_argument(
        "--mode",
        choices=["build", "query", "both"],
        default="both",
        help="Which animation to produce (default: both).",
    )
    an.add_argument(
        "--save",
        action="store_true",
        help="Save animations to assets/figures/ instead of "
        "showing them interactively. Filenames are derived "
        "from the log stem and stage.",
    )
    an.add_argument(
        "--interval", type=int, default=100, help="Frame interval in ms (default: 100)."
    )

    args = parser.parse_args()

    if args.command == "scatter":
        _scatter_main(args)
    elif args.command == "animate":
        _animate_main(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
