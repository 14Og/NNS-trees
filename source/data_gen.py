"""
data_gen.py — synthetic dataset generator for NNS benchmark

Distributions
-------------
uniform    : uniformly random points in [0, 1]^d
clustered  : Gaussian blobs with equal cluster weights
skewed     : uniform within [0, skew_fraction]^d, rest empty

Output
------
Headerless CSV, one point per row, d columns.
Auto-named as:  <repo_root>/data/<dist>_n<n>_d<dims>_s<seed>.csv
Override with:  -o / --output <path>

Usage examples
--------------
python data_gen.py uniform 10000
python data_gen.py clustered 50000 --dims 2 --n-clusters 8 --seed 0
python data_gen.py skewed 5000 --output data/skewed_custom.csv
python data_gen.py uniform 10000 --dims 8   # for KD-tree ND sweep
"""

import argparse
import sys
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class DataGenerator(ABC):
    """Abstract base for all point-cloud generators."""

# : Subclasses register themselves here via __init_subclass__
    _registry: dict[str, type["DataGenerator"]] = {}

    def __init_subclass__(cls, name: str, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        DataGenerator._registry[name] = cls
        cls._dist_name = name

    @classmethod
    def create(cls, name: str, **kwargs: object) -> "DataGenerator":
        """Factory: look up by distribution name and instantiate."""
        if name not in cls._registry:
            raise ValueError(f"Unknown distribution '{name}'. " f"Available: {list(cls._registry)}")
        return cls._registry[name](**kwargs)  # type: ignore[arg-type]

    @abstractmethod
    def generate(self, n: int, dims: int, rng: np.random.Generator) -> np.ndarray:
        """Return float32 array of shape (n, dims) in [0, 1]^d."""

    @property
    def name(self) -> str:
        return self._dist_name  # type: ignore[attr-defined]


class UniformGenerator(DataGenerator, name="uniform"):
    """Uniformly random in [0, 1]^d."""

    def generate(self, n: int, dims: int, rng: np.random.Generator) -> np.ndarray:
        return rng.random((n, dims), dtype=np.float32)


class ClusteredGenerator(DataGenerator, name="clustered"):
    """
    n_clusters Gaussian blobs with centres drawn in [0.1, 0.9]^d.
    Points outside [0, 1]^d are clipped to the canonical bounding box.
    """

    def __init__(self, n_clusters: int = 5, cluster_std: float = 0.05) -> None:
        self.n_clusters = n_clusters
        self.cluster_std = cluster_std

    def generate(self, n: int, dims: int, rng: np.random.Generator) -> np.ndarray:
#All data lives in[0, 1] ^ d.Keep centroids at least 2σ from each wall
#so blobs aren't badly clipped — with default std=0.05 this gives [0.1, 0.9].
        margin = 2.0 * self.cluster_std
        if margin >= 0.5:
            raise ValueError(
                f"cluster_std={self.cluster_std} too large: "
                f"2σ margin ({margin:.2f}) leaves no room for cluster centres in [0,1]."
            )
        centres = rng.uniform(margin, 1.0 - margin, size=(self.n_clusters, dims)).astype(np.float32)

        sizes = np.full(self.n_clusters, n // self.n_clusters, dtype=int)
        sizes[: n % self.n_clusters] += 1

        parts = [
            rng.normal(loc=centres[i], scale=self.cluster_std, size=(sizes[i], dims)).astype(
                np.float32
            )
            for i in range(self.n_clusters)
        ]
        pts = np.concatenate(parts, axis=0)
        np.clip(pts, 0.0, 1.0, out=pts)
        rng.shuffle(pts)
        return pts


class SkewedGenerator(DataGenerator, name="skewed"):
    """
    All points concentrated in [0, skew_fraction]^d.
    Worst case for spatial-midpoint splits (QuadTree).
    """

    def __init__(self, skew_fraction: float = 0.1) -> None:
        self.skew_fraction = skew_fraction

    def generate(self, n: int, dims: int, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(0.0, self.skew_fraction, size=(n, dims)).astype(np.float32)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate synthetic point datasets for NNS benchmarks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "distribution",
        choices=list(DataGenerator._registry),
        help="Point distribution to generate.",
    )
    p.add_argument(
        "n",
        type=int,
        help="Number of points to generate.",
    )
    p.add_argument(
        "--dims",
        "-d",
        type=int,
        default=2,
        metavar="D",
        help="Dimensionality (default: 2). Use >2 for KD-tree ND sweep.",
    )
    p.add_argument(
        "--seed",
        "-s",
        type=int,
        default=42,
        metavar="S",
        help="Random seed for reproducibility (default: 42).",
    )
    p.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Output file path. "
            "Defaults to data/<dist>_n<n>_d<dims>_s<seed>.csv "
            "relative to the project root."
        ),
    )
#Clustered - only knobs
    p.add_argument(
        "--n-clusters",
        type=int,
        default=5,
        metavar="K",
        help="Number of Gaussian clusters (clustered only, default: 5).",
    )
    p.add_argument(
        "--cluster-std",
        type=float,
        default=0.05,
        metavar="σ",
        help="Std dev of each Gaussian blob (default: 0.05 = 5%% of bbox).",
    )
#Skewed - only knob
    p.add_argument(
        "--skew-fraction",
        type=float,
        default=0.1,
        metavar="F",
        help="Fraction of bbox used for skewed distribution (default: 0.1).",
    )
    return p


def default_output_path(dist: str, n: int, dims: int, seed: int) -> Path:
    repo_root = Path(__file__).resolve().parent.parent
    data_dir = repo_root / "data"
    data_dir.mkdir(exist_ok=True)
    return data_dir / f"{dist}_n{n}_d{dims}_s{seed}.csv"


def main() -> None:
    args = build_parser().parse_args()

    if args.n <= 0:
        print(f"error: n must be positive, got {args.n}", file=sys.stderr)
        sys.exit(1)
    if args.dims <= 0:
        print(f"error: dims must be positive, got {args.dims}", file=sys.stderr)
        sys.exit(1)

    kwargs: dict[str, object] = {}
    if args.distribution == "clustered":
        kwargs = {"n_clusters": args.n_clusters, "cluster_std": args.cluster_std}
    elif args.distribution == "skewed":
        kwargs = {"skew_fraction": args.skew_fraction}

    generator = DataGenerator.create(args.distribution, **kwargs)
    rng = np.random.default_rng(args.seed)
    pts = generator.generate(args.n, args.dims, rng)

    out_path: Path = args.output or default_output_path(
        args.distribution, args.n, args.dims, args.seed
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_path, pts, delimiter=",", fmt="%.6f")

    print(
        f"Generated {len(pts):,} {args.distribution} points "
        f"(d={args.dims}, seed={args.seed}) → {out_path}"
    )


if __name__ == "__main__":
    main()
