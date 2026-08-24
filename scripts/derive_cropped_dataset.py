"""Slice an existing 128-particle JetClass tree down to N particles.

No ROOT reprocessing. The builder crops with `ak.pad_none(a, maxlen, clip=True)`
(`src/data/build_jetclass_dataset.py:48`) -- a PREFIX crop -- and the constituent
axis is exactly pT-sorted, measured rather than assumed:
`scripts/particle_multiplicity.py` Q1 reports 100.00% of 7,869,149 adjacent pairs
non-increasing, across all ten classes. So

    X_N == X_128[:, :N, :]

bit-exactly, and the N-particle dataset the builder would spend hours producing
from ROOT can be sliced out of the h5 tree that already exists.

mean.npy / std.npy MUST be recomputed and are NOT copied. `compute_welford_stats`
flattens over the particle axis (`build_jetclass_dataset.py:155`), so the padded
fraction enters the statistics directly: ~69% of slots are padding at 128
particles (mean multiplicity 39.3) but only ~39% at 64. Reusing the 128 stats
would z-score the cropped data against the wrong mean. This script calls the
builder's own Welford implementation so the statistics are produced by exactly
the same code path as a real build.

Reads the source read-only; writes only into the new directory.

Usage:
    python scripts/derive_cropped_dataset.py --num_particles 64 --dry_run
    python scripts/derive_cropped_dataset.py --num_particles 64
"""

import argparse
import os
import sys

import h5py
import numpy as np
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.build_jetclass_dataset import compute_welford_stats  # noqa: E402

PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")


def crop_one(src_path, dst_path, num_particles, batch_size, dry_run):
    """Copy src -> dst keeping only the leading `num_particles` slots."""
    with h5py.File(src_path, "r") as fin:
        X, y = fin["particle_features"], fin["label"]
        n, p_src, f = X.shape

        if num_particles > p_src:
            raise ValueError(
                f"{src_path}: cannot crop to {num_particles} from {p_src} particles"
            )

        out_bytes = n * num_particles * f * np.dtype(X.dtype).itemsize
        print(
            f"  {os.path.basename(src_path):22s} "
            f"({n:,}, {p_src}, {f}) -> ({n:,}, {num_particles}, {f})  "
            f"{out_bytes / 1e9:6.2f} GB"
        )
        if dry_run:
            return

        with h5py.File(dst_path, "w") as fout:
            dX = fout.create_dataset(
                "particle_features",
                shape=(n, num_particles, f),
                dtype=X.dtype,
                chunks=(min(batch_size, n), num_particles, f),
            )
            dy = fout.create_dataset("label", shape=y.shape, dtype=y.dtype)

            for start in tqdm(
                range(0, n, batch_size), desc=f"  crop {os.path.basename(src_path)}"
            ):
                end = min(start + batch_size, n)
                dX[start:end] = X[start:end, :num_particles, :]
                dy[start:end] = y[start:end]


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--num_particles", type=int, required=True,
                   help="target particle count, e.g. 64")
    p.add_argument("--num_feats", type=int, default=17)
    p.add_argument("--src_particles", type=int, default=128,
                   help="source particle count to slice from (default: 128)")
    p.add_argument("--src_dir", type=str, default=None,
                   help="override the source directory entirely")
    p.add_argument("--files", type=str, default="train_part0.h5,test.h5",
                   help="comma-separated h5 filenames to crop "
                        "(default: train_part0.h5,test.h5 -- what "
                        "`--train_parts 0` actually reads)")
    p.add_argument("--batch_size", type=int, default=5000)
    p.add_argument("--dry_run", action="store_true",
                   help="report shapes and output sizes, write nothing")
    args = p.parse_args()

    ff = f"{args.num_feats}f"
    src_dir = args.src_dir or os.path.join(
        PROCESSED_DIR, "jetclass", str(args.src_particles), ff
    )
    dst_dir = os.path.join(PROCESSED_DIR, "jetclass", str(args.num_particles), ff)

    print(f"source      {src_dir}")
    print(f"destination {dst_dir}")
    if args.dry_run:
        print("DRY RUN -- nothing will be written\n")
    else:
        os.makedirs(dst_dir, exist_ok=True)
        print()

    names = [n.strip() for n in args.files.split(",") if n.strip()]
    missing = [n for n in names if not os.path.exists(os.path.join(src_dir, n))]
    if missing:
        raise SystemExit(f"missing in {src_dir}: {', '.join(missing)}")

    for name in names:
        crop_one(
            os.path.join(src_dir, name),
            os.path.join(dst_dir, name),
            args.num_particles,
            args.batch_size,
            args.dry_run,
        )

    train_paths = [os.path.join(dst_dir, n) for n in names if n.startswith("train_part")]
    if args.dry_run:
        print(f"\nwould recompute mean.npy/std.npy from {len(train_paths)} train shard(s)")
        return
    if not train_paths:
        raise SystemExit(
            "no train_part*.h5 among --files, so mean.npy/std.npy cannot be "
            "computed. The 128-particle stats are NOT valid here -- see module docstring."
        )

    print(f"\nrecomputing Welford statistics from {len(train_paths)} train shard(s)")
    compute_welford_stats(train_paths, dst_dir, num_feats=args.num_feats,
                          batch_size=args.batch_size)
    print(f"\ndone. train with --num_particles {args.num_particles}")


if __name__ == "__main__":
    main()
