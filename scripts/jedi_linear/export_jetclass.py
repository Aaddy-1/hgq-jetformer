"""Write the SCRAMJet JetClass subsets in the layout JEDI-linear's loader reads.

Upstream `get_data` (`third_party/JEDI-linear/src/dataloader.py`) reads
`<datapath>/150c-train.h5` and `<datapath>/150c-test.h5`, with keys `feature`
(jets, particles, features) and `label` (jets,). This script writes the JetClass
jets that the SCRAMJet runs train and test on to those two names, so the vendored
code runs unmodified:

- train: `max_samples // n_classes` contiguous jets per class, each block starting
  at the class's first index in `train_part0.h5`. This is the in-memory selection
  in `src/training/train.py:306-313`, and `--train_parts 0` / the in-memory path
  reads only that shard (`train.py:294`).
- test: the same rule on `test.h5` with `max_test_samples`, as in
  `src/training/evaluate.py:118-126`.

Raw features are copied as float32, unnormalized. Upstream standardizes inside
`get_data`. The N=64 runs use these same files, because upstream's
`[:, :n_constituents]` is the same prefix slice as `scripts/derive_cropped_dataset.py`.

Reads the source read-only; writes only into --out_dir.

Usage:
    python scripts/jedi_linear/export_jetclass.py --dry_run
    python scripts/jedi_linear/export_jetclass.py
"""

import argparse
import os

import h5py
import numpy as np
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")


def class_blocks(labels, n_select):
    """(class, start, stop) blocks exactly as train.py / evaluate.py slice them."""
    classes = np.unique(labels)
    quota = n_select // len(classes)
    blocks = []
    for cls in classes:
        start = int(np.where(labels == cls)[0][0])
        blocks.append((int(cls), start, start + quota))
    return blocks


def export_one(src_path, dst_path, n_select, batch_size, dry_run):
    with h5py.File(src_path, "r") as fin:
        X, y = fin["particle_features"], fin["label"]
        n, p, f = X.shape
        labels = y[:]

        # train.py / evaluate.py take every jet when the request covers the file
        if n_select < n:
            blocks = class_blocks(labels, n_select)
        else:
            blocks = [(-1, 0, n)]
        n_out = sum(stop - start for _, start, stop in blocks)

        print(f"  {os.path.basename(src_path)}: ({n:,}, {p}, {f}) {X.dtype} -> "
              f"{os.path.basename(dst_path)} ({n_out:,}, {p}, {f})")
        for cls, start, stop in blocks:
            counts = np.bincount(labels[start:stop].astype(np.int64))
            pure = cls < 0 or counts[cls] == stop - start
            print(f"    class {cls:>2}: [{start:,}, {stop:,})  "
                  f"{stop - start:,} jets  labels pure: {'yes' if pure else 'NO ' + str(counts.tolist())}")
        if dry_run:
            return

        with h5py.File(dst_path, "w") as fout:
            dX = fout.create_dataset("feature", shape=(n_out, p, f), dtype=np.float32,
                                     chunks=(min(batch_size, n_out), p, f))
            dy = fout.create_dataset("label", shape=(n_out,), dtype=y.dtype)
            fout.attrs["source"] = os.path.abspath(src_path)
            fout.attrs["blocks"] = np.array([(s, e) for _, s, e in blocks], dtype=np.int64)

            pos = 0
            for _, start, stop in blocks:
                for s in tqdm(range(start, stop, batch_size),
                              desc=f"    write [{start:,}, {stop:,})", leave=False):
                    e = min(s + batch_size, stop)
                    dX[pos:pos + e - s] = X[s:e].astype(np.float32)
                    dy[pos:pos + e - s] = y[s:e]
                    pos += e - s
            assert pos == n_out


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--src_dir", type=str,
                   default=os.path.join(PROCESSED_DIR, "jetclass", "128", "17f"),
                   help="JetClass tree holding train_part0.h5 and test.h5")
    p.add_argument("--out_dir", type=str,
                   default=os.path.join(PROCESSED_DIR, "jedi_linear_jetclass"),
                   help="the `datapath` of the jetclass-*.yaml configs")
    p.add_argument("--max_samples", type=int, default=2000000,
                   help="train.py --max_samples of the SCRAMJet runs (default 2,000,000)")
    p.add_argument("--max_test_samples", type=int, default=2000000,
                   help="train.py --max_test_samples of the SCRAMJet runs (default 2,000,000)")
    p.add_argument("--batch_size", type=int, default=20000)
    p.add_argument("--dry_run", action="store_true",
                   help="report the selected blocks, write nothing")
    args = p.parse_args()

    print(f"source      {args.src_dir}")
    print(f"destination {args.out_dir}")
    if args.dry_run:
        print("DRY RUN -- nothing will be written\n")
    else:
        os.makedirs(args.out_dir, exist_ok=True)
        print()

    jobs = [("train_part0.h5", "150c-train.h5", args.max_samples),
            ("test.h5", "150c-test.h5", args.max_test_samples)]
    missing = [s for s, _, _ in jobs if not os.path.exists(os.path.join(args.src_dir, s))]
    if missing:
        raise SystemExit(f"missing in {args.src_dir}: {', '.join(missing)}")

    for src, dst, n_select in jobs:
        export_one(os.path.join(args.src_dir, src), os.path.join(args.out_dir, dst),
                   n_select, args.batch_size, args.dry_run)

    if not args.dry_run:
        print(f"\ndone. datapath for the jetclass-*.yaml configs: {args.out_dir}")


if __name__ == "__main__":
    main()
