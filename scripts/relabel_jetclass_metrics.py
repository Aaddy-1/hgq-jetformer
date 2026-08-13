"""
One-off repair: rename mislabelled per-class keys in JetClass metrics JSONs.

Background
----------
`JETCLASS_CLASSES` in src/training/train.py declared an ordering that did not
match `label_list` in src/data/build_jetclass_dataset.py, which is the source
of truth for the integer labels written to disk. Metrics are emitted as
`class_name = classes[i]` for class index i, so the *values* were always
indexed correctly -- only the names attached to them were wrong.

That makes this a lossless, position-preserving rename: no metric is
recomputed, and no value moves.

Usage
-----
    python scripts/relabel_jetclass_metrics.py --dry-run
    python scripts/relabel_jetclass_metrics.py
    python scripts/relabel_jetclass_metrics.py --no-backup
"""

import argparse
import glob
import json
import os
import shutil

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The incorrect ordering previously declared in train.py. A file is only
# rewritten if its per-class keys match this exactly, in this order.
OLD_CLASSES = [
    "g",
    "q",
    "W_qq",
    "Z_qq",
    "t_bqq",
    "H_bb",
    "H_cc",
    "H_gg",
    "H_4q",
    "H_qq",
]

# Ground truth: label_list in src/data/build_jetclass_dataset.py
NEW_CLASSES = [
    "QCD",
    "H_bb",
    "H_cc",
    "H_gg",
    "H_4q",
    "H_qql",
    "Z_qq",
    "W_qq",
    "t_bqq",
    "t_bl",
]

SEARCH_GLOBS = [
    ".agents/*/outputs/*/*_metrics.json",
    "experiment/*/outputs/*/*_metrics.json",
    "outputs/*/*_metrics.json",
    "*_metrics.json",
]


def find_candidates():
    paths = []
    for pattern in SEARCH_GLOBS:
        paths.extend(glob.glob(os.path.join(PROJECT_ROOT, pattern)))
    return sorted(set(paths))


def relabel_file(path, dry_run=False, backup=True):
    """Returns one of: 'relabelled', 'skipped', 'already-correct', 'error'."""
    try:
        with open(path) as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"  ERROR      {os.path.relpath(path, PROJECT_ROOT)}: {exc}")
        return "error"

    per_class = data.get("performance", {}).get("per_class_metrics")
    if not isinstance(per_class, dict):
        return "skipped"

    keys = list(per_class.keys())
    rel = os.path.relpath(path, PROJECT_ROOT)

    if keys == NEW_CLASSES:
        print(f"  OK         {rel} (already correct)")
        return "already-correct"

    if keys != OLD_CLASSES:
        print(f"  SKIP       {rel} (unrecognised keys: {keys})")
        return "skipped"

    # Positional rename; values are carried across untouched.
    data["performance"]["per_class_metrics"] = {
        NEW_CLASSES[i]: per_class[old] for i, old in enumerate(OLD_CLASSES)
    }

    if dry_run:
        print(f"  WOULD FIX  {rel}")
        return "relabelled"

    if backup:
        shutil.copy2(path, path + ".bak")

    with open(path, "w") as fh:
        json.dump(data, fh, indent=2)
        fh.write("\n")

    print(f"  RELABELLED {rel}")
    return "relabelled"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would change without writing anything",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not write a .bak copy alongside each modified file",
    )
    args = parser.parse_args()

    paths = find_candidates()
    print(f"Scanning {len(paths)} metrics file(s) under {PROJECT_ROOT}\n")

    counts = {}
    for path in paths:
        result = relabel_file(path, dry_run=args.dry_run, backup=not args.no_backup)
        counts[result] = counts.get(result, 0) + 1

    print(
        "\nSummary: "
        + ", ".join(f"{v} {k}" for k, v in sorted(counts.items()))
        + (" (dry run - nothing written)" if args.dry_run else "")
    )


if __name__ == "__main__":
    main()
