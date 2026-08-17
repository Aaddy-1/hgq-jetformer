#!/usr/bin/env python3
"""Regenerates ONLY test.h5, leaving the train shards and normalization stats alone.

`process_jetclass_root_dir` has no test-only mode: it rebuilds the ten
train_part*.h5 shards and then recomputes mean.npy/std.npy from them
(build_jetclass_dataset.py:365). New normalization statistics would silently
invalidate every accuracy measured against the old ones, so running the full
builder to repair one file is not an option.

This script reuses the builder's own file-classification and feature-extraction
code -- it does not paraphrase them -- and executes only the test-set block
(build_jetclass_dataset.py:315-356). It never touches train_part*.h5,
mean.npy or std.npy, and never deletes a ROOT file.

Writes to test.h5.rebuilding and renames only on success, so a failed or
interrupted run cannot leave a half-written file where the dataset should be.

The entry count is read from each tree's header, which does NOT prove the file's
data baskets are readable -- a ROOT file damaged by storage failure can report
num_entries fine and then raise OSError mid-conversion. Verify the source files
independently (`dd if=<file> of=/dev/null`) before trusting a rebuild.

Usage:
    python -m scripts.rebuild_test_h5 --input_dir datasets/JetClass/Pythia
    python -m scripts.rebuild_test_h5 --dry_run          # list what it would do
"""

import os
import re
import sys
import argparse

import numpy as np
import h5py
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.build_jetclass_dataset import (  # noqa: E402
    PROCESSED_DIR,
    build_features_and_labels,
)

try:
    import uproot
except ImportError:
    uproot = None


def classify_test_files(input_dir):
    """Selects the test ROOT files exactly as the builder does.

    Transcribed from build_jetclass_dataset.py:210-257. The `sorted()` and the
    classification predicate must match the original, or the rebuilt file will
    hold different jets in a different order and no accuracy measured against
    the old test.h5 will remain comparable.
    """
    all_root_files = []
    for root, _dirs, files in os.walk(input_dir):
        for f in files:
            if f.endswith(".root"):
                all_root_files.append(os.path.join(root, f))
    all_root_files = sorted(all_root_files)

    if not all_root_files:
        raise FileNotFoundError(f"No .root files found in {input_dir}")

    test_files = []
    for rfile in all_root_files:
        fname = os.path.basename(rfile)
        parent_dir = os.path.basename(os.path.dirname(rfile))
        if (
            "test" in fname.lower()
            or "val" in fname.lower()
            or "test" in parent_dir.lower()
            or "val" in parent_dir.lower()
        ):
            test_files.append(rfile)

    # Same fallback as the builder: with no explicitly-named test files, the
    # last 20% of the sorted list becomes the test set.
    if not test_files:
        n_test = max(1, int(len(all_root_files) * 0.2))
        test_files = all_root_files[-n_test:]

    return all_root_files, test_files


def load_exclusions(path):
    """Basenames of ROOT files to skip, one per line. Blank lines and # ignored.

    Storage failure damages individual ROOT files, not whole classes: a file can
    report its entry count from the tree header and still raise OSError when a
    data basket is read (see the note in build_jetclass_dataset). Excluding the
    known-bad files before per-class selection lets the rebuild fall through to
    the next readable file of that class instead of aborting mid-write.
    """
    if not path:
        return set()
    excluded = set()
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#"):
                excluded.add(os.path.basename(line))
    return excluded


def report_slice_impact(test_files, excluded):
    """Warns when an excluded file held jets evaluate.py would actually score.

    evaluate.py reads the first 200,000 jets of each class at the default
    max_test_samples, i.e. that class's first two files. Excluding a file at
    position 3 or later is invisible to the measurement; excluding one at
    position 1 or 2 shifts the class's evaluated jets to a different sample,
    which is a provenance change and must not pass silently.
    """
    position = {}
    shifted = []
    for rfile in test_files:
        cls = re.sub(r"_\d+\.root$", "", os.path.basename(rfile))
        position[cls] = position.get(cls, 0) + 1
        if os.path.basename(rfile) in excluded and position[cls] <= 2:
            shifted.append((os.path.basename(rfile), cls, position[cls]))
    for fname, cls, pos in shifted:
        print(
            f"[Rebuild] WARNING: {fname} is file #{pos} of class {cls}, inside the "
            f"200,000 jets evaluate.py scores. That class will be measured on a "
            f"DIFFERENT sample than a build from the intact files."
        )
    return shifted


def limit_files_per_class(test_files, n_per_class):
    """Keeps the first n ROOT files of each class, in the original sorted order.

    evaluate.py does not read the whole test set: for each class it takes a
    contiguous `max_test_samples // n_classes` slice beginning at that class's
    first occurrence (evaluate.py:119-127). With the default 2,000,000 that is
    200,000 jets per class -- the class's first two 100k-jet ROOT files. So a
    test.h5 built from the first n >= 3 files of each class yields the identical
    slice, and therefore the identical accuracy, at a fraction of the size.

    The margin matters: n must be strictly greater than 2 so that the rebuilt
    file holds more than max_test_samples jets, keeping evaluate.py on its
    slice-sampling branch rather than the whole-file branch, which concatenates
    classes in a different order.

    Class is taken from the filename with its trailing index stripped
    (HToBB_120.root -> HToBB). Sorted order puts test_20M ahead of val_5M, so
    the retained files are the same ones the full build would have placed first.
    """
    kept, seen = [], {}
    for rfile in test_files:
        cls = re.sub(r"_\d+\.root$", "", os.path.basename(rfile))
        if seen.get(cls, 0) < n_per_class:
            seen[cls] = seen.get(cls, 0) + 1
            kept.append(rfile)
    return kept, seen


def rebuild_test_h5(
    input_dir,
    num_particles=128,
    num_feats=17,
    dry_run=False,
    output_dir=None,
    files_per_class=0,
    exclude_files=None,
):
    if uproot is None:
        raise ImportError("uproot is required. pip install uproot awkward")

    canonical_dir = os.path.join(
        PROCESSED_DIR, "jetclass", str(num_particles), f"{num_feats}f"
    )
    # An explicit output_dir exists so the rebuild can be written to a healthy
    # volume when the canonical one cannot hold the file. The temp file and the
    # final rename both stay inside output_dir, because os.rename cannot cross
    # a filesystem boundary. Link the result back into canonical_dir afterwards.
    relocated = output_dir is not None and os.path.abspath(
        output_dir
    ) != os.path.abspath(canonical_dir)
    output_dir = output_dir or canonical_dir
    os.makedirs(output_dir, exist_ok=True)

    test_h5_path = os.path.join(output_dir, "test.h5")
    tmp_h5_path = test_h5_path + ".rebuilding"

    all_root_files, test_files = classify_test_files(input_dir)
    print(f"[Rebuild] {len(all_root_files)} ROOT files found, {len(test_files)} classified as test")

    # Exclusion runs before per-class selection so that --files_per_class keeps
    # the first N *readable* files of each class rather than the first N overall.
    excluded = load_exclusions(exclude_files)
    if excluded:
        report_slice_impact(test_files, excluded)
        present = {os.path.basename(f) for f in test_files}
        unmatched = sorted(excluded - present)
        if unmatched:
            print(f"[Rebuild] WARNING: not in the test set, ignored: {', '.join(unmatched)}")
        before = len(test_files)
        test_files = [f for f in test_files if os.path.basename(f) not in excluded]
        print(f"[Rebuild] EXCLUDED {before - len(test_files)} file(s) listed in {exclude_files}")

    if files_per_class > 0:
        if files_per_class < 3:
            print("[Rebuild] ABORT: --files_per_class must be at least 3, or the rebuilt")
            print("[Rebuild] file holds <= max_test_samples jets and evaluate.py switches")
            print("[Rebuild] to its whole-file branch instead of per-class slice sampling.")
            return 1
        test_files, per_class = limit_files_per_class(test_files, files_per_class)
        print(
            f"[Rebuild] SUBSET: first {files_per_class} file(s) of each of "
            f"{len(per_class)} classes -> {len(test_files)} files"
        )
        odd = sorted(k for k, v in per_class.items() if v != files_per_class)
        if odd:
            print(f"[Rebuild] WARNING: fewer files than asked for: {', '.join(odd)}")

    # Guard the two files a full rebuild would have overwritten. These are always
    # checked in the canonical directory: they are what the model was normalized
    # against, and they stay there whether or not the rebuild is relocated.
    for name in ("mean.npy", "std.npy"):
        p = os.path.join(canonical_dir, name)
        print(f"[Rebuild] preserving {name}: {'present' if os.path.exists(p) else 'MISSING'}")

    if relocated:
        print(f"[Rebuild] writing to {output_dir}")
        print(f"[Rebuild] canonical dir is {canonical_dir} -- link test.h5 back after this run")

    total_test_samples = 0
    unreadable = []
    for rfile in tqdm(test_files, desc="Counting entries"):
        try:
            total_test_samples += uproot.open(rfile)["tree"].num_entries
        except Exception as e:
            unreadable.append((rfile, str(e)))

    if unreadable:
        print(f"\n[Rebuild] WARNING: {len(unreadable)} test ROOT files are unreadable:")
        for rfile, err in unreadable[:10]:
            print(f"    {os.path.basename(rfile)}: {err[:120]}")
        print("[Rebuild] The rebuilt file would be short by their contents. Aborting.")
        print("[Rebuild] Re-run once the storage is healthy, or the test set will not")
        print("[Rebuild] match the one every earlier accuracy was measured on.")
        return 1

    print(f"[Rebuild] total test jets: {total_test_samples:,}")

    if files_per_class > 0:
        print(
            f"[Rebuild] evaluate with --max_test_samples strictly below "
            f"{total_test_samples:,}; at or above it evaluate.py reads the whole "
            f"file instead of slicing per class, and the result is not comparable "
            f"with metrics measured on the full test set."
        )

    if dry_run:
        print(f"[Rebuild] dry run -- would write {tmp_h5_path} then rename to {test_h5_path}")
        return 0

    with h5py.File(tmp_h5_path, "w") as fout:
        dset_X = fout.create_dataset(
            "particle_features",
            shape=(total_test_samples, num_particles, num_feats),
            dtype=np.float32,
            compression="lzf",
            chunks=True,
        )
        dset_y = fout.create_dataset(
            "label",
            shape=(total_test_samples,),
            dtype=np.int64,
            compression="lzf",
            chunks=True,
        )

        write_idx = 0
        for rfile in tqdm(test_files, desc="Converting test ROOT files"):
            tree = uproot.open(rfile)["tree"]
            X_arr, y_arr = build_features_and_labels(tree, max_particles=num_particles)
            n_entries = X_arr.shape[0]
            dset_X[write_idx : write_idx + n_entries] = X_arr
            dset_y[write_idx : write_idx + n_entries] = y_arr
            write_idx += n_entries

    if write_idx != total_test_samples:
        print(f"[Rebuild] ABORT: wrote {write_idx:,} of {total_test_samples:,} expected jets.")
        print(f"[Rebuild] Leaving {tmp_h5_path} in place for inspection; test.h5 untouched.")
        return 1

    # Read the new file back end to end before it replaces anything. The old
    # file failed exactly this way, so the rebuild is not trusted until it has
    # survived the same test.
    print("[Rebuild] verifying readback...")
    with h5py.File(tmp_h5_path, "r") as f:
        d = f["label"]
        for i in range(0, d.shape[0], 1_000_000):
            _ = d[i : i + 1_000_000]
        dx = f["particle_features"]
        for i in range(0, dx.shape[0], 200_000):
            _ = dx[i : i + 200_000]
    print("[Rebuild] readback OK")

    if os.path.exists(test_h5_path):
        damaged = test_h5_path + ".damaged"
        os.rename(test_h5_path, damaged)
        print(f"[Rebuild] previous file moved to {damaged}")
    os.rename(tmp_h5_path, test_h5_path)
    print(f"[Rebuild] wrote {write_idx:,} jets to {test_h5_path}")
    print("[Rebuild] train_part*.h5, mean.npy and std.npy were not modified.")
    if relocated:
        canonical_test = os.path.join(canonical_dir, "test.h5")
        print("[Rebuild] to put it where the training code looks for it:")
        print(f"[Rebuild]     ln -sfn {test_h5_path} {canonical_test}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Regenerate only test.h5")
    parser.add_argument("-i", "--input_dir", type=str, default="datasets/JetClass/Pythia")
    parser.add_argument("--num_particles", type=int, default=128)
    parser.add_argument("--num_feats", type=int, default=17)
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Classify files and count jets without writing anything",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default=None,
        help=(
            "Write test.h5 here instead of data/processed/jetclass/<P>/<F>f/. "
            "Use when the canonical volume cannot hold the file; link the result "
            "back afterwards."
        ),
    )
    parser.add_argument(
        "--files_per_class",
        type=int,
        default=0,
        help=(
            "Build from only the first N ROOT files of each class (minimum 3). "
            "evaluate.py slices 200,000 contiguous jets per class from each "
            "class's first occurrence, so N=3 reproduces the identical slice in "
            "a ~5 GiB file instead of 41 GiB. 0 (default) builds the full set."
        ),
    )
    parser.add_argument(
        "--exclude_files",
        type=str,
        default=None,
        help=(
            "Path to a text file listing ROOT filenames to skip, one per line. "
            "Applied before --files_per_class, so selection falls through to the "
            "next readable file of that class. Use for files that fail to read."
        ),
    )
    args = parser.parse_args()

    sys.exit(
        rebuild_test_h5(
            input_dir=args.input_dir,
            num_particles=args.num_particles,
            num_feats=args.num_feats,
            dry_run=args.dry_run,
            output_dir=args.output_dir,
            files_per_class=args.files_per_class,
            exclude_files=args.exclude_files,
        )
    )
