#!/usr/bin/env python3
"""
scripts/diagnose_train_val_gap.py
---------------------------------
Evaluates a saved checkpoint, in inference mode, on the *training* split -- which
`evaluate.py` cannot do (it is hardwired to test.h5).

Why this exists
===============
Across EXP-24/EXP-25 the training-mode metric and the validation metric decouple
completely once BetaPID begins compressing, and never re-couple:

    epochs (EXP-25 s42)   train_loss  val_loss  train_acc  val_acc     gap
    1-70 (pre-compression)     0.833     0.863     0.7291   0.7133   +0.016
    101-200                    0.914     1.831     0.7193   0.5138   +0.206
    901-1000                   0.849     1.484     0.7286   0.5492   +0.179

Train *loss* keeps falling while train *accuracy* never moves off 0.72-0.73. Two
mutually exclusive readings, ~18 accuracy points apart:

  (A) Artifact  -- HGQ's training=True forward pass is not the deployed function,
      so the train metric never pays the rounding cost. Nothing is recoverable;
      validation is the only honest signal.
  (B) Overfitting -- 18 real points are on the table. Implausible on its face at
      285,709 params over 1.8M samples with train accuracy *flat*, but untested.

Running the *deployed* (inference-mode, calibrated) model over training data
separates them: ~0.72 means (B), ~0.65 means (A).

Protocol
========
trace_minmax is ALWAYS applied -- an uncalibrated model is not the deployed model,
so there is no uncalibrated arm. The model is re-loaded per invocation so
calibration can never leak between splits.

Run --split test FIRST as a harness check: it must reproduce the accuracy already
recorded in that run's metrics.json to within ~1 point. Nothing else this script
prints is believable until that passes.

This script writes NOTHING. It must never call resolve_experiment_paths(), which
would overwrite <experiment>/outputs/quantized/128_17f_metrics.json.

Usage:
    python scripts/diagnose_train_val_gap.py \
        --model_path .agents/EXP-25_FULLCURVE_SEED42/models/quantized/128_17f.keras \
        --split test --seed 42 --max_samples 2000000
"""

import os
import sys
import argparse

import numpy as np

# Must precede any keras import. train.py/evaluate.py are tensorflow-only.
os.environ["KERAS_BACKEND"] = "tensorflow"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import h5py
import keras
from hgq.utils import trace_minmax

from src.data.dataset import JetFormerDataGenerator
from src.training.train import (
    HLS4ML_CLASSES,
    JETCLASS_CLASSES,
    PROCESSED_DIR,
    evaluate,
)


def resolve_data_paths(dataset, num_particles, num_feats):
    """Mirrors the path resolution in evaluate.py:68-78."""
    if dataset == "jetclass":
        base_path = os.path.join(
            PROCESSED_DIR, "jetclass", str(num_particles), f"{num_feats}f"
        )
        if not os.path.exists(base_path):
            base_path = os.path.join(PROCESSED_DIR, "jetclass", str(num_particles), "17f")
        import glob

        part_files = sorted(glob.glob(os.path.join(base_path, "train_part*.h5")))
        train_h5_path = part_files[0] if part_files else os.path.join(base_path, "train.h5")
    else:
        base_path = os.path.join(PROCESSED_DIR, str(num_particles), f"{num_feats}f")
        train_h5_path = os.path.join(base_path, "train.h5")
    return base_path, train_h5_path, os.path.join(base_path, "test.h5")


def detect_keys(h5_path):
    """Mirrors train.py:274-288."""
    with h5py.File(h5_path, "r") as f:
        if "jetConstituentList" in f:
            x_key = "jetConstituentList"
        elif "particle_features" in f:
            x_key = "particle_features"
        else:
            x_key = list(f.keys())[0]
        y_key = "jets" if "jets" in f else ("label" if "label" in f else list(f.keys())[1])
    return x_key, y_key


def build_split_indices(train_h5_path, x_key, y_key, max_samples, seed, val_ratio):
    """Reconstructs train.py's train/val split as *global HDF5 row indices*.

    This is a transcription of train.py:290-330 with one substitution: it tracks
    row indices instead of feature data, so the split can be reproduced exactly
    without materialising 17.4 GB of features.

    Any drift from train.py here silently invalidates the whole diagnostic --
    the val arm would no longer be the data the run actually validated on. That
    is what the --split test harness check and the assertions below guard.
    """
    with h5py.File(train_h5_path, "r") as f:
        y_all = f[y_key][:]
        total_train_samples = f[x_key].shape[0]

    if max_samples is not None and max_samples < total_train_samples:
        # train.py:294-311 -- fast 10-block contiguous slice, `quota` per class.
        unique_classes = np.unique(y_all)
        quota = max_samples // len(unique_classes)
        idx_chunks = []
        for cls in unique_classes:
            cls_indices = np.where(y_all == cls)[0]
            start_i = cls_indices[0]
            idx_chunks.append(np.arange(start_i, start_i + quota))
        global_idx = np.concatenate(idx_chunks)
    else:
        global_idx = np.arange(total_train_samples)

    # train.py:323-329 -- permute, then val takes the head and train the tail.
    perm = np.random.default_rng(seed).permutation(len(global_idx))
    global_idx = global_idx[perm]

    val_size = int(len(global_idx) * val_ratio)
    train_idx, val_idx = global_idx[val_size:], global_idx[:val_size]

    assert len(val_idx) == val_size, "val split size drifted from train.py"
    assert not (set(train_idx.tolist()) & set(val_idx.tolist())), (
        "train and val splits overlap -- split reconstruction is wrong"
    )
    return train_idx, val_idx


def load_rows(h5_path, x_key, y_key, indices, num_feats, chunk=2000):
    """Reads the selected rows, in ascending order.

    The train/val indices are a permutation, so they are scattered across ~2M
    rows. h5py fancy-indexing over that many scattered points degrades to
    near-per-element reads and takes tens of minutes, so instead the sorted
    indices are walked in chunks and each chunk is fetched as one contiguous
    slice, then subset in RAM. At ~10% density a 2,000-index chunk spans ~20,000
    rows (~174 MB at 128x17 float32), which keeps peak memory bounded while the
    I/O stays sequential.

    Order is ascending rather than the permuted order; x and y use the same
    order, so correspondence is preserved and accuracy is order-independent.
    """
    order = np.sort(indices)
    xs, ys = [], []
    with h5py.File(h5_path, "r") as f:
        dx, dy = f[x_key], f[y_key]
        for i in range(0, len(order), chunk):
            block = order[i : i + chunk]
            lo, hi = int(block[0]), int(block[-1]) + 1
            offs = block - lo
            xs.append(dx[lo:hi][offs])
            ys.append(dy[lo:hi][offs])
    x = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    if num_feats is not None and num_feats < x.shape[-1]:
        x = x[:, :, :num_feats]
    return x, y


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a checkpoint in inference mode on train/val/test splits"
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--dataset", type=str, default="jetclass")
    parser.add_argument("--num_particles", type=int, default=128)
    parser.add_argument("--num_feats", type=int, default=17)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="The --seed the run was trained with. Selects the train/val split.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=2000000,
        help="The --max_samples the run was trained with. Selects the train/val split.",
    )
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Fixed at 0.1 in train.py:1190")
    parser.add_argument("--n_eval", type=int, default=200000)
    parser.add_argument(
        "--calib_seed",
        type=int,
        default=42,
        help="Matches evaluate.py's --calib_seed so calibration is identical.",
    )
    parser.add_argument(
        "--quantize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply trace_minmax WRAP calibration (default: True). --no-quantize "
        "only for unquantized checkpoints, which have nothing to calibrate.",
    )
    args = parser.parse_args()

    classes = JETCLASS_CLASSES if args.dataset == "jetclass" else HLS4ML_CLASSES
    base_path, train_h5_path, test_h5_path = resolve_data_paths(
        args.dataset, args.num_particles, args.num_feats
    )

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {args.model_path}")

    # Re-loaded on every invocation so calibration never carries across splits.
    print(f"Loading model from: {args.model_path}")
    model = keras.models.load_model(args.model_path, compile=False)
    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["sparse_categorical_accuracy"],
    )

    if args.split == "test":
        x_key, y_key = detect_keys(test_h5_path)
        # evaluate.py:113-128 slices `quota` contiguous rows from the head of each
        # class block; reproduced here so the harness check is comparable.
        with h5py.File(test_h5_path, "r") as f:
            n_total = f[x_key].shape[0]
            y_test_all = f[y_key][:]
        unique_classes = np.unique(y_test_all)
        quota = min(args.n_eval, n_total) // len(unique_classes)
        sel = np.concatenate(
            [
                np.arange(np.where(y_test_all == cls)[0][0], np.where(y_test_all == cls)[0][0] + quota)
                for cls in unique_classes
            ]
        )
        source_path = test_h5_path
    else:
        x_key, y_key = detect_keys(train_h5_path)
        train_idx, val_idx = build_split_indices(
            train_h5_path, x_key, y_key, args.max_samples, args.seed, args.val_ratio
        )
        chosen = train_idx if args.split == "train" else val_idx
        print(
            f"[Split] Reconstructed from seed={args.seed}, max_samples={args.max_samples:,}, "
            f"val_ratio={args.val_ratio}: train={len(train_idx):,} val={len(val_idx):,}"
        )
        # Already permuted, so the head is class-balanced in expectation.
        sel = chosen[: args.n_eval]
        source_path = train_h5_path

    print(f"[Data] Loading {len(sel):,} rows from {os.path.basename(source_path)} ({args.split} split)...")
    x, y = load_rows(source_path, x_key, y_key, sel, args.num_feats)

    if args.quantize:
        print("\n[HGQ] WRAP-mode calibration (identical to evaluate.py:136-146)...")
        calib_gen = JetFormerDataGenerator(
            h5_path=train_h5_path,
            stats_dir=base_path,
            batch_size=args.batch_size,
            shuffle=True,
            num_feats=args.num_feats,
            in_memory=False,
            seed=args.calib_seed,
        )
        it = iter(calib_gen)
        x_calib = np.concatenate([next(it)[0] for _ in range(10)], axis=0)
        trace_minmax(model, x_calib)
        print(f"[HGQ] Calibrated on {len(x_calib):,} samples (calib_seed={args.calib_seed}).")

    gen = JetFormerDataGenerator(
        h5_path=source_path,
        stats_dir=base_path,
        batch_size=args.batch_size,
        shuffle=False,
        num_feats=args.num_feats,
        in_memory=True,
        preloaded_data=(x, y),
        augment_rotation=False,
        seed=args.seed,
    )
    outputs = model.predict(gen)
    labels = gen.y_data[gen.indices]

    # evaluate() already prints the per-class accuracy/AUC table.
    acc, _class_accs, _aucs = evaluate(outputs, labels, classes)

    print("\n" + "=" * 62)
    print(f"  model           : {args.model_path}")
    print(f"  split           : {args.split}")
    print(f"  samples         : {len(labels):,}")
    print(f"  calibrated      : {args.quantize} (calib_seed={args.calib_seed})")
    print(f"  OVERALL ACCURACY: {acc:.5f}")
    print("=" * 62 + "\n")


if __name__ == "__main__":
    main()
