#!/usr/bin/env python3
"""
Standalone JetClass Dataset Preprocessing Pipeline for JetFormer.

This script parses raw JetClass ROOT files (downloaded via download_jetclass.py),
extracts all 17 particle-level kinematic and PID features and 10 jet class labels,
pads/crops to 128 particles, and streams them into standardized HDF5 files.

Output Structure:
    data/processed/jetclass/128/17f/
        ├── train.h5
        ├── test.h5
        ├── mean.npy  (17-element float vector)
        └── std.npy   (17-element float vector)

Usage:
    python -m src.data.build_jetclass_dataset --input_dir datasets/JetClass/Pythia
"""

import os
import time
import argparse
import numpy as np
import h5py
from tqdm import tqdm

# Import ROOT file processing dependencies
try:
    import uproot
    import awkward as ak
except ImportError:
    uproot = None
    ak = None


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")


def _pad(a, maxlen, value=0, dtype="float32"):
    if isinstance(a, np.ndarray) and a.ndim >= 2 and a.shape[1] == maxlen:
        return a
    elif isinstance(a, ak.Array):
        if a.ndim == 1:
            a = ak.unflatten(a, 1)
        a = ak.fill_none(ak.pad_none(a, maxlen, clip=True), value)
        return ak.values_astype(a, dtype)
    else:
        x = (np.ones((len(a), maxlen)) * value).astype(dtype)
        for idx, s in enumerate(a):
            if not len(s):
                continue
            trunc = s[:maxlen].astype(dtype)
            x[idx, : len(trunc)] = trunc
        return x


def _clip(a, a_min, a_max):
    try:
        return np.clip(a, a_min, a_max)
    except ValueError:
        return ak.unflatten(np.clip(ak.flatten(a), a_min, a_max), ak.num(a))


def build_features_and_labels(tree, max_particles=128):
    """
    Extracts 17 particle-level features and 10 jet class labels from a ROOT TTree.
    Matches the exact feature engineering logic from jetclass_dataset.py.
    """
    a = tree.arrays(filter_name=["part_*", "jet_pt", "jet_energy", "label_*"])

    # 1. Derived Kinematic Features
    a["part_mask"] = ak.ones_like(a["part_energy"])
    a["part_pt"] = np.hypot(a["part_px"], a["part_py"])
    a["part_pt_log"] = np.log(a["part_pt"])
    a["part_e_log"] = np.log(a["part_energy"])
    a["part_logptrel"] = np.log(a["part_pt"] / a["jet_pt"])
    a["part_logerel"] = np.log(a["part_energy"] / a["jet_energy"])
    a["part_deltaR"] = np.hypot(a["part_deta"], a["part_dphi"])
    a["part_d0"] = np.tanh(a["part_d0val"])
    a["part_dz"] = np.tanh(a["part_dzval"])

    # 2. List of all 17 Particle Features
    feature_list = [
        "part_pt_log",
        "part_e_log",
        "part_logptrel",
        "part_logerel",
        "part_deltaR",
        "part_charge",
        "part_isChargedHadron",
        "part_isNeutralHadron",
        "part_isPhoton",
        "part_isElectron",
        "part_isMuon",
        "part_d0",
        "part_d0err",
        "part_dz",
        "part_dzerr",
        "part_deta",
        "part_dphi",
    ]

    pf_features = np.stack(
        [_pad(a[n], maxlen=max_particles).to_numpy() for n in feature_list], axis=2
    )  # (N, max_particles, 17)

    # 3. List of 10 Jet Classes
    label_list = [
        "label_QCD",
        "label_Hbb",
        "label_Hcc",
        "label_Hgg",
        "label_H4q",
        "label_Hqql",
        "label_Zqq",
        "label_Wqq",
        "label_Tbqq",
        "label_Tbl",
    ]
    labels_onehot = np.stack(
        [a[n].to_numpy().astype("int") for n in label_list], axis=1
    )  # (N, 10)
    labels = labels_onehot.argmax(axis=1)  # Convert one-hot to scalar index (0-9)

    return pf_features.astype(np.float32), labels.astype(np.int64)


def compute_welford_stats(train_h5_path, save_dir, num_feats=17, batch_size=5000):
    """
    Computes global mean and std for the 17 features using Welford's algorithm.
    """
    print(f"\n[Welford] Computing feature statistics for {train_h5_path}...")
    start_time = time.time()

    n = 0
    mean = None
    M2 = None

    with h5py.File(train_h5_path, "r") as fin:
        X = fin["particle_features"]
        n_samples = X.shape[0]

        for start in tqdm(range(0, n_samples, batch_size), desc="Computing Welford Stats"):
            end = min(start + batch_size, n_samples)
            batch_X_flat = X[start:end].reshape(-1, num_feats)
            batch_n = batch_X_flat.shape[0]

            if batch_n == 0:
                continue

            batch_mean = np.mean(batch_X_flat, axis=0)
            batch_M2 = np.sum((batch_X_flat - batch_mean) ** 2, axis=0)

            if mean is None:
                mean = batch_mean
                M2 = batch_M2
                n = batch_n
            else:
                delta = batch_mean - mean
                total_n = n + batch_n
                mean = mean + delta * (batch_n / total_n)
                M2 = M2 + batch_M2 + (delta**2) * n * batch_n / total_n
                n = total_n

    std = np.sqrt(M2 / (n - 1 + 1e-8))

    mean_path = os.path.join(save_dir, "mean.npy")
    std_path = os.path.join(save_dir, "std.npy")

    np.save(mean_path, mean)
    np.save(std_path, std)

    print(f"[Welford] Saved mean.npy and std.npy to {save_dir}")
    print(f"[Welford] Mean: {mean}")
    print(f"[Welford] Std:  {std}")
    print(f"[Welford] Time taken: {time.time() - start_time:.2f}s")


def process_jetclass_root_dir(input_dir, num_particles=128, num_feats=17, batch_size=5000):
    """
    Scans input directory for JetClass ROOT files, converts ROOT files
    to standardized HDF5 files (train.h5 & test.h5), and computes Welford statistics.
    """
    if uproot is None or ak is None:
        raise ImportError(
            "uproot and awkward are required to process ROOT files. "
            "Please install them via: pip install uproot awkward"
        )

    output_dir = os.path.join(PROCESSED_DIR, "jetclass", str(num_particles), f"{num_feats}f")
    os.makedirs(output_dir, exist_ok=True)

    # Locate ROOT files in input directory
    all_root_files = []
    for root, dirs, files in os.walk(input_dir):
        for f in files:
            if f.endswith(".root"):
                all_root_files.append(os.path.join(root, f))

    all_root_files = sorted(all_root_files)
    if not all_root_files:
        raise FileNotFoundError(
            f"No .root files found in {input_dir}. "
            f"Please run download_jetclass.py first to download JetClass ROOT files."
        )

    print(f"[JetClass] Found {len(all_root_files)} ROOT files in {input_dir}")

    # Split files into train (80%) and test (20%)
    n_files = len(all_root_files)
    n_test = max(1, int(n_files * 0.2))
    n_train = n_files - n_test

    file_splits = {
        "train": all_root_files[:n_train],
        "test": all_root_files[n_train:],
    }

    for split, root_files in file_splits.items():
        output_h5_path = os.path.join(output_dir, f"{split}.h5")
        print(f"\n[JetClass/{split}] Processing {len(root_files)} ROOT files -> {output_h5_path}")

        # Pass 1: Count total samples across all trees in split
        total_samples = 0
        for rfile in root_files:
            try:
                tree = uproot.open(rfile)["tree"]
                total_samples += tree.num_entries
            except Exception as e:
                print(f"Warning: Could not read {rfile}: {e}")

        print(f"[JetClass/{split}] Total events: {total_samples}")

        # Pass 2: Process and write to HDF5
        with h5py.File(output_h5_path, "w") as fout:
            dset_X = fout.create_dataset(
                "particle_features",
                shape=(total_samples, num_particles, num_feats),
                dtype=np.float32,
                compression="lzf",
                chunks=True,
            )
            dset_y = fout.create_dataset(
                "label",
                shape=(total_samples,),
                dtype=np.int64,
                compression="lzf",
                chunks=True,
            )

            write_idx = 0
            for rfile in tqdm(root_files, desc=f"Converting {split} ROOT files"):
                try:
                    tree = uproot.open(rfile)["tree"]
                    X_arr, y_arr = build_features_and_labels(tree, max_particles=num_particles)
                    n_entries = X_arr.shape[0]

                    dset_X[write_idx : write_idx + n_entries] = X_arr
                    dset_y[write_idx : write_idx + n_entries] = y_arr
                    write_idx += n_entries
                except Exception as e:
                    print(f"\nError reading {rfile}: {e}")

            print(f"[JetClass/{split}] Saved {write_idx} jets to {output_h5_path}")

    # Compute Welford statistics on training set only
    train_h5_path = os.path.join(output_dir, "train.h5")
    compute_welford_stats(train_h5_path, output_dir, num_feats=num_feats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Standalone JetClass HDF5 Dataset")
    parser.add_argument(
        "-i",
        "--input_dir",
        type=str,
        default="datasets/JetClass/Pythia",
        help="Input directory containing JetClass .root files (default: datasets/JetClass/Pythia)",
    )
    parser.add_argument(
        "--num_particles",
        type=int,
        default=128,
        help="Number of particles per jet (default: 128)",
    )
    parser.add_argument(
        "--num_feats",
        type=int,
        default=17,
        help="Number of features per particle (default: 17)",
    )

    args = parser.parse_args()

    process_jetclass_root_dir(
        input_dir=args.input_dir,
        num_particles=args.num_particles,
        num_feats=args.num_feats,
    )
