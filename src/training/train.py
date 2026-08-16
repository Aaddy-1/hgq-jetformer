import os
from contextlib import ExitStack
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score
import keras
import json

# Route Keras to the installed backend
os.environ["KERAS_BACKEND"] = "tensorflow"

# If using JAX: Prevent the backend from pre-allocating 100% of VRAM,
# which can crash out-of-core data loaders.
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.90"

# HGQ2 Imports
from hgq.config import QuantizerConfigScope, LayerConfigScope
from hgq.utils import trace_minmax
from hgq.utils.sugar.beta_pid import BetaPID
from hgq.regularizers import MonoL1
from hgq.utils.sugar.early_stopping_ebops import EarlyStoppingWithEbopsThres

# Relative imports
from src.data.dataset import (
    JetFormerDataGenerator,
    detect_hardware_and_strategy,
    get_stratified_indices,
)
from src.model.jetformer import build_hgq_jetformer
from src.training.onecyclelr import OneCycleLR, build_lr_schedule
from src.training.callbacks import (
    QATEarlyStoppingAndCheckpoint,
    EbopsCaptureCallback,
    AdaptiveReduceLROnPlateau,
    FinalEpochCheckpoint,
)

# Path variables
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define classes for each supported dataset
HLS4ML_CLASSES = ["Gluon", "Light_quarks", "W_boson", "Z_boson", "Top_quark"]

# Order is significant: index i must name label_list[i] in
# src/data/build_jetclass_dataset.py, which is the source of truth for the
# integer labels written to disk. Keep the two lists in sync.
JETCLASS_CLASSES = [
    "QCD",    # label_QCD  - merged light quark / gluon
    "H_bb",   # label_Hbb
    "H_cc",   # label_Hcc
    "H_gg",   # label_Hgg
    "H_4q",   # label_H4q
    "H_qql",  # label_Hqql - H -> qq'lv
    "Z_qq",   # label_Zqq
    "W_qq",   # label_Wqq
    "t_bqq",  # label_Tbqq
    "t_bl",   # label_Tbl  - t -> blv
]

# Default epoch after which EBOPs and val_loss are expected to have stabilized
# under PID control. Overridable per run via --ebops_warmup_epoch; this name is
# also exported by callbacks.py, so keep the two in sync if the default changes.
EBOPS_WARMUP_EPOCH = 75

# The two EBOPs controls are distinct and must not be set equal.
#
#   target_ebops     - the BetaPID *setpoint*; the value the controller regulates
#                      to, and therefore what decides where a run lands on the
#                      EBOPs axis. EXP-18_REPRODUCTION settled at 348,977 against
#                      a 350,000 setpoint while the 450,000 gate never bound.
#   ebops_threshold  - the gate QATEarlyStoppingAndCheckpoint applies before it
#                      will save at all.
#
# Regulation oscillates around the setpoint -- archived runs sit on both sides of
# it (348,977 below; EXP-22_STANDARD_FIX at 442,233 against the same 350,000
# target, 26% above). If the gate equals the setpoint, roughly half of converged
# epochs become ineligible to save, and because the callback only resets its
# patience counter on a successful save, those blocked epochs also burn patience.
# Keeping ~20% headroom admits normal ripple while still catching genuine runaway
# (cf. EXP-18_lowerwarmup_seed44 at 6,854,896).
DEFAULT_TARGET_EBOPS = 450000.0
DEFAULT_EBOPS_THRESHOLD = 550000.0



def set_global_seed(seed: int = 42):
    """Sets random seeds across Python, NumPy, TensorFlow, and Keras for exact reproducibility."""
    import random
    import tensorflow as tf

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    keras.utils.set_random_seed(seed)
    print(f"[Seed] Global random seed set to {seed} (Python, NumPy, TensorFlow, Keras)")


def get_run_provenance():
    """Records which machine, card and commit produced a run.

    EXP-24 and EXP-25 used different GPUs for the same seed (s42: GPU 3 then 2;
    s43: GPU 2 then 3), which is why their curves diverge at epoch 1 despite an
    identical --seed. That was recoverable only because it was remembered -- no
    artifact recorded it. The server is shared, so card assignment cannot be
    controlled and the confound has to be *recorded* rather than eliminated.

    Never raises: a provenance failure must not take down a training run, so
    every probe degrades to None/[] instead.
    """
    import socket
    import subprocess

    provenance = {
        "hostname": None,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "git_commit": None,
        "git_dirty": None,
        "gpus": [],
    }

    try:
        provenance["hostname"] = socket.gethostname()
    except Exception:
        pass

    try:
        provenance["git_commit"] = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=PROJECT_ROOT,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
        # Uncommitted edits mean the commit hash alone does not identify the code
        # that ran -- scripts/analyze_ebops.py has carried local changes for weeks.
        provenance["git_dirty"] = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=PROJECT_ROOT,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        pass

    try:
        # Imported inside the try so a missing/broken TF cannot break a run, and
        # so the function stays callable on machines without TensorFlow.
        import tensorflow as tf

        # Listed *after* CUDA_VISIBLE_DEVICES is applied, so this is the card the
        # run actually used, not every card on the box.
        for dev in tf.config.list_physical_devices("GPU"):
            details = tf.config.experimental.get_device_details(dev)
            cc = details.get("compute_capability")
            provenance["gpus"].append(
                {
                    "name": details.get("device_name"),
                    "compute_capability": ".".join(str(c) for c in cc) if cc else None,
                }
            )
    except Exception:
        pass

    return provenance





def extract_model_metadata(
    model, best_ebops, best_epoch, num_test_samples=None, artifacts=None
):
    layers_metadata = []
    for layer in model.layers:
        try:
            out_shape = str(layer.output.shape)
        except (AttributeError, ValueError):
            out_shape = "N/A"

        layers_metadata.append(
            {
                "name": layer.name,
                "type": layer.__class__.__name__,
                "output_shape": out_shape,
                "params": int(layer.count_params()),
            }
        )
    metadata = {
        "ebops": best_ebops,
        "best_epoch": best_epoch,
        "total_parameters": int(model.count_params()),
        "layers": layers_metadata,
    }
    if num_test_samples is not None:
        metadata["num_test_samples"] = int(num_test_samples)
    # One EBOPs figure per *file* rather than one per run. The single top-level
    # "ebops" field cannot express a run that wrote more than one checkpoint,
    # which is how EXP-18_lowerwarmup_seed44's metadata came to claim 365,644
    # while the file on disk held 6,854,896 -- undetected for three days.
    if artifacts:
        metadata["artifacts"] = artifacts
    return metadata


def setup_data_generators(
    num_particles,
    num_feats,
    batch_size,
    val_ratio=0.1,
    dataset="hls4ml",
    train_parts=None,
    max_samples=None,
    in_memory=False,
    max_test_samples=2000000,
    augment_rotation=False,
    seed=42,
):
    if dataset == "jetclass":
        base_path = os.path.join(
            PROCESSED_DIR, "jetclass", str(num_particles), f"{num_feats}f"
        )
        if not os.path.exists(base_path):
            base_path = os.path.join(
                PROCESSED_DIR, "jetclass", str(num_particles), "17f"
            )

        # Find train_part*.h5 files or fallback to train.h5
        if train_parts is not None:
            train_h5_paths = [
                os.path.join(base_path, f"train_part{p}.h5") for p in train_parts
            ]
        else:
            # Detect all available train_part*.h5 files in base_path
            import glob

            part_files = sorted(glob.glob(os.path.join(base_path, "train_part*.h5")))
            if part_files:
                train_h5_paths = part_files
            else:
                train_h5_paths = [os.path.join(base_path, "train.h5")]
    else:
        base_path = os.path.join(PROCESSED_DIR, str(num_particles), f"{num_feats}f")
        train_h5_paths = [os.path.join(base_path, "train.h5")]

    test_h5_path = os.path.join(base_path, "test.h5")

    print("BASE PATH:", base_path)
    print("================================")
    print("TRAIN H5 PATHS:", train_h5_paths)
    if augment_rotation:
        print(
            "[Dataset] Training augmentation enabled: Dynamic SO(2) rotation in (deta, dphi) plane"
        )

    import h5py

    # Detect x_key and total sample count
    total_train_samples = 0
    x_key = None
    y_key = None
    for p in train_h5_paths:
        with h5py.File(p, "r") as f:
            if x_key is None:
                if "jetConstituentList" in f:
                    x_key = "jetConstituentList"
                elif "particle_features" in f:
                    x_key = "particle_features"
                else:
                    x_key = list(f.keys())[0]
                y_key = (
                    "jets"
                    if "jets" in f
                    else ("label" if "label" in f else list(f.keys())[1])
                )
            total_train_samples += f[x_key].shape[0]

    if in_memory:
        with h5py.File(train_h5_paths[0], "r") as f:
            y_all = f[y_key][:]

            if max_samples is not None and max_samples < total_train_samples:
                unique_classes = np.unique(y_all)
                quota = max_samples // len(unique_classes)
                sample_bytes = num_particles * num_feats * 4
                ram_gb = (max_samples * sample_bytes) / (1024**3)
                print(
                    f"[Dataset] Fast 10-block slice pre-loading: {quota:,} contiguous samples/class ({max_samples:,} total = {ram_gb:.2f} GB RAM)..."
                )

                x_chunks, y_chunks = [], []
                for cls in unique_classes:
                    cls_indices = np.where(y_all == cls)[0]
                    start_i = cls_indices[0]
                    x_chunks.append(f[x_key][start_i : start_i + quota])
                    y_chunks.append(f[y_key][start_i : start_i + quota])

                shared_x = np.concatenate(x_chunks, axis=0)
                shared_y = np.concatenate(y_chunks, axis=0)
                del x_chunks, y_chunks, y_all
            else:
                sample_bytes = num_particles * num_feats * 4
                ram_gb = (total_train_samples * sample_bytes) / (1024**3)
                print(
                    f"[Dataset] Pre-loading full dataset into RAM ({total_train_samples:,} samples = {ram_gb:.2f} GB RAM)..."
                )
                shared_x = f[x_key][:]
                shared_y = y_all
                del y_all

        perm = np.random.default_rng(seed).permutation(len(shared_y))
        shared_x = shared_x[perm]
        shared_y = shared_y[perm]

        val_size = int(len(shared_y) * val_ratio)
        train_x, val_x = shared_x[val_size:], shared_x[:val_size]
        train_y, val_y = shared_y[val_size:], shared_y[:val_size]
        del shared_x, shared_y

        print(
            f"[Dataset] Pre-load complete. Splitting train ({len(train_y):,}) / val ({len(val_y):,})..."
        )

        train_gen = JetFormerDataGenerator(
            h5_path=train_h5_paths,
            stats_dir=base_path,
            batch_size=batch_size,
            shuffle=True,
            num_feats=num_feats,
            preloaded_data=(train_x, train_y),
            augment_rotation=augment_rotation,
            # Without this the generator builds np.random.default_rng(None),
            # which draws from OS entropy and does NOT inherit np.random.seed().
            # Batch order would then differ between runs at identical --seed.
            seed=seed,
        )
        val_gen = JetFormerDataGenerator(
            h5_path=train_h5_paths,
            stats_dir=base_path,
            batch_size=batch_size,
            shuffle=False,
            num_feats=num_feats,
            preloaded_data=(val_x, val_y),
            augment_rotation=False,
            seed=seed,
        )
    else:
        # Build train/val index splits without reading features into RAM.
        # Labels only (~0.5% of feature volume), so this stays low-footprint.
        #
        # Two constraints, both already solved by the in_memory branch above:
        #   1. JetFormerDataGenerator keeps self.indices ordered and shuffles
        #      block_order instead, so contiguous runs hit the fast HDF5 slice
        #      path. Scattered indices would force slow fancy-indexing.
        #   2. The raw files are class-ordered on disk, so a plain head/tail
        #      split would yield a class-skewed validation set.
        # Splitting each class's contiguous block individually satisfies both.
        y_parts = []
        for p in train_h5_paths:
            with h5py.File(p, "r") as f:
                y_part = f[y_key][:]
            if y_part.ndim > 1 and y_part.shape[-1] > 1:
                y_part = np.argmax(y_part, axis=-1)
            y_parts.append(y_part.astype(np.int64))
        y_all = np.concatenate(y_parts)
        del y_parts

        unique_classes = np.unique(y_all)
        per_class_quota = (
            max_samples // len(unique_classes)
            if max_samples is not None and max_samples < len(y_all)
            else None
        )

        train_chunks, val_chunks = [], []
        for cls in unique_classes:
            cls_indices = np.where(y_all == cls)[0]
            if per_class_quota is not None:
                cls_indices = cls_indices[:per_class_quota]
            val_size = int(len(cls_indices) * val_ratio)
            # Val taken from the head, train from the tail, matching the
            # in_memory split above.
            val_chunks.append(cls_indices[:val_size])
            train_chunks.append(cls_indices[val_size:])

        train_indices = np.concatenate(train_chunks)
        val_indices = np.concatenate(val_chunks)
        del y_all, train_chunks, val_chunks

        print(
            f"[Dataset] Block-stream stratified split across "
            f"{len(unique_classes)} classes: train ({len(train_indices):,}) / "
            f"val ({len(val_indices):,})"
        )

        train_gen = JetFormerDataGenerator(
            h5_path=train_h5_paths,
            stats_dir=base_path,
            batch_size=batch_size,
            shuffle=True,
            indices=train_indices,
            num_feats=num_feats,
            in_memory=False,
            augment_rotation=augment_rotation,
            seed=seed,
        )
        val_gen = JetFormerDataGenerator(
            h5_path=train_h5_paths,
            stats_dir=base_path,
            batch_size=batch_size,
            shuffle=False,
            indices=val_indices,
            num_feats=num_feats,
            in_memory=False,
            augment_rotation=False,
            seed=seed,
        )

    # test_gen ALWAYS streams sequentially from disk (never pre-loaded)
    with h5py.File(test_h5_path, "r") as f:
        key = (
            "jetConstituentList"
            if "jetConstituentList" in f
            else (
                "particle_features" if "particle_features" in f else list(f.keys())[0]
            )
        )
        total_test_samples = f[key].shape[0]

    if max_test_samples is not None and max_test_samples > 0:
        test_indices = np.arange(min(max_test_samples, total_test_samples))
    else:
        test_indices = None

    test_gen = JetFormerDataGenerator(
        h5_path=test_h5_path,
        stats_dir=base_path,
        batch_size=batch_size,
        shuffle=False,
        indices=test_indices,
        num_feats=num_feats,
        in_memory=False,
        seed=seed,
    )
    return train_gen, val_gen, test_gen


def save_final_evaluation(acc, class_accs, aucs, classes, metadata, config, filepath):
    results = {
        "configuration": config,
        "performance": {"overall_accuracy": float(acc), "per_class_metrics": {}},
        "metadata": metadata,
    }
    for i, class_name in enumerate(classes):
        results["performance"]["per_class_metrics"][class_name] = {
            "accuracy": float(class_accs[i]) if not np.isnan(class_accs[i]) else None,
            "auc": float(aucs[i]) if aucs[i] is not None else None,
        }
    with open(filepath, "w") as f:
        json.dump(results, f, indent=4)


def save_loss_acc(
    history_dict,
    num_particles,
    num_feats,
    output_path,
    ebops=None,
    beta=None,
    best_epochs=None,
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_kwargs = {
        "train_losses": np.array(history_dict["loss"]),
        "val_losses": np.array(history_dict["val_loss"]),
        "train_accs": np.array(history_dict.get("sparse_categorical_accuracy", [])),
        "val_accs": np.array(history_dict.get("val_sparse_categorical_accuracy", [])),
    }
    if ebops is not None and len(ebops) > 0:
        save_kwargs["ebops"] = np.array(ebops)
    if beta is not None and len(beta) > 0:
        save_kwargs["beta"] = np.array(beta)
    if best_epochs is not None and len(best_epochs) > 0:
        save_kwargs["best_epochs"] = np.array(best_epochs)

    np.savez(output_path, **save_kwargs)
    print(f"Loss, accuracy, and QAT metrics saved to {output_path}")


def plot_loss_acc(history_dict, num_particles, num_feats, plot_path):
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    epochs = np.arange(len(history_dict["loss"]))

    plt.figure(figsize=(6, 6))
    plt.subplot(2, 1, 1)
    plt.plot(epochs, history_dict["loss"], label="Train Loss")
    plt.plot(epochs, history_dict["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(
        epochs, history_dict.get("sparse_categorical_accuracy", []), label="Train Acc"
    )
    plt.plot(
        epochs, history_dict.get("val_sparse_categorical_accuracy", []), label="Val Acc"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"Loss and accuracy plots saved to {plot_path}")


def plot_ebops_beta(
    history_dict,
    best_epochs=None,
    target_ebops=DEFAULT_TARGET_EBOPS,
    warmup_epoch=10,
    plot_path=None,
):
    """Plots model EBOPs and BetaPID values across epochs.

    Subplot 1: Model EBOPs vs Epoch (linear scale, target EBOPs reference line, warmup line, best checkpoint markers)
    Subplot 2: BetaPID (beta) vs Epoch (log scale, warmup line, best checkpoint markers)
    """
    ebops = history_dict.get("ebops")
    beta = history_dict.get("beta")

    if ebops is None or len(ebops) == 0:
        print("[QAT Plot] No EBOPs data found in history. Skipping EBOPs/BetaPID plot.")
        return

    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    epochs = np.arange(len(ebops))
    best_epochs = best_epochs or []

    plt.figure(figsize=(8, 8))

    # --- Subplot 1: EBOPs ---
    plt.subplot(2, 1, 1)
    plt.plot(epochs, ebops, label="Model EBOPs", color="#1f77b4", linewidth=2)
    plt.axhline(
        y=target_ebops,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"Target EBOPs ({int(target_ebops):,})",
    )
    if warmup_epoch is not None and warmup_epoch < len(epochs):
        plt.axvline(
            x=warmup_epoch,
            color="gray",
            linestyle=":",
            linewidth=1.5,
            label=f"PID Warmup End (Epoch {warmup_epoch})",
        )

    # Plot Best Model Checkpoint markers
    best_ebops_pts = [ebops[ep] for ep in best_epochs if ep < len(ebops)]
    if best_ebops_pts:
        plt.scatter(
            best_epochs,
            best_ebops_pts,
            color="gold",
            edgecolors="black",
            s=80,
            zorder=5,
            marker="*",
            label="New Best Checkpoint",
        )

    plt.xlabel("Epoch")
    plt.ylabel("EBOPs")
    plt.title("Model EBOPs over Training")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper right")

    # --- Subplot 2: BetaPID ---
    plt.subplot(2, 1, 2)
    if beta is not None and len(beta) > 0:
        plt.plot(epochs, beta, label="BetaPID (Beta)", color="#ff7f0e", linewidth=2)
        plt.yscale("log")
        if warmup_epoch is not None and warmup_epoch < len(epochs):
            plt.axvline(
                x=warmup_epoch,
                color="gray",
                linestyle=":",
                linewidth=1.5,
                label=f"PID Warmup End (Epoch {warmup_epoch})",
            )
        best_beta_pts = [beta[ep] for ep in best_epochs if ep < len(beta)]
        if best_beta_pts:
            plt.scatter(
                best_epochs,
                best_beta_pts,
                color="gold",
                edgecolors="black",
                s=80,
                zorder=5,
                marker="*",
                label="New Best Checkpoint",
            )
        plt.ylabel("Beta (log scale)")
    else:
        plt.text(0.5, 0.5, "Beta data not available", ha="center", va="center")
        plt.ylabel("Beta")

    plt.xlabel("Epoch")
    plt.title("BetaPID Parameter over Training")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"EBOPs and BetaPID plot saved to {plot_path}")


def evaluate(outputs: np.ndarray, labels: np.ndarray, classes: list):
    pred_labels = outputs.argmax(axis=1)
    acc = accuracy_score(labels, pred_labels)
    n_classes = outputs.shape[1]

    class_accs = []
    for i in range(n_classes):
        idx = labels == i
        if idx.sum() > 0:
            class_acc = accuracy_score(labels[idx], pred_labels[idx])
        else:
            class_acc = float("nan")
        class_accs.append(class_acc)

    try:
        y_true_onehot = np.eye(n_classes)[labels.astype(int)]
        aucs = roc_auc_score(y_true_onehot, outputs, average=None, multi_class="ovr")
    except Exception:
        aucs = [None] * n_classes

    print(f"Total Accuracy: {acc:.4f}")
    for i in range(n_classes):
        class_name = classes[i]
        auc_str = f"{aucs[i]:.4f}" if aucs[i] is not None else "N/A"
        acc_str = f"{class_accs[i]:.4f}" if not np.isnan(class_accs[i]) else "N/A"
        print(f"Class {i} ({class_name}): Accuracy={acc_str}, AUC={auc_str}")

    return acc, class_accs, aucs


def resolve_experiment_paths(experiment: str, quantize: bool) -> tuple[str, str]:
    if experiment:
        exp_root = os.path.join(PROJECT_ROOT, "experiment", experiment)
        current_model_dir = os.path.join(exp_root, "models")
        current_output_dir = os.path.join(exp_root, "outputs")
    else:
        current_model_dir = MODEL_DIR
        current_output_dir = OUTPUT_DIR

    if quantize:
        current_model_dir = os.path.join(current_model_dir, "quantized")
        current_output_dir = os.path.join(current_output_dir, "quantized")
    else:
        current_model_dir = os.path.join(current_model_dir, "unquantized")
        current_output_dir = os.path.join(current_output_dir, "unquantized")

    os.makedirs(current_model_dir, exist_ok=True)
    os.makedirs(current_output_dir, exist_ok=True)
    return current_model_dir, current_output_dir


def build_callbacks(
    early_stopping_patience: int,
    quantize: bool,
    model_path: str = None,
    use_adaptive_lr: bool = False,
    target_ebops: float = DEFAULT_TARGET_EBOPS,
    ebops_threshold: float = DEFAULT_EBOPS_THRESHOLD,
    ebops_warmup_epoch: int = EBOPS_WARMUP_EPOCH,
    save_final_epoch: bool = False,
):
    callbacks = []
    final_cb = None

    if quantize:
        # --- Quantized Training Callbacks (Unified EarlyStopping & Checkpoint) ---
        qat_cb = QATEarlyStoppingAndCheckpoint(
            ebops_threshold=ebops_threshold,
            patience=early_stopping_patience,
            min_delta=1e-3,
            start_from_epoch=ebops_warmup_epoch,
            model_path=model_path,
        )
        callbacks.append(qat_cb)

        if use_adaptive_lr:
            print(
                "[Callbacks] Enabling dynamic noise-bounded AdaptiveReduceLROnPlateau"
            )
            callbacks.append(
                AdaptiveReduceLROnPlateau(
                    factor=0.8,
                    patience=40,
                    cooldown=40,
                    min_lr=1e-5,
                    window=8,
                    c_safety=1.25,
                    min_delta_floor=1e-4,
                    start_from_epoch=ebops_warmup_epoch,
                )
            )
        else:
            callbacks.append(
                keras.callbacks.ReduceLROnPlateau(
                    monitor="val_sparse_categorical_accuracy",
                    mode="max",
                    factor=0.8,
                    patience=40,
                    min_lr=1e-5,
                    cooldown=40,
                    min_delta=1e-4,
                )
            )

        callbacks.append(
            BetaPID(
                p=1,
                i=0.1,
                d=0,
                target_ebops=target_ebops,
                init_beta=1e-10,
                warmup=10,
                max_beta=5e-6,
                damp_beta_on_target=0.5,
            )
        )
        ebops_capture = qat_cb
    else:
        # --- Non-quantized Training Callbacks (Original) ---
        if early_stopping_patience > 0:
            callbacks.append(
                keras.callbacks.EarlyStopping(
                    monitor="val_sparse_categorical_accuracy",
                    mode="max",
                    patience=early_stopping_patience,
                    min_delta=1e-4,
                    restore_best_weights=True,
                )
            )
        callbacks.append(
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_sparse_categorical_accuracy",
                mode="max",
                factor=0.8,
                patience=5,
                min_lr=1e-4,
            )
        )
        ebops_capture = EbopsCaptureCallback(
            start_from_epoch=0,
            model_path=model_path,
        )
        callbacks.append(ebops_capture)

    # Applies to both paths: the last-epoch model is written to its own file,
    # derived inside the callback from model_path, so it can never overwrite the
    # gated checkpoint.
    if save_final_epoch and model_path:
        final_cb = FinalEpochCheckpoint(checkpoint_path=model_path)
        callbacks.append(final_cb)

    return callbacks, ebops_capture, final_cb


def run_post_training_pipeline(
    model,
    train_gen,
    test_gen,
    quantize: bool,
    save: bool,
    model_path: str,
    eval_results_path: str,
    best_ebops: float,
    best_epoch: int,
    config: dict,
    classes: list = None,
    artifacts: list = None,
):
    if classes is None:
        classes = HLS4ML_CLASSES

    # Invoke standalone evaluation pipeline from evaluate.py
    from src.training.evaluate import run_standalone_evaluation

    print(
        "\n[Post-Training] Transitioning to standalone evaluation pipeline (evaluate.py)..."
    )
    run_standalone_evaluation(
        num_particles=config.get("num_particles", 128),
        num_feats=config.get("in_dim", 17),
        batch_size=config.get("batch_size", 256),
        max_test_samples=config.get("max_test_samples", 2000000),
        experiment=config.get("experiment"),
        quantize=quantize,
        dataset=config.get("dataset", "jetclass"),
        model_path=model_path,
        in_memory=True,
        best_ebops=best_ebops,
        best_epoch=best_epoch,
        artifacts=artifacts,
    )


def train(
    num_particles: int = 16,
    num_feats: int = 3,
    do_train: bool = True,
    val_ratio: float = 0.1,
    num_epochs: int = 250,
    early_stopping_patience: int = 150,
    num_transformers: int = 1,
    embbed_dim: int = 32,
    num_heads: int = 2,
    use_cls_token: bool = False,
    use_linformer: bool = True,
    activation: str = "ReLU",
    normalization: str = "Batch",
    batch_size: int = 256,
    dropout: float = 0.0,
    save: bool = True,
    model_path: str = None,
    plot_path: str = None,
    output_path: str = None,
    experiment: str = None,
    quantize: bool = True,
    dataset: str = "hls4ml",
    train_parts: list = None,
    max_samples: int = None,
    in_memory: bool = False,
    max_test_samples: int = 2000000,
    augment_rotation: bool = False,
    seed: int = 42,
    use_adaptive_lr: bool = False,
    target_ebops: float = DEFAULT_TARGET_EBOPS,
    ebops_threshold: float = DEFAULT_EBOPS_THRESHOLD,
    ebops_warmup_epoch: int = EBOPS_WARMUP_EPOCH,
    save_final_epoch: bool = False,
):
    # Set global random seeds
    set_global_seed(seed)

    # Resolve class registry based on dataset
    classes = JETCLASS_CLASSES if dataset == "jetclass" else HLS4ML_CLASSES
    train_gen, val_gen, test_gen = setup_data_generators(
        num_particles=num_particles,
        num_feats=num_feats,
        batch_size=batch_size,
        val_ratio=val_ratio,
        dataset=dataset,
        train_parts=train_parts,
        max_samples=max_samples,
        in_memory=in_memory,
        max_test_samples=max_test_samples,
        augment_rotation=augment_rotation,
        seed=seed,
    )

    current_model_dir, current_output_dir = resolve_experiment_paths(
        experiment, quantize
    )

    if model_path is None:
        model_path = os.path.join(
            current_model_dir, f"{num_particles}_{num_feats}f.keras"
        )
    if output_path is None:
        output_path = os.path.join(
            current_output_dir, f"{num_particles}_{num_feats}f_loss_acc.npz"
        )
    if plot_path is None:
        plot_path = os.path.join(
            current_output_dir, f"{num_particles}_{num_feats}f_plot.png"
        )
    ebops_beta_plot_path = os.path.join(
        current_output_dir, f"{num_particles}_{num_feats}f_ebops_beta.png"
    )

    eval_results_path = os.path.join(
        current_output_dir, f"{num_particles}_{num_feats}f_metrics.json"
    )

    optimizer = keras.optimizers.AdamW(learning_rate=1e-3)

    # --- Quantizer & Layer Scopes ---
    # Quantized path: separate kernel (weights) and datalane (activations) scopes
    # matching the reference sub-microsecond transformers notebook.
    # Unquantized path: original single scope preserved for non-HGQ training.
    stack = ExitStack()
    if quantize:
        kernel_scope = QuantizerConfigScope(
            k0=1, b0=8, i0=1, br=MonoL1(1e-8), overflow_mode="WRAP"
        )
        datalane_scope = QuantizerConfigScope(
            place="datalane", k0=1, f0=6, fr=MonoL1(1e-8), ir=MonoL1(1e-8)
        )
        layer_scope = LayerConfigScope(enable_ebops=True, beta0=1e-10)
        stack.enter_context(kernel_scope)
        stack.enter_context(datalane_scope)
        stack.enter_context(layer_scope)
    else:
        quant_scope = QuantizerConfigScope(
            place="all", default_q_type="kbi", overflow_mode="WRAP", br=MonoL1(1e-8)
        )
        layer_scope = LayerConfigScope(enable_ebops=True, beta0=1e-10)
        stack.enter_context(quant_scope)
        stack.enter_context(layer_scope)

    with stack:
        config = {
            "in_dim": num_feats,
            "embed_dim": embbed_dim,
            "num_heads": num_heads,
            "num_classes": len(classes),
            "num_transformers": num_transformers,
            "use_cls_token": use_cls_token,
            "use_linformer": use_linformer,
            "dropout": dropout,
            "num_particles": num_particles,
            "activation": activation,
            "normalization": normalization,
            "quantize": quantize,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "early_stopping_patience": early_stopping_patience,
            "experiment": experiment,
            "dataset": dataset,
            "max_samples": max_samples,
            "max_test_samples": max_test_samples,
            "augment_rotation": augment_rotation,
            "seed": seed,
            "target_ebops": target_ebops,
            "ebops_threshold": ebops_threshold,
            "ebops_warmup_epoch": ebops_warmup_epoch,
            "save_final_epoch": save_final_epoch,
            "use_adaptive_lr": use_adaptive_lr,
            "provenance": get_run_provenance(),
        }

        print("[DEBUG] Model Args: ")
        for k, v in config.items():
            print(f"  {k}={v}")

        model = build_hgq_jetformer(
            in_dim=num_feats,
            embed_dim=embbed_dim,
            num_heads=num_heads,
            num_classes=len(classes),
            num_transformers=num_transformers,
            dropout=dropout,
            num_particles=num_particles,
            activation=activation,
            normalization=normalization,
            quantize=quantize,
            use_linformer=use_linformer,
            use_cls_token=use_cls_token,
        )

        print("=================MODEL SUMMARY=================")
        model.summary()

        model.compile(
            optimizer=optimizer,
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["sparse_categorical_accuracy"],
        )

        callbacks, ebops_capture, final_cb = build_callbacks(
            early_stopping_patience,
            quantize,
            model_path=model_path if save else None,
            use_adaptive_lr=use_adaptive_lr,
            target_ebops=target_ebops,
            ebops_threshold=ebops_threshold,
            ebops_warmup_epoch=ebops_warmup_epoch,
            save_final_epoch=save_final_epoch,
        )

        if do_train:
            print(
                f"Starting training for {num_particles} particles, {num_feats} features with early stopping {early_stopping_patience} and {'with' if quantize else 'without'} quantization ... "
            )
            history = model.fit(
                train_gen,
                validation_data=val_gen,
                epochs=num_epochs,
                callbacks=callbacks,
            )

            if save:
                save_loss_acc(
                    history.history,
                    num_particles,
                    num_feats,
                    output_path,
                    ebops=history.history.get("ebops"),
                    beta=history.history.get("beta"),
                    best_epochs=ebops_capture.best_epochs,
                )
                plot_loss_acc(history.history, num_particles, num_feats, plot_path)
                if quantize:
                    plot_ebops_beta(
                        history.history,
                        best_epochs=ebops_capture.best_epochs,
                        target_ebops=target_ebops,
                        # BetaPID's own warmup (when beta starts being applied),
                        # which is distinct from --ebops_warmup_epoch (when the
                        # checkpoint callback starts considering saves).
                        warmup_epoch=10,
                        plot_path=ebops_beta_plot_path,
                    )
                else:
                    print(
                        "[QAT Plot] Unquantized training mode: skipping EBOPs and BetaPID plot generation."
                    )

        # One entry per checkpoint actually written, each carrying its own EBOPs
        # as measured at save time.
        artifacts = []
        if save and model_path and ebops_capture.best_epoch is not None:
            artifacts.append(
                {
                    "path": os.path.relpath(model_path, PROJECT_ROOT),
                    "kind": "gated_best",
                    "epoch": ebops_capture.best_epoch,
                    "ebops": ebops_capture.best_ebops,
                }
            )
        # final_epoch is only set once an epoch has actually ended, so this stays
        # empty when do_train is False and no file was written.
        if final_cb is not None and final_cb.final_epoch is not None:
            artifacts.append(
                {
                    "path": os.path.relpath(final_cb.model_path, PROJECT_ROOT),
                    "kind": "final_epoch",
                    "epoch": final_cb.final_epoch,
                    "ebops": final_cb.final_ebops,
                }
            )

        run_post_training_pipeline(
            model,
            train_gen,
            test_gen,
            quantize,
            save,
            model_path,
            eval_results_path,
            ebops_capture.best_ebops,
            ebops_capture.best_epoch,
            config,
            classes=classes,
            artifacts=artifacts,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train HGQJetFormer")
    parser.add_argument(
        "--dataset",
        type=str,
        default="hls4ml",
        choices=["hls4ml", "jetclass"],
        help="Target dataset: hls4ml (default) or jetclass",
    )
    parser.add_argument(
        "--num_particles", type=int, default=None, help="Number of jet constituents"
    )
    parser.add_argument(
        "--num_feats",
        type=int,
        default=None,
        help="Number of features per constituent",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=250, help="Total training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=256, help="Training batch size"
    )
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate")
    parser.add_argument(
        "--train_parts",
        type=int,
        nargs="+",
        default=None,
        help="Specific training part indices to train on (e.g. --train_parts 0 or --train_parts 0 1 2)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum total samples to use for training + validation (e.g. 2000000)",
    )
    parser.add_argument(
        "--experiment", type=str, default=None, help="Name of the experiment folder"
    )
    parser.add_argument(
        "--quantize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable HGQ2 quantization",
    )
    parser.add_argument(
        "--in_memory",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Pre-load dataset into RAM for ultra-fast training (default: True for max_samples <= 5M or single part)",
    )
    parser.add_argument(
        "--num_transformers",
        type=int,
        default=1,
        help="Number of Transformer blocks (default: 1, set 3 for legacy JetFormer)",
    )
    parser.add_argument(
        "--embed_dim",
        type=int,
        default=32,
        help="Embedding and hidden dimension (default: 32)",
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=2,
        help="Number of attention heads (default: 2)",
    )
    parser.add_argument(
        "--use_cls_token",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use CLS token injection and extraction (default: False)",
    )
    parser.add_argument(
        "--use_linformer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use QLinformerAttention instead of standard QMultiHeadAttention (default: True)",
    )
    parser.add_argument(
        "--max_test_samples",
        type=int,
        default=2000000,
        help="Maximum test set samples for evaluation (default: 2,000,000)",
    )
    parser.add_argument(
        "--augment_rotation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable dynamic on-the-fly SO(2) rotation data augmentation in (deta, dphi) plane",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for weight initialization and dataset shuffling (default: 42)",
    )
    parser.add_argument(
        "--use_adaptive_lr",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use dynamic noise-bounded AdaptiveReduceLROnPlateau callback instead of standard ReduceLROnPlateau",
    )
    parser.add_argument(
        "--target_ebops",
        type=float,
        default=DEFAULT_TARGET_EBOPS,
        help=f"BetaPID setpoint: the EBOPs value the controller regulates to (default: {DEFAULT_TARGET_EBOPS:,.0f})",
    )
    parser.add_argument(
        "--ebops_threshold",
        type=float,
        default=DEFAULT_EBOPS_THRESHOLD,
        help=f"Checkpoint gate: models are only saved when EBOPs <= this (default: {DEFAULT_EBOPS_THRESHOLD:,.0f}). Keep above --target_ebops so normal PID ripple does not block saves.",
    )
    parser.add_argument(
        "--ebops_warmup_epoch",
        type=int,
        default=EBOPS_WARMUP_EPOCH,
        help=f"Epoch from which the checkpoint/early-stopping callback starts considering saves (default: {EBOPS_WARMUP_EPOCH}). Distinct from BetaPID's own warmup.",
    )
    parser.add_argument(
        "--save_final_epoch",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Additionally save the last-epoch model to <stem>_final.keras, alongside the gated checkpoint",
    )
    # Set >= --num_epochs to disable early stopping and observe the full training
    # curve. EXP-24 stopped both seeds at anchor+150 while validation accuracy was
    # still climbing out of its trough (seed 42: 0.384 @ ep202 -> 0.557 @ ep246;
    # seed 43: 0.401 @ ep196 -> 0.626 @ ep261), so whether the recovery ever clears
    # the early anchor has never actually been observed.
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=150,
        help="Epochs without a new saved checkpoint before training stops (default: 150). Set >= --num_epochs to disable.",
    )
    args = parser.parse_args()

    # Set global random seed immediately upon parsing arguments
    set_global_seed(args.seed)

    # Resolve dataset-specific defaults
    if args.dataset == "jetclass":
        num_particles = args.num_particles if args.num_particles is not None else 128
        num_feats = args.num_feats if args.num_feats is not None else 17
    else:
        num_particles = args.num_particles if args.num_particles is not None else 16
        num_feats = args.num_feats if args.num_feats is not None else 3

    # Dynamic hardware auto-detection for RAM & GPU budget
    approx_samples = args.max_samples if args.max_samples is not None else 10000000
    if args.in_memory is not None:
        in_memory = args.in_memory
    else:
        strategy = detect_hardware_and_strategy(
            num_samples=approx_samples,
            num_particles=num_particles,
            num_feats=num_feats,
            ram_safety_ratio=0.50,
        )
        in_memory = strategy == "FULL_RAM"

    train(
        num_particles=num_particles,
        num_feats=num_feats,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        num_transformers=args.num_transformers,
        embbed_dim=args.embed_dim,
        num_heads=args.num_heads,
        use_cls_token=args.use_cls_token,
        use_linformer=args.use_linformer,
        early_stopping_patience=args.early_stopping_patience,
        dropout=args.dropout,
        val_ratio=0.1,
        experiment=args.experiment,
        quantize=args.quantize,
        dataset=args.dataset,
        train_parts=args.train_parts,
        max_samples=args.max_samples,
        in_memory=in_memory,
        max_test_samples=args.max_test_samples,
        augment_rotation=args.augment_rotation,
        seed=args.seed,
        use_adaptive_lr=args.use_adaptive_lr,
        target_ebops=args.target_ebops,
        ebops_threshold=args.ebops_threshold,
        ebops_warmup_epoch=args.ebops_warmup_epoch,
        save_final_epoch=args.save_final_epoch,
    )
