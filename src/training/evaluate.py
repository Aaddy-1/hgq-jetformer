import os
import argparse
import json
import glob
import gc
import numpy as np
import keras
from sklearn.metrics import accuracy_score, roc_auc_score

os.environ["KERAS_BACKEND"] = "tensorflow"
from hgq.utils import trace_minmax

from src.data.dataset import (
    JetFormerDataGenerator,
    detect_hardware_and_strategy,
    get_stratified_indices,
)
from src.training.train import (
    HLS4ML_CLASSES,
    JETCLASS_CLASSES,
    PROJECT_ROOT,
    PROCESSED_DIR,
    resolve_experiment_paths,
    evaluate,
    extract_model_metadata,
    save_final_evaluation,
)


def run_standalone_evaluation(
    num_particles: int = 128,
    num_feats: int = 17,
    batch_size: int = 256,
    max_test_samples: int = 2000000,
    experiment: str = None,
    quantize: bool = True,
    dataset: str = "jetclass",
    model_path: str = None,
    in_memory: bool = True,
    best_ebops: float = None,
    best_epoch: int = None,
):
    classes = JETCLASS_CLASSES if dataset == "jetclass" else HLS4ML_CLASSES
    current_model_dir, current_output_dir = resolve_experiment_paths(
        experiment, quantize
    )

    if model_path is None:
        model_path = os.path.join(
            current_model_dir, f"{num_particles}_{num_feats}f.keras"
        )

    eval_results_path = os.path.join(
        current_output_dir, f"{num_particles}_{num_feats}f_metrics.json"
    )

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")

    print(f"Loading model from: {model_path}")
    model = keras.models.load_model(model_path, compile=False)
    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["sparse_categorical_accuracy"],
    )

    if dataset == "jetclass":
        base_path = os.path.join(PROCESSED_DIR, "jetclass", str(num_particles), f"{num_feats}f")
        if not os.path.exists(base_path):
            base_path = os.path.join(PROCESSED_DIR, "jetclass", str(num_particles), "17f")
        part_files = sorted(glob.glob(os.path.join(base_path, "train_part*.h5")))
        train_h5_path = part_files[0] if part_files else os.path.join(base_path, "train.h5")
    else:
        base_path = os.path.join(PROCESSED_DIR, str(num_particles), f"{num_feats}f")
        train_h5_path = os.path.join(base_path, "train.h5")

    test_h5_path = os.path.join(base_path, "test.h5")

    import h5py

    preloaded = None
    with h5py.File(test_h5_path, "r") as f:
        key = "jetConstituentList" if "jetConstituentList" in f else ("particle_features" if "particle_features" in f else list(f.keys())[0])
        y_key = "jets" if "jets" in f else ("label" if "label" in f else list(f.keys())[1])
        total_test_samples = f[key].shape[0]

    eval_samples = max_test_samples if (max_test_samples is not None and max_test_samples > 0 and max_test_samples < total_test_samples) else total_test_samples

    # 1. Hardware Detection BEFORE any data preloading
    if in_memory:
        if eval_samples > 5000000:
            print(f"\n[Evaluate] Large sample count detected ({eval_samples:,} > 5,000,000).")
            print("[Evaluate] Automatically selecting Option 2 (CHUNKED_RAM) for 100% OOM safety.")
            strategy = "CHUNKED_RAM"
        else:
            strategy = detect_hardware_and_strategy(
                num_samples=eval_samples,
                num_particles=num_particles,
                num_feats=num_feats,
                ram_safety_ratio=0.50,
            )
    else:
        strategy = "SEQUENTIAL_DISK_STREAM"

    preloaded = None
    test_indices = None

    # 2. Pre-load into RAM ONLY if FULL_RAM is selected
    if strategy == "FULL_RAM":
        with h5py.File(test_h5_path, "r") as f:
            if eval_samples < total_test_samples:
                print(f"[Evaluate] Fast 10-block slice sampling for {eval_samples:,} test samples across all 10 classes...")
                y_test_all = f[y_key][:]
                unique_classes = np.unique(y_test_all)
                quota = eval_samples // len(unique_classes)
                x_chunks, y_chunks = [], []
                idx_chunks = []
                for cls in unique_classes:
                    cls_indices = np.where(y_test_all == cls)[0]
                    start_i = cls_indices[0]
                    x_chunks.append(f[key][start_i : start_i + quota])
                    y_chunks.append(f[y_key][start_i : start_i + quota])
                    idx_chunks.append(np.arange(start_i, start_i + quota))
                shared_x = np.concatenate(x_chunks, axis=0)
                shared_y = np.concatenate(y_chunks, axis=0)
                preloaded = (shared_x, shared_y)
                test_indices = np.concatenate(idx_chunks)
                del y_test_all
            else:
                preloaded = (f[key][:], f[y_key][:])
                test_indices = np.arange(total_test_samples)

    if quantize:
        print("\n[HGQ] Initiating activation profiling for WRAP mode calibration...")
        train_gen = JetFormerDataGenerator(
            h5_path=train_h5_path,
            stats_dir=base_path,
            batch_size=batch_size,
            shuffle=True,
            num_feats=num_feats,
            in_memory=False,
        )
        it = iter(train_gen)
        x_calib = np.concatenate([next(it)[0] for _ in range(10)], axis=0)
        trace_minmax(model, x_calib)
        print("[HGQ] Profiling complete. Integer boundaries calibrated.")

    print(f"\nExecuting Inference on Test Set ({eval_samples:,} samples)...")

    outputs = None
    labels = None

    # Tier 1: FULL_RAM
    if strategy == "FULL_RAM":
        print("\n[Evaluate] Strategy Selected: Option 1 (FULL_RAM - Entire dataset fits safely in RAM)")
        try:
            test_gen = JetFormerDataGenerator(
                h5_path=test_h5_path,
                stats_dir=base_path,
                batch_size=batch_size,
                shuffle=False,
                indices=test_indices if preloaded is None else None,
                num_feats=num_feats,
                in_memory=True,
                preloaded_data=preloaded,
            )
            outputs = model.predict(test_gen)
            labels = np.concatenate([y for _, y in test_gen], axis=0)
        except (MemoryError, Exception) as e:
            print(f"[Evaluate] Option 1 (FULL_RAM) failed with error: {e}")
            print("[Evaluate] Stepping down to Option 2 (CHUNKED_RAM)...")
            strategy = "CHUNKED_RAM"

    # Tier 2: CHUNKED_RAM (Fallback if FULL_RAM is unavailable or fails)
    if strategy == "CHUNKED_RAM":
        print("\n[Evaluate] Strategy Selected: Option 2 (CHUNKED_RAM - Available RAM < Full Dataset)")
        print("[Evaluate] Evaluating in 1,000,000-sample in-memory chunks (~8.70 GB RAM per chunk)...")
        chunk_size = 1000000
        all_outputs = []
        all_labels = []
        try:
            num_chunks = int(np.ceil(eval_samples / chunk_size))
            with h5py.File(test_h5_path, "r") as f:
                for chunk_idx in range(num_chunks):
                    c_start = chunk_idx * chunk_size
                    c_end = min((chunk_idx + 1) * chunk_size, eval_samples)
                    print(f"[Evaluate] [Chunk {chunk_idx + 1}/{num_chunks}] Slicing samples {c_start:,} to {c_end:,} into RAM...")
                    c_x = f[key][c_start:c_end]
                    c_y = f[y_key][c_start:c_end]
                    chunk_gen = JetFormerDataGenerator(
                        h5_path=test_h5_path,
                        stats_dir=base_path,
                        batch_size=batch_size,
                        shuffle=False,
                        num_feats=num_feats,
                        in_memory=True,
                        preloaded_data=(c_x, c_y),
                    )
                    c_out = model.predict(chunk_gen)
                    c_labels = np.concatenate([y for _, y in chunk_gen], axis=0)
                    all_outputs.append(c_out)
                    all_labels.append(c_labels)
                    del c_x, c_y, chunk_gen, c_out, c_labels
                    gc.collect()
            outputs = np.concatenate(all_outputs, axis=0)
            labels = np.concatenate(all_labels, axis=0)
        except (MemoryError, Exception) as e:
            print(f"[Evaluate] Option 2 (CHUNKED_RAM) failed with error: {e}")
            print("[Evaluate] Stepping down to Option 3 (SEQUENTIAL_DISK_STREAM)...")
            strategy = "SEQUENTIAL_DISK_STREAM"

    # Tier 3: SEQUENTIAL_DISK_STREAM (Final Fallback if Tiers 1 & 2 fail)
    if strategy == "SEQUENTIAL_DISK_STREAM" or outputs is None:
        print("\n[Evaluate] Strategy Selected: Option 3 (SEQUENTIAL_DISK_STREAM)")
        print("[Evaluate] WARNING: Options 1 & 2 unavailable or failed. Streaming batch-by-batch from HDF5 disk...")
        test_gen = JetFormerDataGenerator(
            h5_path=test_h5_path,
            stats_dir=base_path,
            batch_size=batch_size,
            shuffle=False,
            indices=test_indices,
            num_feats=num_feats,
            in_memory=False,
        )
        outputs = model.predict(test_gen)
        labels = np.concatenate([y for _, y in test_gen], axis=0)

    test_acc, test_class_accs, test_aucs = evaluate(outputs, labels, classes)

    metadata = extract_model_metadata(model, best_ebops=best_ebops, best_epoch=best_epoch, num_test_samples=len(labels))
    config = {
        "num_particles": num_particles,
        "num_feats": num_feats,
        "batch_size": batch_size,
        "max_test_samples": max_test_samples,
        "experiment": experiment,
        "quantize": quantize,
        "dataset": dataset,
    }

    save_final_evaluation(
        test_acc,
        test_class_accs,
        test_aucs,
        classes,
        metadata,
        config,
        eval_results_path,
    )
    print(f"\nMetrics and evaluation metadata saved to: {eval_results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate JetFormer Model")
    parser.add_argument(
        "--num_particles", type=int, default=128, help="Number of jet constituents"
    )
    parser.add_argument(
        "--num_feats", type=int, default=17, help="Number of features per constituent"
    )
    parser.add_argument(
        "--batch_size", type=int, default=256, help="Evaluation batch size"
    )
    parser.add_argument(
        "--max_test_samples",
        type=int,
        default=2000000,
        help="Maximum test samples for evaluation (0 for uncapped full dataset)",
    )
    parser.add_argument(
        "--dataset", type=str, default="jetclass", help="Dataset name (jetclass or hls4ml)"
    )
    parser.add_argument(
        "--experiment", type=str, default=None, help="Experiment folder name"
    )
    parser.add_argument(
        "--quantize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable HGQ2 quantization mode",
    )
    parser.add_argument(
        "--model_path", type=str, default=None, help="Explicit path to model file"
    )
    parser.add_argument(
        "--in_memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pre-load test set into RAM for ultra-fast evaluation (default: True)",
    )
    args = parser.parse_args()

    run_standalone_evaluation(
        num_particles=args.num_particles,
        num_feats=args.num_feats,
        batch_size=args.batch_size,
        max_test_samples=args.max_test_samples,
        experiment=args.experiment,
        quantize=args.quantize,
        dataset=args.dataset,
        model_path=args.model_path,
        in_memory=args.in_memory,
    )
