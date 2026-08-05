import os
import argparse
import json
import numpy as np
import keras
from sklearn.metrics import accuracy_score, roc_auc_score

os.environ["KERAS_BACKEND"] = "tensorflow"
from hgq.utils import trace_minmax

from src.data.dataset import JetFormerDataGenerator
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
        train_h5_path = os.path.join(base_path, "train_0.h5")
    else:
        base_path = os.path.join(PROCESSED_DIR, str(num_particles), f"{num_feats}f")
        train_h5_path = os.path.join(base_path, "train.h5")

    test_h5_path = os.path.join(base_path, "test.h5")

    import h5py

    with h5py.File(test_h5_path, "r") as f:
        key = "jetConstituentList" if "jetConstituentList" in f else ("particle_features" if "particle_features" in f else list(f.keys())[0])
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
    )

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

    print(f"\nExecuting Inference on Test Set ({len(test_gen.indices):,} samples)...")
    outputs = model.predict(test_gen)
    labels = np.concatenate([y for _, y in test_gen], axis=0)
    test_acc, test_class_accs, test_aucs = evaluate(outputs, labels, classes)

    metadata = extract_model_metadata(model, best_ebops=None, best_epoch=None, num_test_samples=len(labels))
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
        help="Maximum test samples for evaluation (default: 2,000,000)",
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
    )
