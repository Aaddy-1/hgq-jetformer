import os
import sys
import json
import glob
import argparse
import numpy as np
import h5py
import keras
from scipy.special import softmax
from sklearn.metrics import accuracy_score, roc_auc_score

os.environ["KERAS_BACKEND"] = "tensorflow"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
HLS4ML_CLASSES = ["q", "g", "W", "Z", "t"]


def load_and_pad_hls4ml_data(h5_path, max_samples=260000):
    """
    Loads HLS4ML test dataset and adapts inputs to EXP-18 shape (N, 128, 17).
    Channel mapping:
      Feature 0 (pT)       -> Target Channel 0  (part_pt_log)
      Feature 1 (deta)     -> Target Channel 15 (part_deta)
      Feature 2 (dphi)     -> Target Channel 16 (part_dphi)
      Channels 1..14       -> 0.0 (zero-padded)
    """
    print(f"[Zero-Shot] Loading HLS4ML test data from: {h5_path}")
    with h5py.File(h5_path, "r") as f:
        x_key = "jetConstituentList" if "jetConstituentList" in f else ("particle_features" if "particle_features" in f else list(f.keys())[0])
        y_key = "jets" if "jets" in f else ("label" if "label" in f else list(f.keys())[1])

        raw_x = f[x_key][:max_samples]  # Shape: (N, num_particles, num_feats)
        raw_y = f[y_key][:max_samples]  # Shape: (N, 5) or scalar labels

    n_samples, in_particles, in_feats = raw_x.shape
    print(f"[Zero-Shot] Raw HLS4ML input shape: ({n_samples}, {in_particles}, {in_feats})")

    # Construct padded tensor for EXP-18: (N, 128, 17)
    padded_x = np.zeros((n_samples, 128, 17), dtype=np.float32)

    # 1. Truncate/Pad sequence length to 128
    effective_particles = min(in_particles, 128)

    # 2. Map kinematic features
    if in_feats == 3:
        # pT, deta, dphi
        padded_x[:, :effective_particles, 0] = raw_x[:, :effective_particles, 0]   # pT
        padded_x[:, :effective_particles, 15] = raw_x[:, :effective_particles, 1]  # deta
        padded_x[:, :effective_particles, 16] = raw_x[:, :effective_particles, 2]  # dphi
    elif in_feats >= 16:
        mapped_feats = min(in_feats, 17)
        padded_x[:, :effective_particles, :mapped_feats] = raw_x[:, :effective_particles, :mapped_feats]
    else:
        padded_x[:, :effective_particles, :in_feats] = raw_x[:, :effective_particles, :in_feats]

    # Convert one-hot labels to integer class indices if necessary
    if raw_y.ndim > 1:
        labels = np.argmax(raw_y, axis=1)
    else:
        labels = raw_y.astype(int)

    return padded_x, labels


def evaluate_exp18_zeroshot(model_path, h5_path, max_samples=260000, output_json=None):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"HLS4ML test dataset not found at: {h5_path}")

    print(f"\n[Zero-Shot] Loading model from: {model_path}")
    model = keras.models.load_model(model_path, compile=False)

    padded_x, labels = load_and_pad_hls4ml_data(h5_path, max_samples=max_samples)

    print(f"[Zero-Shot] Running inference on {len(padded_x):,} samples...")
    raw_logits = model.predict(padded_x, batch_size=512)

    # EXP-18 JetClass Logit Slicing for HLS4ML 5 classes:
    # 0: q / QCD (JetClass index 0)
    # 1: g / QCD (JetClass index 0)
    # 2: W (JetClass Wqq index 7)
    # 3: Z (JetClass Zqq index 6)
    # 4: t (JetClass Tbqq index 8)
    hls4ml_logit_indices = [0, 0, 7, 6, 8]
    sliced_logits = raw_logits[:, hls4ml_logit_indices]

    # Apply Softmax across the 5 target classes
    probs = softmax(sliced_logits, axis=1)
    preds = np.argmax(probs, axis=1)

    overall_acc = float(accuracy_score(labels, preds))

    # Per-class metrics
    per_class = {}
    for i, class_name in enumerate(HLS4ML_CLASSES):
        idx = (labels == i)
        c_acc = float(accuracy_score(labels[idx], preds[idx])) if np.sum(idx) > 0 else 0.0
        try:
            one_hot_y = (labels == i).astype(int)
            c_auc = float(roc_auc_score(one_hot_y, probs[:, i]))
        except Exception:
            c_auc = 0.0

        per_class[class_name] = {"accuracy": c_acc, "auc": c_auc}

    print("\n" + "=" * 55)
    print(f"ZERO-SHOT EVALUATION REPORT (`EXP-18` on HLS4ML)")
    print("=" * 55)
    print(f"Overall Accuracy: {overall_acc * 100:.2f}%")
    print("-" * 55)
    for class_name, m in per_class.items():
        print(f"  Class {class_name:<5}: Acc = {m['accuracy']*100:6.2f}%, AUC = {m['auc']:.4f}")
    print("=" * 55)

    results = {
        "model_path": model_path,
        "h5_path": h5_path,
        "num_samples": len(labels),
        "overall_accuracy": overall_acc,
        "per_class_metrics": per_class,
    }

    if output_json:
        os.makedirs(os.path.dirname(output_json), exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(results, f, indent=4)
        print(f"[Zero-Shot] Metrics saved to: {output_json}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zero-Shot Evaluation of EXP-18 on HLS4ML Dataset")
    parser.add_argument(
        "--model_path",
        type=str,
        default=os.path.join(PROJECT_ROOT, ".agents", "jet", "128_17f.keras"),
        help="Path to pre-trained EXP-18 Keras model file",
    )
    parser.add_argument(
        "--h5_path",
        type=str,
        default=os.path.join(PROJECT_ROOT, "data", "processed", "32", "3f", "test.h5"),
        help="Path to HLS4ML test HDF5 file",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=260000,
        help="Maximum test samples to evaluate",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=os.path.join(PROJECT_ROOT, ".agents", "jet", "zeroshot_hls4ml_metrics.json"),
        help="Output path for JSON results",
    )
    args = parser.parse_args()

    model_p = args.model_path
    if not os.path.exists(model_p):
        candidates = glob.glob(os.path.join(PROJECT_ROOT, "**", "128_17f.keras"), recursive=True)
        if candidates:
            model_p = candidates[0]
            print(f"[Auto-Detect] Found model file at: {model_p}")

    evaluate_exp18_zeroshot(
        model_path=model_p,
        h5_path=args.h5_path,
        max_samples=args.max_samples,
        output_json=args.output_json,
    )
