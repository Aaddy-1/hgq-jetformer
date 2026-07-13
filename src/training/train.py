import os
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
from src.data.dataset import JetFormerDataGenerator
from src.model.jetformer import build_hgq_jetformer
from src.training.onecyclelr import OneCycleLR, build_lr_schedule

# Path variables
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define classes for the 150-particle dataset
CLASSES = ["Gluon", "Light_quarks", "W_boson", "Z_boson", "Top_quark"]

# Shared training constant: epoch after which EBOPs and val_loss
# are expected to have stabilized under PID control.
EBOPS_WARMUP_EPOCH = 75


class EbopsCaptureCallback(keras.callbacks.Callback):
    """Captures the EBOPs and accuracy metadata of the best model.

    Note: Model saving is handled by EarlyStoppingWithEbopsThres
    (restore_best_weights=True). This callback only tracks metadata.
    """

    def __init__(self, start_from_epoch=EBOPS_WARMUP_EPOCH):
        super().__init__()
        self.best_val_acc = -float("inf")
        self.best_ebops = None
        self.best_epoch = None
        self.start_from_epoch = start_from_epoch

    def _get_ebops(self):
        ebops = 0.0
        found = False
        for layer in self.model.layers:
            if hasattr(layer, "ebops"):
                ebops += float(layer.ebops)
                found = True
        return ebops if found else None

    def on_epoch_end(self, epoch, logs=None):
        if epoch < self.start_from_epoch:
            return

        logs = logs or {}
        val_acc = logs.get("val_sparse_categorical_accuracy")
        if val_acc is not None and val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.best_ebops = self._get_ebops()
            self.best_epoch = epoch


def extract_model_metadata(model, best_ebops, best_epoch):
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
    return {
        "ebops": best_ebops,
        "best_epoch": best_epoch,
        "total_parameters": int(model.count_params()),
        "layers": layers_metadata,
    }


def setup_data_generators(num_particles, num_feats, batch_size, val_ratio=0.1):
    base_path = os.path.join(PROCESSED_DIR, str(num_particles), f"{num_feats}f")
    train_h5_path = os.path.join(base_path, "train.h5")
    test_h5_path = os.path.join(base_path, "test.h5")

    print("BASE PATH:", base_path)
    print("================================")
    print("TRAIN H5 PATH:", train_h5_path)

    import h5py

    with h5py.File(train_h5_path, "r") as f:
        total_train_samples = f["jetConstituentList"].shape[0]

    indices = np.random.permutation(total_train_samples)
    val_size = int(total_train_samples * val_ratio)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_gen = JetFormerDataGenerator(
        h5_path=train_h5_path,
        stats_dir=base_path,
        batch_size=batch_size,
        shuffle=True,
        indices=train_indices,
    )
    val_gen = JetFormerDataGenerator(
        h5_path=train_h5_path,
        stats_dir=base_path,
        batch_size=batch_size,
        shuffle=False,
        indices=val_indices,
    )
    test_gen = JetFormerDataGenerator(
        h5_path=test_h5_path, stats_dir=base_path, batch_size=batch_size, shuffle=False
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


def save_loss_acc(history_dict, num_particles, num_feats, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez(
        output_path,
        train_losses=np.array(history_dict["loss"]),
        val_losses=np.array(history_dict["val_loss"]),
        train_accs=np.array(history_dict.get("sparse_categorical_accuracy", [])),
        val_accs=np.array(history_dict.get("val_sparse_categorical_accuracy", [])),
    )
    print(f"Loss and accuracy saved to {output_path}")


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


def build_callbacks(early_stopping_patience: int, quantize: bool):
    callbacks = []

    if quantize:
        # --- Quantized Training Callbacks (Reference-aligned) ---
        callbacks.append(
            EarlyStoppingWithEbopsThres(
                ebops_threshold=450000,
                monitor="val_loss",
                patience=150,
                mode="min",
                restore_best_weights=True,
                start_from_epoch=EBOPS_WARMUP_EPOCH,
            )
        )
        callbacks.append(
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                mode="min",
                factor=0.8,
                patience=50,
                min_lr=1e-5,
                cooldown=200,
                min_delta=0.05,
            )
        )
        callbacks.append(
            BetaPID(
                p=1,
                i=0.1,
                d=0,
                target_ebops=350000.0,
                init_beta=1e-10,
                warmup=10,
                max_beta=5e-6,
                damp_beta_on_target=0.5,
            )
        )
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
        start_from_epoch=EBOPS_WARMUP_EPOCH if quantize else 0
    )
    callbacks.append(ebops_capture)

    return callbacks, ebops_capture


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
):
    # Best weights are already restored by EarlyStoppingWithEbopsThres
    # (restore_best_weights=True) or keras.callbacks.EarlyStopping.

    if quantize:
        # --- Diagnostic: Evaluate BEFORE trace_minmax ---
        print("\n[Diagnostic] Evaluating model BEFORE trace_minmax...")
        pre_outputs = model.predict(test_gen)
        pre_labels = np.concatenate([y for _, y in test_gen], axis=0)
        pre_acc = accuracy_score(pre_labels, pre_outputs.argmax(axis=1))
        print(f"[Diagnostic] Pre-trace_minmax accuracy: {pre_acc:.4f}")

        print("\n[HGQ] Initiating activation profiling for WRAP mode calibration...")
        it = iter(train_gen)
        x_calib = np.concatenate([next(it)[0] for _ in range(10)], axis=0)
        trace_minmax(model, x_calib)
        print("[HGQ] Profiling complete. Integer boundaries calibrated.")

    print("\nExecuting Final Inference on Test Set...")
    outputs = model.predict(test_gen)
    labels = np.concatenate([y for _, y in test_gen], axis=0)
    test_acc, test_class_accs, test_aucs = evaluate(outputs, labels, CLASSES)

    if quantize:
        print(f"\n[Diagnostic] trace_minmax accuracy delta: {test_acc - pre_acc:+.4f}")

    if save:
        if model_path:
            model.save(model_path)
            print(f"Final model saved to: {model_path}")
        if eval_results_path:
            metadata = extract_model_metadata(model, best_ebops, best_epoch)
            save_final_evaluation(
                test_acc,
                test_class_accs,
                test_aucs,
                CLASSES,
                metadata,
                config,
                eval_results_path,
            )
            print(f"Final metrics and metadata saved to: {eval_results_path}")


def train(
    num_particles: int = 150,
    num_feats: int = 16,
    do_train: bool = True,
    val_ratio: float = 0.1,
    num_epochs: int = 25,
    early_stopping_patience: int = 0,
    num_transformers: int = 3,
    embbed_dim: int = 64,
    num_heads: int = 2,
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
):
    train_gen, val_gen, test_gen = setup_data_generators(
        num_particles=num_particles,
        num_feats=num_feats,
        batch_size=batch_size,
        val_ratio=val_ratio,
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

    eval_results_path = os.path.join(
        current_output_dir, f"{num_particles}_{num_feats}f_metrics.json"
    )

    optimizer = keras.optimizers.AdamW(learning_rate=1e-3)

    quant_scope = QuantizerConfigScope(
        place="all", default_q_type="kbi", overflow_mode="WRAP", br=MonoL1(1e-8)
    )
    layer_scope = LayerConfigScope(enable_ebops=True, beta0=1e-10)

    with quant_scope, layer_scope:
        config = {
            "in_dim": num_feats,
            "embed_dim": embbed_dim,
            "num_heads": num_heads,
            "num_classes": len(CLASSES),
            "num_transformers": num_transformers,
            "dropout": dropout,
            "num_particles": num_particles,
            "activation": activation,
            "normalization": normalization,
            "quantize": quantize,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "early_stopping_patience": early_stopping_patience,
        }

        print("[DEBUG] Model Args: ")
        for k, v in config.items():
            print(f"  {k}={v}")

        model = build_hgq_jetformer(
            in_dim=num_feats,
            embed_dim=embbed_dim,
            num_heads=num_heads,
            num_classes=len(CLASSES),
            num_transformers=num_transformers,
            dropout=dropout,
            num_particles=num_particles,
            activation=activation,
            normalization=normalization,
            quantize=quantize,
        )

        print("=================MODEL SUMMARY=================")
        model.summary()

        model.compile(
            optimizer=optimizer,
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["sparse_categorical_accuracy"],
        )

        callbacks, ebops_capture = build_callbacks(early_stopping_patience, quantize)

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
                save_loss_acc(history.history, num_particles, num_feats, output_path)
                plot_loss_acc(history.history, num_particles, num_feats, plot_path)

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
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train HGQJetFormer")
    parser.add_argument(
        "--num_particles", type=int, default=8, help="Number of jet constituents"
    )
    parser.add_argument(
        "--num_feats",
        type=int,
        default=3,
        choices=[3, 16],
        help="Number of features per constituent",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=25, help="Total training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=256, help="Training batch size"
    )
    parser.add_argument(
        "--experiment", type=str, default=None, help="Name of the experiment folder"
    )
    parser.add_argument(
        "--quantize",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable HGQ2 quantization",
    )
    args = parser.parse_args()

    train(
        num_particles=args.num_particles,
        num_feats=args.num_feats,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        num_transformers=3,
        embbed_dim=16,
        early_stopping_patience=100,
        val_ratio=0.1,
        experiment=args.experiment,
        quantize=args.quantize,
    )
