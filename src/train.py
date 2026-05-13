import os

# Route Keras to the installed backend
os.environ["KERAS_BACKEND"] = "tensorflow"  # or "tensorflow"

# If using JAX: Prevent the backend from pre-allocating 100% of VRAM,
# which can crash out-of-core data loaders.
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.90"

import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score
import keras
import json

# HGQ2 Imports
from hgq.config import QuantizerConfigScope, LayerConfigScope
from hgq.utils import trace_minmax

# Relative imports
from src.dataset import JetFormerDataGenerator
from src.models.jetformer import build_hgq_jetformer
from src.onecyclelr import OneCycleLR
from hgq.utils.sugar.beta_pid import BetaPID

# from hgq.regularizers import MonoL1
from hgq.utils.sugar import FreeEBOPs

# Resolves to the 'src' directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Steps up one level to the project root
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Map target directories to the root
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define classes for the 150-particle dataset
CLASSES = ["Gluon", "Light_quarks", "W_boson", "Z_boson", "Top_quark"]


def build_lr_schedule(max_lr, total_steps, pct_start=0.2):
    # PyTorch OneCycleLR defaults: div_factor=25.0, final_div_factor=1e4
    return OneCycleLR(
        max_lr=max_lr,
        total_steps=total_steps,
        pct_start=pct_start,
        div_factor=25.0,
        final_div_factor=1e4,
    )


def setup_data_generators(num_particles, num_feats, batch_size, val_ratio=0.1):
    base_path = os.path.join(PROCESSED_DIR, str(num_particles), f"{num_feats}f")
    train_h5_path = os.path.join(base_path, "train.h5")
    test_h5_path = os.path.join(base_path, "test.h5")

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


def save_final_evaluation(acc, class_accs, aucs, classes, filepath):
    """
    Serializes test set metrics to a JSON file.
    """
    results = {"overall_accuracy": float(acc), "per_class_metrics": {}}

    for i, class_name in enumerate(classes):
        results["per_class_metrics"][class_name] = {
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
        train_accs=np.array(history_dict["sparse_categorical_accuracy"]),
        val_accs=np.array(history_dict["val_sparse_categorical_accuracy"]),
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
    plt.plot(epochs, history_dict["sparse_categorical_accuracy"], label="Train Acc")
    plt.plot(epochs, history_dict["val_sparse_categorical_accuracy"], label="Val Acc")
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

    os.makedirs(current_model_dir, exist_ok=True)
    os.makedirs(current_output_dir, exist_ok=True)

    total_steps = len(train_gen) * num_epochs
    # lr_schedule = build_lr_schedule(max_lr=1e-3, total_steps=total_steps)
    optimizer = keras.optimizers.AdamW(
        learning_rate=1e-3, weight_decay=1e-4, global_clipnorm=1.0
    )

    # Instantiate scopes BEFORE the context manager
    quant_scope = QuantizerConfigScope(
        place="all", default_q_type="kbi", overflow_mode="SAT_SYM"
    )
    layer_scope = LayerConfigScope(enable_ebops=True, beta0=5e-8)

    # Wrap model instantiation and training in HGQ2 Scopes
    with quant_scope, layer_scope:

        # 1. Instantiate the Statically Traceable Graph
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

        # The manual forward pass (model(physical_batch)) is removed.
        # The Keras Functional API pre-compiles the topological footprint.

        # Output the fully instantiated architectural footprint
        print("=================MODEL SUMMARY=================")
        model.summary()

        model.compile(
            optimizer=optimizer,
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["sparse_categorical_accuracy"],
        )

        callbacks = []
        if early_stopping_patience > 0:
            callbacks.append(
                keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=early_stopping_patience,
                    min_delta=1e-4,
                    restore_best_weights=True,
                )
            )

        # Add this inside your training block before model.fit()
        callbacks.append(
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.8, patience=5, min_lr=1e-4
            )
        )

        # ebops = FreeEBOPs()
        # callbacks.append(ebops)

        if quantize:
            callbacks.append(BetaPID(target_ebops=350000.0, warmup=5))

        if save:
            if model_path is None:
                model_path = os.path.join(
                    current_model_dir, f"{num_particles}_{num_feats}f.keras"
                )
            # Initialize eval path here to ensure global function scope
            eval_results_path = os.path.join(
                current_output_dir, f"{num_particles}_{num_feats}f_metrics.json"
            )
            callbacks.append(
                keras.callbacks.ModelCheckpoint(
                    filepath=model_path, monitor="val_loss", save_best_only=True
                )
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
                if output_path is None:
                    output_path = os.path.join(
                        current_output_dir, f"{num_particles}_{num_feats}f_loss_acc.npz"
                    )
                if plot_path is None:
                    plot_path = os.path.join(
                        current_output_dir, f"{num_particles}_{num_feats}f_plot.png"
                    )

                save_loss_acc(history.history, num_particles, num_feats, output_path)
                plot_loss_acc(history.history, num_particles, num_feats, plot_path)

        # Post-Training Activation Profiling for WRAP mode safety
        # if quantize:
        #     print("\nProfiling activations for WRAP mode bit allocation...")

        #     # 1. Define a symbolic Keras Input matching the batch geometry
        #     symbolic_input = keras.Input(shape=(num_particles, num_feats), name="profiler_input")

        #     # 2. Route the symbolic input through the trained subclassed model
        #     symbolic_output = model(symbolic_input)

        #     # 3. Instantiate a strict Functional Model wrapper to expose the .outputs attribute
        #     functional_model = keras.Model(inputs=symbolic_input, outputs=symbolic_output)

        #     # Create a small subset generator for profiling
        #     def profiler_generator(gen, steps=10):
        #         iterator = iter(gen)
        #         for _ in range(steps):
        #             yield next(iterator)

        #     # Execute the profiler on a fraction of the data
        #     trace_minmax(functional_model, profiler_generator(train_gen, steps=10))

        #     # # 4. Execute the HGQ profiler on the wrapped graph
        #     # trace_minmax(functional_model, train_gen)

        print("\nLoading Best Checkpoint Weights...")
        # Explicitly restore the optimal weights saved by ModelCheckpoint
        if save and model_path is not None:
            model.load_weights(model_path)

        print("\nExecuting Inference on Test Set...")
        outputs = model.predict(test_gen)

        # Extract true labels directly from the un-shuffled generator
        labels = np.concatenate([y for _, y in test_gen], axis=0)

        test_acc, test_class_accs, test_aucs = evaluate(outputs, labels, CLASSES)

        # Persist to disk
        if save:
            save_final_evaluation(
                test_acc, test_class_accs, test_aucs, CLASSES, eval_results_path
            )
            print(f"Final metrics saved to: {eval_results_path}")

        # evaluate(outputs, labels, CLASSES)


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
        default=True,
        help="Enable HGQ2 quantization",
    )
    args = parser.parse_args()

    train(
        num_particles=args.num_particles,
        num_feats=args.num_feats,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        num_transformers=3,
        early_stopping_patience=15,  # Enforced 0 to allow OneCycleLR decay phase
        val_ratio=0.1,
        experiment=args.experiment,
        quantize=args.quantize,
    )
