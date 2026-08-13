import os

import keras
from keras import ops

# Shared constant for PID warmup epoch
EBOPS_WARMUP_EPOCH = 75



def get_model_ebops(model):
    """Calculates top-level model EBOPs matching BetaPID calculation."""
    ebops = 0.0
    found = False
    for layer in model.layers:
        if hasattr(layer, "ebops") and layer.ebops is not None:
            val = float(ops.convert_to_numpy(layer.ebops))
            ebops += val
            found = True
    return ebops if found else None


class QATEarlyStoppingAndCheckpoint(keras.callbacks.Callback):
    """Unified Early Stopping and Model Checkpointing Callback for QAT.

    Guarantees that:
    1. Checkpoints are ONLY saved when EBOPs <= ebops_threshold AND val_acc > best_val_acc + min_delta.
    2. Patience counter (patience_wait) tracks consecutive epochs since the last valid checkpoint was saved.
    3. Training stops immediately when patience_wait >= patience.
    """

    def __init__(
        self,
        ebops_threshold: float = 450000.0,
        patience: int = 150,
        min_delta: float = 1e-3,
        start_from_epoch: int = EBOPS_WARMUP_EPOCH,
        model_path: str = None,
    ):
        super().__init__()
        self.ebops_threshold = ebops_threshold
        self.patience = patience
        self.min_delta = min_delta
        self.start_from_epoch = start_from_epoch
        self.model_path = model_path

        self.best_val_acc = -float("inf")
        self.best_ebops = None
        self.best_epoch = None
        self.best_epochs = []
        self.patience_wait = 0

    def on_epoch_end(self, epoch, logs=None):
        if epoch < self.start_from_epoch:
            return

        logs = logs or {}
        val_acc = logs.get("val_sparse_categorical_accuracy")
        ebops = get_model_ebops(self.model)

        is_compliant = ebops is None or ebops <= self.ebops_threshold
        is_improvement = val_acc is not None and (
            val_acc > self.best_val_acc + self.min_delta
        )

        if is_compliant and is_improvement:
            self.best_val_acc = val_acc
            self.best_ebops = ebops
            self.best_epoch = epoch
            self.best_epochs.append(epoch)
            self.patience_wait = 0  # Reset patience counter ONLY when a new best model is saved!

            if self.model_path:
                self.model.save(self.model_path)
                print(
                    f"\n[Checkpoint] Saved new best EBOP-compliant model "
                    f"(val_acc: {val_acc:.4f}, ebops: {ebops if ebops is not None else 0:.0f}) "
                    f"at Epoch {epoch+1} to {self.model_path}"
                )
        else:
            self.patience_wait += 1

        if self.patience_wait >= self.patience:
            best_epoch_str = f"Epoch {self.best_epoch+1}" if self.best_epoch is not None else "N/A"
            best_val_str = f"{self.best_val_acc:.4f}" if self.best_val_acc != -float("inf") else "N/A"
            best_ebops_str = f"{self.best_ebops:.0f}" if self.best_ebops is not None else "N/A"
            print(
                f"\n[EarlyStopping] No new EBOP-compliant best model saved for {self.patience} epochs. "
                f"Stopping training at Epoch {epoch+1}. Best model was at {best_epoch_str} "
                f"(val_acc: {best_val_str}, ebops: {best_ebops_str})."
            )
            self.model.stop_training = True


class FinalEpochCheckpoint(keras.callbacks.Callback):
    """Saves the last-epoch weights alongside -- never on top of -- the gated checkpoint.

    Commit 52e42c6 saved the final epoch unconditionally to the *same* path the
    EBOP-gated checkpoint used, silently overwriting it. That is how
    EXP-18_lowerwarmup_seed44 came to hold a 6,854,896-EBOP model while its
    metrics recorded 365,644 for best_epoch 417 -- the metadata described a state
    that no longer existed on disk.

    To make that failure impossible rather than merely unlikely, this callback
    takes the *gated* checkpoint path and derives its own output from it
    (`<stem>_final<ext>`). The caller cannot accidentally point both at the same
    file.
    """

    def __init__(self, checkpoint_path: str = None):
        super().__init__()
        self.model_path = None
        if checkpoint_path:
            stem, ext = os.path.splitext(checkpoint_path)
            self.model_path = f"{stem}_final{ext or '.keras'}"
        self.final_ebops = None
        self.final_epoch = None

    def on_epoch_end(self, epoch, logs=None):
        self.final_epoch = epoch

    def on_train_end(self, logs=None):
        if not self.model_path:
            return
        self.final_ebops = get_model_ebops(self.model)
        self.model.save(self.model_path)
        ebops_str = f"{self.final_ebops:.0f}" if self.final_ebops is not None else "N/A"
        epoch_str = f"{self.final_epoch + 1}" if self.final_epoch is not None else "N/A"
        print(
            f"\n[FinalEpoch] Saved last-epoch model (epoch {epoch_str}, "
            f"ebops: {ebops_str}) to {self.model_path}"
        )


class EbopsCaptureCallback(keras.callbacks.Callback):
    """Captures EBOPs and accuracy metadata for unquantized training runs."""

    def __init__(self, start_from_epoch=0, model_path=None):
        super().__init__()
        self.best_val_acc = -float("inf")
        self.best_ebops = None
        self.best_epoch = None
        self.best_epochs = []
        self.start_from_epoch = start_from_epoch
        self.model_path = model_path

    def on_epoch_end(self, epoch, logs=None):
        if epoch < self.start_from_epoch:
            return

        logs = logs or {}
        val_acc = logs.get("val_sparse_categorical_accuracy")
        ebops = get_model_ebops(self.model)

        if val_acc is not None and val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.best_ebops = ebops
            self.best_epoch = epoch
            self.best_epochs.append(epoch)
            if self.model_path:
                self.model.save(self.model_path)
                print(
                    f"\n[Checkpoint] Saved new best model "
                    f"(val_acc: {val_acc:.4f}) at Epoch {epoch+1} to {self.model_path}"
                )


class AdaptiveReduceLROnPlateau(keras.callbacks.Callback):
    """Simple, debuggable Adaptive ReduceLROnPlateau.

    Dynamically bounds min_delta by c * std_dev(val_acc over window) to prevent
    noise jitter from prematurely resetting LR patience.
    """

    def __init__(
        self,
        factor: float = 0.8,
        patience: int = 40,
        cooldown: int = 40,
        min_lr: float = 1e-5,
        window: int = 8,
        c_safety: float = 1.25,
        min_delta_floor: float = 1e-4,
        start_from_epoch: int = EBOPS_WARMUP_EPOCH,
    ):
        super().__init__()
        import numpy as np
        self.np = np
        self.factor = factor
        self.patience = patience
        self.cooldown = cooldown
        self.min_lr = min_lr
        self.window = window
        self.c_safety = c_safety
        self.min_delta_floor = min_delta_floor
        self.start_from_epoch = start_from_epoch

        self.acc_history = []
        self.patience_wait = 0
        self.cooldown_wait = 0
        self.best_acc = -float("inf")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current_lr = float(ops.convert_to_numpy(self.model.optimizer.learning_rate))
        logs["lr"] = current_lr
        logs["learning_rate"] = current_lr

        current_acc = logs.get("val_sparse_categorical_accuracy")
        if current_acc is None:
            return

        self.acc_history.append(current_acc)

        # 1. Warmup bypass: skip LR adjustments while BetaPID compresses model
        if epoch < self.start_from_epoch:
            self.best_acc = max(self.best_acc, current_acc)
            return

        # 2. Cooldown lockout
        if self.cooldown_wait > 0:
            self.cooldown_wait -= 1
            return

        # 3. Calculate dynamic min_delta from rolling noise window
        if len(self.acc_history) >= self.window:
            recent_vals = self.acc_history[-self.window :]
            noise_std = float(self.np.std(recent_vals))
            dynamic_min_delta = max(self.min_delta_floor, self.c_safety * noise_std)
        else:
            noise_std = 0.0
            dynamic_min_delta = self.min_delta_floor

        # 4. Check improvement against dynamic min_delta
        if current_acc > self.best_acc + dynamic_min_delta:
            self.best_acc = current_acc
            self.patience_wait = 0
        else:
            self.patience_wait += 1

        # Debug printout per epoch including current LR
        print(
            f" [LR-Debug] Ep {epoch+1}: acc={current_acc:.4f}, best={self.best_acc:.4f}, "
            f"std={noise_std:.5f}, min_delta={dynamic_min_delta:.5f}, wait={self.patience_wait}/{self.patience}, lr={current_lr:.6f}"
        )

        # 5. Trigger LR reduction when patience is exhausted
        if self.patience_wait >= self.patience:
            old_lr = current_lr
            if old_lr > self.min_lr:
                new_lr = max(old_lr * self.factor, self.min_lr)
                self.model.optimizer.learning_rate = new_lr
                self.cooldown_wait = self.cooldown
                self.patience_wait = 0
                logs["lr"] = new_lr
                logs["learning_rate"] = new_lr
                print(
                    f"\n>>> [AdaptiveReduceLROnPlateau] Epoch {epoch+1}: Reducing LR from {old_lr:.6f} to {new_lr:.6f} "
                    f"(dynamic min_delta was {dynamic_min_delta:.5f}).\n"
                )

