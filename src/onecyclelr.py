import math
import keras
from keras import ops

@keras.saving.register_keras_serializable()
class OneCycleLR(keras.optimizers.schedules.LearningRateSchedule):
    """
    1:1 PyTorch Parity for OneCycleLR.
    Executes cosine annealing for both the warmup and decay phases.
    """
    def __init__(
        self, 
        max_lr: float, 
        total_steps: int, 
        pct_start: float = 0.2, 
        div_factor: float = 25.0, 
        final_div_factor: float = 1e4, 
        **kwargs
    ):
        super().__init__(**kwargs)
        self.max_lr = float(max_lr)
        self.total_steps = float(total_steps)
        self.pct_start = float(pct_start)
        self.div_factor = float(div_factor)
        self.final_div_factor = float(final_div_factor)

        # PyTorch default boundary calculations
        self.initial_lr = self.max_lr / self.div_factor
        self.min_lr = self.initial_lr / self.final_div_factor
        self.step_size_up = float(math.floor(self.total_steps * self.pct_start))
        self.step_size_down = self.total_steps - self.step_size_up

    def __call__(self, step):
        pi_tensor = ops.cast(math.pi, dtype=step.dtype)
        step = ops.cast(step, dtype="float32")
        total_steps = ops.cast(self.total_steps, dtype="float32")
        step_size_up = ops.cast(self.step_size_up, dtype="float32")
        step_size_down = ops.cast(self.step_size_down, dtype="float32")

        # Clamp step to prevent schedule drifting past total_steps
        step = ops.minimum(step, total_steps)

        # Phase 1: Warmup (Cosine annealing from initial_lr up to max_lr)
        phase_1_progress = step / step_size_up
        phase_1_lr = self.initial_lr + (self.max_lr - self.initial_lr) * 0.5 * (1.0 - ops.cos(pi_tensor * phase_1_progress))

        # Phase 2: Decay (Cosine annealing from max_lr down to min_lr)
        phase_2_progress = (step - step_size_up) / step_size_down
        phase_2_lr = self.min_lr + (self.max_lr - self.min_lr) * 0.5 * (1.0 + ops.cos(pi_tensor * phase_2_progress))

        # Condition routing
        return ops.where(step <= step_size_up, phase_1_lr, phase_2_lr)

    def get_config(self):
        return {
            "max_lr": self.max_lr,
            "total_steps": self.total_steps,
            "pct_start": self.pct_start,
            "div_factor": self.div_factor,
            "final_div_factor": self.final_div_factor,
        }