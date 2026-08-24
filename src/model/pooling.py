"""Masked multi-statistic pooling for the aggregation step (B1).

The default aggregation is `GlobalAveragePooling1D` over all `num_particles`
slots. Two things are wrong with that at this budget:

  1. It is one statistic. The transformer's output is a (P, E) table and the
     mean collapses it with the bluntest available operator: two jets with very
     different constituent structure can average identically.
  2. It averages over PADDING. A padded slot holds literal 0.0 in the h5, but
     `mean.npy`/`std.npy` are computed with padding included
     (`build_jetclass_dataset.py:155` flattens over the particle axis), so
     z-scoring maps it to a NONZERO constant, and `embedding_projection` has a
     bias, so it cannot map that constant to zero either. Every padded slot
     therefore carries some constant vector v_pad into the pool, and
     `GlobalAveragePooling1D` divides by P regardless of how many slots are
     real:

         pooled = (1/P) [ sum_real + (P - n) * v_pad ]

     which is an affine function of the multiplicity n.

That second point cuts both ways, and it is the reason `n` is part of the
output below rather than an optional extra. The pooled vector currently leaks
multiplicity for free, and multiplicity is strongly discriminative here --
measured on the 128 test set, mean constituent count runs from 27.6 (t_bl) to
56.4 (H_gg), a 2x spread. Masking DELETES that accidental signal. If it is not
re-supplied explicitly the arm is expected to regress; that is the single most
likely way it fails.

Sum and max are the statistics that make this worth doing and they are also the
two that padding destroys outright (max can return v_pad itself; sum
accumulates (P - n) copies of it), which is why the mask is bundled here rather
than offered separately.

Cost. EBOPs counts multipliers. `max` and `min` are comparators and `sum` is an
adder tree, so the width increase from E to 3E + 1 is close to free on the
meter; the only multipliers introduced are the two mask gatings (P*E products
against a 1-bit mask, i.e. AND gates) and the E products of the mean's
reciprocal.

Functional style throughout, per CLAUDE.md -- no Layer subclasses, so the whole
thing serializes as ordinary graph ops and `evaluate.py`'s `load_model` needs no
custom_objects.
"""

import numpy as np
from keras import ops

# The five particle-type one-hots (isChargedHadron, isNeutralHadron, isPhoton,
# isElectron, isMuon) as written by `build_jetclass_dataset.py:85-108`. Exactly
# one is set for a real constituent and all five are zero on padding, which is
# what makes the mask below exact rather than learned.
DEFAULT_MASK_COLS = (6, 11)


def particle_mask_threshold(mean, std, mask_cols=DEFAULT_MASK_COLS, eps=1e-8):
    """The constant that separates a real constituent from padding, exactly.

    `dataset.py` applies `(x - mean) / (std + eps)`, which is affine and
    therefore injective, so the one-hot structure survives normalization even
    though the literal 0/1 values do not. Summing the five type columns of a
    normalized row gives:

        padding      ->  A          where A = sum_c (-mean_c / std_c)
        real type j  ->  A + 1/std_j

    Every `1/std_j` is strictly positive, so A is the strict minimum over all
    six possible rows and a single threshold placed half a step above it
    separates them for every jet, forever. This is a property of the encoding,
    not a fitted boundary -- there is nothing here to drift under QAT.

    `mean` and `std` are the full-width arrays as saved by
    `compute_welford_stats`; they MUST come from the same directory the run
    trains against, since the padded fraction (and hence the statistics) differ
    between a 128 and a 64 tree.
    """
    lo, hi = mask_cols
    mu = np.asarray(mean, dtype=np.float64)[lo:hi]
    sd = np.asarray(std, dtype=np.float64)[lo:hi] + eps
    if mu.shape[0] != 5:
        raise ValueError(
            f"mask_cols={mask_cols} selected {mu.shape[0]} columns of the "
            f"{np.asarray(mean).shape[0]}-feature stats; expected the 5 "
            f"particle-type one-hots. --rich_pool requires the 17-feature "
            f"JetClass layout."
        )
    pad_sum = float(np.sum(-mu / sd))
    step = float(np.min(1.0 / sd))
    return pad_sum + 0.5 * step


def apply_particle_mask(inputs, mask_cols=DEFAULT_MASK_COLS, threshold=None):
    """(B, P, F) normalized inputs -> (B, P, 1) mask, 1.0 real / 0.0 padding."""
    if threshold is None:
        raise ValueError(
            "apply_particle_mask needs a threshold from particle_mask_threshold(); "
            "it is derived from the dataset's mean/std and has no safe default."
        )
    lo, hi = mask_cols
    type_sum = ops.sum(inputs[:, :, lo:hi], axis=-1, keepdims=True)
    return ops.cast(type_sum > threshold, inputs.dtype)


def apply_rich_pooling(
    x,
    inputs,
    num_particles,
    mask_cols=DEFAULT_MASK_COLS,
    mask_threshold=None,
):
    """(B, P, E) -> (B, 3E + 1): [masked sum | masked max | masked mean | n/P].

    `inputs` is the model's raw (normalized) input tensor, not `x` -- the mask
    is derived from the one-hot columns, which only exist upstream of the
    embedding.
    """
    mask = apply_particle_mask(inputs, mask_cols, mask_threshold)  # (B, P, 1)
    gated = x * mask                                               # padding -> 0

    total = ops.sum(gated, axis=1)                                 # (B, E)
    n = ops.sum(mask, axis=1)                                      # (B, 1)

    # Masked max. Rather than a magic -inf sentinel (which needs a constant wide
    # enough to sit below every real activation -- unknowable, and wrong if it
    # ever fails), floor the padded slots at the jet's own per-channel minimum.
    # That minimum is <= every real value by construction, so the max is EXACT,
    # and min/max are comparators so this costs no multipliers.
    floor = ops.min(x, axis=1, keepdims=True)                      # (B, 1, E)
    for_max = gated + (1.0 - mask) * floor
    maximum = ops.max(for_max, axis=1)                             # (B, E)

    # Masked mean = total / n. A general divider is expensive, but n is an
    # integer in [1, P], so this is a P-entry reciprocal ROM plus E multiplies.
    # Index 0 is 0.0 and is unreachable: a jet with zero constituents does not
    # exist in JetClass, and if one did, total would be zero anyway.
    recip = np.zeros(num_particles + 1, dtype="float32")
    recip[1:] = 1.0 / np.arange(1, num_particles + 1, dtype="float32")
    inv_n = ops.take(
        ops.convert_to_tensor(recip), ops.cast(ops.squeeze(n, axis=-1), "int32"), axis=0
    )
    mean = total * ops.expand_dims(inv_n, axis=-1)                 # (B, E)

    # n normalized to [0, 1] so it sits in the same range as the activations
    # beside it; handing the head a raw 6-bit integer next to O(1) values would
    # force the concatenated quantizer to carry integer bits it needs nowhere
    # else.
    n_feat = n / float(num_particles)                              # (B, 1)

    return ops.concatenate([total, maximum, mean, n_feat], axis=-1)
