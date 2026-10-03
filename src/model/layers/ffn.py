import keras
from hgq.layers import QDense, QBatchNormalization, QEinsumDenseBatchnorm, Quantizer


def apply_hgq_feed_forward(
    x,
    in_dim,
    multiplication=2,
    activation="ReLU",
    normalization="Layer",
    momentum=0.9,
    quantize=True,
    prefix="ffn",
    training=False,
    out_activation=True,
    fused_bn=False,
):
    hidden_dim = in_dim * multiplication
    dense_cls = QDense if quantize else keras.layers.Dense
    activation_fn = keras.activations.get(activation.lower())

    from ..initializers import get_parity_initializer
    parity_initializer = get_parity_initializer()

    def apply_norm(tensor, name_suffix):
        if not quantize:
            if normalization == "Batch":
                return keras.layers.BatchNormalization(
                    axis=-1, momentum=momentum, epsilon=1e-5, name=f"{prefix}_{name_suffix}"
                )(tensor, training=training)
            else:
                return keras.layers.LayerNormalization(
                    axis=-1, name=f"{prefix}_{name_suffix}"
                )(tensor)
        else:
            # [ABLATION] QBatchNormalization removed from quantized path.
            # return QBatchNormalization(
            #     axis=-1, momentum=momentum, epsilon=1e-5, name=f"{prefix}_{name_suffix}"
            # )(tensor, training=training)
            return tensor

    def apply_dense(tensor, units, name):
        # --fused_bn: BatchNorm on the dense output, folded into the kernel and bias
        # (see embedding.py). The fused layer needs a bias to carry the BatchNorm
        # shift, so it gains one where the plain dense has none. Quantized path only.
        if quantize and fused_bn:
            return QEinsumDenseBatchnorm(
                "bnc,cC->bnC",
                (tensor.shape[1], units),
                bias_axes="C",
                kernel_initializer=parity_initializer,
                name=name,
            )(tensor)
        return dense_cls(
            units,
            use_bias=False,
            kernel_initializer=parity_initializer,
            name=name,
        )(tensor)

    # Block 1: Norm -> Linear (Expansion) -> Activation
    x = apply_norm(x, "norm1")
    x = apply_dense(x, hidden_dim, f"{prefix}_expand")

    # [CLEAN] The LUT quantizers around each ReLU are commented out. ffn_contract's
    # own input quantizer (enable_iq=True) quantizes the activation's output, so
    # lut_in -> ReLU -> lut_out -> contract.iq rounded the same tensor three times.
    # ReLU is a native op in da4ml, not a table, so there is no LUT space to bound.
    # See .agents/analysis/size_gap_research_2026-10-01/C_audit.md E4.
    # if quantize:
    #     x = Quantizer(name=f"{prefix}_lut_in_1")(x)  # Bounds the LUT Address Space

    x = activation_fn(x)

    # if quantize:
    #     x = Quantizer(name=f"{prefix}_lut_out_1")(x)  # Bounds the LUT Value Space

    # Block 2: Norm -> Linear (Contraction) -> Activation (--ffn_out_activation)
    x = apply_norm(x, "norm2")
    x = apply_dense(x, in_dim, f"{prefix}_contract")

    # The FFN output feeds the residual QAdd, whose own input quantizer bounds it
    # (as attn_residual does for the attention output), with or without the
    # activation. Default True is the original FFN.
    if out_activation:
        # [CLEAN] Commented out for the same reason as lut_in_1 / lut_out_1:
        # ffn_residual's branch input quantizer quantizes the activation's output.
        # if quantize:
        #     x = Quantizer(name=f"{prefix}_lut_in_2")(x)  # Bounds the LUT Address Space

        x = activation_fn(x)

        # if quantize:
        #     x = Quantizer(name=f"{prefix}_lut_out_2")(x)  # Bounds the LUT Value Space

    return x
