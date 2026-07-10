import keras
from hgq.layers import QDense, QBatchNormalization, Quantizer


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

    # Block 1: Norm -> Linear (Expansion) -> Activation
    x = apply_norm(x, "norm1")
    x = dense_cls(
        hidden_dim,
        use_bias=False,
        kernel_initializer=parity_initializer,
        name=f"{prefix}_expand",
    )(x)

    if quantize:
        x = Quantizer(name=f"{prefix}_lut_in_1")(x)  # Bounds the LUT Address Space

    x = activation_fn(x)

    if quantize:
        x = Quantizer(name=f"{prefix}_lut_out_1")(x)  # Bounds the LUT Value Space

    # Block 2: Norm -> Linear (Contraction) -> Activation
    x = apply_norm(x, "norm2")
    x = dense_cls(
        in_dim,
        use_bias=False,
        kernel_initializer=parity_initializer,
        name=f"{prefix}_contract",
    )(x)

    if quantize:
        x = Quantizer(name=f"{prefix}_lut_in_2")(x)  # Bounds the LUT Address Space

    x = activation_fn(x)

    if quantize:
        x = Quantizer(name=f"{prefix}_lut_out_2")(x)  # Bounds the LUT Value Space

    return x
