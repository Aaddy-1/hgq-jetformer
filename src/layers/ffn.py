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
    training=False
):
    hidden_dim = in_dim * multiplication
    dense_cls = QDense if quantize else keras.layers.Dense
    activation_fn = keras.activations.get(activation.lower())

    parity_initializer = keras.initializers.VarianceScaling(
        scale=1 / 3, mode="fan_in", distribution="uniform"
    )

    def apply_norm(tensor, name_suffix):
        if quantize:
            return QBatchNormalization(
                axis=-1, epsilon=1e-5, name=f"{prefix}_{name_suffix}"
            )(tensor, training=training)
        if normalization == "Batch":
            return keras.layers.BatchNormalization(
                axis=-1, momentum=momentum, epsilon=1e-5, name=f"{prefix}_{name_suffix}"
            )(tensor, training=training)
        else:
            return keras.layers.LayerNormalization(
                axis=-1, name=f"{prefix}_{name_suffix}"
            )(tensor)

    # Block 1: Norm -> Linear (Expansion) -> Activation
    x = apply_norm(x, "norm1")
    x = dense_cls(
        hidden_dim, 
        use_bias=False, 
        kernel_initializer=parity_initializer, 
        name=f"{prefix}_expand"
    )(x)
    x = activation_fn(x)
    
    if quantize:
        x = Quantizer(name=f"{prefix}_quantizer1")(x)

    # Block 2: Norm -> Linear (Contraction) -> Activation
    x = apply_norm(x, "norm2")
    x = dense_cls(
        in_dim, 
        use_bias=False, 
        kernel_initializer=parity_initializer, 
        name=f"{prefix}_contract"
    )(x)
    x = activation_fn(x)
    
    if quantize:
        x = Quantizer(name=f"{prefix}_quantizer2")(x)
        x = QBatchNormalization(
            axis=-1, epsilon=1e-5, name=f"{prefix}_residual_align"
        )(x, training=training)

    return x