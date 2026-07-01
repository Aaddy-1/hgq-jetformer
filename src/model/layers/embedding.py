import keras
from hgq.layers import QDense, Quantizer


def apply_hgq_embedding(x, in_dim, embedding_dim, quantize=True, prefix="embedding"):
    parity_initializer = keras.initializers.VarianceScaling(
        scale=1 / 3, mode="fan_in", distribution="uniform"
    )

    dense_cls = QDense if quantize else keras.layers.Dense

    x = dense_cls(
        embedding_dim,
        kernel_initializer=parity_initializer,
        bias_initializer=parity_initializer,
        name=f"{prefix}_projection",
    )(x)

    # if quantize:
    #     x = Quantizer(name=f"{prefix}_quantizer")(x)

    return x
