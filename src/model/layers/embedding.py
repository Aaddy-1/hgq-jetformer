import keras
from hgq.layers import QDense, QEinsumDenseBatchnorm, Quantizer


def apply_hgq_embedding(
    x, in_dim, embedding_dim, quantize=True, prefix="embedding", fused_bn=False
):
    from ..initializers import get_parity_initializer
    parity_initializer = get_parity_initializer()

    dense_cls = QDense if quantize else keras.layers.Dense

    # --fused_bn: BatchNorm on the projection's output, folded into it (gamma/sigma
    # into the kernel before the kernel quantizer, the shift into the bias), so the
    # deployed layer is still one dense. Quantized path only.
    if quantize and fused_bn:
        x = QEinsumDenseBatchnorm(
            "bnc,cC->bnC",
            (x.shape[1], embedding_dim),
            bias_axes="C",
            kernel_initializer=parity_initializer,
            bias_initializer=parity_initializer,
            name=f"{prefix}_projection",
        )(x)
    else:
        x = dense_cls(
            embedding_dim,
            kernel_initializer=parity_initializer,
            bias_initializer=parity_initializer,
            name=f"{prefix}_projection",
        )(x)

    if quantize:
        x = Quantizer(name=f"{prefix}_quantizer")(x)

    return x
