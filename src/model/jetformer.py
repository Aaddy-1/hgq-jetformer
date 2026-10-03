import keras
from keras import ops
from keras import layers
from hgq.layers import QBatchNormalization, Quantizer, QDense, QEinsumDenseBatchnorm

# Assuming imports from your architecture definitions:
from .layers.embedding import apply_hgq_embedding
from .layers.transformer import apply_hgq_transformer_block

from hgq.layers import Quantizer
from keras import ops


@keras.saving.register_keras_serializable()
class PrependCLSToken(keras.layers.Layer):
    def __init__(self, embed_dim, quantize=True, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.quantize = quantize

    def build(self, input_shape):
        self.cls_token = self.add_weight(
            name="cls_token",
            shape=(1, 1, self.embed_dim),
            initializer="random_normal",
            trainable=True,
        )
        if self.quantize:
            self.token_quantizer = Quantizer(name="cls_quantizer")
        super().build(input_shape)

    def call(self, x):
        batch_size = ops.shape(x)[0]
        tokens = ops.broadcast_to(self.cls_token, (batch_size, 1, self.embed_dim))

        if self.quantize:
            tokens = self.token_quantizer(tokens)

        return ops.concatenate([tokens, x], axis=1)


def build_hgq_jetformer(
    in_dim=3,
    embed_dim=32,
    num_heads=2,
    num_classes=5,
    num_transformers=1,
    proj_dim_k=2,
    dropout=0.0,
    num_particles=16,
    activation="ReLU",
    normalization="Batch",
    quantize=True,
    use_linformer=True,
    use_cls_token=False,
    floor_attn_datalane=False,
    ffn_out_activation=True,
    head_activation=False,
    fused_bn=False,
):
    # 1. Explicit Input Definition
    inputs = keras.Input(shape=(num_particles, in_dim), name="input_particles")

    # 2. Input Embedding Projection
    x = apply_hgq_embedding(
        inputs,
        in_dim=in_dim,
        embedding_dim=embed_dim,
        quantize=quantize,
        prefix="embedding",
        fused_bn=fused_bn,
    )

    # 3. Optional CLS Token Injection
    if use_cls_token:
        dummy_slice = x[:, 0:1, :]
        zero_slice = dummy_slice * 0.0
        raw_cls_tokens = keras.layers.Dense(
            units=embed_dim,
            use_bias=True,
            kernel_initializer="zeros",
            bias_initializer="random_normal",
            trainable=True,
            name="cls_token_weight",
        )(zero_slice)

        if quantize:
            quantized_cls_tokens = Quantizer(name="cls_quantizer")(raw_cls_tokens)
        else:
            quantized_cls_tokens = raw_cls_tokens

        x = keras.layers.Concatenate(axis=1, name="cls_token_injection")(
            [quantized_cls_tokens, x]
        )

    # 4. Pass through Transformer Encoder blocks
    for i in range(num_transformers):
        x = apply_hgq_transformer_block(
            x,
            in_dim=embed_dim,
            latent_dim=embed_dim,
            num_heads=num_heads,
            proj_dim_k=proj_dim_k,
            dropout=dropout,
            num_particles=num_particles,
            activation=activation,
            normalization=normalization,
            quantize=quantize,
            use_linformer=use_linformer,
            block_name=f"transformer_block_{i}",
            floor_attn_datalane=floor_attn_datalane,
            ffn_out_activation=ffn_out_activation,
            fused_bn=fused_bn,
        )

    # 5. Aggregation
    if use_cls_token:
        raw_slice = x[:, 0, :]
        pooled = keras.layers.Activation("linear", name="extract_cls")(raw_slice)
    else:
        pooled = keras.layers.GlobalAveragePooling1D(name="linformer_pool")(x)

    # 6. Dense Projection -> Activation (--head_activation) -> Classifier Head
    from .initializers import get_parity_initializer

    parity_initializer = get_parity_initializer()
    dense_cls = QDense if quantize else keras.layers.Dense

    def apply_head_dense(tensor, units, name):
        # --fused_bn: BatchNorm on the dense output, folded into the kernel and bias
        # (see layers/embedding.py). Applied to both head layers, logits included.
        # Quantized path only.
        if quantize and fused_bn:
            return QEinsumDenseBatchnorm(
                "bc,cC->bC",
                units,
                bias_axes="C",
                kernel_initializer=parity_initializer,
                name=name,
            )(tensor)
        return dense_cls(
            units,
            kernel_initializer=parity_initializer,
            name=name,
        )(tensor)

    embed_dense = apply_head_dense(pooled, embed_dim, "embed_dense")

    # embed_dense and classifier_head are both linear, so with nothing between them
    # they compose into one embed_dim -> num_classes map and embed_dense adds no
    # capacity. With head_activation=True the head becomes a one-hidden-layer MLP.
    # It runs once per jet rather than once per particle, and the activation itself
    # carries no EBOPs. classifier_head's own input quantizer bounds the
    # activation's output. Default False is the original linear head.
    if head_activation:
        embed_dense = keras.activations.get(activation.lower())(embed_dense)

    # [CLEAN] Commented out: classifier_head quantizes its input itself
    # (enable_iq=True), so this was a second rounding of the same tensor.
    # See .agents/analysis/size_gap_research_2026-10-01/C_audit.md E4.
    # if quantize:
    #     embed_dense = Quantizer(name="embed_dense_quantizer")(embed_dense)

    logits = apply_head_dense(embed_dense, num_classes, "classifier_head")

    # 7. Compile Static Graph
    return keras.Model(inputs=inputs, outputs=logits, name="HGQJetFormer")
