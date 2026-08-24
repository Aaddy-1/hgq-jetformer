import keras
from keras import ops
from keras import layers
from hgq.layers import QBatchNormalization, Quantizer, QDense

# Assuming imports from your architecture definitions:
from .layers.embedding import apply_hgq_embedding
from .layers.transformer import apply_hgq_transformer_block
from .pooling import DEFAULT_MASK_COLS, apply_rich_pooling

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
    rich_pool=False,
    head_width=None,
    mask_cols=DEFAULT_MASK_COLS,
    mask_threshold=None,
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
        )

    # 5. Aggregation
    #
    # [B1] --rich_pool replaces the single mean with [sum | max | mean | n],
    # masked against padding. See pooling.py for why the mask is bundled with it
    # and why n has to be handed back explicitly. Mutually exclusive with the
    # CLS token, which is a different aggregation strategy entirely (it reads
    # one slot rather than reducing over them, so there is nothing to mask).
    if use_cls_token:
        if rich_pool:
            raise ValueError(
                "--rich_pool and use_cls_token are two different aggregation "
                "strategies; enable at most one."
            )
        raw_slice = x[:, 0, :]
        pooled = keras.layers.Activation("linear", name="extract_cls")(raw_slice)
    elif rich_pool:
        pooled = apply_rich_pooling(
            x,
            inputs,
            num_particles=num_particles,
            mask_cols=mask_cols,
            mask_threshold=mask_threshold,
        )
    else:
        pooled = keras.layers.GlobalAveragePooling1D(name="linformer_pool")(x)

    # 6. Dense Projection & Classifier Head
    #
    # [B2] --head_width N widens this from embed_dim -> embed_dim -> num_classes
    # to N -> N -> num_classes. Everything here runs ONCE per jet, whereas every
    # layer above runs once per particle, so a multiply here costs 1/num_particles
    # of a multiply there. Measured on E4_EMBED16_SEED42 the default head is 416
    # MACs of ~240,600 in the model -- 0.17% -- which is why this is the cheapest
    # capacity available and why it had never been worth flagging until the
    # particle axis came down.
    from .initializers import get_parity_initializer

    dense_cls = QDense if quantize else keras.layers.Dense
    hidden_units = embed_dim if head_width is None else head_width

    # parity_initializer must stay per-layer (CLAUDE.md): a single shared object
    # breaks Keras graph generation and hardware compilation.
    embed_dense = dense_cls(
        hidden_units,
        kernel_initializer=get_parity_initializer(),
        name="embed_dense",
    )(pooled)

    if quantize:
        embed_dense = Quantizer(name="embed_dense_quantizer")(embed_dense)

    # The second hidden layer exists only under --head_width. With the flag off
    # this branch is skipped and the graph is byte-for-byte the previous one, so
    # E4_EMBED16_SEED42/43/44 and C1_CROP64_SEED42 stay valid controls.
    if head_width is not None:
        embed_dense = dense_cls(
            hidden_units,
            kernel_initializer=get_parity_initializer(),
            name="head_dense_2",
        )(embed_dense)
        if quantize:
            embed_dense = Quantizer(name="head_dense_2_quantizer")(embed_dense)

    logits = dense_cls(
        num_classes,
        kernel_initializer=get_parity_initializer(),
        name="classifier_head",
    )(embed_dense)

    # 7. Compile Static Graph
    return keras.Model(inputs=inputs, outputs=logits, name="HGQJetFormer")
