import keras
from keras import ops
from keras import layers
from hgq.layers import QBatchNormalization, Quantizer

# Assuming imports from your architecture definitions:
from ..layers.embedding import apply_hgq_embedding
from ..layers.transformer import apply_hgq_transformer_block


@keras.saving.register_keras_serializable()
class PrependCLSToken(keras.layers.Layer):
    """Isolated trainable parameter layer to avoid subclassing the main model."""

    def __init__(self, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim

    def build(self, input_shape):
        self.cls_token = self.add_weight(
            name="cls_token",
            shape=(1, 1, self.embed_dim),
            initializer=keras.initializers.RandomNormal(mean=0.0, stddev=1.0),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x):
        batch_size = ops.shape(x)[0]
        cls_tokens = ops.broadcast_to(self.cls_token, (batch_size, 1, self.embed_dim))
        return ops.concatenate([cls_tokens, x], axis=1)


def build_hgq_jetformer(
    in_dim=16,
    embed_dim=128,
    num_heads=2,
    num_classes=5,
    num_transformers=4,
    dropout=0.0,
    num_particles=30,
    activation="ReLU",
    normalization="Batch",
    quantize=True,
):
    # 1. Explicit Input Definition
    inputs = keras.Input(shape=(num_particles, in_dim), name="input_particles")

    # 2. Input Embedding
    # Note: Preserving your original logic where the embedding is unquantized
    # if you explicitly pass quantize=False, otherwise it follows the global flag.
    x = apply_hgq_embedding(
        inputs,
        in_dim=in_dim,
        embedding_dim=embed_dim,
        quantize=quantize,  # Match your original HGQJetFormer hardcoded value, or change to `quantize`
        prefix="embedding",
    )

    # 3. Prepare and Prepend CLS token
    # Extract a single spatial token to capture the dynamic batch dimension natively.
    # Shape becomes: (Batch, 1, embed_dim)
    dummy_slice = x[:, 0:1, :]

    # Nullify the slice. Keras translates this cleanly to an ops.multiply node.
    # Shape remains: (Batch, 1, embed_dim), but all values are 0.0
    zero_slice = dummy_slice * 0.0

    # Inject the trainable CLS parameter via a standard Dense bias.
    # The kernel multiplication evaluates to zero; only the bias (the CLS token) remains.
    cls_tokens = layers.Dense(
        units=embed_dim,
        use_bias=True,
        kernel_initializer="zeros",
        bias_initializer="random_normal",
        trainable=True,
        name="cls_token_weight",
    )(zero_slice)

    # Prepend the token to the sequence
    x = layers.Concatenate(axis=1, name="cls_token_injection")([cls_tokens, x])

    if quantize:
        x = Quantizer(name="entry_quantizer")(x)

    # 4. Pass through Transformer Encoder blocks
    for i in range(num_transformers):
        x = apply_hgq_transformer_block(
            x,
            in_dim=embed_dim,
            latent_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            num_particles=num_particles,
            activation=activation,
            normalization=normalization,
            quantize=quantize,
            block_name=f"transformer_block_{i}",
        )

    # 5. Extract CLS token output (Index 0)
    # Lambda layer ensures the slicing operation is named and traceable
    # Native topological slicing.
    # Keras automatically translates this into a static, serializable Slice op.
    raw_slice = x[:, 0, :]

    # Anchor the structural name for Alkaid's routing trace.
    # The 'linear' activation is a mathematical identity (f(x) = x).
    # It incurs zero hardware cost and will be optimized out by Alkaid
    # while preserving the "extract_cls" boundary marker in the netlist.
    cls_out = keras.layers.Activation("linear", name="extract_cls")(raw_slice)

    # 6. Final Normalization
    if quantize:
        # cls_out = QBatchNormalization(axis=-1, name="final_norm", epsilon=1e-5)(cls_out)
        cls_out = cls_out
    elif normalization == "Batch":
        cls_out = keras.layers.BatchNormalization(
            axis=-1, name="final_norm", momentum=0.9, epsilon=1e-5
        )(cls_out)
    elif normalization == "Layer":
        cls_out = keras.layers.LayerNormalization(axis=-1, name="final_norm")(cls_out)
    else:
        cls_out = keras.layers.BatchNormalization(
            axis=-1, name="final_norm", momentum=0.9, epsilon=1e-5
        )(cls_out)

    # 7. Classification Head
    parity_initializer = keras.initializers.VarianceScaling(
        scale=1 / 3, mode="fan_in", distribution="uniform"
    )

    dense_cls = keras.layers.Dense
    logits = dense_cls(
        num_classes,
        kernel_initializer=parity_initializer,
        name="classifier_head",
    )(cls_out)

    # 8. Compile Static Graph
    return keras.Model(inputs=inputs, outputs=logits, name="HGQJetFormer")
