import keras
from keras import ops
from keras import layers
from hgq.layers import QBatchNormalization, Quantizer, QDense

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
    # withClsFunctional
    # ==========================================
    # FUNCTIONAL CLS TOKEN INJECTION
    # ==========================================
    # 1. Isolate a single quantized spatial vector.
    # The native Python slice is preserved by compile_hls.py
    dummy_slice = x[:, 0:1, :]

    # 2. Zero out the spatial data to create a blank canvas.
    zero_slice = dummy_slice * 0.0

    # 3. Inject the trainable token parameters using a standard primitive.
    # Because x is already quantized from the embedding, this avoids the cmvm crash.
    raw_cls_tokens = keras.layers.Dense(
        units=embed_dim,
        use_bias=True,
        kernel_initializer="zeros",
        bias_initializer="random_normal",
        trainable=True,
        name="cls_token_weight",
    )(zero_slice)

    # from hgq.layers import Quantizer

    # 4. Explicitly bound the float32 bias output back into the integer domain
    # before it enters the Transformer blocks.
    if quantize:
        quantized_cls_tokens = Quantizer(name="cls_quantizer")(raw_cls_tokens)
    else:
        quantized_cls_tokens = raw_cls_tokens

    # 5. Route the purely quantized tensors into the sequence block.
    x = keras.layers.Concatenate(axis=1, name="cls_token_injection")(
        [quantized_cls_tokens, x]
    )
    # ==========================================

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
    # TODO: Investigate this claim
    # It incurs zero hardware cost and will be optimized out by Alkaid
    # while preserving the "extract_cls" boundary marker in the netlist.
    cls_out = keras.layers.Activation("linear", name="extract_cls")(raw_slice)

    # 6. Final Normalization
    if not quantize:
        if normalization == "Batch":
            cls_out = keras.layers.BatchNormalization(
                axis=-1, name="final_norm", momentum=0.9, epsilon=1e-5
            )(cls_out)
        elif normalization == "Layer":
            cls_out = keras.layers.LayerNormalization(axis=-1, name="final_norm")(cls_out)
        else:
            cls_out = keras.layers.BatchNormalization(
                axis=-1, name="final_norm", momentum=0.9, epsilon=1e-5
            )(cls_out)
    else:
        cls_out = QBatchNormalization(
            axis=-1, momentum=0.9, epsilon=1e-5, name="final_norm"
        )(cls_out)

    from .initializers import get_parity_initializer
    parity_initializer = get_parity_initializer()

    dense_cls = QDense if quantize else keras.layers.Dense
    logits = dense_cls(
        num_classes,
        kernel_initializer=parity_initializer,
        name="classifier_head",
    )(cls_out)

    # 8. Compile Static Graph
    return keras.Model(inputs=inputs, outputs=logits, name="HGQJetFormer")
