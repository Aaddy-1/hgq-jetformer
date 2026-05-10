import keras
from keras import ops
from hgq.layers import QDense, QBatchNormalization, Quantizer

# Relative imports from your src directory structure
from ..layers.embedding import HGQEmbedding
from ..layers.transformer import HGQTransformerBlock


class HGQJetFormer(keras.Model):
    def __init__(
        self,
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
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.normalization = normalization
        self.quantize = quantize

        # 1. Input Embedding
        self.embedding = HGQEmbedding(
            in_dim=in_dim, embedding_dim=embed_dim, quantize=False
        )

        # 2. Transformer Encoder Stack
        self.transformers = [
            HGQTransformerBlock(
                in_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                num_particles=num_particles,
                activation=activation,
                normalization=normalization,
                quantize=quantize,
                name=f"transformer_block_{i}",
            )
            for i in range(num_transformers)
        ]

        # Force Keras to track the entire stack in the computational graph
        # transformer_list = [
        #     HGQTransformerBlock(
        #         in_dim=embed_dim,
        #         num_heads=num_heads,
        #         dropout=dropout,
        #         num_particles=num_particles,
        #         activation=activation,
        #         normalization=normalization,
        #         quantize=quantize,
        #         name=f"transformer_block_{i}",
        #     )
        #     for i in range(num_transformers)
        # ]

        # # 2. Transformer Encoder Stack
        # self.transformer_blocks = []
        # for i in range(num_transformers):
        #     block = HGQTransformerBlock(
        #         in_dim=embed_dim,
        #         num_heads=num_heads,
        #         dropout=dropout,
        #         num_particles=num_particles,
        #         activation=activation,
        #         normalization=normalization,
        #         quantize=quantize,
        #         name=f"transformer_block_{i}",
        #     )
        #     # CRITICAL: Manually register each block as a top-level attribute
        #     # This forces the HGQ callback to 'see' the layers
        #     setattr(self, f"block_{i}", block)
        #     self.transformer_blocks.append(block)
        
        # This explicit wrapper registers the layers and their EBOP metrics
        # self.transformers = keras.Sequential(transformer_list, name="transformer_stack")

        # 3. Final Normalization (Applied to CLS token output)
        if self.quantize:
            self.final_norm = QBatchNormalization(axis=-1, name="final_norm", epsilon=1e-5)
        elif self.normalization == "Batch":
            self.final_norm = keras.layers.BatchNormalization(
                axis=-1, name="final_norm", momentum=0.9, epsilon=1e-5
            )
        elif self.normalization == "Layer":
            self.final_norm = keras.layers.LayerNormalization(
                axis=-1, name="final_norm"
            )
        else:
            self.final_norm = keras.layers.BatchNormalization(
                axis=-1, name="final_norm", momentum=0.9, epsilon=1e-5
            )

        self.parity_initializer = keras.initializers.VarianceScaling(
            scale=1 / 3, mode="fan_in", distribution="uniform"
        )

        # 4. Classification Head
        dense_cls = keras.layers.Dense
        self.classifier = dense_cls(
            num_classes,
            kernel_initializer=self.parity_initializer,
            name="classifier_head",
        )

        if self.quantize:
            # Establishes the radix metadata bridge for the entire transformer stack
            self.entry_quantizer = Quantizer(name="entry_quantizer")

    def build(self, input_shape):
        # Initialize the learned CLS token: (1, 1, embed_dim)
        self.cls_token = self.add_weight(
            name="cls_token",
            shape=(1, 1, self.embed_dim),
            initializer=keras.initializers.RandomNormal(mean=0.0, stddev=1.0),
            # initializer=keras.initializers.TruncatedNormal(stddev=0.02),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x, training=False):
        batch_size = ops.shape(x)[0]

        # Step 1: Embedding Projection
        x = self.embedding(x, training=training)

        # Step 2: Prepare and Prepend CLS token
        # Broadcast CLS token to match batch size: (Batch, 1, embed_dim)
        cls_tokens = ops.broadcast_to(self.cls_token, (batch_size, 1, self.embed_dim))
        x = ops.concatenate([cls_tokens, x], axis=1)

        if self.quantize:
            x = self.entry_quantizer(x)

        # Step 3: Pass through Transformer Encoder blocks
        for transformer in self.transformers:
            x = transformer(x, training=training)
        # x = self.transformers(x, training=training)  # Pass through the
        # Step 3: Pass through Transformer Encoder blocks
        # for block in self.transformer_blocks:
        #     x = block(x, training=training)

        # Step 4: Extract CLS token output (Index 0)
        # Shape: (Batch, embed_dim)
        cls_out = x[:, 0, :]

        # Step 5: Final Normalization and Head
        cls_out = self.final_norm(cls_out, training=training)
        logits = self.classifier(cls_out)

        # Step 6: Return raw logits (Softmax will be applied in loss function)
        return logits
