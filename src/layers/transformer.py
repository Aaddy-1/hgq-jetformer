import keras
from .attention import HGQSelfAttention
from .ffn import HGQFeedForward
from hgq.layers import QAdd

class HGQTransformerBlock(keras.layers.Layer):
    def __init__(
        self,
        in_dim,
        latent_dim=None,
        num_heads=1,
        dropout=0.0,
        num_particles=30,
        activation="ReLU",
        normalization="Layer",
        momentum=0.9,
        quantize=True,
        **kwargs
    ):
        super().__init__(**kwargs)

        # 1. Attention Block
        self.self_attention = HGQSelfAttention(
            in_dim=in_dim,
            latent_dim=latent_dim,
            num_heads=num_heads,
            num_particles=num_particles,
            normalization=normalization,
            momentum=momentum,
            quantize=quantize,
        )

        # 2. Modular Feed-Forward Block
        self.ffn = HGQFeedForward(
            in_dim=in_dim,
            multiplication=2,
            activation=activation,
            normalization=normalization,
            momentum=momentum,
            quantize=quantize,
        )

        self.dropout = keras.layers.Dropout(dropout)

        self.q_add = QAdd(name="residual_add") if quantize else keras.layers.Add()

    def call(self, x, training=False):
        # Step 1: Self-Attention (Residual is internal to our HGQSelfAttention)
        attn_out = self.self_attention(x, training=training)

        # Step 2: Feed-Forward
        ffn_out = self.ffn(attn_out, training=training)

        # Step 3: Final Residual (out = x_after_attn + ffn_output)
        # Mirroring the PyTorch: out = x + out5
        x = self.q_add([attn_out, ffn_out])

        # Step 4: Final Dropout
        return self.dropout(x, training=training)
