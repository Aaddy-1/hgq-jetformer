import keras
from hgq.layers import QDense


class HGQFeedForward(keras.layers.Layer):
    """
    Position-wise feed-forward layer with HGQ2 integration.
    Replicates the 'Double-Norm' structure from the reference design.
    """

    def __init__(
        self,
        in_dim,
        multiplication=2,
        activation="silu",
        normalization="Layer",
        quantize=True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.in_dim = in_dim
        self.multiplication = multiplication
        self.quantize = quantize

        # Internal dimension: expansion factor
        hidden_dim = in_dim * multiplication

        # Determine whether to use quantized or standard dense layers
        dense_cls = QDense if quantize else keras.layers.Dense

        def get_norm(name):
            if normalization == "Batch":
                return keras.layers.BatchNormalization(axis=-1, name=name)
            else:
                return keras.layers.LayerNormalization(axis=-1, name=name)

        # Layer Definitions based on the nn.Sequential reference
        self.norm1 = get_norm("ffn_norm1")
        self.dense1 = dense_cls(hidden_dim, use_bias=False, name="ffn_expand")

        self.norm2 = get_norm("ffn_norm2")
        self.dense2 = dense_cls(in_dim, use_bias=False, name="ffn_contract")

        # Activation (Keras 'silu' is equivalent to torch.nn.SiLU)
        self.activation_fn = keras.activations.get(activation)

    def call(self, x, training=False):
        # x shape: (batch, seq_len, in_dim)

        # Block 1: Norm -> Activation -> Linear (Expansion)
        x = self.norm1(x, training=training)
        x = self.activation_fn(x)
        x = self.dense1(x)

        # Block 2: Norm -> Activation -> Linear (Contraction)
        x = self.norm2(x, training=training)
        x = self.activation_fn(x)
        x = self.dense2(x)

        return x
