import keras
from hgq.layers import QDense


@keras.saving.register_keras_serializable()
class HGQFeedForward(keras.layers.Layer):
    """
    Position-wise feed-forward layer with HGQ2 integration.
    Replicates the 'Double-Norm' structure from the reference design.
    """

    def __init__(
        self,
        in_dim,
        multiplication=2,
        activation="ReLU",
        normalization="Layer",
        momentum=0.9,
        quantize=True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.in_dim = in_dim
        self.multiplication = multiplication
        self.activation = activation
        self.normalization = normalization
        self.momentum = momentum
        self.quantize = quantize

        # Internal dimension: expansion factor
        hidden_dim = in_dim * multiplication

        # Determine whether to use quantized or standard dense layers
        dense_cls = QDense if quantize else keras.layers.Dense

        def get_norm(name):
            if normalization == "Batch":
                return keras.layers.BatchNormalization(
                    axis=-1, name=name, momentum=momentum
                )
            else:
                return keras.layers.LayerNormalization(axis=-1, name=name)

        # Layer Definitions based on the nn.Sequential reference
        self.norm1 = get_norm("ffn_norm1")
        self.dense1 = dense_cls(
            hidden_dim,
            use_bias=False,
            kernel_initializer="he_uniform",
            name="ffn_expand",
        )

        self.norm2 = get_norm("ffn_norm2")
        self.dense2 = dense_cls(
            in_dim, use_bias=False, kernel_initializer="he_uniform", name="ffn_contract"
        )

        # Activation (Keras 'silu' is equivalent to torch.nn.SiLU)
        self.activation_fn = keras.activations.get(activation.lower())

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

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "in_dim": self.in_dim,
                "multiplication": self.multiplication,
                "activation": self.activation,
                "normalization": self.normalization,
                "momentum": self.momentum,
                "quantize": self.quantize,
            }
        )
        return config
