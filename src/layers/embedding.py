import keras
from hgq.layers import QDense, Quantizer


class HGQEmbedding(keras.layers.Layer):
    """
    Embedding layer with HGQ2 integration.
    Projects raw particle features into the latent space.
    """

    def __init__(self, in_dim, embedding_dim, quantize=True, **kwargs):
        super().__init__(**kwargs)
        self.in_dim = in_dim
        self.embedding_dim = embedding_dim
        self.quantize = quantize

        self.parity_initializer = keras.initializers.VarianceScaling(
            scale=1 / 3, mode="fan_in", distribution="uniform"
        )

        dense_cls = QDense if quantize else keras.layers.Dense

        self.dense_embedding = dense_cls(
            embedding_dim,
            kernel_initializer=self.parity_initializer,
            # Replicates PyTorch's default uniform bias initialization
            bias_initializer=self.parity_initializer,
            name="embedding_projection",
        )

        if self.quantize:
            self.quantizer = Quantizer(name="embedding_quantizer")
        else:
            self.quantizer = None

    def call(self, x, training=False):
        # x shape: (batch, num_particles, in_dim)
        x = self.dense_embedding(x)
        if self.quantizer is not None:
            x = self.quantizer(x)
        return x
