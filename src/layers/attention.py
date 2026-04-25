import keras
from keras import ops
from hgq.layers import QDense, QSoftmax


class HGQSelfAttention(keras.layers.Layer):
    def __init__(
        self,
        in_dim,
        latent_dim=None,
        num_heads=1,
        num_particles=30,
        normalization="Layer",
        momentum=0.9,
        quantize=True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.in_dim = in_dim
        self.latent_dim = latent_dim if latent_dim is not None else in_dim
        self.num_heads = num_heads
        self.head_dim = self.latent_dim // num_heads
        self.normalization = normalization
        self.momentum = momentum
        self.num_particles = num_particles
        self.quantize = quantize

        # 1. Normalization Selection
        if self.normalization == "Batch":
            self.norm = keras.layers.BatchNormalization(axis=-1, momentum=momentum)
        elif self.normalization == "Layer":
            self.norm = keras.layers.LayerNormalization(axis=-1)
        else:
            self.norm = None

        # 2. Q, K, V Projections (Replacing nn.Linear)
        # Jetformer uses bias=False for Q, K, V
        dense_cls = QDense if quantize else keras.layers.Dense

        self.q_proj = dense_cls(
            self.latent_dim,
            use_bias=False,
            kernel_initializer="he_uniform",
            name="query",
        )
        self.k_proj = dense_cls(
            self.latent_dim, use_bias=False, kernel_initializer="he_uniform", name="key"
        )
        self.v_proj = dense_cls(
            self.latent_dim,
            use_bias=False,
            kernel_initializer="he_uniform",
            name="value",
        )

        # 3. Output Projection (Uses bias by default in PyTorch)
        self.out_proj = dense_cls(
            self.in_dim,
            kernel_initializer="he_uniform",
            bias_initializer=keras.initializers.VarianceScaling(
                scale=1 / 3, mode="fan_in", distribution="uniform"
            ),
            name="output",
        )

        # 4. Attention Softmax
        # Using QSoftmax ensures bit-accuracy if the library requires it
        self.softmax = QSoftmax(axis=-1)

    def call(self, x, training=False):
        # x shape: (batch, seq_len, in_dim)
        residual = x

        # 1. Input Normalization
        if self.norm is not None:
            x = self.norm(x, training=training)

        # 2. Generate Q, K, V
        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)

        # 3. Reshape for Multi-Head: (batch, seq_len, heads, head_dim)
        # We use ops.reshape to stay backend-agnostic
        shape = ops.shape(x)
        batch_size, seq_len = shape[0], shape[1]

        queries = ops.reshape(
            queries, (batch_size, seq_len, self.num_heads, self.head_dim)
        )
        keys = ops.reshape(keys, (batch_size, seq_len, self.num_heads, self.head_dim))
        values = ops.reshape(
            values, (batch_size, seq_len, self.num_heads, self.head_dim)
        )

        # 4. Energy Calculation: Attention softmax(Q^T * K)
        # PyTorch: "nqhc,nkhc->nhqk"
        # (n:batch, q:query_seq, k:key_seq, h:heads, c:head_dim)
        energy = ops.einsum("nqhc,nkhc->nhqk", queries, keys)

        # Scale and Softmax
        scale = ops.cast(self.head_dim, x.dtype) ** 0.5
        attention = self.softmax(energy / scale)

        # 5. Context Vector Calculation
        # PyTorch: "nhql,nlhc->nqhc"
        out = ops.einsum("nhql,nlhc->nqhc", attention, values)

        # 6. Re-combine heads and Final Projection
        out = ops.reshape(out, (batch_size, seq_len, self.latent_dim))
        out = self.out_proj(out)

        # 7. Residual Connection
        return out + residual
