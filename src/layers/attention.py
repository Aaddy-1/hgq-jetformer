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
        # This is exactly how weights are initialized in Pytorch's nn.Linear by default
        self.parity_initializer = keras.initializers.VarianceScaling(
            scale=1 / 3, mode="fan_in", distribution="uniform"
        )

        # 1. Normalization Selection
        if self.normalization == "Batch":
            self.norm = keras.layers.BatchNormalization(
                axis=-1, momentum=momentum, epsilon=1e-5
            )
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
            kernel_initializer=self.parity_initializer,
            name="query",
        )
        self.k_proj = dense_cls(
            self.latent_dim,
            use_bias=False,
            kernel_initializer=self.parity_initializer,
            name="key",
        )
        self.v_proj = dense_cls(
            self.latent_dim,
            use_bias=False,
            kernel_initializer=self.parity_initializer,
            name="value",
        )
        # Add this after your Q, K, V projections
        seq_len_with_cls = self.num_particles + 1
        
        if self.normalization == "Batch":
            # axis=1 maps to the flattened seq_len*seq_len dimension
            self.pre_exp_norm = keras.layers.BatchNormalization(
                axis=1, momentum=self.momentum, epsilon=1e-5, name="pre_exp_norm"
            )
        elif self.normalization == "Layer":
            self.pre_exp_norm = keras.layers.LayerNormalization(
                axis=1, name="pre_exp_norm"
            )
        else:
            self.pre_exp_norm = None


        # 3. Output Projection (Uses bias by default in PyTorch)
        self.out_proj = dense_cls(
            self.in_dim,
            kernel_initializer=self.parity_initializer,
            bias_initializer=self.parity_initializer,
            name="output",
        )

        # 4. Attention Softmax
        # Using QSoftmax ensures bit-accuracy if the library requires it
        self.softmax = keras.layers.Softmax(axis=-1)

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

        # 4.5 The Variance Injector (pre_exp_norm)
        # if self.pre_exp_norm is not None:
        #     seq_len_with_cls = self.num_particles + 1
            
        #     # Reshape from (batch, heads, seq, seq) to (batch, heads, seq * seq)
        #     energy_flat = ops.reshape(
        #         energy, (batch_size, self.num_heads, seq_len_with_cls * seq_len_with_cls)
        #     )
            
        #     # Transpose to (batch, seq * seq, heads) to match PyTorch channel dim
        #     energy_transposed = ops.transpose(energy_flat, (0, 2, 1))
            
        #     # Apply Normalization across batch and heads
        #     energy_normed = self.pre_exp_norm(energy_transposed, training=training)
            
        #     # Transpose back and reshape
        #     energy_restored = ops.transpose(energy_normed, (0, 2, 1))
        #     energy_post = ops.reshape(
        #         energy_restored, (batch_size, self.num_heads, seq_len_with_cls, seq_len_with_cls)
        #     )
        # else:
        #     energy_post = energy

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
