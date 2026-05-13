import keras
from keras import ops
from hgq.layers import QDense, QSoftmax, QBatchNormalization, Quantizer, QAdd, QEinsum

def apply_hgq_self_attention(
    x,
    in_dim,
    latent_dim=None,
    num_heads=1,
    num_particles=30,
    normalization="Layer",
    momentum=0.9,
    quantize=True,
    prefix="self_attn",
    training=False
):
    latent_dim = latent_dim if latent_dim is not None else in_dim
    head_dim = latent_dim // num_heads
    inv_scale = head_dim ** -0.5

    parity_initializer = keras.initializers.VarianceScaling(
        scale=1 / 3, mode="fan_in", distribution="uniform"
    )

    residual = x

    # 1. Input Normalization
    if quantize:
        x = QBatchNormalization(axis=-1, epsilon=1e-5, name=f"{prefix}_input_norm")(x, training=training)
    elif normalization == "Batch":
        x = keras.layers.BatchNormalization(axis=-1, momentum=momentum, epsilon=1e-5, name=f"{prefix}_input_norm")(x, training=training)
    elif normalization == "Layer":
        x = keras.layers.LayerNormalization(axis=-1, name=f"{prefix}_input_norm")(x)

    # 2. Generate Q, K, V Projections
    dense_cls = QDense if quantize else keras.layers.Dense
    
    queries = dense_cls(latent_dim, use_bias=False, kernel_initializer=parity_initializer, name=f"{prefix}_query")(x)
    keys = dense_cls(latent_dim, use_bias=False, kernel_initializer=parity_initializer, name=f"{prefix}_key")(x)
    values = dense_cls(latent_dim, use_bias=False, kernel_initializer=parity_initializer, name=f"{prefix}_value")(x)

    # 3. Reshape for Multi-Head
    shape = ops.shape(x)
    batch_size, seq_len = shape[0], shape[1]

    queries = ops.reshape(queries, (batch_size, seq_len, num_heads, head_dim))
    keys = ops.reshape(keys, (batch_size, seq_len, num_heads, head_dim))
    values = ops.reshape(values, (batch_size, seq_len, num_heads, head_dim))

    # 4. Energy Calculation & Scaling (Deterministic MatMul)
    # Transpose to (batch_size, num_heads, seq_len, head_dim)
    q_t = ops.transpose(queries, axes=(0, 2, 1, 3))
    # Transpose to (batch_size, num_heads, head_dim, seq_len)
    k_t = ops.transpose(keys, axes=(0, 2, 3, 1))
    
    # Matmul: (B, H, Q, C) @ (B, H, C, K) -> (B, H, Q, K)
    energy = ops.matmul(q_t, k_t)
    scaled_energy = energy * inv_scale

    if quantize:
        scaled_energy = Quantizer(name=f"{prefix}_energy_quantizer")(scaled_energy)

    # Softmax
    softmax_cls = QSoftmax if quantize else keras.layers.Softmax
    attention = softmax_cls(axis=-1, name=f"{prefix}_softmax")(scaled_energy)

    # 5. Context Vector Calculation (Deterministic MatMul)
    # Transpose to (batch_size, num_heads, seq_len, head_dim)
    v_t = ops.transpose(values, axes=(0, 2, 1, 3))
    
    # Matmul: (B, H, Q, L) @ (B, H, L, C) -> (B, H, Q, C)
    out = ops.matmul(attention, v_t)
    
    # Transpose back to (batch_size, seq_len, num_heads, head_dim)
    out = ops.transpose(out, axes=(0, 2, 1, 3))

    if quantize:
        out = Quantizer(name=f"{prefix}_out_quantizer")(out)

    # 6. Re-combine heads and Final Projection
    out = ops.reshape(out, (batch_size, seq_len, latent_dim))
    
    out = dense_cls(
        in_dim, 
        kernel_initializer=parity_initializer, 
        bias_initializer=parity_initializer, 
        name=f"{prefix}_output"
    )(out)

    # 7. Residual Alignment 
    if quantize:
        out = QBatchNormalization(axis=-1, epsilon=1e-5, name=f"{prefix}_residual_align")(out, training=training)

    # 8. Residual Addition
    add_cls = QAdd if quantize else keras.layers.Add
    return add_cls(name=f"{prefix}_residual_add")([out, residual])