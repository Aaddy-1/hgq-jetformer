import keras
from hgq.layers import QAdd
# Assuming you have imported your new functional definitions:
from .attention import apply_hgq_self_attention
from .ffn import apply_hgq_feed_forward

def apply_hgq_transformer_block(
    x,
    in_dim,
    latent_dim=None,
    num_heads=1,
    dropout=0.0,
    num_particles=30,
    activation="ReLU",
    normalization="Layer",
    momentum=0.9,
    quantize=True,
    block_name="transformer_block",
    training=False
):
    """
    Constructs a statically traceable Transformer block on tensor 'x'.
    """
    # Step 1: Self-Attention (Using the functional builder)
    attn_out = apply_hgq_self_attention(
        x,
        in_dim=in_dim,
        latent_dim=latent_dim,
        num_heads=num_heads,
        num_particles=num_particles,
        normalization=normalization,
        momentum=momentum,
        quantize=quantize,
        prefix=f"{block_name}_attention",
        training=training
    )

    # Step 2: Feed-Forward (Using the functional builder)
    ffn_out = apply_hgq_feed_forward(
        attn_out,
        in_dim=in_dim,
        multiplication=2,
        activation=activation,
        normalization=normalization,
        momentum=momentum,
        quantize=quantize,
        prefix=f"{block_name}_ffn",
        training=training
    )

    # Step 3: Final Residual
    if quantize:
        x_res = QAdd(name=f"{block_name}_residual_add")([attn_out, ffn_out])
    else:
        x_res = keras.layers.Add(name=f"{block_name}_residual_add")([attn_out, ffn_out])

    # Step 4: Final Dropout
    return keras.layers.Dropout(dropout, name=f"{block_name}_dropout")(x_res, training=training)