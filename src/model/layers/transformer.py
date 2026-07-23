import keras
from hgq.layers import (
    QAdd,
    QBatchNormalization,
    QMultiHeadAttention,
    QLinformerAttention,
    Quantizer,
)

# Assuming you have imported your new functional definitions:
# from .attention import apply_hgq_self_attention
from .ffn import apply_hgq_feed_forward


def apply_hgq_transformer_block(
    x,
    in_dim,
    latent_dim=None,
    num_heads=1,
    proj_dim_k=2,
    dropout=0.0,
    num_particles=30,
    activation="ReLU",
    normalization="Layer",
    momentum=0.9,
    quantize=True,
    use_linformer=True,
    block_name="transformer_block",
    training=False,
):
    latent_dim = latent_dim if latent_dim is not None else in_dim
    head_dim = latent_dim // num_heads

    # 1. Pre-Normalization (Replacing the internal norm of HGQSelfAttention)
    if not quantize:
        if normalization == "Batch":
            norm_x = keras.layers.BatchNormalization(
                axis=-1, momentum=momentum, epsilon=1e-5, name=f"{block_name}_attn_norm"
            )(x, training=training)
        else:
            norm_x = keras.layers.LayerNormalization(
                axis=-1, name=f"{block_name}_attn_norm"
            )(x)
    else:
        # [ABLATION] QBatchNormalization removed from quantized path.
        # norm_x = QBatchNormalization(
        #     axis=-1, momentum=momentum, epsilon=1e-5, name=f"{block_name}_attn_norm"
        # )(x, training=training)
        norm_x = x

    # 2. Library-Native Quantized Attention
    # QMultiHeadAttention automatically instantiates Q/K/V QDense layers,
    # handles the softmax, and manages internal Quantizer traces perfectly.
    #
    # MHA-specific quantizer scope (quantized path only):
    #   - overflow_mode='SAT': Prevents attention scores from wrapping around
    #     (WRAP mode would catastrophically corrupt softmax outputs).
    #   - round_mode='RND': Standard rounding for attention computations.
    #   - bc=MinMax(1, 8): Constrains attention bit-widths to [1, 8] bits,
    #     preventing the optimizer from pruning attention heads to 0-bits
    #     (which would reduce the Transformer to a trivial DeepSets model).
    if quantize:
        from hgq.config import QuantizerConfigScope
        from hgq.constraints import MinMax

        mha_scope = QuantizerConfigScope(
            k0=1, i0=1, f0=6,
            round_mode="RND",
            overflow_mode="SAT",
            bc=MinMax(1, 8),
        )
        with mha_scope:
            if use_linformer:
                attn_out = QLinformerAttention(
                    num_heads=num_heads,
                    key_dim=head_dim,
                    value_dim=head_dim,
                    proj_dim_k=proj_dim_k,
                    output_shape=in_dim,
                    name=f"{block_name}_linformer_attn",
                )(norm_x, norm_x, training=training)
            else:
                attn_out = QMultiHeadAttention(
                    num_heads=num_heads,
                    key_dim=head_dim,
                    value_dim=head_dim,
                    output_shape=in_dim,
                    name=f"{block_name}_q_attention",
                )(norm_x, norm_x, training=training)
    else:
        attn_out = keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=head_dim,
            value_dim=head_dim,
            output_shape=in_dim,
            name=f"{block_name}_q_attention",
        )(norm_x, norm_x, training=training)

    # if quantize:
    #     # Establishes strict fractional geometry for the tiny attention update without scaling variance
    #     attn_out = Quantizer(name=f"{block_name}_attn_fract_align")(attn_out)

    # 3. First Residual Connection (attn_out = self_attention(x) + x)
    add_cls = QAdd if quantize else keras.layers.Add

    res_x = add_cls(name=f"{block_name}_attn_residual")([attn_out, x])

    # 4. Feed-Forward
    ffn_out = apply_hgq_feed_forward(
        res_x,
        in_dim=in_dim,
        multiplication=2,
        activation=activation,
        normalization=normalization,
        momentum=momentum,
        quantize=quantize,
        prefix=f"{block_name}_ffn",
        training=training,
    )



    # 5. Second Residual Connection (out = ffn(attn_out) + attn_out)
    res_out = add_cls(name=f"{block_name}_ffn_residual")([ffn_out, res_x])

    # 6. Final Dropout
    return keras.layers.Dropout(dropout, name=f"{block_name}_dropout")(
        res_out, training=training
    )
