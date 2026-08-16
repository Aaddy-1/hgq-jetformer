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
    floor_attn_datalane=False,
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
    #
    #     CAVEAT: bc only reaches KBI-typed quantizers -- weights, biases, tables.
    #     Datalane (activation) quantizers are KIF-typed and have no `b`, so bc is
    #     silently ignored there and ic/fc fall through to the library defaults
    #     MinMax(-23, 23) / MinMax(-24, 24). Measured on EXP-25_FULLCURVE_SEED42:
    #     every bc-constrained quantizer is 0.0% pruned, while 63.3% of
    #     q_einsum_dense_iq's channels sit at k+i+f <= 0. The DeepSets collapse this
    #     scope was written to prevent has been happening through the activations.
    #     See floor_attn_datalane below and knowledge_base.md Part I 4.2 / 5.8.
    if quantize:
        from hgq.config import QuantizerConfigScope
        from hgq.constraints import MinMax

        import contextlib

        mha_scope = QuantizerConfigScope(
            k0=1, i0=1, f0=6,
            round_mode="RND",
            overflow_mode="SAT",
            bc=MinMax(1, 8),
        )

        # The prune test is on the SUM k+i+f, and hgq.constraints offers no sum
        # constraint, so the floors must be chosen such that their minima sum above
        # zero. fc>=1 alone still leaves k+(-23)+1 = -22; ic>=0 with fc>=0 leaves k,
        # which is 0 whenever k=0 -- and k is data-driven (get_any_k), so post-ReLU
        # non-negative channels can reach it. ic>=0 AND fc>=1 gives a minimum of 1
        # regardless of k. Ceilings stay at the library defaults so the A/B against
        # floor_attn_datalane=False moves exactly one variable.
        #
        # place="datalane" is REQUIRED, not tidiness. KBIConfig also has an `ic`
        # field, so an unplaced scope silently applies this floor to the weight
        # quantizers too -- and since KBI derives f = b - i, forcing i upward to 0
        # *removes* fractional bits from weights. Verified: without the place
        # argument, query_kq goes from ic=None to ic=(0,23).
        scopes = contextlib.ExitStack()
        scopes.enter_context(mha_scope)
        if floor_attn_datalane:
            scopes.enter_context(
                QuantizerConfigScope(
                    place="datalane", ic=MinMax(0, 23), fc=MinMax(1, 24)
                )
            )

        with scopes:
            if use_linformer:
                attn_out = QLinformerAttention(
                    num_heads=num_heads,
                    key_dim=head_dim,
                    value_dim=head_dim,
                    lin_kv_proj_dim=proj_dim_k,
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
