from math import log2

import keras
from hgq.config import LayerConfigScope, QuantizerConfig, QuantizerConfigScope
from hgq.constraints import MinMax
from hgq.layers import QAdd, QEinsumDense, QEinsumDenseBatchnorm, QSum
from hgq.regularizers import MonoL1

if keras.backend.backend() == 'jax':
    import jax

    jax.config.update('jax_default_matmul_precision', 'tensorfloat32')


def get_mlpm(conf, uq1: bool = False):
    iq_conf = QuantizerConfig(place='datalane', round_mode='RND')
    iq_default = QuantizerConfig(place='datalane')

    N = conf.n_constituents
    n = 3 if conf.pt_eta_phi else 16
    heterogeneous_axis = None if not uq1 else (-1,)

    with QuantizerConfigScope(place='datalane', heterogeneous_axis=heterogeneous_axis):
        inp = keras.layers.Input((N, n))

        x1 = QEinsumDenseBatchnorm(
            'bnc,cC->bnC', (N, 16), bias_axes='C', activation='relu', iq_conf=iq_conf
        )(inp)
        x1 = QEinsumDenseBatchnorm(
            'bnc,cC->bnC', (N, n), bias_axes='C', activation='relu'
        )(x1)
        x2 = QEinsumDenseBatchnorm('bnc,nN->bNc', (N, n), bias_axes='N')(x1)
        x = QAdd(iq_confs=(iq_conf, iq_default))([inp, x2])
        x = QEinsumDenseBatchnorm(
            'bnc,cC->bnC', (N, 16), bias_axes='C', activation='relu'
        )(x)

        x = QEinsumDenseBatchnorm(
            'bnc,cC->bnC', (N, 16), bias_axes='C', activation='relu'
        )(x)
        x = QEinsumDense('bnc,n->bc', 16)(x)

        x = QEinsumDenseBatchnorm('bc,cC->bC', 16, bias_axes='C', activation='relu')(x)
        x = QEinsumDenseBatchnorm('bc,cC->bC', 16, bias_axes='C', activation='relu')(x)
        x = QEinsumDenseBatchnorm('bc,cC->bC', 16, bias_axes='C', activation='relu')(x)
        out = QEinsumDenseBatchnorm('bc,cC->bC', 5, bias_axes='C')(x)

    model = keras.Model(inputs=inp, outputs=out)
    return model


def get_mlp(conf):
    iq_conf = QuantizerConfig(place='datalane', round_mode='RND')

    N = conf.n_constituents
    n = 3 if conf.pt_eta_phi else 16

    inp = keras.layers.Input((N, n))

    x = keras.layers.Flatten()(inp)
    x = QEinsumDenseBatchnorm(
        'bc->bC', 64, bias_axes='C', activation='relu', iq_conf=iq_conf
    )(x)
    x = QEinsumDenseBatchnorm('bc->bC', 64, bias_axes='C', activation='relu')(x)
    x = QEinsumDenseBatchnorm('bc->bC', 64, bias_axes='C', activation='relu')(x)
    x = QEinsumDenseBatchnorm('bc->bC', 64, bias_axes='C', activation='relu')(x)
    out = QEinsumDenseBatchnorm('bc->bC', 5, bias_axes='C')(x)
    model = keras.Model(inputs=inp, outputs=out)
    return model


def get_gnn(conf, uq1: bool = False):
    N = conf.n_constituents
    n = 3 if conf.pt_eta_phi else 16
    heterogeneous_axis = None if not uq1 else (-1,)

    with (
        QuantizerConfigScope(place=('weight', 'bias'), overflow_mode='SAT_SYM'),
        QuantizerConfigScope(place='datalane', heterogeneous_axis=heterogeneous_axis),
    ):
        inp = keras.layers.Input((N, n))

        pool_scale = 2.0 ** -round(log2(N))
        x = QEinsumDenseBatchnorm(
            'bnc,cC->bnC', (N, 64), bias_axes='C', activation='relu'
        )(inp)
        s = QEinsumDenseBatchnorm(
            'bnc,cC->bnC',
            (N, 64),
            bias_axes='C',
            activation='relu',
        )(x)
        d = QEinsumDenseBatchnorm(
            'bnc,cC->bnC', (1, 64), bias_axes='C', activation='relu'
        )(QSum(axes=1, scale=pool_scale, keepdims=True)(x))
        x = QAdd()([s, d])

        x = QEinsumDenseBatchnorm(
            'bnc,cC->bnC',
            (N, 64),
            bias_axes='C',
            activation='relu',
        )(x)
        x = QSum(axes=1, scale=1 / 16, keepdims=False)(x)
        x = QEinsumDenseBatchnorm('bc,cC->bC', 64, bias_axes='C', activation='relu')(x)
        x = QEinsumDenseBatchnorm('bc,cC->bC', 32, bias_axes='C', activation='relu')(x)
        x = QEinsumDenseBatchnorm('bc,cC->bC', 16, bias_axes='C', activation='relu')(x)
        out = QEinsumDenseBatchnorm('bc,cC->bC', 5, bias_axes='C')(x)

    model = keras.Model(inputs=inp, outputs=out)
    return model


def get_model(conf):
    scope0 = QuantizerConfigScope(
        default_q_type='kbi',
        b0=conf.model.init_bw_k,
        overflow_mode='wrap',
        i0=0,
        fr=MonoL1(conf.model.k_bw_l1_reg),
        ir=MonoL1(conf.model.k_bw_l1_reg),
    )
    scope1 = QuantizerConfigScope(
        default_q_type='kif',
        place='datalane',
        overflow_mode='wrap',
        f0=conf.model.init_bw_a,
        fr=MonoL1(conf.model.a_bw_l1_reg),
        ic=MinMax(0, 12),
    )
    scope2 = LayerConfigScope(beta0=0)
    with scope0, scope1, scope2:
        match conf.model_class:
            case 'mlp_mixer':
                model = get_mlpm(conf)
            case 'mlp_mixer_uq1':  # Uniform quantization on ax1 (particle ax)
                model = get_mlpm(conf, uq1=True)
            case 'mlp':
                model = get_mlp(conf)
            case 'gnn':
                model = get_gnn(conf)
            case 'gnn_uq1':
                model = get_gnn(conf, uq1=True)
            case _:
                raise ValueError(f'Unknown model class: {conf.model_class}')

        return model
