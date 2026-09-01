"""Apply HGQ's zero-bit prune mask during training, behind a straight-through
estimator.

HGQ masks channels whose learnt fixed-point format holds no bits at all
(``k + i + f <= 0``) only when ``training`` is falsy -- see
``FixedPointQuantizerBase.call`` in
``hgq/quantizer/internal/fixed_point_quantizer.py:140-148`` of hgq 0.1.8::

    ret = self.stateless_quantizer(inputs, k, i, f, training is True, self.seed_gen)
    if not training:
        ret = ops.where(k + i + f > 0, ret, ops.zeros_like(ret))

So a dead channel is fully alive for every step of ``fit()`` and a hard zero the
moment the model is evaluated or compiled to RTL. Training and validation score
different functions, which is the measured cause of the train/val divergence
(.agents/DEMO_MASK_{ON,OFF}: bit-identical training curves through epoch 164,
validation diverging by up to +0.01805 over that same window).

WHY THE MASK CANNOT SIMPLY BE MOVED INTO TRAINING
-------------------------------------------------
``ops.where`` is a hard gate. Masking a channel during training drives its output
to a constant, so the data-path gradient ``d out/d f`` collapses to zero. The
EBOPs gradient is already zero there -- KIF's ``bits`` is ``ops.relu(i + f)``,
whose derivative vanishes once ``i + f <= 0``. That leaves only ``MonoL1``
(``l1 * ops.sum(x)``, a *signed* sum, l1=1e-8), a constant +1e-8 push downward
that never dies. ``{k+i+f <= 0}`` becomes absorbing: ``f`` falls forever until it
clamps at the library default MinMax(-24, 24), and a channel that dies can never
come back.

THE SURROGATE
-------------
Take the forward *value* of the masked tensor and the *derivative* of the
unmasked one::

    ops.stop_gradient(masked) + (ret - ops.stop_gradient(ret))

    forward     =  masked + ret - ret   =  masked
    derivative  =    0    + dret -  0   =  dret

This is HGQ's own idiom, used for every rounding mode at
``quantizers/fixed_point/_fixed_point_ops.py:26-28``::

    return ops.stop_gradient(xq) + (x - ops.stop_gradient(x))

The mask is the one non-differentiable op in the library left without it.

THREE PROPERTIES
----------------
1. No-op on live channels. Where ``k+i+f > 0``, ``masked`` IS ``ret``, so the
   expression collapses to ``ret`` and both the forward value and the gradient are
   bit-identical to stock HGQ. Only dead channels are touched.

2. Patched on the KBI and KIF subclasses, not the shared base. For
   WRAP + trainable quantizers both subclasses return early during training
   (``:275-305`` and ``:399-427``) and never reach the base ``call``, so patching
   the base alone would silently miss them at exactly the moment this must act.
   KBI is defensive only: under the scopes in train.py no KBI quantizer can reach
   ``k+i+f <= 0`` (weights reduce to ``k + b >= 1`` given bc=MinMax(0,23), k0=1),
   so today this fires solely on KIF datalanes. It costs nothing and survives a
   later config change.

3. ``training is not True`` admits real training only. The calibration sentinel
   ``TrainingFlagWrapper('tracing')`` (hgq/utils/minmax_trace.py:24-32) fails the
   identity test and falls through to the original method, so ``trace_minmax``
   behaves identically with the patch on or off and the two arms stay comparable.

The rebind is on the class, so it also takes effect on already-constructed
quantizers -- but apply it before the model is built anyway.
"""


def patch_prune_mask_with_ste():
    """Rebind FixedPointQuantizer{KBI,KIF}.call to mask dead channels during
    training with a straight-through estimator. Idempotent."""
    from keras import ops

    from hgq.quantizer.internal.fixed_point_quantizer import (
        FixedPointQuantizerKBI,
        FixedPointQuantizerKIF,
    )

    def wrap(cls):
        original = cls.call
        if getattr(original, "_ste_patched", False):
            return

        def call(self, inputs, training=None):
            ret = original(self, inputs, training)
            if training is not True:
                return ret  # inference masks itself; 'tracing' stays stock
            k, i, f = self.kif
            shape = ops.shape(inputs)
            k = self.bw_mapper.bw_to_x(k, shape)
            i = self.bw_mapper.bw_to_x(i, shape)
            f = self.bw_mapper.bw_to_x(f, shape)
            masked = ops.where(k + i + f > 0, ret, ops.zeros_like(ret))
            # Value of `masked`, derivative of `ret`. On a live channel the two
            # are the same tensor and this is exactly a no-op.
            return ops.stop_gradient(masked) + (ret - ops.stop_gradient(ret))

        call._ste_patched = True
        cls.call = call

    wrap(FixedPointQuantizerKBI)
    wrap(FixedPointQuantizerKIF)
