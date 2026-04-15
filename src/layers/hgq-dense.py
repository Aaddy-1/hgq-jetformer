import keras
from hgq.layers import QDense


def get_dense(units, quantize=True, name=None, activation=None, **kwargs):
    """
    A factory function to return either an HGQ-quantized or a standard Dense layer.

    Args:
        units (int): Dimensionality of the output space.
        quantize (bool): If True, returns an HGQ QDense layer. If False, returns standard Keras Dense.
        name (str): Name for the layer.
        activation (str): Activation function to use.
        **kwargs: Additional arguments passed to the underlying layer.
    """
    if quantize:
        # Returns the library's QDense.
        # Note: Its behavior is governed by the ConfigScope in your model file.
        return QDense(units=units, activation=activation, name=name, **kwargs)
    else:
        # Returns standard Keras Dense for your 'vanilla' baseline.
        return keras.layers.Dense(
            units=units, activation=activation, name=name, **kwargs
        )
