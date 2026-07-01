import keras

def get_parity_initializer():
    """
    Returns the standard parity initializer used throughout the HGQ model.
    Instantiating it per-layer prevents graph generation issues in Keras/Alkaid.
    """
    return keras.initializers.VarianceScaling(
        scale=1 / 3, mode="fan_in", distribution="uniform"
    )
