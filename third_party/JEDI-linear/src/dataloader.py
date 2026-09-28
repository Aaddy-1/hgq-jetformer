import os
import pickle as pkl
from pathlib import Path

import numpy as np

from keras import ops


def get_data(data_path: Path, n_constituents: int, ptetaphi: bool):
    import h5py as h5

    with h5.File(data_path / "150c-train.h5") as f:
        X_train_val = np.array(f["feature"][:, :n_constituents])  # type: ignore
        y_train_val = np.array(f["label"])
    with h5.File(data_path / "150c-test.h5") as f:
        X_test = np.array(f["feature"][:, :n_constituents])  # type: ignore
        y_test = np.array(f["label"])
    labels = "gqWZt"

    X_train_val = X_train_val.astype(np.float32)
    y_train_val = y_train_val.astype(np.int32)
    X_test = X_test.astype(np.float32)
    y_test = y_test.astype(np.int32)

    scale = np.std(X_train_val, axis=(0, 1), keepdims=True)
    shift = np.mean(X_train_val, axis=(0, 1), keepdims=True)
    X_train_val = (X_train_val - shift) / scale  # type: ignore
    X_test = (X_test - shift) / scale  # type: ignore

    order = np.arange(len(X_train_val))
    np.random.shuffle(order)
    print(len(X_train_val))
    X_train_val, y_train_val = X_train_val[order], y_train_val[order]
    if ptetaphi:
        X_train_val = X_train_val[..., [5, 8, 11]]
        X_test = X_test[..., [5, 8, 11]]
    return X_train_val, X_test, y_train_val, y_test
