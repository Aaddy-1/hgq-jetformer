from pathlib import Path
import json

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from hgq.utils import trace_minmax
import pickle as pkl


def plot_history(histry: dict, metrics=("loss", "val_loss"), ylabel="Loss", logy=False):
    fig, ax = plt.subplots()
    for metric in metrics:
        ax.plot(histry[metric], label=metric)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale("log")
    ax.legend()
    return fig, ax


def test(model, save_path: Path, X_train, X, Y):
    results = {}
    Y = np.array(Y)
    pbar = tqdm(list(save_path.glob("ckpts/*.keras")))
    (save_path / "models").mkdir(exist_ok=True)
    X_train = np.array(X_train, np.float32)
    for ckpt in pbar:
        model.load_weights(ckpt)

        trace_minmax(model, X_train, batch_size=16384)
        pred = model.predict(X, batch_size=16384, verbose=0)

        acc = np.mean(np.argmax(pred, axis=1) == Y.ravel())
        ebops = sum(
            float(layer.ebops) for layer in model.layers if hasattr(layer, "ebops")
        )

        # print(f'Test accuracy: {acc:.5%} @ {mul_bops} BOPs')
        results[ckpt.name] = {"acc": acc, "ebops": ebops}
        model.save(save_path / "models" / f"{ckpt.stem}.keras")
        pbar.set_description(f"Test accuracy: {acc:.5%} @ {ebops:.0f} EBOPs")
    with open(save_path / "test_acc.json", "w") as f:
        json.dump(results, f)
