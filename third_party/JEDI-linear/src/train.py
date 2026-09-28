from pathlib import Path

import numpy as np

from matplotlib import pyplot as plt

from hgq.utils.sugar import BetaScheduler, ParetoFront, PBar, PieceWiseSchedule

from hgq.utils.sugar import FreeEBOPs, Dataset
import keras
import pickle as pkl


def train_hgq(model: keras.Model, X, Y, Xs, Ys, conf):
    save_path = Path(conf.save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    pred = model.predict(Xs, batch_size=2048, verbose=0)  # type: ignore
    hgq_acc_1 = np.mean(np.argmax(pred, axis=1) == np.array(Ys).ravel())
    print(f"pre-training HGQ accuracy: {hgq_acc_1:.2%}")

    with open(save_path / "pretrain_acc.txt", "w") as f:
        f.write(f"pre-training HGQ accuracy: {hgq_acc_1:.2%}\n")

    print("Compiling model & registering callbacks...")
    opt = keras.optimizers.Adam()
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics = ["accuracy"]
    model.compile(optimizer=opt, loss=loss, metrics=metrics)  # type: ignore

    assert conf.train.cdr_args["t_mul"] == 1
    first_decay_steps = conf.train.cdr_args["first_decay_steps"]
    initial_learning_rate = conf.train.cdr_args["initial_learning_rate"]
    t_mul = conf.train.cdr_args["t_mul"]
    m_mul = conf.train.cdr_args["m_mul"]
    alpha = conf.train.cdr_args["alpha"]
    alpha_steps = conf.train.cdr_args["alpha_steps"]

    def cosine_decay_restarts(global_step):
        from math import cos, pi

        n_cycle = 1
        cycle_step = global_step
        cycle_len = first_decay_steps
        while cycle_step >= cycle_len:
            cycle_step -= cycle_len
            cycle_len *= t_mul
            n_cycle += 1

        cycle_t = min(cycle_step / (cycle_len - alpha_steps), 1)
        lr = alpha + 0.5 * (initial_learning_rate - alpha) * (
            1 + cos(pi * cycle_t)
        ) * m_mul ** max(n_cycle - 1, 0)
        return lr

    scheduler = keras.callbacks.LearningRateScheduler(cosine_decay_restarts)

    pbar = PBar(
        metric="loss: {loss:.2f}/{val_loss:.2f} - acc: {accuracy:.2%}/{val_accuracy:.2%} - lr:{learning_rate:.2e} - beta: {beta:.2e}"
    )

    terminate_on_nan = keras.callbacks.TerminateOnNaN()

    save = ParetoFront(
        path=save_path / "ckpts",
        fname_format="epoch={epoch}-acc={accuracy:.2%}-val_acc={val_accuracy:.2%}-EBOPs={ebops}.keras",
        metrics=["val_accuracy", "ebops"],
        enable_if=lambda x: x["val_accuracy"] > 0.5,
        sides=[1, -1],
    )

    ebops = FreeEBOPs()
    beta_sched = BetaScheduler(PieceWiseSchedule(conf.beta.intervals))

    callbacks = [scheduler, beta_sched, ebops, save, pbar, terminate_on_nan]

    batch_size = conf.train.bsz

    val_split = 0.1
    val_size = int(len(X) * val_split)
    X, Y = X.astype(np.float16), Y.astype(np.int32)
    X_val, Y_val = X[:val_size], Y[:val_size]
    X_train, Y_train = X[val_size:], Y[val_size:]

    dataset_train = Dataset(
        X_train, Y_train, batch_size=batch_size, drop_last=True, device="gpu:0"
    )
    dataset_val = Dataset(
        X_val, Y_val, batch_size=batch_size, drop_last=True, device="gpu:0"
    )

    model.fit(
        dataset_train,
        epochs=conf.train.epochs,
        validation_data=dataset_val,
        callbacks=callbacks,
        verbose=0,
    )  # type: ignore
    history = model.history.history  # type: ignore
    with open(save_path / "history.pkl", "wb") as f:
        f.write(pkl.dumps(history))

    model.save(save_path / "last.h5")

    return model, history
