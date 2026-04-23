import numpy as np
import h5py
import os
import time
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data PATH
DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")


def _read_h5_files(name="train", batch_size=5000):
    if name == "train":
        input_path = os.path.join(DATA_DIR, "train")
        output_path = os.path.join(DATA_DIR, "merged_train.h5")
        file_list = sorted(os.listdir(os.path.join(DATA_DIR, "train")))
    elif name == "test":
        input_path = os.path.join(DATA_DIR, "test")
        output_path = os.path.join(DATA_DIR, "merged_test.h5")
        file_list = sorted(os.listdir(os.path.join(DATA_DIR, "test")))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    start_time = time.time()

    total_samples = 0
    sample_shape = None
    target_shape = None
    for file in file_list:
        with h5py.File(os.path.join(input_path, file), "r") as data:
            jets = data["jetConstituentList"]
            targets = data["jets"][:, -6:-1]
            total_samples += jets.shape[0]
            if sample_shape is None:
                sample_shape = jets.shape[1:]
                target_shape = targets.shape[1:]

    with h5py.File(output_path, "w") as f_out:
        dset_X = f_out.create_dataset(
            "jetConstituentList",
            shape=(total_samples, *sample_shape),
            dtype=np.float32,
            compression="gzip",
            chunks=True,
        )
        dset_y = f_out.create_dataset(
            "jets",
            shape=(total_samples, *target_shape),
            dtype=np.float32,
            compression="gzip",
            chunks=True,
        )

        write_idx = 0

        with tqdm(file_list, desc="Merging h5 files") as t:
            for file in t:
                t.set_postfix(file=file)
                with h5py.File(os.path.join(input_path, file), "r") as data:
                    jets = data["jetConstituentList"][...]
                    targets = data["jets"][:, -6:-1]
                    num = jets.shape[0]
                    for start in range(0, num, batch_size):
                        end = min(start + batch_size, num)
                        dset_X[write_idx : write_idx + (end - start)] = jets[start:end]
                        dset_y[write_idx : write_idx + (end - start)] = targets[
                            start:end
                        ]
                        write_idx += end - start

        print("Final shape:", dset_X.shape, dset_y.shape)
        print(f"Saved merged result to {output_path}")
        print("Time taken:", time.time() - start_time, "s")


def _filter(name="train", batch_size=5000):
    if name == "train":
        input_path = os.path.join(DATA_DIR, "merged_train.h5")
        output_path = os.path.join(DATA_DIR, "filtered_train.h5")
    elif name == "test":
        input_path = os.path.join(DATA_DIR, "merged_test.h5")
        output_path = os.path.join(DATA_DIR, "filtered_test.h5")

    pt_min = 2.0
    pt_index = 5
    start_time = time.time()

    with h5py.File(input_path, "r") as fin:
        X = fin["jetConstituentList"]
        y = fin["jets"]
        n_samples, n_constit, n_feat = X.shape

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with h5py.File(output_path, "w") as fout:
            dset_X = fout.create_dataset(
                "jetConstituentList",
                shape=(0, n_constit, n_feat),
                maxshape=(None, n_constit, n_feat),
                dtype=np.float32,
                chunks=True,
                compression="lzf",
            )
            dset_y = fout.create_dataset(
                "jets",
                shape=(0, y.shape[1]),
                maxshape=(None, y.shape[1]),
                dtype=np.float32,
                chunks=True,
                compression="lzf",
            )

            write_idx = 0
            for start in tqdm(range(0, n_samples, batch_size), desc="Filtering"):
                end = min(start + batch_size, n_samples)
                batch_X = X[start:end]
                batch_y = y[start:end]

                filtered_X_list = []
                filtered_y_list = []

                for jet, label in zip(batch_X, batch_y):
                    valid = jet[jet[:, pt_index] >= pt_min]

                    if len(valid) == 0:
                        continue

                    pad = np.zeros(
                        (n_constit - valid.shape[0], n_feat), dtype=np.float32
                    )
                    padded = np.vstack([valid, pad])

                    filtered_X_list.append(padded)
                    filtered_y_list.append(label)

                if not filtered_X_list:
                    continue

                filtered_X_batch = np.stack(filtered_X_list)
                filtered_y_batch = np.stack(filtered_y_list)

                batch_len = filtered_X_batch.shape[0]
                dset_X.resize(write_idx + batch_len, axis=0)
                dset_y.resize(write_idx + batch_len, axis=0)

                dset_X[write_idx : write_idx + batch_len] = filtered_X_batch
                dset_y[write_idx : write_idx + batch_len] = filtered_y_batch

                write_idx += batch_len

            print(f"Saved filtered result to {output_path}")
            print(f"Filtered jets shape: {dset_X.shape}")
            print(f"Filtered targets shape: {dset_y.shape}")

    print("Time taken:", time.time() - start_time, "s")


def customize_dataset(num_particles, feats: list = [5, 8, 11], name="train"):
    assert num_particles <= 150, "num_particles should be less than or equal to 150"
    assert len(feats) <= 16, "feats should be less than or equal to 16"

    filtered_train_path = os.path.join(DATA_DIR, "filtered_train.h5")
    filtered_test_path = os.path.join(DATA_DIR, "filtered_test.h5")

    if not (os.path.exists(filtered_train_path) and os.path.exists(filtered_test_path)):
        _read_h5_files(name="train")
        _read_h5_files(name="test")
        _filter(name="train")
        _filter(name="test")

    if name == "train":
        input_path = os.path.join(DATA_DIR, "filtered_train.h5")
        output_path = os.path.join(
            PROCESSED_DIR, str(num_particles), f"{len(feats)}f", "train.h5"
        )
    elif name == "test":
        input_path = os.path.join(DATA_DIR, "filtered_test.h5")
        output_path = os.path.join(
            PROCESSED_DIR, str(num_particles), f"{len(feats)}f", "test.h5"
        )

    batch_size = 5000

    with h5py.File(input_path, "r") as fin:
        X = fin["jetConstituentList"]
        y = fin["jets"]
        n_samples, n_constit, n_feat = X.shape

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with h5py.File(output_path, "w") as fout:
            dset_X = fout.create_dataset(
                "jetConstituentList",
                shape=(0, num_particles, len(feats)),
                maxshape=(None, num_particles, len(feats)),
                dtype=np.float32,
                chunks=True,
                compression="lzf",
            )
            dset_y = fout.create_dataset(
                "jets",
                shape=(0, y.shape[1]),
                maxshape=(None, y.shape[1]),
                dtype=np.float32,
                chunks=True,
                compression="lzf",
            )

            write_idx = 0
            for start in tqdm(range(0, n_samples, batch_size), desc="Cropping"):
                end = min(start + batch_size, n_samples)

                batch_X = X[start:end, :num_particles, :][:, :, feats]
                batch_y = y[start:end]

                batch_len = batch_X.shape[0]

                dset_X.resize(write_idx + batch_len, axis=0)
                dset_y.resize(write_idx + batch_len, axis=0)

                dset_X[write_idx : write_idx + batch_len] = batch_X
                dset_y[write_idx : write_idx + batch_len] = batch_y

                write_idx += batch_len

            print(f"Saved customized result to {output_path}")


def compute_and_save_welford_stats(num_particles, num_feats, batch_size=5000):
    """
    Computes and serializes the global mean and standard deviation of the dataset
    using Welford's online algorithm to prevent memory overflow and catastrophic cancellation.
    """
    train_path = os.path.join(
        PROCESSED_DIR, str(num_particles), f"{num_feats}f", "train.h5"
    )
    save_dir = os.path.dirname(train_path)

    print(f"Computing Welford statistics for {train_path}...")
    start_time = time.time()

    n = 0
    mean = None
    M2 = None

    with h5py.File(train_path, "r") as fin:
        X = fin["jetConstituentList"]
        n_samples = X.shape[0]

        for start in tqdm(range(0, n_samples, batch_size), desc="Welford Stats"):
            end = min(start + batch_size, n_samples)

            # Extract and flatten batch: (B, P, F) -> (B*P, F)
            batch_X_flat = X[start:end].reshape(-1, num_feats)
            batch_n = batch_X_flat.shape[0]

            if batch_n == 0:
                continue

            # Compute local batch statistics
            batch_mean = np.mean(batch_X_flat, axis=0)
            batch_M2 = np.sum((batch_X_flat - batch_mean) ** 2, axis=0)

            # Initialize or merge global statistics
            if mean is None:
                mean = batch_mean
                M2 = batch_M2
                n = batch_n
            else:
                delta = batch_mean - mean
                total_n = n + batch_n

                mean = mean + delta * (batch_n / total_n)
                M2 = M2 + batch_M2 + (delta**2) * n * batch_n / total_n
                n = total_n

    # Compute unbiased sample standard deviation (Bessel's correction)
    std = np.sqrt(M2 / (n - 1 + 1e-8))

    # Serialize to disk
    mean_path = os.path.join(save_dir, "mean.npy")
    std_path = os.path.join(save_dir, "std.npy")

    np.save(mean_path, mean)
    np.save(std_path, std)

    print(f"Saved mean.npy and std.npy to {save_dir}")
    print("Time taken:", time.time() - start_time, "s")


if __name__ == "__main__":
    # The reference paper evaluates the 150-particle baseline using all 16 features.
    target_particles = 150
    target_feats = list(range(16))

    # 1. Structural Preprocessing
    customize_dataset(num_particles=target_particles, feats=target_feats, name="train")
    customize_dataset(num_particles=target_particles, feats=target_feats, name="test")

    # 2. Statistical Preprocessing (Executing solely on the training split to prevent data leakage)
    compute_and_save_welford_stats(
        num_particles=target_particles, num_feats=len(target_feats)
    )
