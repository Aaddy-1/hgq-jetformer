import numpy as np
import h5py
import os
import keras


class JetFormerDataGenerator(keras.utils.PyDataset):
    """
    Keras 3 PyDataset for batched HDF5 streaming.
    Handles lazy loading, dynamic shuffling, and on-the-fly normalization
    using offline Welford statistics.
    """

    def __init__(
        self,
        h5_path,
        stats_dir,
        batch_size=256,
        shuffle=True,
        indices=None,
        x_key="jetConstituentList",
        y_key="jets",
        num_feats=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.h5_path = h5_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_feats = num_feats

        self.mean = np.load(os.path.join(stats_dir, "mean.npy"))
        self.std = np.load(os.path.join(stats_dir, "std.npy"))

        if self.num_feats is not None and self.num_feats < len(self.mean):
            self.mean = self.mean[: self.num_feats]
            self.std = self.std[: self.num_feats]

        with h5py.File(self.h5_path, "r") as f:
            # Auto-detect HDF5 keys for cross-dataset compatibility
            if x_key not in f:
                x_key = "particle_features" if "particle_features" in f else list(f.keys())[0]
            if y_key not in f:
                y_key = "label" if "label" in f else list(f.keys())[1]

            self.x_key = x_key
            self.y_key = y_key
            total_length = f[self.x_key].shape[0]

        # Parity Fix: Allow external subsetting
        if indices is not None:
            self.indices = np.array(indices)
            self.length = len(self.indices)
        else:
            self.indices = np.arange(total_length)
            self.length = total_length

        self._h5_file = None
        self.on_epoch_end()

    def _get_file(self):
        # Thread-safe lazy initialization for Keras multiprocessing
        if self._h5_file is None:
            self._h5_file = h5py.File(self.h5_path, "r")
        return self._h5_file

    def __len__(self):
        return int(np.ceil(self.length / self.batch_size))

    def __getitem__(self, idx):
        f = self._get_file()

        start_idx = idx * self.batch_size
        end_idx = min((idx + 1) * self.batch_size, self.length)
        batch_indices = self.indices[start_idx:end_idx]

        if self.shuffle:
            # h5py requires monotonically increasing indices for multi-index selection
            sorted_indices = np.sort(batch_indices)
            x_batch = f[self.x_key][sorted_indices]
            y_batch = f[self.y_key][sorted_indices]

            # Revert to the randomized order
            restore_order = np.argsort(np.argsort(batch_indices))
            x_batch = x_batch[restore_order]
            y_batch = y_batch[restore_order]
        else:
            # Contiguous slice (faster I/O)
            x_batch = f[self.x_key][start_idx:end_idx]
            y_batch = f[self.y_key][start_idx:end_idx]

        # On-the-fly feature slicing for feature ablation experiments
        if self.num_feats is not None and self.num_feats < x_batch.shape[-1]:
            x_batch = x_batch[:, :, : self.num_feats]

        # 1. Statistical Normalization (Z-score)
        x_batch = (x_batch - self.mean) / (self.std + 1e-8)

        # 2. Target Formulation (Convert one-hot to sparse categorical indices)
        # Handle both one-hot encoded labels (HLS4ML) and integer labels (JetClass)
        if y_batch.ndim > 1 and y_batch.shape[-1] > 1:
            y_batch = np.argmax(y_batch, axis=-1)

        return x_batch.astype(np.float32), y_batch.astype(np.int64)

    def on_epoch_end(self):
        # Shuffle index map at the end of each epoch
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __del__(self):
        if self._h5_file is not None:
            try:
                self._h5_file.close()
            except Exception:
                pass
