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
        if isinstance(h5_path, (list, tuple)):
            self.h5_paths = list(h5_path)
        else:
            self.h5_paths = [h5_path]

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_feats = num_feats

        self.mean = np.load(os.path.join(stats_dir, "mean.npy"))
        self.std = np.load(os.path.join(stats_dir, "std.npy"))

        if self.num_feats is not None and self.num_feats < len(self.mean):
            self.mean = self.mean[: self.num_feats]
            self.std = self.std[: self.num_feats]

        # Inspect first file to detect keys and count total lengths across files
        self.file_lengths = []
        with h5py.File(self.h5_paths[0], "r") as f:
            if x_key not in f:
                x_key = "particle_features" if "particle_features" in f else list(f.keys())[0]
            if y_key not in f:
                y_key = "label" if "label" in f else list(f.keys())[1]

            self.x_key = x_key
            self.y_key = y_key

        total_length = 0
        for p in self.h5_paths:
            with h5py.File(p, "r") as f:
                l = f[self.x_key].shape[0]
                self.file_lengths.append(l)
                total_length += l

        self.cum_lengths = np.cumsum([0] + self.file_lengths)

        # Parity Fix: Allow external subsetting
        if indices is not None:
            self.indices = np.array(indices)
            self.length = len(self.indices)
        else:
            self.indices = np.arange(total_length)
            self.length = total_length

        self._h5_files = {}
        self.on_epoch_end()

    def _get_file(self, file_idx):
        # Thread-safe lazy initialization for Keras multiprocessing
        if file_idx not in self._h5_files or self._h5_files[file_idx] is None:
            self._h5_files[file_idx] = h5py.File(self.h5_paths[file_idx], "r")
        return self._h5_files[file_idx]

    def __len__(self):
        return int(np.ceil(self.length / self.batch_size))

    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        end_idx = min((idx + 1) * self.batch_size, self.length)
        batch_indices = self.indices[start_idx:end_idx]

        # Group batch indices by file index
        x_chunks = []
        y_chunks = []
        chunk_orders = []

        for f_idx in range(len(self.h5_paths)):
            f_start = self.cum_lengths[f_idx]
            f_end = self.cum_lengths[f_idx + 1]

            # Mask for indices falling into this file
            mask = (batch_indices >= f_start) & (batch_indices < f_end)
            if not np.any(mask):
                continue

            sub_indices = batch_indices[mask] - f_start
            sub_positions = np.where(mask)[0]

            f = self._get_file(f_idx)

            if self.shuffle:
                sorted_sub = np.sort(sub_indices)
                x_sub = f[self.x_key][sorted_sub]
                y_sub = f[self.y_key][sorted_sub]
                restore = np.argsort(np.argsort(sub_indices))
                x_sub = x_sub[restore]
                y_sub = y_sub[restore]
            else:
                x_sub = f[self.x_key][sub_indices]
                y_sub = f[self.y_key][sub_indices]

            x_chunks.append(x_sub)
            y_chunks.append(y_sub)
            chunk_orders.append(sub_positions)

        # Combine chunks back into original batch order
        if len(x_chunks) == 1:
            x_batch = x_chunks[0]
            y_batch = y_chunks[0]
        else:
            total_b = len(batch_indices)
            sample_x_shape = x_chunks[0].shape[1:]
            sample_y_shape = y_chunks[0].shape[1:]

            x_batch = np.empty((total_b, *sample_x_shape), dtype=x_chunks[0].dtype)
            y_batch = np.empty((total_b, *sample_y_shape), dtype=y_chunks[0].dtype)

            for x_sub, y_sub, pos in zip(x_chunks, y_chunks, chunk_orders):
                x_batch[pos] = x_sub
                y_batch[pos] = y_sub

        # On-the-fly feature slicing for feature ablation experiments
        if self.num_feats is not None and self.num_feats < x_batch.shape[-1]:
            x_batch = x_batch[:, :, : self.num_feats]

        # 1. Statistical Normalization (Z-score)
        x_batch = (x_batch - self.mean) / (self.std + 1e-8)

        # 2. Target Formulation (Convert one-hot to sparse categorical indices)
        if y_batch.ndim > 1 and y_batch.shape[-1] > 1:
            y_batch = np.argmax(y_batch, axis=-1)

        return x_batch.astype(np.float32), y_batch.astype(np.int64)

    def on_epoch_end(self):
        # Shuffle index map at the end of each epoch
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __del__(self):
        if hasattr(self, "_h5_files"):
            for f in self._h5_files.values():
                if f is not None:
                    try:
                        f.close()
                    except Exception:
                        pass
