import numpy as np
import h5py
import os
import keras


def detect_hardware_and_strategy(num_samples, num_particles=128, num_feats=17, ram_safety_ratio=0.50):
    """
    Auto-detects active GPUs and available system RAM.
    Determines whether to use FULL_RAM, CHUNK_BUFFER, or BLOCK_DISK_STREAM strategy.
    """
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    num_gpus = len(gpus)
    gpu_names = [g.name for g in gpus]

    try:
        import psutil
        avail_bytes = psutil.virtual_memory().available
    except ImportError:
        try:
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if "MemAvailable:" in line:
                        avail_bytes = int(line.split()[1]) * 1024
                        break
        except Exception:
            avail_bytes = 16 * (1024 ** 3)

    ram_budget_bytes = int(avail_bytes * ram_safety_ratio)
    sample_bytes = num_particles * num_feats * 4  # float32
    required_bytes = num_samples * sample_bytes

    avail_gb = avail_bytes / (1024 ** 3)
    budget_gb = ram_budget_bytes / (1024 ** 3)
    req_gb = required_bytes / (1024 ** 3)

    print("================ HARDWARE AUTO-DETECTION ================")
    print(f"[Hardware] GPUs Detected: {num_gpus} ({', '.join(gpu_names) if gpu_names else 'CPU/Default'})")
    print(f"[Hardware] System Available RAM: {avail_gb:.2f} GB | Safety Budget ({int(ram_safety_ratio*100)}%): {budget_gb:.2f} GB")
    print(f"[Hardware] Dataset Memory Required: {req_gb:.2f} GB ({num_samples} samples)")

    if ram_budget_bytes >= required_bytes:
        strategy = "FULL_RAM"
        print("[Hardware] Strategy Selected: FULL_RAM (Full dataset fits safely in RAM budget)")
    else:
        strategy = "BLOCK_DISK_STREAM"
        print(f"[Hardware] Strategy Selected: BLOCK_DISK_STREAM (Low RAM footprint < 10 MB)")
    print("=========================================================")

    return strategy


def get_stratified_indices(y_labels: np.ndarray, max_samples: int, seed: int = 42) -> np.ndarray:
    """
    Computes exact stratified indices across all unique classes in y_labels.
    Ensures every class receives an identical quota of max_samples // n_classes.
    """
    if y_labels.ndim > 1 and y_labels.shape[-1] > 1:
        y_labels = np.argmax(y_labels, axis=-1)

    rng = np.random.default_rng(seed)
    unique_classes = np.unique(y_labels)
    n_classes = len(unique_classes)
    per_class_quota = max_samples // n_classes

    print(
        f"[Stratified] Sampling exactly {per_class_quota:,} items per class "
        f"across {n_classes} classes ({per_class_quota * n_classes:,} total samples)..."
    )

    stratified_indices = []
    for cls in unique_classes:
        cls_indices = np.where(y_labels == cls)[0]
        quota = min(per_class_quota, len(cls_indices))
        selected = rng.choice(cls_indices, size=quota, replace=False)
        stratified_indices.append(selected)

    result = np.concatenate(stratified_indices)
    rng.shuffle(result)
    return result


class JetFormerDataGenerator(keras.utils.PyDataset):
    """
    Keras 3 PyDataset for batched HDF5 streaming and high-performance RAM caching.
    Handles in-memory RAM caching, contiguous block HDF5 streaming, dynamic shuffling,
    and on-the-fly normalization using offline Welford statistics.
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
        in_memory=False,
        preloaded_data=None,
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
        self.in_memory = in_memory

        self.mean = np.load(os.path.join(stats_dir, "mean.npy"))
        self.std = np.load(os.path.join(stats_dir, "std.npy"))

        if self.num_feats is not None and self.num_feats < len(self.mean):
            self.mean = self.mean[: self.num_feats]
            self.std = self.std[: self.num_feats]

        self.x_data = None
        self.y_data = None

        if preloaded_data is not None:
            # Use pre-loaded numpy arrays directly (shared single-read path)
            self._init_from_preloaded(preloaded_data)
            return

        # HDF5 file-based initialization (detect keys and file lengths)
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

        if indices is not None:
            self.indices = np.array(indices)
            self.length = len(self.indices)
        else:
            self.indices = np.arange(total_length)
            self.length = total_length

        if self.in_memory:
            self._preload_into_ram()
        else:
            self._h5_files = {}
            self.num_blocks = int(np.ceil(self.length / self.batch_size))
            self.block_order = np.arange(self.num_blocks)
            self.on_epoch_end()

    def _init_from_preloaded(self, preloaded_data):
        """Initialize from pre-loaded numpy arrays. Applies normalization and label conversion."""
        raw_x, raw_y = preloaded_data

        if self.num_feats is not None and self.num_feats < raw_x.shape[-1]:
            raw_x = raw_x[:, :, : self.num_feats]

        # Z-score normalization
        raw_x = (raw_x - self.mean) / (self.std + 1e-8)

        # Convert one-hot to sparse categorical indices
        if raw_y.ndim > 1 and raw_y.shape[-1] > 1:
            raw_y = np.argmax(raw_y, axis=-1)

        self.x_data = raw_x.astype(np.float32)
        self.y_data = raw_y.astype(np.int64)
        self.in_memory = True
        self.indices = np.arange(len(self.x_data))
        self.length = len(self.x_data)
        if self.shuffle:
            np.random.shuffle(self.indices)
        print(f"[JetFormerDataGenerator] Preloaded data ready. Shape: {self.x_data.shape}")

    def _preload_into_ram(self):
        max_needed = int(np.max(self.indices)) + 1 if self.indices is not None and len(self.indices) > 0 else None
        print(f"[JetFormerDataGenerator] Pre-loading {self.length} samples (reading up to index {max_needed if max_needed else 'end'}) into RAM...")
        x_list = []
        y_list = []
        loaded_so_far = 0

        for p in self.h5_paths:
            with h5py.File(p, "r") as f:
                file_len = f[self.x_key].shape[0]
                if max_needed is not None and max_needed <= loaded_so_far:
                    break
                slice_end = min(file_len, max_needed - loaded_so_far) if max_needed is not None else file_len
                x_list.append(f[self.x_key][:slice_end])
                y_list.append(f[self.y_key][:slice_end])
                loaded_so_far += slice_end

        all_x = np.concatenate(x_list, axis=0) if len(x_list) > 1 else x_list[0]
        all_y = np.concatenate(y_list, axis=0) if len(y_list) > 1 else y_list[0]

        if self.indices is not None:
            all_x = all_x[self.indices]
            all_y = all_y[self.indices]

        if self.num_feats is not None and self.num_feats < all_x.shape[-1]:
            all_x = all_x[:, :, : self.num_feats]

        all_x = (all_x - self.mean) / (self.std + 1e-8)

        if all_y.ndim > 1 and all_y.shape[-1] > 1:
            all_y = np.argmax(all_y, axis=-1)

        self.x_data = all_x.astype(np.float32)
        self.y_data = all_y.astype(np.int64)
        self.indices = np.arange(len(self.x_data))
        self.length = len(self.x_data)
        if self.shuffle:
            np.random.shuffle(self.indices)
        print(f"[JetFormerDataGenerator] Pre-load complete. Shape: {self.x_data.shape}")

    def _get_file(self, file_idx):
        if file_idx not in self._h5_files or self._h5_files[file_idx] is None:
            self._h5_files[file_idx] = h5py.File(self.h5_paths[file_idx], "r")
        return self._h5_files[file_idx]

    def __len__(self):
        return int(np.ceil(self.length / self.batch_size))

    def __getitem__(self, idx):
        if self.in_memory:
            start_idx = idx * self.batch_size
            end_idx = min((idx + 1) * self.batch_size, self.length)
            batch_indices = self.indices[start_idx:end_idx]
            return self.x_data[batch_indices], self.y_data[batch_indices]

        # Low-RAM Contiguous Block Disk Streaming (< 10 MB RAM)
        b_idx = self.block_order[idx]
        start_idx = b_idx * self.batch_size
        end_idx = min((b_idx + 1) * self.batch_size, self.length)
        batch_indices = self.indices[start_idx:end_idx]

        x_chunks = []
        y_chunks = []
        chunk_orders = []

        for f_idx in range(len(self.h5_paths)):
            f_start = self.cum_lengths[f_idx]
            f_end = self.cum_lengths[f_idx + 1]

            mask = (batch_indices >= f_start) & (batch_indices < f_end)
            if not np.any(mask):
                continue

            sub_indices = batch_indices[mask] - f_start
            sub_positions = np.where(mask)[0]

            f = self._get_file(f_idx)
            min_i, max_i = np.min(sub_indices), np.max(sub_indices)

            # Efficient slice vs fancy indexing read from HDF5
            if (max_i - min_i + 1) > len(sub_indices) * 4:
                sort_idx = np.argsort(sub_indices)
                sorted_sub = sub_indices[sort_idx]
                inv_sort = np.argsort(sort_idx)
                x_sub = f[self.x_key][sorted_sub][inv_sort]
                y_sub = f[self.y_key][sorted_sub][inv_sort]
            else:
                rel_indices = sub_indices - min_i
                x_sub = f[self.x_key][min_i : max_i + 1][rel_indices]
                y_sub = f[self.y_key][min_i : max_i + 1][rel_indices]

            x_chunks.append(x_sub)
            y_chunks.append(y_sub)
            chunk_orders.append(sub_positions)

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

        if self.shuffle and len(x_batch) > 1:
            perm = np.random.permutation(len(x_batch))
            x_batch = x_batch[perm]
            y_batch = y_batch[perm]

        if self.num_feats is not None and self.num_feats < x_batch.shape[-1]:
            x_batch = x_batch[:, :, : self.num_feats]

        x_batch = (x_batch - self.mean) / (self.std + 1e-8)

        if y_batch.ndim > 1 and y_batch.shape[-1] > 1:
            y_batch = np.argmax(y_batch, axis=-1)

        return x_batch.astype(np.float32), y_batch.astype(np.int64)

    def on_epoch_end(self):
        if self.shuffle:
            if self.in_memory:
                np.random.shuffle(self.indices)
            elif hasattr(self, "block_order"):
                np.random.shuffle(self.block_order)

    def __del__(self):
        if hasattr(self, "_h5_files"):
            for f in self._h5_files.values():
                if f is not None:
                    try:
                        f.close()
                    except Exception:
                        pass
