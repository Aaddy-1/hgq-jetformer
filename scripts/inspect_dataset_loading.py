import os
import h5py
import numpy as np

# Find processed JetClass dataset path
processed_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed")
jetclass_path = os.path.join(processed_dir, "jetclass", "128", "17f", "train_part0.h5")

if not os.path.exists(jetclass_path):
    print(f"File not found at: {jetclass_path}")
    # Search for any h5 file in processed_dir
    h5_files = []
    for root, dirs, files in os.walk(processed_dir):
        for f in files:
            if f.endswith(".h5") and "train" in f:
                h5_files.append(os.path.join(root, f))
    if h5_files:
        jetclass_path = h5_files[0]
        print(f"Using alternative found file: {jetclass_path}")
    else:
        print("No HDF5 files found in data/processed/")
        exit(1)

print(f"=== Inspecting HDF5 File: {jetclass_path} ===")
with h5py.File(jetclass_path, "r") as f:
    keys = list(f.keys())
    print(f"Keys in HDF5: {keys}")
    
    x_key = "particle_features" if "particle_features" in f else keys[0]
    y_key = "jets" if "jets" in f else ("label" if "label" in f else keys[1])
    
    print(f"Using x_key: '{x_key}', y_key: '{y_key}'")
    print(f"x shape: {f[x_key].shape}, dtype: {f[x_key].dtype}")
    print(f"y shape: {f[y_key].shape}, dtype: {f[y_key].dtype}")
    
    # Load first 100,000 label entries for quick verification
    slice_n = min(100000, f[y_key].shape[0])
    y_slice = f[y_key][:slice_n]
    
    print("\n--- TEST 1: Direct np.unique(y_slice) on raw label array ---")
    direct_unique = np.unique(y_slice)
    print(f"np.unique(y_slice) result: {direct_unique}")
    print(f"Number of unique classes detected: {len(direct_unique)}")
    
    print("\n--- TEST 2: np.unique(np.argmax(y_slice, axis=-1)) ---")
    if y_slice.ndim > 1:
        argmax_labels = np.argmax(y_slice, axis=-1)
        argmax_unique = np.unique(argmax_labels)
        print(f"np.unique(np.argmax(y_slice, axis=-1)) result: {argmax_unique}")
        print(f"Number of unique classes detected: {len(argmax_unique)}")
        
        counts = {cls: np.sum(argmax_labels == cls) for cls in argmax_unique}
        print(f"Per-class counts in first {slice_n:,} samples: {counts}")
    else:
        print("y_slice is 1D integer labels already.")

    print("\n--- TEST 3: Inspecting np.where(y_slice == 0) vs np.where(argmax == 0) ---")
    if y_slice.ndim > 1:
        raw_where_0 = np.where(y_slice == 0)[0]
        argmax_where_0 = np.where(argmax_labels == 0)[0]
        print(f"First 10 indices where raw y_slice == 0: {raw_where_0[:10]}")
        print(f"First 10 indices where argmax(y_slice) == 0: {argmax_where_0[:10]}")
        print(f"Total matches for raw y_slice == 0: {len(raw_where_0):,}")
        print(f"Total matches for argmax == 0: {len(argmax_where_0):,}")
