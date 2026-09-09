# SCRAMJet

A quantization-aware transformer for jet tagging in the CERN LHC Level-1 Trigger. The model is
trained with HGQ2 high-granularity quantization-aware training, in which every parameter learns
its own fixed-point precision, and compiled to register-transfer level hardware for a Xilinx
Virtex UltraScale+ FPGA.

This repository accompanies the MSc thesis *Quantization-Aware Transformers for Jet Tagging in
the LHC Level-1 Trigger* (Imperial College London, 2026). The tag `v1.0-thesis` marks the state
of the code at submission.

## Installation

The project targets Python 3.12. Create the conda environment (named `tf_keras`):

```bash
conda env create -f environment.yml
conda activate tf_keras
```

Or install into an existing environment with pip:

```bash
pip install -r requirements.txt
```

The main dependencies are Keras 3.14 and TensorFlow 2.19 for the model, `hgq2` 0.1.9 for
quantization-aware training, and `alkaid` for RTL generation. Alkaid runs on JAX and is only
needed for the hardware compilation step.

## Project structure

```
.
├── src/
│   ├── model/
│   │   ├── jetformer.py               # Functional model builder (quantized and float paths)
│   │   ├── initializers.py            # Per-layer parity initializer
│   │   └── layers/
│   │       ├── embedding.py           # Constituent embedding
│   │       ├── transformer.py         # Linformer / multi-head attention block
│   │       └── ffn.py                 # Position-wise feed-forward
│   ├── training/
│   │   ├── train.py                   # Main entry point: training + post-training evaluation
│   │   ├── evaluate.py                # Standalone evaluation of a saved checkpoint
│   │   ├── callbacks.py               # EBOPs-gated checkpointing, BetaPID, adaptive LR
│   │   └── onecyclelr.py              # One-cycle learning-rate schedule
│   ├── data/
│   │   ├── dataset.py                 # PyDataset loader and SO(2) augmentation
│   │   ├── download_jetclass.py       # Fetch the raw JetClass ROOT files
│   │   ├── build_jetclass_dataset.py  # Convert JetClass ROOT to HDF5
│   │   ├── jetclass_dataset.py        # JetClass feature construction
│   │   └── build_dataset.py           # HDF5 builder for the hls4ml dataset
│   └── hardware/
│       ├── compile_hls.py             # Native graph rebuild and Alkaid RTL generation
│       └── list_alkaid_plugins.py     # Inspect the available Alkaid backends
├── scripts/
│   ├── analyze_ebops.py               # Per-layer EBOPs breakdown and plot
│   ├── particle_multiplicity.py       # Constituent-count distributions
│   ├── derive_cropped_dataset.py      # Slice an N-particle tree out of the 128-particle one
│   ├── diagnose_train_val_gap.py      # Train/validation divergence diagnostic
│   ├── eval_zeroshot_hls4ml.py        # Zero-shot transfer to the hls4ml dataset
│   ├── fetch_jetclass_test.py         # Resumable download of the JetClass test archives
│   ├── rebuild_test_h5.py             # Rebuild a damaged test.h5
│   └── relabel_jetclass_metrics.py    # Re-map class labels in a metrics file
├── notebooks/                         # Exploratory notebook
├── repositoryModel/                   # Small checked-in example models
├── compile.sh                         # Wrapper around `alkaid convert`
├── environment.yml
└── requirements.txt
```

## Datasets

**JetClass** — 128 constituents, 17 features, 10 classes. The primary benchmark.
<https://doi.org/10.5281/zenodo.6619768>

```bash
python -m src.data.download_jetclass JetClass                          # downloads into datasets/
python -m src.data.build_jetclass_dataset --input_dir datasets/JetClass/Pythia
```

**hls4ml LHC jet** — 16 constituents, 3 features, 5 classes. A run takes 10–15 minutes rather
than several hours, which makes it the practical dataset for ablations.
<https://doi.org/10.5281/zenodo.3602260>

```bash
python -m src.data.build_dataset --num_particles 16 --num_feats 3
```

Training will fail if the matching dataset has not been built first.

## Usage

### Training

```bash
python -m src.training.train [flags]
```

Quantization-aware training is enabled by default. Pass `--no-quantize` to train the
floating-point baseline through the same pipeline.

**Dataset and run size**

| Flag | Default | Meaning |
|---|---|---|
| `--dataset` | `hls4ml` | `hls4ml` or `jetclass` |
| `--num_particles` | 16 / 128 | Constituents per jet; resolves per dataset |
| `--num_feats` | 3 / 17 | Features per constituent; resolves per dataset |
| `--max_samples` | all | Cap on training + validation samples, e.g. `2000000` |
| `--train_parts` | all | Specific JetClass part indices, e.g. `--train_parts 0 1 2` |
| `--in_memory` | auto | Pre-load the dataset into RAM |
| `--experiment` | none | Route all artifacts under `experiment/<name>/` |

**Architecture**

| Flag | Default | Meaning |
|---|---|---|
| `--embed_dim` | 32 | Embedding and hidden dimension |
| `--num_transformers` | 1 | Transformer blocks (3 for the legacy JetFormer) |
| `--num_heads` | 2 | Attention heads |
| `--use_linformer` | on | Low-rank attention instead of standard multi-head |
| `--proj_dim_k` | 2 | Linformer summary slots the particle axis is compressed to |
| `--use_cls_token` | off | CLS token instead of global average pooling |
| `--dropout` | 0.0 | Dropout rate |

**Optimisation and the resource budget**

| Flag | Default | Meaning |
|---|---|---|
| `--num_epochs` | 250 | Total training epochs |
| `--batch_size` | 256 | Training batch size |
| `--seed` | 42 | Seed for initialization and shuffling |
| `--early_stopping_patience` | 150 | Epochs without a saved checkpoint before stopping |
| `--target_ebops` | 450000 | BetaPID setpoint the controller regulates to |
| `--ebops_threshold` | 550000 | Checkpoint gate; models save only below this |
| `--ebops_warmup_epoch` | 75 | Epoch from which checkpointing begins |
| `--floor_attn_datalane` | off | Floor attention activation bit-widths so channels cannot prune to zero |
| `--augment_rotation` | off | On-the-fly SO(2) rotation in the (deta, dphi) plane |
| `--use_adaptive_lr` | off | Noise-bounded adaptive LR plateau callback |
| `--save_final_epoch` | off | Also save the last-epoch model as `<stem>_final.keras` |

Run `python -m src.training.train --help` for the complete list.

Train the deployed configuration on JetClass:

```bash
python -m src.training.train \
    --dataset jetclass \
    --num_particles 128 --num_feats 17 \
    --max_samples 2000000 \
    --experiment my_run
```

A quick ablation on the small dataset:

```bash
python -m src.training.train --dataset hls4ml --num_epochs 50 --experiment ablation_k8 --proj_dim_k 8
```

### Evaluation

Training runs its own evaluation at the end. To re-evaluate a saved checkpoint:

```bash
python -m src.training.evaluate --experiment my_run --dataset jetclass
python -m src.training.evaluate --model_path experiment/my_run/models/quantized/128_17f.keras
```

`--calib_seed` (default 42) seeds the WRAP-mode calibration sample. Keep it fixed across runs so
accuracies stay comparable; it is independent of `--seed`.

### EBOPs analysis

Per-layer breakdown of where the bit-operation budget is spent:

```bash
python scripts/analyze_ebops.py --model_path experiment/my_run/models/quantized/128_17f.keras
python scripts/analyze_ebops.py --model_path <path.keras> --sublayers
```

### Hardware compilation

`compile.sh` wraps `alkaid convert`, pins JAX to the CPU and tees the log into the output
directory. The second argument is the output directory:

```bash
./compile.sh experiment/my_run/models/quantized/128_17f.keras rtl_out/
```

The generated Verilog and the compile log land in `rtl_out/`.

## Where results are written

Without `--experiment`, artifacts go to `models/` and `outputs/` at the repository root. With
`--experiment NAME`, everything is routed under `experiment/NAME/` instead, which is the
recommended way to keep runs from overwriting each other. Quantized and float runs are kept in
separate subdirectories:

```
experiment/NAME/
├── models/quantized/128_17f.keras
└── outputs/quantized/
    ├── 128_17f_metrics.json          # accuracy, per-class AUC, full run configuration
    ├── 128_17f_loss_acc.npz          # loss and accuracy curves
    ├── 128_17f_plot.png              # training curves
    └── 128_17f_ebops_beta.png        # EBOPs and BetaPID beta over training
```

File stems follow `{num_particles}_{num_feats}f`. Every empirical number reported in the thesis
is read from one of these `metrics.json` files or from a Vivado report.

## Branches

`main` holds the final state of the work. The other branches are the experimental arcs the
project ran through and are kept so the history of the work is visible; `feat/proj-dim-k`, for
example, carries the `--rich_pool` and `--head_width` reallocation experiments that are not on
`main`.

## License

See [LICENSE](LICENSE).
