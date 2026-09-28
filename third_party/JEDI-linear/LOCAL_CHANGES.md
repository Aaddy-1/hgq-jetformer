# Local changes to JEDI-linear

Source: https://github.com/calad0i/JEDI-linear at
`48032802068bf1e4878b5b119a6e8a87512c733e` (2025-11-29), Apache-2.0 (`LICENSE`).

The tree was vendored byte-for-byte in commit `928a767`. `official_models.tar.gz` and
`official_result_reports.tar.gz` were left out: they are artifacts, not code. Run
`git diff 928a767 -- third_party/JEDI-linear` to see every change listed here.

## Changes

**`src/model.py`: two lines in `get_gnn`, plus a notice at the top of the file.**

```python
n = conf.get('n_features', 3 if conf.pt_eta_phi else 16)                                # was: 3 if conf.pt_eta_phi else 16
out = QEinsumDenseBatchnorm('bc,cC->bC', conf.get('n_classes', 5), bias_axes='C')(x)   # was: 5
```

JetClass has 17 features and 10 classes; upstream hardcodes 3 or 16 features and 5
classes. The upstream configs do not set these keys, so they build the upstream graph.
`get_mlpm`, `get_mlp` and `get_model` are unchanged.

**New configs in `configs/`, each derived from an upstream config.** Every key not
listed below is identical to the base.

| Config | Base | Changed keys |
|---|---|---|
| `hls4ml-official-n128-f16.yaml` | `sweep-n128-f16.yaml` | `datapath`, `save_path` |
| `hls4ml-parity-n128-f16.yaml` | `sweep-n128-f16.yaml` | `datapath`, `save_path`, `train.epochs: 2000` |
| `jetclass-n{128,64}-f17-seed{42,43,44}.yaml` | `sweep-n{128,64}-f16.yaml` | `seed`, `datapath`, `save_path`, added `n_features: 17` and `n_classes: 10` |
| `jetclass-smoke-n128.yaml` | `jetclass-n128-f17-seed42.yaml` | `save_path`, `train.epochs: 2`, last β knot written as `7000` |

- **2000 epochs.** The β schedule interpolates between knots, and with `epochs: 2000` the
  knots become `[0, 2000, 2000]`. For epochs 0–1999 this gives the same β as the
  7000-epoch schedule, and the learning-rate schedule is indexed by epoch. The published
  N=128 checkpoint is from epoch 1585.
- **Smoke config.** The last β knot is written as a literal so that its two epochs use
  production β. It is used only to measure memory and time per epoch.

Paths in these configs are relative to this directory, so run `jet_classifier.py` from
here.

## Unchanged

`jet_classifier.py`, `src/dataloader.py`, `src/train.py`, `src/test.py`,
`src/syn_test.py`, `prepare_dataset.py`, `prepare_dataset.sh`, `environment.yml`,
`requirements.txt` and the upstream configs.

## Data

`get_data` reads `<datapath>/150c-train.h5` and `<datapath>/150c-test.h5`, with keys
`feature` and `label`.
- **hls4ml:** these come from upstream `prepare_dataset.py`.
- **JetClass:** `scripts/jedi_linear/export_jetclass.py` writes them under the same names.
  It uses the 2M-jet train and test subsets that the SCRAMJet runs use.

## Outside this directory

`scripts/jedi_linear/select_and_evaluate.py` takes the training outputs. It picks the
best-val_acc Pareto checkpoint under an EBOPs threshold and computes accuracy and macro
AUC on logits. With `--verilog` it runs upstream `syn_test_verilog` on that checkpoint.
It imports upstream modules and does not modify them.
