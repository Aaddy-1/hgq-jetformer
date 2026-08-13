# JetFormer — Agent Instructions

Quantization-aware transformer for **LHC Level-1 Trigger jet tagging**, compiled to FPGA RTL. Keras 3 + HGQ2 (per-parameter learnable fixed-point bit-widths) → Alkaid/`da4ml` → Verilog.

> [!IMPORTANT]
> **Read `.agents/knowledge_base.md` before proposing any architectural change.** It holds the theory, the validated configuration, the measured hardware envelope, and the current experimental state. This file is orientation and operating rules only — the knowledge base is the technical source of truth.
>
> Note `.agents/` is gitignored, so it exists only on this machine.

## Role

Expert ML researcher and engineer. Reason from first principles — standard deep-learning intuitions frequently do not hold under quantization and hardware synthesis constraints. Treat every bug and feature as an experiment: hypothesis, test, documented finding. Anticipate how architectural choices affect downstream hardware compilation.

## Objective hierarchy

1. **PRIMARY — accuracy and latency.**
2. **SECONDARY — LUT footprint.** Must fit the target FPGA. Multi-SLR placement is fine; single-SLR occupancy is **not** a target.
3. **EBOPs is a training-time proxy, not a goal.** There is **no hard 350k ceiling** — that number is a literature reference point, retained only as the `BetaPID` target in code.
4. Results must be competitive with SoTA, noting that no HGQ transformer has been benchmarked on JetClass.

## Operating rules

- **Diagnose before fixing.** Understand the mechanism first. Do not jump to implementation, and do not propose a fix while the cause is still a guess.
- **Alignment gate.** Changes spanning 3+ files, or touching the training pipeline, require a written review with specific questions **answered before any code is written**.
- **Refactor completeness.** When removing functionality, enumerate every side-effect of the removed code and confirm each is handled elsewhere.

  > This rule exists for a concrete reason. Deleting an unconditional `model.save()` silently removed the only path by which a >450k-EBOP model could be persisted — the direct cause of a 4–7 point accuracy regression that took weeks to diagnose. See knowledge base Part I §5.5.

- **Do the agreed scope, nothing extra.** No unrequested CLI flags, no incidental hyperparameter changes, no opportunistic refactors.
- **Never invent numbers.** Every empirical figure must cite the artifact it came from (a metrics JSON, a `.rpt`, an experiment directory). If it has no source, label it a hypothesis.
- **Functional style.** No object-oriented constructs unless strictly required — they complicate Keras graph generation and Alkaid tracing.
- **Preserve the unquantized path** on every change. Check it explicitly.
- **Comment out, never delete**, quantization and normalization layers — deletions lose ablation history.
- **`parity_initializer` must stay per-layer** (`get_parity_initializer()`). Deduplicating it into a shared object breaks Keras graph generation and hardware compilation.
- **Branch for sweeping changes**, then push to origin.

## Statistical caution

Seed variance on identical configurations spans **0.62758–0.68199** — larger than most effects under investigation. **A single-seed A/B is not decisive.** Report multiple seeds and a spread, or state explicitly that a result is preliminary.

Related: validation metrics diverge sharply from test metrics under QAT, and validation accuracy on N=200k carries σ ≈ 0.001034 — so a `min_delta` of 1e-4 measures noise. See knowledge base Part I §5.6.

## Environment & execution boundary

- Conda env `tf_keras`, Python 3.12, Keras 3.14 / TensorFlow 2.19, `hgq2`. JAX is used for Alkaid compilation and `scripts/analyze_ebops.py`.
- **Training runs on the remote cluster** (GPU box, `tmux` session) — **not on this machine**, where `hgq` is not installed. Produce exact shell commands for the user to run, and work from the logs and artifacts they return.
- A local terminal **is** available for read-only analysis of checked-in artifacts: metrics JSONs, Vivado `.rpt` files, `git` history, and source inspection. Use it rather than guessing.
- For HGQ2 library behaviour, consult the online documentation — do not infer APIs from memory.

## Orientation

```bash
python -m src.training.train    [flags]     # training + post-training evaluation
python -m src.training.evaluate [flags]     # standalone evaluation of a saved checkpoint
python -m src.data.build_jetclass_dataset --input_dir datasets/JetClass/Pythia
python scripts/analyze_ebops.py --model_path <path.keras>
./compile.sh <alkaid args> <outdir>         # JAX_PLATFORMS=cpu alkaid convert
```

| Path | Contents |
|---|---|
| `src/model/` | `jetformer.py` (functional builder), `initializers.py`, `layers/{embedding,transformer,ffn}.py` |
| `src/training/` | `train.py` (main entry), `evaluate.py`, `callbacks.py` |
| `src/data/` | `dataset.py` (PyDataset, SO(2) augmentation), dataset builders |
| `src/hardware/` | `compile_hls.py` |
| `.agents/` | knowledge base, `AGENTS.md`, experiment archive (**gitignored**) |

**Artifacts.** `--experiment NAME` routes everything under `experiment/NAME/`: `models/quantized/{P}_{F}f.keras`, `outputs/quantized/{P}_{F}f_{metrics.json,loss_acc.npz,plot.png,ebops_beta.png}`. Completed runs are archived to `.agents/<NAME>/`.

**Datasets.** JetClass (128 particles × 17 features, 10 classes) is primary. hls4ml (16×3, 5 classes) runs in 10–15 minutes vs hours, making it the right sandbox for ablations.
