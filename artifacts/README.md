# Artifacts

Pipeline outputs behind the numbers in the MSc thesis *Quantization-Aware Transformers for
Jet Tagging in the LHC Level-1 Trigger*. Every file was written by the pipeline when the run
finished. Nothing has been edited or regenerated.

```
metrics/     57 JSONs, one per evaluated run - configuration, provenance,
             per-class and overall accuracy/AUC, EBOPs, best epoch, layer shapes
compiled/    22 metadata.json - target device, clock period, timing uncertainty,
             I/O delays, adder-graph cost, pipeline cycles
synthesis/   full Vivado report sets for the two designs of Table 5.4
               64const/   the operating point (Section 4.6)
               128const/  the design it is cropped from
```

## Finding a number

| Thesis | Files in `metrics/` |
|---|---|
| Table 5.1 — 500k, D=32, control | `E1_CONTROL_SEED42`, `E1_CONTROL_SEED43`, `E2_CONTROL_SEED44` |
| Table 5.1 — 500k, D=32, floored | `E1_FLOORED_SEED42`, `E1_FLOORED_SEED43`, `E2_FLOORED_SEED44` |
| Table 5.1 — 350k, D=32, control | `E3_CONTROL_350K_SEED42` |
| Table 5.1 — 350k, D=32, floored | `E3_FLOORED_350K_TIGHT_SEED42`, `E3_FLOORED_350K_TIGHT_SEED43` |
| Table 5.1 — 350k, D=16, floored | `E4_EMBED16_SEED42`, `E4_EMBED16_SEED43`, `E4_EMBED16_SEED44` |
| Table 5.1 — 350k, D=16, 64 const. | `C1_CROP64_SEED42`, `C1_CROP64_SEED43`, `C1_CROP64_SEED44` |
| Table 5.1 — unquantized ceiling | `EXP-22_FP32_CONTROL` |
| Table 4.1 | `A1_K8_SEED42`, `C1_CROP64_K8_SEED42` |
| Table 5.3 — 500k, D=16, floored | `E4_EMBED16_500K_SEED42` |
| Table 5.5 | `E3_CONTROL_350K_SEED42`, `E3_FLOORED_350K_TIGHT_SEED42` |
| Tables 5.2, 5.4 and Section 4.6 (operating point) | `C1_CROP64_SEED42` |
| Section 5.5.3 (projection rank k=1) | `A1_K1_SEED42` |
| Section 5.5.4 (depth) | `C2_DEPTH2_SEED42`, `C2_DEPTH2_SEED43` |
| Section 5.5.5 (pooling) | `C2_RICHPOOL64_SEED42` |

The remaining files are exploratory or superseded runs that back no figure in the report.
Where a run appears twice, the `_1` file is the completed evaluation.

## Two things visible in the data

- Four runs were produced from a working tree with uncommitted changes (`git_dirty: true`):
  `C1_CROP64_SEED43`, `C1_CROP64_SEED44`, `DEMO_MASK_OFF`, `DEMO_MASK_ON`. Their recorded
  commit identifies the base state, not the exact tree. The seed carried through to hardware,
  `C1_CROP64_SEED42`, was clean.
- 34 of the 57 files predate the `configuration.training` block added in commit `07af92c`,
  so their full hyperparameter set is not recoverable from the artifact alone. The
  architectural fields are present in all of them.

## Not included

Trained `.keras` checkpoints (~1 MB each) and generated Verilog (~28 MB per design) are not
in the repository. Available on request.
