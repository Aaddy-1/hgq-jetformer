# Artifacts

Recorded outputs behind the numbers in the MSc thesis *Quantization-Aware Transformers for
Jet Tagging in the LHC Level-1 Trigger*. Every file here was written by the pipeline at the
time the run finished; nothing has been edited or regenerated.

```
metrics/     one JSON per evaluated run: configuration, provenance, per-class and overall
             accuracy/AUC, EBOPs, best epoch, layer shapes
compiled/    metadata.json per compiled design: target device, clock period, timing
             uncertainty, I/O delays, adder-graph cost, pipeline cycles
synthesis/   full Vivado report sets for the two designs of Table 5.4
               64const/  operating point (Section 4.6)
               128const/ the design it is cropped from
```

## Provenance

Each metrics file carries a `configuration.provenance` block recording the hostname, the
visible CUDA device, the GPU model, the git commit and whether the working tree was clean.
Two limitations are visible in the data and are not hidden here:

- **4 runs were produced from a working tree with uncommitted changes**
  (`git_dirty: true`): C1_CROP64_SEED43, C1_CROP64_SEED44, DEMO_MASK_OFF, DEMO_MASK_ON.
  Their recorded commit therefore identifies the base state, not the exact tree.
- **34 of 57 files predate the `configuration.training` block** added in commit
  `07af92c`, so their full hyperparameter set is not recoverable from the artifact alone;
  the architectural fields (`num_particles`, `num_feats`, `dataset`, `quantize`) are present.

## Runs cited in the thesis

| File | Thesis | Configuration | Accuracy | EBOPs | Clean | Commit |
|---|---|---|---|---|---|---|
| `A1_K1_SEED42__128_17f_metrics.json` | Sec 5.5.3 | projection rank k=1 | 0.69760 | 336,022 | yes | `ef79fdac` |
| `C2_DEPTH2_SEED42__64_17f_metrics.json` | Sec 5.5.4 | second transformer block | 0.71682 | 356,010 | yes | `b83267f2` |
| `C2_DEPTH2_SEED43__64_17f_metrics.json` | Sec 5.5.4 | second transformer block | 0.70946 | 353,498 | yes | `a97db1ea` |
| `C2_RICHPOOL64_SEED42__64_17f_metrics.json` | — | earlier partial evaluation of the same run, superseded by `_1` | 0.64050 | — | yes | `a97db1ea` |
| `C2_RICHPOOL64_SEED42__64_17f_metrics_1.json` | Sec 5.5.5 | rich pooling variant | 0.70959 | 354,314 | yes | `a97db1ea` |
| `A1_K8_SEED42__128_17f_metrics.json` | Table 4.1 | 128 constituents, k=8 | 0.59034 | 355,392 | yes | `ef79fdac` |
| `C1_CROP64_K8_SEED42__64_17f_metrics.json` | Table 4.1 | 64 constituents, k=8 | 0.70279 | 356,156 | yes | `b83267f2` |
| `C1_CROP64_SEED43__64_17f_metrics.json` | Table 5.1 | 350k D=16 64-const (n=3 mean 0.71434) | 0.72145 | 358,843 | **no** | `598ae6df` |
| `C1_CROP64_SEED44__64_17f_metrics.json` | Table 5.1 | 350k D=16 64-const (n=3 mean 0.71434) | 0.71058 | 349,655 | **no** | `598ae6df` |
| `E1_CONTROL_SEED42__128_17f_metrics.json` | Table 5.1 | 500k D=32 control (n=3 mean 0.68207) | 0.65318 | 516,131 | yes | `0949cc78` |
| `E1_CONTROL_SEED43__128_17f_metrics.json` | Table 5.1 | 500k D=32 control (n=3 mean 0.68207) | 0.68516 | 525,697 | yes | `0949cc78` |
| `E1_FLOORED_SEED42__128_17f_metrics.json` | Table 5.1 | 500k D=32 floored (n=3 mean 0.72089) | 0.72100 | 533,746 | yes | `0949cc78` |
| `E1_FLOORED_SEED43__128_17f_metrics.json` | Table 5.1 | 500k D=32 floored (n=3 mean 0.72089) | 0.71502 | 535,738 | yes | `eaf67359` |
| `E2_CONTROL_SEED44__128_17f_metrics.json` | Table 5.1 | 500k D=32 control (n=3 mean 0.68207) | 0.70786 | 521,571 | yes | `eaf67359` |
| `E2_FLOORED_SEED44__128_17f_metrics.json` | Table 5.1 | 500k D=32 floored (n=3 mean 0.72089) | 0.72664 | 516,025 | yes | `eaf67359` |
| `E3_FLOORED_350K_TIGHT_SEED43__128_17f_metrics.json` | Table 5.1 | 350k D=32 floored (n=2 mean 0.69460) | 0.69440 | 348,965 | yes | `eaf67359` |
| `E4_EMBED16_SEED43__128_17f_metrics.json` | Table 5.1 | 350k D=16 floored (the val/test outlier of Sec 5.7) | 0.66115 | 349,062 | yes | `eaf67359` |
| `E4_EMBED16_SEED44__128_17f_metrics.json` | Table 5.1 | 350k D=16 floored (n=3 mean 0.68606) | 0.70072 | 358,633 | yes | `eaf67359` |
| `EXP-22_FP32_CONTROL__128_17f_metrics.json` | Table 5.1 | unquantized ceiling 0.76690 | 0.76690 | — | — | `—` |
| `E4_EMBED16_500K_SEED42__128_17f_metrics.json` | Table 5.3 | 500k D=16 floored (single run) | 0.70491 | 513,216 | yes | `eaf67359` |
| `C1_CROP64_SEED42__64_17f_metrics.json` | — | earlier partial evaluation of the same run, superseded by `_1` | 0.67316 | — | yes | `a97db1ea` |
| `C1_CROP64_SEED42__64_17f_metrics_1.json` | Tables 5.1, 5.2, 5.4; Sec 4.6 | operating point; the seed carried to hardware | 0.71100 | 353,885 | yes | `b83267f2` |
| `E4_EMBED16_SEED42__128_17f_metrics.json` | Tables 5.1, 5.3; Sec 5.5.3 | 350k D=16 floored; also the k=1 ablation baseline | 0.69632 | 358,419 | yes | `eaf67359` |
| `E3_CONTROL_350K_SEED42__128_17f_metrics.json` | Tables 5.1, 5.5 | 350k D=32 control; the matched-EBOPs control arm | 0.66379 | 344,050 | yes | `eaf67359` |
| `E3_FLOORED_350K_TIGHT_SEED42__128_17f_metrics.json` | Tables 5.1, 5.5 | 350k D=32 floored; the matched-EBOPs floored arm | 0.69480 | 343,627 | yes | `eaf67359` |

Where an experiment appears twice, the `_1` file is the completed evaluation and the
un-suffixed file is an earlier partial pass retained for completeness.

Table 5.1's means and standard deviations reproduce exactly from the rows above; the two
synthesis report sets give 453,288 CLB LUTs (26.23%) and 540,050 (31.25%), matching Table 5.4.

## Runs not cited in the thesis

Exploratory, superseded or dead-end runs, retained so the record is complete rather than
filtered. They back no figure in the report.

| File | Constituents | Dataset | Accuracy | EBOPs |
|---|---|---|---|---|
| `DEMO_MASK_OFF__16_3f_metrics.json` | 16 | hls4ml | 0.71071 | 250,408 |
| `DEMO_MASK_ON__16_3f_metrics.json` | 16 | hls4ml | 0.70192 | 250,351 |
| `E3_FLOORED_350K_SEED42__128_17f_metrics.json` | 128 | jetclass | 0.69532 | 418,959 |
| `EXP-17__128_17f_metrics.json` | 128 | jetclass | 0.17953 | — |
| `EXP-18__128_17f_metrics.json` | 128 | jetclass | 0.68199 | — |
| `EXP-18__jetmetrics.json` | 128 | jetclass | 0.66203 | 498,070 |
| `EXP-18_lowerwarmup_seed44__128_17f_metrics.json` | 128 | jetclass | 0.72429 | 365,644 |
| `EXP-18_rotate_seed45__128_17f_metrics.json` | 128 | jetclass | 0.62586 | 351,571 |
| `EXP-18_rotate_seed46__128_17f_metrics.json` | 128 | jetclass | 0.60370 | 404,703 |
| `EXP-18_seed43__128_17f_metrics.json` | 128 | jetclass | 0.62758 | — |
| `EXP-18_seed44__128_17f_metrics.json` | 128 | jetclass | 0.66929 | — |
| `EXP-22_ADAPTIVE_LR_SEED42__128_17f_metrics.json` | 128 | jetclass | 0.60826 | 428,791 |
| `EXP-22_CLS_TOKEN__128_17f_metrics.json` | 128 | jetclass | 0.64396 | 342,136 |
| `EXP-22_NEWCONFIG_VALIDATION__128_17f_metrics.json` | 128 | jetclass | 0.63811 | 353,884 |
| `EXP-22_STANDARD_FIX_SEED42__128_17f_metrics.json` | 128 | jetclass | 0.64318 | 442,233 |
| `EXP-22_Seed43__128_17f_metrics.json` | 128 | jetclass | 0.61700 | 375,700 |
| `EXP-24_TARGET500K_SEED42__128_17f_metrics.json` | 128 | jetclass | 0.70376 | 578,286 |
| `EXP-24_TARGET500K_SEED42_FINALEVAL__128_17f_metrics.json` | 128 | jetclass | 0.54219 | — |
| `EXP-24_TARGET500K_SEED43__128_17f_metrics.json` | 128 | jetclass | 0.68163 | 525,697 |
| `EXP-24_TARGET500K_SEED43_FINALEVAL__128_17f_metrics.json` | 128 | jetclass | 0.60164 | — |
| `EXP-25_FULLCURVE_SEED42__128_17f_metrics.json` | 128 | jetclass | 0.65366 | 516,131 |
| `EXP-25_FULLCURVE_SEED42_FINALEVAL__128_17f_metrics.json` | 128 | jetclass | 0.58699 | — |
| `EXP-25_FULLCURVE_SEED43__128_17f_metrics.json` | 128 | jetclass | 0.65043 | 509,392 |
| `EXP-25_FULLCURVE_SEED43_FINALEVAL__128_17f_metrics.json` | 128 | jetclass | 0.61323 | — |
| `hgq_Jetformer__32embedidim_metrics.json` | 8 | — | 0.56748 | 667,729 |
| `hgq_Jetformer__8_3f_metrics_64dim.json` | 8 | — | 0.39880 | 913,189 |
| `hgq_Jetformer__linformergapmetrics.json` | 16 | — | 0.70577 | 345,770 |
| `outputs__16_3f_metrics.json` | 16 | — | 0.71765 | 376,510 |
| `outputs__8_3f_metrics.json` | 8 | — | 0.65093 | 285,791 |
| `outputs__8_3f_metrics_1.json` | 8 | — | 0.56748 | 667,729 |
| `outputs__16_3f_metrics_1.json` | 16 | — | 0.70117 | 344,751 |
| `outputs__16_3f_metrics_2.json` | 16 | — | 0.52476 | 336,210 |

## Compiled designs

22 `metadata.json` files under `compiled/`, named by experiment. Each records the
compilation parameters quoted in Appendix A.2.

## Not included here

Trained `.keras` checkpoints (~1 MB each) and generated Verilog (~28 MB per design) are not
in the repository. They are available on request.

