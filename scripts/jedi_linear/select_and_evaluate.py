"""Pick one checkpoint from a JEDI-linear Pareto front and evaluate it on test.

Upstream training (`third_party/JEDI-linear/src/train.py`) saves every checkpoint
that is non-dominated on (val_accuracy, EBOPs) to `<save_path>/ckpts/`. Upstream
`test.py` (`jet_classifier.py -r test`) then runs `trace_minmax` on each one, records
its test accuracy in `<save_path>/test_acc.json`, and saves the calibrated model to
`<save_path>/models/`. This script runs after that:

1. Selection: the checkpoint with the highest val_accuracy whose training-time
   EBOPs is <= --ebops_threshold. This is the SCRAMJet checkpoint rule (EBOPs under
   the gate, best val_accuracy). The test set plays no part in the choice. Exact
   values come from `history.pkl` when the run finished, otherwise from the
   filename, where val_accuracy is rounded to 0.01%.
2. Evaluation: the calibrated `models/<stem>.keras` is run on the test split
   returned by upstream `get_data`, so the normalization is the one used in
   training. Accuracy, per-class accuracy and one-vs-rest AUC are computed on the
   logits with the formula of `src/training/train.py:641-668`, which is how the
   SCRAMJet numbers are computed. The accuracy must equal that checkpoint's
   `test_acc.json` entry.
3. --reference: the parity verdict against another run's test_acc.json, e.g. the
   re-evaluated published checkpoint. Tolerance is PARITY_TOLERANCE, fixed before
   the parity run.
4. --verilog: the selected model is staged alone in `<save_path>/selected/`, and
   upstream `syn_test_verilog` runs on it unmodified.

Writes `<save_path>/selected_metrics.json`. Upstream modules are imported, not
modified. Run in the `jedi-linear` env.

Usage:
    KERAS_BACKEND=jax python scripts/jedi_linear/select_and_evaluate.py \\
        --config third_party/JEDI-linear/configs/hls4ml-parity-n128-f16.yaml \\
        --ebops_threshold 142664 --reference experiment/JEDI_OFFICIAL_EVAL_N128/test_acc.json
"""

import argparse
import json
import os
import pickle
import platform
import re
import shutil
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
VENDORED = PROJECT_ROOT / "third_party" / "JEDI-linear"

PARITY_TOLERANCE = 0.005  # 0.5 accuracy points, fixed before the parity run

CKPT_RE = re.compile(
    r"^epoch=(?P<epoch>\d+)-acc=(?P<acc>[\d.]+)%-val_acc=(?P<val_acc>[\d.]+)%-EBOPs=(?P<ebops>[\d.]+)\.keras$"
)


def parse_ckpt(name):
    m = CKPT_RE.match(name)
    if m is None:
        return None
    return {
        "name": name,
        "epoch": int(m["epoch"]),
        "val_acc": float(m["val_acc"]) / 100,
        "ebops": float(m["ebops"]),
    }


def select_checkpoint(ckpts, ebops_threshold, history=None):
    """Best val_acc with EBOPs <= threshold; ties go to fewer EBOPs, then the earlier epoch."""
    rows = []
    for c in ckpts:
        row = dict(c)
        if history is not None:
            e = c["epoch"]
            row["val_acc"] = float(history["val_accuracy"][e])
            row["ebops"] = float(history["ebops"][e])
            if round(row["ebops"]) != round(c["ebops"]):
                raise SystemExit(f"history EBOPs {row['ebops']} != filename EBOPs for {c['name']}")
        rows.append(row)
    eligible = [r for r in rows if r["ebops"] <= ebops_threshold]
    if not eligible:
        return None, rows
    best = min(eligible, key=lambda r: (-r["val_acc"], r["ebops"], r["epoch"]))
    return best, rows


def classification_metrics(outputs, labels):
    """src/training/train.py:641-668, on logits."""
    from sklearn.metrics import accuracy_score, roc_auc_score

    pred = outputs.argmax(axis=1)
    n_classes = outputs.shape[1]
    acc = float(accuracy_score(labels, pred))
    class_accs = []
    for i in range(n_classes):
        idx = labels == i
        class_accs.append(float(accuracy_score(labels[idx], pred[idx])) if idx.sum() > 0 else float("nan"))
    onehot = np.eye(n_classes)[labels.astype(int)]
    aucs = [float(a) for a in roc_auc_score(onehot, outputs, average=None, multi_class="ovr")]
    return acc, class_accs, aucs


def provenance():
    def version(pkg):
        try:
            return metadata.version(pkg)
        except metadata.PackageNotFoundError:
            return None

    def git(*args):
        try:
            return subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True,
                                  text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = git("status", "--porcelain", "--untracked-files=no")
    return {
        "hostname": platform.node(),
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python": platform.python_version(),
        "packages": {p: version(p) for p in ("keras", "hgq2", "jax", "da4ml", "numpy", "scikit-learn")},
        "keras_backend": os.environ.get("KERAS_BACKEND"),
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty_tracked": bool(status) if status is not None else None,
    }


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", type=str, required=True,
                   help="the JEDI-linear yaml the run was trained with")
    p.add_argument("--ebops_threshold", type=float, required=True,
                   help="EBOPs gate for the selection (360000 for JetClass, 142664 for parity)")
    p.add_argument("--reference", type=str, default=None,
                   help="test_acc.json of the reference run for the parity verdict")
    p.add_argument("--verilog", action="store_true",
                   help="run upstream syn_test_verilog on the selected checkpoint")
    args = p.parse_args()

    from omegaconf import OmegaConf

    conf = OmegaConf.load(args.config)
    save_path = (VENDORED / conf.save_path).resolve()
    data_path = (VENDORED / conf.datapath).resolve()
    print(f"save_path {save_path}\ndatapath  {data_path}")

    ckpts = [c for c in (parse_ckpt(f.name) for f in sorted((save_path / "ckpts").glob("*.keras"))) if c]
    if not ckpts:
        raise SystemExit(f"no checkpoints in {save_path / 'ckpts'}")
    history = None
    if (save_path / "history.pkl").exists():
        with open(save_path / "history.pkl", "rb") as f:
            history = pickle.load(f)
    best, rows = select_checkpoint(ckpts, args.ebops_threshold, history)
    print(f"{len(rows)} Pareto checkpoints; values from {'history.pkl' if history else 'filenames'}")
    if best is None:
        lowest = min(rows, key=lambda r: r["ebops"])
        raise SystemExit(f"no checkpoint with EBOPs <= {args.ebops_threshold:,.0f}; "
                         f"lowest is {lowest['ebops']:,.0f} at epoch {lowest['epoch']}")
    stem = best["name"][: -len(".keras")]
    print(f"selected {best['name']}\n  epoch {best['epoch']}  val_acc {best['val_acc']:.5f}  "
          f"EBOPs {best['ebops']:,.0f} (<= {args.ebops_threshold:,.0f})")

    with open(save_path / "test_acc.json") as f:
        test_acc = json.load(f)
    if best["name"] not in test_acc:
        raise SystemExit(f"{best['name']} missing from test_acc.json: run `jet_classifier.py -r test` first")

    sys.path.insert(0, str(VENDORED))
    import keras
    # Imported as jet_classifier.py imports it. That registers the HGQ layers for
    # load_model, and it sets JAX matmul precision to tensorfloat32, the setting
    # under which test.py wrote test_acc.json.
    import src.model  # noqa: F401
    from src.dataloader import get_data

    np.random.seed(conf.seed)  # jet_classifier.py seeds numpy before get_data
    X_train, X_test, y_train, y_test = get_data(data_path, conf.n_constituents, ptetaphi=conf.pt_eta_phi)
    del X_train, y_train
    y_test = np.asarray(y_test)

    model = keras.models.load_model(save_path / "models" / f"{stem}.keras", compile=False)
    outputs = np.asarray(model.predict(X_test, batch_size=16384, verbose=0))
    acc, class_accs, aucs = classification_metrics(outputs, y_test)
    upstream_acc = float(test_acc[best["name"]]["acc"])
    print(f"test accuracy {acc:.5f}  (upstream test_acc.json {upstream_acc:.5f})  macro AUC {np.mean(aucs):.5f}")
    if abs(acc - upstream_acc) > 1e-6:
        raise SystemExit("accuracy does not reproduce test_acc.json; the saved model and the test pass differ")

    result = {
        "config": str(Path(args.config).resolve()),
        "seed": conf.seed,
        "n_constituents": conf.n_constituents,
        "selection": {
            "rule": "max val_accuracy with training-time EBOPs <= threshold",
            "ebops_threshold": args.ebops_threshold,
            "source": "history.pkl" if history else "filenames",
            "n_pareto_checkpoints": len(rows),
            "checkpoint": best["name"],
            "epoch": best["epoch"],
            "val_accuracy": best["val_acc"],
            "ebops_train": best["ebops"],
            "ebops_after_trace_minmax": float(test_acc[best["name"]]["ebops"]),
        },
        "performance": {
            "overall_accuracy": acc,
            "macro_auc": float(np.mean(aucs)),
            "per_class_metrics": {str(i): {"accuracy": a, "auc": u}
                                  for i, (a, u) in enumerate(zip(class_accs, aucs))},
            "num_test_samples": int(len(y_test)),
        },
        "provenance": provenance(),
    }

    if args.reference:
        with open(args.reference) as f:
            ref = json.load(f)
        if len(ref) != 1:
            raise SystemExit(f"--reference must hold one checkpoint, found {len(ref)}")
        (ref_name, ref_entry), = ref.items()
        ref_acc = float(ref_entry["acc"])
        diff = acc - ref_acc
        verdict = "PASS" if abs(diff) <= PARITY_TOLERANCE else "FAIL"
        result["parity"] = {"reference": str(Path(args.reference).resolve()), "reference_checkpoint": ref_name,
                            "reference_accuracy": ref_acc, "difference": diff,
                            "tolerance": PARITY_TOLERANCE, "verdict": verdict}
        print(f"parity: ours {acc:.5f}  reference {ref_acc:.5f}  diff {100 * diff:+.2f} points  "
              f"(tolerance {100 * PARITY_TOLERANCE:.1f})  {verdict}")

    if args.verilog:
        from src.syn_test import syn_test_verilog

        staged = save_path / "selected"
        (staged / "models").mkdir(parents=True, exist_ok=True)
        shutil.copy2(save_path / "models" / f"{stem}.keras", staged / "models" / f"{stem}.keras")
        with open(staged / "test_acc.json", "w") as f:
            json.dump({best["name"]: test_acc[best["name"]]}, f)
        syn_test_verilog(staged, X_test, y_test, N=None)
        with open(staged / "test_acc.json") as f:
            syn = json.load(f)[best["name"]]
        result["verilog"] = {k: syn.get(k) for k in ("verilog_acc", "da_est_LUT", "da_est_FF")}
        result["verilog"]["bit_exact_accuracy"] = syn.get("verilog_acc") == acc
        print(f"verilog: acc {syn.get('verilog_acc')}  da_est_LUT {syn.get('da_est_LUT')}  "
              f"da_est_FF {syn.get('da_est_FF')}  matches keras: {syn.get('verilog_acc') == acc}")

    out = save_path / "selected_metrics.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
