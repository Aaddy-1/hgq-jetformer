"""What does cropping the particle axis actually cost?

`--num_particles N` selects a dataset directory; the builder produces it with
`ak.pad_none(a, maxlen, clip=True)`, i.e. a PREFIX crop with zero padding
(`src/data/build_jetclass_dataset.py:42,107`). So the N-particle dataset is
exactly `X_128[:, :N, :]` and this script measures, from the existing 128
dataset, what that prefix throws away.

Three questions, in order of importance:

  1. Are constituents pT-ordered? A prefix crop is only cheap if they are.
     If they are not, cropping keeps an arbitrary subset and everything below
     is moot. MEASURED HERE, not assumed.
  2. How much of the jet's pT survives the leading N? This is the physically
     meaningful loss, not the particle count.
  3. Does the loss fall unevenly across classes? Truncation that mostly hurts
     the already-weak high-multiplicity classes (t_bqq, H_4q) is a worse trade
     than the average suggests.

The h5 stores RAW features; z-scoring happens in the generator
(`src/data/dataset.py:215`), so raw column semantics apply here:

    col  2 = part_logptrel = log(part_pt / jet_pt)  -> pt fraction = exp(col2)
    cols 6..10 = isChargedHadron, isNeutralHadron, isPhoton, isElectron, isMuon
                 one-hot over real particles, all-zero on padding
                 -> sum over 6..10 > 0.5 is an EXACT real-constituent mask

Read-only. Writes nothing.

Usage:
    python scripts/particle_multiplicity.py                       # default paths
    python scripts/particle_multiplicity.py --h5 <path> --max_samples 200000
    python scripts/particle_multiplicity.py --dump out.npz        # + arrays for plotting
"""

import argparse
import os

import h5py
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

JETCLASS_CLASSES = [
    "QCD", "H_bb", "H_cc", "H_gg", "H_4q",
    "H_qql", "Z_qq", "W_qq", "t_bqq", "t_bl",
]

MASK_COLS = slice(6, 11)   # the five particle-type one-hots
LOGPTREL_COL = 2

CROPS = [8, 16, 24, 32, 48, 64, 96, 128]


def load(h5_path, max_samples):
    """Stride the whole file uniformly rather than taking a prefix.

    The h5 is written class by class, so `[:max_samples]` returns a single
    class: every multiplicity statistic below silently becomes that one class's
    and Q3 collapses to a single row. Striding by total//max_samples reads the
    same number of jets at the same cost but covers all ten classes evenly.
    """
    with h5py.File(h5_path, "r") as f:
        total = f["particle_features"].shape[0]
        step = max(1, total // max_samples)
        x = f["particle_features"][::step][:max_samples]
        y = f["label"][::step][:max_samples]
    return x, y, total, step


def dump_arrays(path, mult, ptfrac, y, h5_path, total, step, frac_sorted, pair_frac):
    """Write what a plot needs, at every cut rather than only the CROPS rows.

    The console tables above sample eight crop values; a cumulative-retention
    curve needs all of them, so `retention_*` is indexed by cut 0..128 with
    column c holding the leading-c result (column 0 is the empty crop). Per-class
    rows are included so a class breakdown never costs a second pass over the h5.
    """
    total_pt = ptfrac.sum(axis=1)
    cum = np.cumsum(ptfrac, axis=1)
    r = np.divide(cum, total_pt[:, None], out=np.ones_like(cum), where=total_pt[:, None] > 0)
    r = np.concatenate([np.zeros((r.shape[0], 1), dtype=r.dtype), r], axis=1)   # (N, 129)

    n_cls = len(JETCLASS_CLASSES)
    mult_hist_by_class = np.zeros((n_cls, 129), dtype=np.int64)
    retention_by_class = np.zeros((n_cls, 129), dtype=np.float64)
    for i in range(n_cls):
        sel = y == i
        if not sel.any():
            continue
        mult_hist_by_class[i] = np.bincount(mult[sel], minlength=129)[:129]
        retention_by_class[i] = r[sel].mean(axis=0)

    np.savez_compressed(
        path,
        mult=mult.astype(np.int16),
        label=y.astype(np.int8),
        mult_hist=np.bincount(mult, minlength=129)[:129],
        mult_hist_by_class=mult_hist_by_class,
        retention_mean=r.mean(axis=0),
        retention_p1=np.percentile(r, 1, axis=0),
        retention_p5=np.percentile(r, 5, axis=0),
        retention_by_class=retention_by_class,
        classes=np.array(JETCLASS_CLASSES),
        source_h5=np.array(h5_path),
        total_rows=np.array(total),
        stride=np.array(step),
        frac_jets_sorted=np.array(frac_sorted),
        frac_pairs_sorted=np.array(pair_frac),
    )
    print(f"wrote {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--h5", type=str, default=None,
                   help="path to a 128-particle h5 (default: jetclass/128/17f/test.h5)")
    p.add_argument("--max_samples", type=int, default=200000,
                   help="samples to read (default: 200,000 — plenty for these statistics)")
    p.add_argument("--dump", type=str, default=None,
                   help="also write the underlying arrays to this .npz, for plotting")
    args = p.parse_args()

    h5_path = args.h5 or os.path.join(PROCESSED_DIR, "jetclass", "128", "17f", "test.h5")
    print(f"reading {h5_path}")
    x, y, total, step = load(h5_path, args.max_samples)
    n_jets, n_part, n_feat = x.shape
    print(f"{n_jets:,} jets x {n_part} particles x {n_feat} features (raw, unnormalized)")
    print(f"sampled every {step} of {total:,} rows\n")

    # The h5 is class-ordered, so confirm the stride actually spanned it. A
    # sample missing classes makes Q2/Q3 that subset's statistics, not JetClass's.
    counts = [int((y == i).sum()) for i in range(len(JETCLASS_CLASSES))]
    print("  class coverage: " +
          "  ".join(f"{n}={c:,}" for n, c in zip(JETCLASS_CLASSES, counts)))
    missing = [n for n, c in zip(JETCLASS_CLASSES, counts) if c == 0]
    if missing:
        print(f"  WARNING: no jets for {', '.join(missing)} -- Q2/Q3 below are "
              f"NOT representative. Raise --max_samples.")
    print()

    mask = x[:, :, MASK_COLS].sum(axis=2) > 0.5          # (N, 128) exact real-particle mask
    mult = mask.sum(axis=1)                              # (N,) constituents per jet
    ptfrac = np.where(mask, np.exp(x[:, :, LOGPTREL_COL]), 0.0)   # (N, 128) pt / jet_pt

    # ---- Q1: are constituents pT-ordered? -------------------------------------
    print("=" * 72)
    print("Q1. pT ordering (a prefix crop is only cheap if constituents are sorted)")
    print("=" * 72)
    valid = ptfrac[:, :-1] > 0
    nonincreasing = (ptfrac[:, :-1] >= ptfrac[:, 1:] - 1e-9) | ~valid
    frac_sorted = nonincreasing.all(axis=1).mean()
    # a per-adjacent-pair view, so a few ties do not mask a real ordering
    pairs = valid.sum()
    pair_ok = (nonincreasing & valid).sum()
    print(f"  jets fully non-increasing in pT : {100 * frac_sorted:.2f}%")
    print(f"  adjacent pairs non-increasing   : {100 * pair_ok / pairs:.2f}%  ({pair_ok:,}/{pairs:,})")
    if frac_sorted > 0.99:
        print("  -> SORTED. A prefix crop keeps the leading-pT constituents.")
    elif pair_ok / pairs > 0.9:
        print("  -> MOSTLY sorted. Prefix crop is approximately leading-pT.")
    else:
        print("  -> NOT SORTED. A prefix crop keeps an ARBITRARY subset;")
        print("     everything below understates the true cost. Stop and re-sort")
        print("     the dataset by pT before treating --num_particles as a lever.")
    print()

    # ---- Q2: multiplicity and retained pT -------------------------------------
    print("=" * 72)
    print("Q2. What a prefix crop keeps")
    print("=" * 72)
    qs = [1, 5, 25, 50, 75, 95, 99]
    print("  constituent multiplicity percentiles: " +
          "  ".join(f"p{q}={np.percentile(mult, q):.0f}" for q in qs))
    print(f"  mean {mult.mean():.1f}   max {mult.max()}   "
          f"jets hitting the 128 cap: {100 * (mult >= 128).mean():.2f}%\n")

    total_pt = ptfrac.sum(axis=1)
    print(f"  {'crop N':>7}{'jets fully kept':>18}{'mean pT kept':>15}{'p1 pT kept':>13}{'particles lost':>16}")
    for c in CROPS:
        kept_pt = ptfrac[:, :c].sum(axis=1)
        r = np.divide(kept_pt, total_pt, out=np.ones_like(kept_pt), where=total_pt > 0)
        print(f"  {c:>7}{100 * (mult <= c).mean():>17.2f}%{100 * r.mean():>14.3f}%"
              f"{100 * np.percentile(r, 1):>12.2f}%{np.maximum(mult - c, 0).mean():>15.2f}")
    print()

    # ---- Q3: per class ---------------------------------------------------------
    print("=" * 72)
    print("Q3. Per class — does truncation fall unevenly?")
    print("=" * 72)
    shown = [c for c in CROPS if c <= 64]
    print(f"  {'class':<8}{'n':>8}{'mean mult':>11}{'p95':>6}" +
          "".join(f"{('pT@' + str(c)):>10}" for c in shown))
    for i, name in enumerate(JETCLASS_CLASSES):
        sel = y == i
        if not sel.any():
            continue
        m, pf = mult[sel], ptfrac[sel]
        tot = pf.sum(axis=1)
        cells = ""
        for c in shown:
            r = np.divide(pf[:, :c].sum(axis=1), tot, out=np.ones(sel.sum()), where=tot > 0)
            cells += f"{100 * r.mean():>9.2f}%"
        print(f"  {name:<8}{sel.sum():>8,}{m.mean():>11.1f}{np.percentile(m, 95):>6.0f}{cells}")
    print("\n  pT@N = mean fraction of jet pT retained in the leading N constituents.")
    print("  Compare the weakest classes (H_4q, t_bqq, H_gg) against QCD: if they")
    print("  lose materially more, the crop is a worse trade than the average shows.")

    if args.dump:
        print()
        dump_arrays(args.dump, mult, ptfrac, y, h5_path, total, step,
                    frac_sorted, pair_ok / pairs)


if __name__ == "__main__":
    main()
