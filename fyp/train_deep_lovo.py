"""
train_deep_lovo.py - leave-one-video-out evaluation for the 4 deep temporal
architectures, matching what ml_classifier.py does for the 9 sklearn models.

For each architecture it:
  * trains one model per held-out source video (8 folds),
  * pools the 8 held-out confusion matrices and reports a 4-class macro-F1
    (the empty 5th "Unclassified" head that train_mamba always carries is
    dropped, so the number is directly comparable to the sklearn LOVO 0.51),
  * trains one final checkpoint on ALL clips for deployment.

Usage:
  python fyp/train_deep_lovo.py training_csv_v4 --out-dir deep_results_v4
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# train_mamba's LABEL_NAMES order; first 4 are the real classes.
CLASSES = ["Spacing Breakdown", "Delayed Support", "Coordinated Attack",
           "Coordinated Defense"]
ARCHES = ["inceptiontime", "tcn", "bilstm", "mamba"]
FOLDS = ["(1)", "(2)", "(3)", "(4)", "(5)", "(6)", "(plain)", "usa"]


def _macro_f1_4(cm4: np.ndarray) -> tuple[float, list[float]]:
    tp = np.diag(cm4).astype(float)
    support = cm4.sum(1).astype(float)
    pred = cm4.sum(0).astype(float)
    prec = tp / np.clip(pred, 1, None)
    rec = tp / np.clip(support, 1, None)
    f1 = (2 * prec * rec) / np.clip(prec + rec, 1e-12, None)
    return float(f1.mean()), [round(float(x), 3) for x in f1]


def _parse_confusion(stdout: str) -> np.ndarray | None:
    """Pull the 5x5 held-out confusion tensor out of train_mamba's stdout,
    return the top-left 4x4 (real classes)."""
    m = re.search(r"confusion matrix \(rows=true, cols=pred\):\s*\n\s*tensor\(\[(.+?)\]\)",
                  stdout, re.S)
    if not m:
        return None
    nums = [int(x) for x in re.findall(r"-?\d+", m.group(1))]
    n = int(round(len(nums) ** 0.5))
    if n * n != len(nums):
        return None
    cm = np.array(nums).reshape(n, n)
    return cm[:4, :4]


def _run(cmd: list[str], log_path: str) -> str:
    with open(log_path, "w", encoding="utf-8") as fh:
        p = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=ROOT)
    out = open(log_path, encoding="utf-8").read()
    if p.returncode != 0:
        print(f"    [WARN] exit {p.returncode} - see {log_path}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_dir")
    ap.add_argument("--out-dir", default="deep_results_v4")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--arches", nargs="+", default=ARCHES)
    args = ap.parse_args()

    out = os.path.join(ROOT, args.out_dir)
    logs = os.path.join(out, "logs")
    os.makedirs(logs, exist_ok=True)
    py = sys.executable

    summary: dict = {"csv_dir": args.csv_dir, "architectures": {}}
    t0 = time.time()

    for arch in args.arches:
        print(f"\n{'=' * 60}\n{arch.upper()}\n{'=' * 60}")
        n_layers = "6" if arch == "inceptiontime" else "4"
        pooled = np.zeros((4, 4), dtype=int)
        fold_f1: list[float] = []
        fold_rows = []

        for vid in FOLDS:
            tag = f"{arch}_{vid.strip('()')}"
            ck = os.path.join(logs, f"ck_{tag}.pt")
            lg = os.path.join(logs, f"{tag}.log")
            cmd = [py, "fyp/train_mamba.py", args.csv_dir,
                   "--arch", arch, "--test-video", vid, "--augment",
                   "--n_layers", n_layers, "--epochs", str(args.epochs),
                   "--patience", str(args.patience),
                   "--checkpoint", ck,
                   "--history_csv", os.path.join(logs, f"hist_{tag}.csv")]
            ts = time.time()
            sout = _run(cmd, lg)
            cm4 = _parse_confusion(sout)
            if cm4 is None:
                print(f"  {vid:<9} FAILED (no confusion matrix)")
                continue
            pooled += cm4
            f1, per = _macro_f1_4(cm4)
            fold_f1.append(f1)
            fold_rows.append({"video": vid, "n_test": int(cm4.sum()),
                              "macro_f1_4": round(f1, 3), "per_class_f1": per})
            print(f"  {vid:<9} n={cm4.sum():4d}  macroF1(4)={f1:.3f}  "
                  f"({time.time() - ts:.0f}s)")
            try:
                os.remove(ck)
            except OSError:
                pass

        pooled_f1, pooled_per = _macro_f1_4(pooled)
        acc = float(np.diag(pooled).sum() / max(pooled.sum(), 1))
        print(f"  --> POOLED LOVO  macroF1(4)={pooled_f1:.3f}  acc={acc:.3f}  "
              f"mean+/-SD folds={np.mean(fold_f1):.3f}+/-{np.std(fold_f1):.3f}")

        # Final deployment checkpoint on ALL clips (small random val split).
        final_ck = os.path.join(ROOT, f"mamba_v4_{arch}.pt")
        print(f"  training final checkpoint on all clips -> {os.path.basename(final_ck)}")
        _run([py, "fyp/train_mamba.py", args.csv_dir, "--arch", arch,
              "--augment", "--n_layers", n_layers, "--epochs", str(args.epochs),
              "--patience", str(args.patience), "--val_split", "0.15",
              "--test_split", "0.0", "--checkpoint", final_ck,
              "--history_csv", os.path.join(logs, f"hist_{arch}_final.csv")],
             os.path.join(logs, f"{arch}_final.log"))

        summary["architectures"][arch] = {
            "pooled_lovo_macro_f1_4class": round(pooled_f1, 4),
            "pooled_lovo_accuracy": round(acc, 4),
            "fold_mean_macro_f1_4class": round(float(np.mean(fold_f1)), 4),
            "fold_sd": round(float(np.std(fold_f1)), 4),
            "pooled_per_class_f1": dict(zip(CLASSES, pooled_per)),
            "pooled_confusion": pooled.tolist(),
            "folds": fold_rows,
            "final_checkpoint": os.path.basename(final_ck),
        }
        with open(os.path.join(out, "deep_results.json"), "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)

    print(f"\n{'=' * 60}\nDEEP LOVO SUMMARY  (4-class macro-F1, comparable to sklearn 0.510)\n{'=' * 60}")
    print(f"  {'arch':<16}{'pooledF1':>10}{'foldMean':>10}{'acc':>8}")
    for a, r in summary["architectures"].items():
        print(f"  {a:<16}{r['pooled_lovo_macro_f1_4class']:>10.3f}"
              f"{r['fold_mean_macro_f1_4class']:>10.3f}{r['pooled_lovo_accuracy']:>8.3f}")
    print(f"\n  total wall time: {(time.time() - t0) / 60:.0f} min")
    print(f"  results: {os.path.join(args.out_dir, 'deep_results.json')}")


if __name__ == "__main__":
    main()
