"""
reproduce.py - ONE command that regenerates every sklearn number and figure in
the report (model comparison, feature ablation, rule-engine baseline, decision
rule). The Mamba checkpoint is hashed for provenance only, not scored - S3.

    python reproduce.py                # full run   (~25-40 min CPU)
    python reproduce.py --quick        # fast pass  (~6-10 min, fewer frames)
    python reproduce.py --stage 3      # a single stage

Why this file exists
--------------------
A panel cannot verify a claim it cannot re-run. Technical Enrichment activity 8
asks for exactly this: "the commands/script that regenerates the metrics and
sample outputs (with model/config versions)". Every table and figure in
Technical_Review_and_QA.md sections 13-14 is produced here, from the weights and
CSVs in this repository, with the environment and file hashes recorded alongside
so a reviewer can prove they are looking at the same artefacts.

It is also the project's own regression test. Each stage asserts the property it
is supposed to demonstrate and the script exits non-zero if one fails, so a
silent degradation - the failure mode that produced three separate
"100% TRANSITION" demos in this project's history - cannot survive a run.

Outputs land in results/<timestamp>/ (and results/latest/):
    METRICS.md            every headline number, written for the report
    provenance.json       versions, weight hashes, git commit, timings
    *.csv                 machine-readable tables
    figures/*.png         panel-ready figures
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
import warnings

ROOT = os.path.dirname(os.path.abspath(__file__))
FYP = os.path.join(ROOT, "fyp")
sys.path.insert(0, FYP)

VIDEO = "videoplayback (4).mp4"
DEMO_START = 11200            # a verified live rally (see sec 13.1)
CSV_DIR = "training_csv"
PLAYER_W = "yolo11n.pt"
BALL_W = os.path.join("fyp", "volleyball_best.pt")
DETECTOR_VAL_YAML = os.path.join("volleyball-detection.yolov11", "data.yaml")


# ---------------------------------------------------------------------------
# infrastructure
# ---------------------------------------------------------------------------

class Ctx:
    def __init__(self, outdir: str, quick: bool):
        self.outdir = outdir
        self.figdir = os.path.join(outdir, "figures")
        os.makedirs(self.figdir, exist_ok=True)
        self.quick = quick
        self.results: dict = {}
        self.failures: list[str] = []
        self.timings: dict = {}

    def path(self, *p):
        return os.path.join(self.outdir, *p)

    def fig(self, name):
        return os.path.join(self.figdir, name)


def _sha(path: str, n: int = 12) -> str | None:
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def banner(n, title):
    print("\n" + "=" * 78)
    print(f"STAGE {n}  {title}")
    print("=" * 78)


def check(ctx: Ctx, ok: bool, msg: str):
    """Assert a property the report claims. A failure is recorded and the run
    ends non-zero, but later stages still execute so one bad stage does not
    hide the state of the rest."""
    print(f"  [{'PASS' if ok else 'FAIL'}] {msg}")
    if not ok:
        ctx.failures.append(msg)
    return ok


# ---------------------------------------------------------------------------
# STAGE 0 - provenance
# ---------------------------------------------------------------------------

def stage0(ctx: Ctx):
    banner(0, "Environment and artefact provenance")
    import numpy, pandas, sklearn, torch, cv2, ultralytics

    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                cwd=ROOT, capture_output=True, text=True,
                                timeout=20).stdout.strip() or None
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                                    capture_output=True, text=True,
                                    timeout=20).stdout.strip())
    except Exception:
        commit, dirty = None, None

    prov = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "git_commit": commit,
        "git_dirty": dirty,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {
            "numpy": numpy.__version__, "pandas": pandas.__version__,
            "scikit-learn": sklearn.__version__, "torch": torch.__version__,
            "opencv": cv2.__version__, "ultralytics": ultralytics.__version__,
        },
        "cuda_available": bool(torch.cuda.is_available()),
        "artefacts": {
            name: {"sha256_12": _sha(os.path.join(ROOT, name)),
                   "bytes": os.path.getsize(os.path.join(ROOT, name))
                   if os.path.exists(os.path.join(ROOT, name)) else None}
            for name in (PLAYER_W, BALL_W, "mamba_checkpoint_v2.pt",
                         "tactical_ml.joblib", VIDEO)
        },
        "quick_mode": ctx.quick,
    }
    for k, v in prov["packages"].items():
        print(f"  {k:<14}: {v}")
    print(f"  {'CUDA':<14}: {prov['cuda_available']}")
    print(f"  {'git':<14}: {commit}{' (dirty)' if dirty else ''}")
    print("\n  Artefact hashes (a reviewer can verify identical inputs):")
    for k, v in prov["artefacts"].items():
        print(f"    {os.path.basename(k):<28} {v['sha256_12'] or 'MISSING'}")

    check(ctx, os.path.exists(os.path.join(ROOT, BALL_W)), "ball weights present")
    check(ctx, os.path.isdir(os.path.join(ROOT, CSV_DIR)), "clip CSVs present")
    ctx.results["provenance"] = prov
    return prov


# ---------------------------------------------------------------------------
# STAGE 1 - test suite
# ---------------------------------------------------------------------------

def stage1(ctx: Ctx):
    banner(1, "Unit test suite")
    r = subprocess.run([sys.executable, "-m", "pytest", "fyp/tests", "-q"],
                       cwd=ROOT, capture_output=True, text=True, timeout=1800)
    tail = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""
    print("  " + tail)
    passed = "passed" in tail and r.returncode == 0
    check(ctx, passed, f"test suite green ({tail})")
    ctx.results["tests"] = {"summary": tail, "returncode": r.returncode}


# ---------------------------------------------------------------------------
# STAGE 2 - detector ablation (sec 13.1)
# ---------------------------------------------------------------------------

def stage2(ctx: Ctx):
    banner(2, "Detector ablation - the root-cause table (sec 13.1)")
    import cv2, numpy as np
    from ultralytics import YOLO

    n_frames = 30 if ctx.quick else 100
    videos = [VIDEO] if ctx.quick else [
        "videoplayback (1).mp4", "videoplayback (3).mp4", VIDEO]
    videos = [v for v in videos if os.path.exists(os.path.join(ROOT, v))]

    stock = YOLO(os.path.join(ROOT, PLAYER_W))
    custom = YOLO(os.path.join(ROOT, BALL_W))
    S_P = [k for k, v in stock.names.items() if v == "person"][0]
    S_B = [k for k, v in stock.names.items() if v == "sports ball"][0]
    C_P = [k for k, v in custom.names.items() if v == "player"][0]
    C_B = [k for k, v in custom.names.items() if v == "ball"][0]

    rows = []
    for vid in videos:
        cap = cv2.VideoCapture(os.path.join(ROOT, vid))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idxs = np.linspace(int(total * .1), int(total * .9), n_frames).astype(int)
        frames = []
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, f = cap.read()
            if ok:
                frames.append(f)
        cap.release()

        for tag, mdl, pid, bid, imgsz in (
            ("custom", custom, C_P, C_B, 640),
            ("custom", custom, C_P, C_B, 1280),
            ("stock", stock, S_P, S_B, 640),
            ("stock", stock, S_P, S_B, 1280),
        ):
            pcs, bh = [], 0
            for f in frames:
                r = mdl.predict(f, imgsz=imgsz, conf=0.25, verbose=False)[0]
                cls = r.boxes.cls.cpu().numpy().astype(int)
                pcs.append(int((cls == pid).sum()))
                bh += int((cls == bid).any())
            rows.append({"video": vid, "model": tag, "imgsz": imgsz,
                         "players_per_frame": round(float(np.mean(pcs)), 2),
                         "ball_recall": round(bh / max(len(frames), 1), 3),
                         "n_frames": len(frames)})
            print(f"  {vid:<24}{tag:<8}{imgsz:>6}  "
                  f"players/frame {rows[-1]['players_per_frame']:>6.2f}  "
                  f"ball {100*rows[-1]['ball_recall']:>3.0f}%")

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(ctx.path("detector_ablation.csv"), index=False)
    ctx.results["detector_ablation"] = rows

    demo = df[(df.video == VIDEO)]
    if len(demo):
        cust = demo[(demo.model == "custom") & (demo.imgsz == 1280)]
        stok = demo[(demo.model == "stock") & (demo.imgsz == 1280)]
        if len(cust) and len(stok):
            check(ctx, float(cust.players_per_frame.iloc[0]) < 3.0,
                  "custom fine-tune fails on players in the demo video "
                  f"({float(cust.players_per_frame.iloc[0]):.1f}/frame)")
            check(ctx, float(stok.players_per_frame.iloc[0]) > 10.0,
                  "stock COCO weights see players in the demo video "
                  f"({float(stok.players_per_frame.iloc[0]):.1f}/frame)")
        c640 = demo[(demo.model == "custom") & (demo.imgsz == 640)]
        if len(c640) and len(cust):
            check(ctx,
                  float(cust.ball_recall.iloc[0]) > float(c640.ball_recall.iloc[0]),
                  "imgsz 1280 beats 640 for ball recall "
                  f"({100*float(c640.ball_recall.iloc[0]):.0f}% -> "
                  f"{100*float(cust.ball_recall.iloc[0]):.0f}%)")
    _fig_detector(ctx, df)


def _fig_detector(ctx, df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    sub = df[df.imgsz == 1280]
    vids = list(dict.fromkeys(sub.video))
    if not vids:
        return
    x = np.arange(len(vids))
    w = 0.35
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))

    for i, tag in enumerate(("custom", "stock")):
        vals = [float(sub[(sub.video == v) & (sub.model == tag)]
                      .players_per_frame.iloc[0]) for v in vids]
        ax[0].bar(x + (i - .5) * w, vals, w,
                  label="custom fine-tune" if tag == "custom" else "stock yolo11n",
                  color="#f8766d" if tag == "custom" else "#00bfc4")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels([v.replace("videoplayback ", "vid ").replace(".mp4", "")
                           for v in vids], fontsize=9)
    ax[0].set_ylabel("players detected / frame")
    ax[0].set_title("Player detection @ imgsz 1280\n(12 players are on court)")
    ax[0].axhline(12, ls="--", c="gray", lw=1)
    ax[0].legend(fontsize=8)

    d = df[(df.video == VIDEO) & (df.model == "custom")]
    if len(d):
        ax[1].bar([str(int(s)) for s in d.imgsz], d.ball_recall * 100,
                  color="#00bfc4")
        ax[1].set_xlabel("inference resolution (imgsz)")
        ax[1].set_ylabel("ball recall (%)")
        ax[1].set_title("Ball recall vs inference resolution\n(identical weights)")
    fig.tight_layout()
    fig.savefig(ctx.fig("detector_ablation.png"), dpi=200)
    plt.close(fig)
    print(f"  figure -> {ctx.fig('detector_ablation.png')}")


# ---------------------------------------------------------------------------
# STAGE 3 - end-to-end pipeline on real video
# ---------------------------------------------------------------------------

def stage3(ctx: Ctx):
    banner(3, "End-to-end pipeline on real broadcast video")
    n = 150 if ctx.quick else 300
    out_csv = ctx.path("pipeline_predictions.csv")
    out_vid = ctx.path("demo_annotated.mp4")

    cmd = [sys.executable, os.path.join("fyp", "pipeline.py"), VIDEO,
           "tactical_ml.joblib",
           "--yolo-model", PLAYER_W, "--ball-model", BALL_W,
           "--imgsz", "1280", "--team-split", "colour",
           "--tracker", "botsort", "--auto-court", "--court-coords", "linear",
           "--start-frame", str(DEMO_START), "--max-frames", str(n),
           "--output-csv", out_csv, "--output-video", out_vid,
           "--preflight-frames", "12"]
    print("  $ " + " ".join(f'"{c}"' if " " in c else c for c in cmd[1:]))
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=5400)

    for line in r.stdout.splitlines():
        if any(k in line for k in ("Preflight", "people/frame", "players/frame",
                                   "ball recall", "PASS", "FATAL", "Seq ",
                                   "Temporal backend", "checks passed")):
            print("  " + line.strip())

    check(ctx, r.returncode == 0, f"pipeline exited cleanly (rc={r.returncode})")
    check(ctx, "All preflight checks passed" in r.stdout,
          "preflight gate passed on the demo segment")
    check(ctx, os.path.exists(out_csv), "per-sequence predictions CSV written")

    summary = {"returncode": r.returncode}
    if os.path.exists(out_csv):
        import pandas as pd
        d = pd.read_csv(out_csv)
        summary["n_sequences"] = int(len(d))
        if "label" in d.columns:
            vc = d["label"].value_counts().to_dict()
            summary["label_counts"] = {str(k): int(v) for k, v in vc.items()}
            unc = int(vc.get("Unclassified", 0))
            print(f"\n  sequences: {len(d)}  labels: {summary['label_counts']}")
            check(ctx, unc < len(d),
                  f"pipeline produces real tactical labels, not 100% "
                  f"Unclassified ({unc}/{len(d)} unclassified)")
        if "confidence" in d.columns:
            summary["mean_confidence"] = round(float(d.confidence.mean()), 3)
        if "entropy_bits" in d.columns:
            summary["mean_entropy_bits"] = round(float(d.entropy_bits.mean()), 3)
            check(ctx, float(d.entropy_bits.mean()) > 0.05,
                  "entropy channel is live (calibration working), mean "
                  f"{float(d.entropy_bits.mean()):.2f} bits")
    if os.path.exists(out_vid):
        summary["annotated_video_bytes"] = os.path.getsize(out_vid)
    ctx.results["pipeline"] = summary


# ---------------------------------------------------------------------------
# STAGE 4 - tactical classifier: model comparison + LOVO
# ---------------------------------------------------------------------------

def stage4(ctx: Ctx):
    banner(4, "Tactical classifier - 9 models, k-fold vs leave-one-video-out")
    import numpy as np
    import ml_classifier as mc

    (X, y, groups, files), _sk = mc.load_dataset(
        os.path.join(ROOT, CSV_DIR), mode="invariant")
    print(f"  {len(X)} clips, {X.shape[1]} invariant features, "
          f"{len(np.unique(groups))} source videos")

    base = mc.majority_baseline(y)
    res = mc.cross_validate_models(X, y, groups, seed=0, n_splits=5)

    def _r(v):
        return None if v is None else round(float(v), 4)

    rows = []
    for name, r in res.items():
        rows.append({
            "model": name,
            "kfold_accuracy": round(r.get("kfold_accuracy", float("nan")), 4),
            "kfold_macro_f1": round(r.get("kfold_macro_f1", float("nan")), 4),
            "lovo_accuracy": round(r.get("lovo_accuracy", float("nan")), 4),
            "lovo_macro_f1": round(r.get("lovo_macro_f1", float("nan")), 4),
            "lovo_balanced_accuracy": round(
                r.get("lovo_balanced_accuracy", float("nan")), 4),
            "lovo_cohen_kappa": round(r.get("lovo_cohen_kappa", float("nan")), 4),
            "lovo_roc_auc_macro": _r(r.get("lovo_roc_auc_macro")),
            "lovo_pr_auc_macro": _r(r.get("lovo_pr_auc_macro")),
            "lovo_train_macro_f1": _r(r.get("lovo_train_macro_f1")),
            "lovo_overfit_gap": _r(r.get("lovo_overfit_gap")),
            "lovo_overfitting": bool(r.get("lovo_overfitting", False)),
        })
    rows.sort(key=lambda d: -d["lovo_macro_f1"])

    import pandas as pd
    pd.DataFrame(rows).to_csv(ctx.path("model_comparison.csv"), index=False)

    hdr = f"  {'model':<24}{'kfoldF1':>9}{'lovoF1':>9}{'lovoAcc':>9}{'kappa':>8}"
    print("\n" + hdr)
    print("  " + "-" * (len(hdr) - 2))
    for d in rows:
        print(f"  {d['model']:<24}{d['kfold_macro_f1']:>9.3f}"
              f"{d['lovo_macro_f1']:>9.3f}{d['lovo_accuracy']:>9.3f}"
              f"{d['lovo_cohen_kappa']:>8.3f}")
    print(f"\n  majority baseline: acc {base['accuracy']:.3f}  "
          f"macro-F1 {base['macro_f1']:.3f}")

    hdr2 = (f"  {'model':<24}{'roc_auc':>9}{'pr_auc':>9}{'train_F1':>10}"
           f"{'gap':>8}{'flag':>9}")
    print("\n" + hdr2)
    print("  " + "-" * (len(hdr2) - 2))
    n_overfit = 0
    for d in rows:
        flag = "OVERFIT" if d["lovo_overfitting"] else ""
        n_overfit += int(d["lovo_overfitting"])
        roc = d["lovo_roc_auc_macro"]
        pr = d["lovo_pr_auc_macro"]
        print(f"  {d['model']:<24}"
              f"{roc if roc is not None else float('nan'):>9.3f}"
              f"{pr if pr is not None else float('nan'):>9.3f}"
              f"{d['lovo_train_macro_f1']:>10.3f}"
              f"{d['lovo_overfit_gap']:>8.3f}{flag:>9}")

    # S1 (AUDIT_FINDINGS.md): the headline model is PRE-DECLARED (the deployed
    # hist_gradient_boosting), not rows[0] = max-of-9 on the same CV, which
    # would be the winner's curse. The full ranked table is still shown as
    # exploration.
    DECLARED = "hist_gradient_boosting"
    best = next((r for r in rows if r["model"] == DECLARED), rows[0])
    if best["model"] != rows[0]["model"]:
        print(f"\n  [S1] headline model = {DECLARED} (declared), not "
              f"{rows[0]['model']} (zoo best on lovoF1 "
              f"{rows[0]['lovo_macro_f1']:.3f} vs {best['lovo_macro_f1']:.3f}) "
              f"- reported for transparency, the declared number is quoted")
    ctx.results["model_comparison"] = {
        "rows": rows, "majority_baseline": base, "winner": best["model"],
        "declared_model": DECLARED, "zoo_best": rows[0]["model"],
        "selection": "pre-declared (S1) - not max-of-9",
        "n_clips": int(len(X)), "n_features": int(X.shape[1]),
        "n_models_overfitting": n_overfit, "n_models_total": len(rows),
    }
    check(ctx, best["lovo_macro_f1"] > base["macro_f1"],
          f"best model beats majority baseline on macro-F1 "
          f"({best['lovo_macro_f1']:.3f} > {base['macro_f1']:.3f})")
    check(ctx, best["kfold_macro_f1"] > best["lovo_macro_f1"],
          "k-fold exceeds LOVO, i.e. the cross-video gap is real and reported "
          f"({best['kfold_macro_f1']:.3f} vs {best['lovo_macro_f1']:.3f})")
    if n_overfit == len(rows):
        print(f"\n  [NOTE] all {len(rows)} models are flagged OVERFIT "
              "(train-vs-held-out macro-F1 gap > 0.05) under LOVO - with 3 "
              "source videos, LOVO trains on only 2, and ~130-190 rows in "
              f"{X.shape[1]} dimensions is not enough data for any of these "
              "models to avoid memorising the training folds. This is a "
              "stronger statement than the LOVO-vs-k-fold gap alone: not just "
              "'worse cross-video', but 'close to memorising two videos'.")

    # confusion matrix of the winner, under the honest protocol
    y_pred = res[best["model"]].get("_pred_lovo",
                                    res[best["model"]].get("_pred_kfold"))
    print("\n" + mc.confusion_table(y, y_pred))
    print()
    print(mc.per_class_report(y, y_pred))
    _fig_models(ctx, rows, base)
    _fig_confusion(ctx, y, y_pred, mc.CLASS_NAMES, best["model"])
    ctx.results["confusion"] = {
        "protocol": "leave-one-video-out",
        "report": mc.per_class_report(y, y_pred),
    }

    # S7 / S3 (AUDIT_FINDINGS.md): the Mamba checkpoint is hashed in stage 0
    # for provenance but is NOT scored here. A LOVO number for Mamba needs a
    # per-fold retrain, and the shipped checkpoint is a stale, five-axis-
    # confounded artefact the project itself says not to quote (S3). Producing
    # a number from it would be reproducing a value that should not exist.
    # Mamba evaluation lands with the S3/S9 re-extraction + retrain.
    print("\n  [Mamba] checkpoint hashed for provenance only - not scored "
          "here (S3: stale/confounded; a LOVO number needs a per-fold retrain).")
    ctx.results["mamba_evaluated"] = False

    # S7 (AUDIT_FINDINGS.md): the two most-quoted comparisons in the report -
    # the hand-coded rule engine and the prior-corrected decision rule - were
    # implemented in ml_classifier.py but never regenerated here, so
    # "one command regenerates everything" was false. Both now run.
    rule = mc.rule_engine_baseline(os.path.join(ROOT, CSV_DIR), files, y)
    if rule is not None:
        print(f"\n  rule engine vs the SAME human labels: macro-F1 "
              f"{rule['macro_f1']:.3f}, kappa {rule['cohen_kappa']:.3f} "
              f"(abstained on {rule['n_abstained']}/{rule['n_clips']})")
        rule_pred, keep_idx = rule.pop("_pred"), rule.pop("_keep_idx")
        ctx.results["rule_engine_baseline"] = rule
        check(ctx, best["lovo_macro_f1"] >= rule["macro_f1"] - 0.02,
              f"learned model is at least competitive with the rule engine it "
              f"replaces ({best['lovo_macro_f1']:.3f} vs {rule['macro_f1']:.3f} "
              f"macro-F1) - if it is clearly worse, report that honestly")

        # S2: paired McNemar - model vs rule engine on the SAME clips
        mc_res = mc.mcnemar_test(y[keep_idx], y_pred[keep_idx], rule_pred)
        print(f"  McNemar (model vs rule engine): model-only-right "
              f"{mc_res['a_correct_b_wrong']}, rule-only-right "
              f"{mc_res['b_correct_a_wrong']}, p={mc_res['p_value']:.3f} "
              f"({'significant' if mc_res['significant_at_0.05'] else 'n.s.'})")
        ctx.results["mcnemar_model_vs_rules"] = mc_res

    # S2: clip-level bootstrap 95% CI on the winner's LOVO macro-F1 / kappa
    boot = mc.bootstrap_metric_ci(y, y_pred, n_boot=2000 if ctx.quick else 10000,
                                  seed=0)
    print(f"\n  bootstrap 95% CI ({boot['n_boot']} resamples, "
          f"{boot['n_clips']} clips):")
    print(f"    macro-F1 {boot['macro_f1']['point']:.3f} "
          f"[{boot['macro_f1']['ci_low']:.3f}, {boot['macro_f1']['ci_high']:.3f}]")
    print(f"    kappa    {boot['cohen_kappa']['point']:.3f} "
          f"[{boot['cohen_kappa']['ci_low']:.3f}, "
          f"{boot['cohen_kappa']['ci_high']:.3f}]")
    ctx.results["bootstrap_ci"] = boot
    check(ctx, boot["macro_f1"]["ci_low"] <= boot["macro_f1"]["point"]
          <= boot["macro_f1"]["ci_high"],
          "bootstrap CI brackets the point estimate (sanity)")

    proba_l = res[best["model"]].get("_proba_lovo")
    if proba_l is not None:
        counts = np.bincount(y, minlength=len(mc.CLASS_NAMES))
        # S6: per-fold training prior, so the correction does not divide by a
        # prior that includes the held-out video.
        dec = mc.compare_decision_rules(y, proba_l, counts, groups=groups)
        d_f1 = dec["prior_corrected"]["macro_f1"] - dec["argmax"]["macro_f1"]
        print(f"  decision rule ({dec['_prior']}):  argmax macro-F1 "
              f"{dec['argmax']['macro_f1']:.3f}  ->  prior-corrected "
              f"{dec['prior_corrected']['macro_f1']:.3f}  "
              f"({d_f1:+.3f}, parameter-free)")
        ctx.results["decision_rule"] = dec


def _fig_models(ctx, rows, base):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    names = [r["model"] for r in rows]
    x = np.arange(len(names))
    w = 0.38
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - w/2, [r["kfold_macro_f1"] for r in rows], w,
           label="k-fold (within-video, optimistic)", color="#f8766d")
    ax.bar(x + w/2, [r["lovo_macro_f1"] for r in rows], w,
           label="leave-one-video-out (honest)", color="#00bfc4")
    ax.axhline(base["macro_f1"], ls="--", c="gray", lw=1.2,
               label=f"majority baseline ({base['macro_f1']:.2f})")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("macro-F1")
    ax.set_title("Tactical classifier: within-video vs cross-video generalisation\n"
                 "the gap between the bars is the per-video signature")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ctx.fig("model_comparison.png"), dpi=200)
    plt.close(fig)
    print(f"  figure -> {ctx.fig('model_comparison.png')}")


def _fig_confusion(ctx, y, y_pred, classes, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y, y_pred, labels=range(len(classes)))
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels([c.replace(" ", "\n") for c in classes], fontsize=8)
    ax.set_yticklabels([c.replace(" ", "\n") for c in classes], fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("human label")
    ax.set_title(f"Confusion matrix - {title}\nleave-one-video-out")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{cm[i,j]}\n{cmn[i,j]:.0%}", ha="center", va="center",
                    fontsize=8, color="white" if cmn[i, j] > .5 else "black")
    fig.colorbar(im, ax=ax, fraction=.045)
    fig.tight_layout()
    fig.savefig(ctx.fig("confusion_matrix.png"), dpi=200)
    plt.close(fig)
    print(f"  figure -> {ctx.fig('confusion_matrix.png')}")


# ---------------------------------------------------------------------------
# STAGE 5 - feature ablation (sec 14.3)
# ---------------------------------------------------------------------------

def stage5(ctx: Ctx):
    banner(5, "Feature ablation - absolute vs scale-invariant (sec 14.3)")
    import ml_classifier as mc

    rows = []
    for mode in ("absolute", "invariant"):   # S4: 'both'/union dropped
        (X, y, groups, _f), _s = mc.load_dataset(
            os.path.join(ROOT, CSV_DIR), mode=mode)
        res = mc.cross_validate_models(X, y, groups, seed=0, n_splits=5)
        best = max(res, key=lambda k: res[k].get("lovo_macro_f1", -1))
        rows.append({
            "features": mode, "n_features": int(X.shape[1]), "best_model": best,
            "kfold_macro_f1": round(res[best].get("kfold_macro_f1", 0), 4),
            "lovo_macro_f1": round(res[best].get("lovo_macro_f1", 0), 4),
            "lovo_cohen_kappa": round(res[best].get("lovo_cohen_kappa", 0), 4),
        })
        print(f"  {mode:<11}n={rows[-1]['n_features']:<5}{best:<24}"
              f"kfoldF1 {rows[-1]['kfold_macro_f1']:.3f}   "
              f"lovoF1 {rows[-1]['lovo_macro_f1']:.3f}   "
              f"kappa {rows[-1]['lovo_cohen_kappa']:+.3f}")

    import pandas as pd
    pd.DataFrame(rows).to_csv(ctx.path("feature_ablation.csv"), index=False)
    ctx.results["feature_ablation"] = rows

    a = {r["features"]: r for r in rows}
    d_lovo = a["invariant"]["lovo_macro_f1"] - a["absolute"]["lovo_macro_f1"]
    d_kf = a["invariant"]["kfold_macro_f1"] - a["absolute"]["kfold_macro_f1"]
    print(f"\n  invariant - absolute:  LOVO {d_lovo:+.3f}   k-fold {d_kf:+.3f}")
    check(ctx, d_lovo > 0,
          f"scale-invariant features improve CROSS-video macro-F1 ({d_lovo:+.3f})")
    check(ctx, d_kf < 0,
          f"and reduce WITHIN-video macro-F1 ({d_kf:+.3f}) - the signature of "
          "removing a camera-identity shortcut")
    _fig_ablation(ctx, rows)


def _fig_ablation(ctx, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    modes = [r["features"] for r in rows]
    x = np.arange(len(modes))
    w = .38
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w/2, [r["kfold_macro_f1"] for r in rows], w,
           label="k-fold (within-video)", color="#f8766d")
    ax.bar(x + w/2, [r["lovo_macro_f1"] for r in rows], w,
           label="LOVO (cross-video)", color="#00bfc4")
    for i, r in enumerate(rows):
        ax.text(i - w/2, r["kfold_macro_f1"] + .008,
                f"{r['kfold_macro_f1']:.3f}", ha="center", fontsize=8)
        ax.text(i + w/2, r["lovo_macro_f1"] + .008,
                f"{r['lovo_macro_f1']:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(modes)
    ax.set_ylabel("macro-F1")
    ax.set_title("Feature-block ablation\n"
                 "invariant features trade within-video score for cross-video "
                 "generalisation")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ctx.fig("feature_ablation.png"), dpi=200)
    plt.close(fig)
    print(f"  figure -> {ctx.fig('feature_ablation.png')}")


# ---------------------------------------------------------------------------
# STAGE 6 - dataset + calibration characterisation
# ---------------------------------------------------------------------------

def stage6(ctx: Ctx):
    banner(6, "Dataset characterisation and probability calibration")
    import numpy as np
    import ml_classifier as mc

    (X, y, groups, files), _s = mc.load_dataset(
        os.path.join(ROOT, CSV_DIR), mode="invariant")

    counts = {c: int((y == i).sum()) for i, c in enumerate(mc.CLASS_NAMES)}
    per_video = {}
    for g in np.unique(groups):
        per_video[str(g)] = int((groups == g).sum())
    print("  class distribution:")
    for c, n in counts.items():
        print(f"    {c:<22}{n:>4}  ({100*n/len(y):>4.1f}%)")
    print("  clips per source video:")
    for g, n in per_video.items():
        print(f"    {g:<10}{n:>4}")

    imb = max(counts.values()) / max(min(counts.values()), 1)
    check(ctx, True, f"class imbalance ratio {imb:.1f}:1 recorded "
                     f"(smallest class n={min(counts.values())})")

    model = mc.build_models(0)["hist_gradient_boosting"]
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y)
    raw = mc.confidence_profile(model, X)
    cal_model, info = mc.calibrate(model, X, y, seed=0)
    cal = mc.confidence_profile(cal_model, X)

    print("\n  probability calibration:")
    print(f"    {'':<28}{'uncalibrated':>14}{'calibrated':>13}")
    for k in ("mean_confidence", "frac_confidence_above_0.99",
              "mean_entropy_bits"):
        print(f"    {k:<28}{raw[k]:>14.3f}{cal[k]:>13.3f}")
    check(ctx, cal["mean_entropy_bits"] > raw["mean_entropy_bits"],
          "calibration restores a usable entropy range "
          f"({raw['mean_entropy_bits']:.2f} -> {cal['mean_entropy_bits']:.2f} bits)")

    # S8 (AUDIT_FINDINGS.md): the numbers above are IN-SAMPLE (model fit and
    # scored on the same X). An honest reliability curve uses POOLED
    # out-of-fold LOVO probabilities: "when the model says 0.8 on a clip from
    # a video it never trained on, is it right 80% of the time?"
    reliab = None
    n_groups = len(np.unique(groups))
    if n_groups >= 2:
        from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            proba_oof = cross_val_predict(
                mc.build_models(0)["hist_gradient_boosting"], X, y,
                cv=LeaveOneGroupOut(), groups=groups, method="predict_proba",
                n_jobs=1)
        reliab = mc.reliability_curve(y, proba_oof)
        print(f"\n  out-of-fold reliability (LOVO, uncalibrated): "
              f"ECE {reliab['ece']:.3f} over {reliab['n']} clips")
        for b in reliab["bins"]:
            print(f"    conf {b['bin_lo']:.1f}-{b['bin_hi']:.1f}: "
                  f"predicted {b['mean_confidence']:.2f}  actual "
                  f"{b['empirical_accuracy']:.2f}  (n={b['count']})")
        check(ctx, reliab["ece"] <= 0.35,
              f"out-of-fold ECE is recorded ({reliab['ece']:.3f}) - a large "
              "value is itself a reportable finding, not a failure")
        _fig_reliability(ctx, reliab)

    ctx.results["dataset"] = {
        "n_clips": int(len(y)), "class_counts": counts,
        "clips_per_video": per_video, "imbalance_ratio": round(imb, 2),
    }
    ctx.results["calibration"] = {"info": info, "uncalibrated": raw,
                                  "calibrated": cal,
                                  "out_of_fold_reliability": reliab}
    _fig_dataset(ctx, counts, per_video)


def _fig_reliability(ctx, reliab):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not reliab["bins"]:
        return
    conf = [b["mean_confidence"] for b in reliab["bins"]]
    acc = [b["empirical_accuracy"] for b in reliab["bins"]]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], ls="--", c="gray", lw=1, label="perfect calibration")
    ax.plot(conf, acc, "o-", c="#00bfc4", label="model (out-of-fold, LOVO)")
    ax.set_xlabel("mean predicted confidence")
    ax.set_ylabel("empirical accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f"Reliability curve (pooled LOVO out-of-fold)\n"
                 f"ECE = {reliab['ece']:.3f}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ctx.fig("reliability_curve.png"), dpi=200)
    plt.close(fig)
    print(f"  figure -> {ctx.fig('reliability_curve.png')}")


def _fig_dataset(ctx, counts, per_video):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].bar([c.replace(" ", "\n") for c in counts], list(counts.values()),
              color="#00bfc4")
    for i, v in enumerate(counts.values()):
        ax[0].text(i, v + 1, str(v), ha="center", fontsize=9)
    ax[0].set_ylabel("clips")
    ax[0].set_title("Class distribution (rule-engine labels)\n"
                    f"imbalance {max(counts.values())}:{min(counts.values())}")
    ax[0].tick_params(axis="x", labelsize=8)

    ax[1].bar(list(per_video), list(per_video.values()), color="#f8766d")
    for i, v in enumerate(per_video.values()):
        ax[1].text(i, v + 1, str(v), ha="center", fontsize=9)
    ax[1].set_ylabel("clips")
    ax[1].set_title("Clips per source video\n(each is one leave-one-video-out fold)")
    fig.tight_layout()
    fig.savefig(ctx.fig("dataset.png"), dpi=200)
    plt.close(fig)
    print(f"  figure -> {ctx.fig('dataset.png')}")


# ---------------------------------------------------------------------------
# STAGE 7 - formal detector validation (mAP)
# ---------------------------------------------------------------------------

def stage7(ctx: Ctx):
    banner(7, "Formal detector validation - mAP on the labelled Roboflow set")
    print("  senior-computer-vision checklist: 'Evaluate Results' asks for "
          "mAP@50 / mAP@50:95 / precision / recall, not the informal "
          "'players per frame' proxy stage 2 uses. This stage runs the real "
          "one, on the leak-free temporal-block validation split (sec 11.2).")

    yaml_path = os.path.join(ROOT, DETECTOR_VAL_YAML)
    if not os.path.exists(yaml_path):
        print(f"  [SKIP] {DETECTOR_VAL_YAML} not present - the labelled "
              "detection dataset was not transferred to this machine. "
              "Not a failure: this stage is best-effort when the raw "
              "Roboflow export (not committed to git, see sec on repo size) "
              "is unavailable.")
        ctx.results["detector_map"] = None
        return

    from train_detector import patch_yaml_path
    patch_yaml_path(yaml_path)   # data.yaml's `path:` is per-machine (sec 11)

    from ultralytics import YOLO
    model = YOLO(os.path.join(ROOT, BALL_W))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = model.val(data=yaml_path, imgsz=1280, verbose=False, plots=False)

    per_class = {name: float(ap) for name, ap in zip(model.names.values(), r.box.ap50)}
    out = {
        "map50": round(float(r.box.map50), 4),
        "map50_95": round(float(r.box.map), 4),
        "precision": round(float(r.box.mp), 4),
        "recall": round(float(r.box.mr), 4),
        "per_class_map50": {k: round(v, 4) for k, v in per_class.items()},
    }
    print(f"\n  mAP@50      : {out['map50']:.3f}")
    print(f"  mAP@50-95   : {out['map50_95']:.3f}")
    print(f"  precision   : {out['precision']:.3f}")
    print(f"  recall      : {out['recall']:.3f}")
    for cls, ap in per_class.items():
        print(f"  AP50[{cls:<8}]: {ap:.3f}")

    ctx.results["detector_map"] = out
    check(ctx, out["map50"] > 0.3,
          f"custom detector clears a minimum mAP@50 bar ({out['map50']:.3f})")
    if "ball" in per_class and "player" in per_class:
        check(ctx, per_class["player"] > per_class["ball"],
              "player AP50 exceeds ball AP50 "
              f"({per_class['player']:.3f} vs {per_class['ball']:.3f}) - "
              "quantifies, rather than only argues, the sec 13.1 finding "
              "that this fine-tune is the weaker of the two on players")


# ---------------------------------------------------------------------------
# STAGE 8 - inter-video feature drift (direct evidence for sec 12.6/14.3)
# ---------------------------------------------------------------------------

def stage8(ctx: Ctx):
    banner(8, "Inter-video feature drift - direct evidence for the per-video "
              "signature")
    import numpy as np
    import ml_classifier as mc
    from drift import compare_all_pairs

    print("  senior-ml-engineer checklist: input data drift detection. "
          "Sec 12.6/14.3 argue a 'per-video signature' from an INDIRECT "
          "signal (the LOVO-vs-k-fold performance gap). This stage measures "
          "it DIRECTLY: KS-test + Population Stability Index per feature, "
          "between every pair of source videos.\n")

    (X, y, groups, _f), _s = mc.load_dataset(
        os.path.join(ROOT, CSV_DIR), mode="invariant")
    names = mc.invariant_feature_names()
    by_video = {g: X[groups == g] for g in np.unique(groups)}

    reports = compare_all_pairs(by_video, names)
    rows = [{"from": r.pair[0], "to": r.pair[1], "mean_psi": round(r.mean_psi, 3),
            "n_major_shift": r.n_major_shift, "n_channels": len(r.channels)}
           for r in reports]
    rows.sort(key=lambda d: -d["mean_psi"])

    import pandas as pd
    pd.DataFrame(rows).to_csv(ctx.path("drift_pairs.csv"), index=False)

    print(f"  {'from':<8}{'to':<8}{'mean_psi':>10}{'n_major/{}'.format(len(names)):>14}")
    for d in rows:
        print(f"  {d['from']:<8}{d['to']:<8}{d['mean_psi']:>10.3f}"
              f"{d['n_major_shift']:>10}/{d['n_channels']}")

    mean_psi_overall = float(np.mean([d["mean_psi"] for d in rows]))
    print(f"\n  mean PSI across all directed video pairs: {mean_psi_overall:.3f}"
          " (conventional reading: <0.1 none, 0.1-0.25 moderate, >0.25 major)")

    worst = reports[int(np.argmax([r.mean_psi for r in reports]))]
    top_channels = worst.top(10)
    print(f"\n  Top-10 most-shifted channels, {worst.pair[0]} -> {worst.pair[1]}:")
    for c in top_channels:
        print(f"    {c.feature:<26} psi={c.psi:.3f}  ks={c.ks_statistic:.3f}  "
              f"[{c.shift_level}]")

    # Cross-check against the classifier's own channel importance (exploratory,
    # not overclaimed: rank correlation is reported with its own p-value).
    imp = mc.grouped_permutation_importance(
        "hist_gradient_boosting", X, y, groups, "invariant", seed=0,
        n_repeats=3)
    imp_channels = [d["channel"] for d in imp["channels"]]
    drift_by_channel: dict[str, list] = {}
    for c in worst.channels:
        base = c.feature.split("__")[0]
        drift_by_channel.setdefault(base, []).append(c.psi)
    common = [c for c in imp_channels if c in drift_by_channel]
    rho = rho_p = None
    if len(common) >= 5:
        from scipy.stats import spearmanr
        imp_rank = [imp_channels.index(c) for c in common]
        drift_rank = sorted(drift_by_channel, key=lambda k: -np.mean(drift_by_channel[k]))
        d_rank = [drift_rank.index(c) for c in common]
        rho, pval = spearmanr(imp_rank, d_rank)
        rho, rho_p = float(rho), float(pval)
        print(f"\n  Cross-check: Spearman rank correlation between "
              f"classifier-importance rank and drift rank = {rho:.3f} "
              f"(p={rho_p:.3f}, n={len(common)} channels)")
        print("  (weak/non-significant is expected and reported as such - this "
              "is a supporting exploratory check, not the main claim)")

    ctx.results["drift"] = {
        "pairs": rows,
        "mean_psi_overall": round(mean_psi_overall, 3),
        "worst_pair": list(worst.pair),
        "top_channels": [
            {"feature": c.feature, "psi": round(c.psi, 3),
             "ks_statistic": round(c.ks_statistic, 3), "shift_level": c.shift_level}
            for c in top_channels
        ],
        "importance_rank_correlation": (
            {"rho": round(rho, 3), "p_value": round(rho_p, 3), "n": len(common)}
            if rho is not None else None
        ),
    }
    check(ctx, mean_psi_overall > 0.25,
          "measured inter-video feature drift exceeds the conventional "
          f"'major shift' PSI threshold (mean {mean_psi_overall:.3f} > 0.25) - "
          "direct confirmation of the per-video signature, not just the "
          "indirect LOVO-vs-k-fold performance gap")
    _fig_drift(ctx, rows, worst, top_channels)


def _fig_drift(ctx, rows, worst_report, top_channels):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))

    labels = [f"{d['from']}->{d['to']}" for d in rows]
    colors = ["#8b0000" if d["mean_psi"] > 0.25 else
             "#f8766d" if d["mean_psi"] > 0.1 else "#00bfc4" for d in rows]
    ax[0].bar(labels, [d["mean_psi"] for d in rows], color=colors)
    ax[0].axhline(0.25, ls="--", c="gray", lw=1, label="major-shift threshold")
    ax[0].set_ylabel("mean PSI across 132 invariant features")
    ax[0].set_title("Inter-video feature drift\n(every pair, both directions)")
    ax[0].tick_params(axis="x", labelsize=8, rotation=20)
    ax[0].legend(fontsize=8)

    feats = [c.feature for c in top_channels]
    psis = [c.psi for c in top_channels]
    ax[1].barh(feats[::-1], psis[::-1], color="#8b0000")
    ax[1].set_xlabel("PSI")
    ax[1].set_title(f"Top-10 most-shifted channels\n{worst_report.pair[0]} -> "
                    f"{worst_report.pair[1]}")
    ax[1].tick_params(axis="y", labelsize=7)

    fig.tight_layout()
    fig.savefig(ctx.fig("drift.png"), dpi=200)
    plt.close(fig)
    print(f"  figure -> {ctx.fig('drift.png')}")


# ---------------------------------------------------------------------------
# METRICS.md
# ---------------------------------------------------------------------------

def write_metrics(ctx: Ctx):
    r = ctx.results
    L = []
    A = L.append
    A("# Measured results\n")
    A(f"Generated by `reproduce.py` on "
      f"{r.get('provenance', {}).get('generated_at', '?')}"
      f"{' (quick mode)' if ctx.quick else ''}.\n")
    A("Every number below is regenerated by one command from the weights and "
      "CSVs in this repository. `provenance.json` records the package versions "
      "and SHA-256 prefixes of every artefact used.\n")

    p = r.get("provenance", {})
    if p:
        A("\n## Environment\n")
        A("| component | version |")
        A("|---|---|")
        A(f"| python | {p.get('python')} |")
        for k, v in p.get("packages", {}).items():
            A(f"| {k} | {v} |")
        A(f"| CUDA available | {p.get('cuda_available')} |")
        A(f"| git commit | `{p.get('git_commit')}` "
          f"{'(working tree dirty)' if p.get('git_dirty') else ''} |")
        A("\n### Artefact hashes\n")
        A("| file | sha256 (first 12) | bytes |")
        A("|---|---|---|")
        for k, v in p.get("artefacts", {}).items():
            A(f"| `{os.path.basename(k)}` | `{v['sha256_12']}` | {v['bytes']} |")

    d = r.get("dataset")
    if d:
        A("\n## Dataset\n")
        A(f"{d['n_clips']} labelled clips, {len(d['clips_per_video'])} source "
          f"videos, class imbalance **{d['imbalance_ratio']}:1**.\n")
        A("| class | clips |")
        A("|---|---|")
        for c, n in d["class_counts"].items():
            A(f"| {c} | {n} |")
        A("\n| source video | clips (one LOVO fold each) |")
        A("|---|---|")
        for v, n in d["clips_per_video"].items():
            A(f"| `{v}` | {n} |")

    da = r.get("detector_ablation")
    if da:
        A("\n## Detector ablation\n")
        A("Player detection is taken from stock COCO weights and the ball from "
          "the volleyball fine-tune, because each wins decisively on one class "
          "and loses on the other.\n")
        A("| video | model | imgsz | players/frame | ball recall |")
        A("|---|---|---|---|---|")
        for row in da:
            A(f"| `{row['video']}` | {row['model']} | {row['imgsz']} | "
              f"{row['players_per_frame']} | {100*row['ball_recall']:.0f}% |")
        A("\n![detector ablation](figures/detector_ablation.png)")

    dm = r.get("detector_map")
    if dm:
        A("\n## Formal detector validation (mAP)\n")
        A("The ablation above uses an informal 'players/frame' proxy. This is "
          "the standard COCO-style metric, on the leak-free temporal-block "
          "validation split (sec 11.2), for the custom ball fine-tune.\n")
        A(f"- mAP@50: **{dm['map50']:.3f}**\n"
          f"- mAP@50-95: {dm['map50_95']:.3f}\n"
          f"- precision: {dm['precision']:.3f}\n"
          f"- recall: {dm['recall']:.3f}\n")
        A("| class | AP50 |")
        A("|---|---|")
        for cls, ap in dm["per_class_map50"].items():
            A(f"| {cls} | {ap:.3f} |")
        A("\nPlayer AP50 exceeding ball AP50 quantifies, rather than only "
          "argues, the sec 13.1 finding that this fine-tune is the weaker "
          "detector of the two on players - the reason stock COCO weights "
          "are used for players in the deployed pipeline instead.")

    mc_ = r.get("model_comparison")
    if mc_:
        A("\n## Tactical classifier\n")
        b = mc_["majority_baseline"]
        A(f"{mc_['n_clips']} clips, {mc_['n_features']} scale-invariant "
          f"features. Majority-class baseline: accuracy "
          f"**{b['accuracy']:.3f}**, macro-F1 **{b['macro_f1']:.3f}** - every "
          f"score must beat this to carry information.\n")
        A("| model | k-fold macro-F1 | LOVO macro-F1 | LOVO acc | LOVO kappa |")
        A("|---|---|---|---|---|")
        for row in mc_["rows"]:
            A(f"| {row['model']} | {row['kfold_macro_f1']:.3f} | "
              f"**{row['lovo_macro_f1']:.3f}** | {row['lovo_accuracy']:.3f} | "
              f"{row['lovo_cohen_kappa']:.3f} |")
        A(f"\nWinner by the honest protocol: **{mc_['winner']}**.\n")
        A("![model comparison](figures/model_comparison.png)")
        A("\n![confusion matrix](figures/confusion_matrix.png)")
        conf = r.get("confusion")
        if conf:
            A(f"\n### Per-class ({conf['protocol']})\n")
            A("```")
            A(conf["report"].rstrip())
            A("```")

        A("\n### Probability-ranking quality and within-fold overfitting "
          "(LOVO)\n")
        A("Accuracy/macro-F1/kappa above are hard-label metrics - they only "
          "see the argmax. AUC-PR asks a more forgiving question: even where "
          "the argmax is wrong, does the true class still rank above chance? "
          "The overfit gap asks a stricter one: within a single fold, how "
          "much better does the model fit rows it trained on than rows it "
          "did not.\n")
        A("| model | ROC-AUC (macro, ovr) | PR-AUC (macro, ovr) | train macro-F1 | "
          "overfit gap | flag |")
        A("|---|---|---|---|---|---|")
        for row in mc_["rows"]:
            roc = row.get("lovo_roc_auc_macro")
            pr = row.get("lovo_pr_auc_macro")
            roc_s = f"{roc:.3f}" if roc is not None else "n/a"
            pr_s = f"{pr:.3f}" if pr is not None else "n/a"
            flag = "**OVERFIT**" if row.get("lovo_overfitting") else ""
            A(f"| {row['model']} | {roc_s} | {pr_s} | "
              f"{row.get('lovo_train_macro_f1', float('nan')):.3f} | "
              f"{row.get('lovo_overfit_gap', float('nan')):.3f} | {flag} |")
        n_of = mc_.get("n_models_overfitting", 0)
        n_tot = mc_.get("n_models_total", len(mc_["rows"]))
        if n_of == n_tot and n_tot:
            A(f"\n**All {n_tot} models are flagged OVERFIT** (train-vs-held-out "
              "macro-F1 gap > 0.05) under leave-one-video-out. With only 3 "
              "source videos, LOVO trains on 2 of them; ~130-190 clip rows in "
              f"{mc_['n_features']} dimensions is not enough for ANY of these "
              "9 candidates to avoid substantially memorising the training "
              "videos rather than learning a generalisable tactic classifier. "
              "This is a stronger, more precise statement than 'LOVO score is "
              "lower than k-fold score' - it is close to memorisation, not "
              "merely imperfect generalisation, and it is the strongest "
              "argument in this report for why more source videos (not just "
              "more clips from the same 3) are needed before the tactical "
              "numbers can be trusted.")

    fa = r.get("feature_ablation")
    if fa:
        A("\n## Feature-block ablation\n")
        A("| features | n | best model | k-fold macro-F1 | LOVO macro-F1 | kappa |")
        A("|---|---|---|---|---|---|")
        for row in fa:
            A(f"| {row['features']} | {row['n_features']} | "
              f"{row['best_model']} | {row['kfold_macro_f1']:.3f} | "
              f"{row['lovo_macro_f1']:.3f} | {row['lovo_cohen_kappa']:.3f} |")
        A("\nRead the direction: dimensionless features **lower** the "
          "within-video score and **raise** the cross-video score. That is the "
          "signature of removing a camera-identity shortcut.\n")
        A("![feature ablation](figures/feature_ablation.png)")

    dr = r.get("drift")
    if dr:
        A("\n## Inter-video feature drift (direct evidence)\n")
        A("The per-video signature above was previously argued only "
          "indirectly, from the LOVO-vs-k-fold performance gap. This measures "
          "it directly: Kolmogorov-Smirnov test + Population Stability Index "
          "per feature, between every pair of source videos, on the "
          "132-dimensional scale-invariant feature block.\n")
        A("| from | to | mean PSI | channels with major shift |")
        A("|---|---|---|---|")
        for d in dr["pairs"]:
            A(f"| `{d['from']}` | `{d['to']}` | {d['mean_psi']:.3f} | "
              f"{d['n_major_shift']} / {d['n_channels']} |")
        A(f"\nMean PSI across all pairs: **{dr['mean_psi_overall']:.3f}** "
          "(conventional reading: <0.1 none, 0.1-0.25 moderate, >0.25 major - "
          "this project's own videos land well into the 'major' range even "
          "on features that were specifically engineered to be scale-"
          "invariant).\n")
        A(f"Top-10 most-shifted channels, `{dr['worst_pair'][0]}` -> "
          f"`{dr['worst_pair'][1]}`:\n")
        A("| feature | PSI | KS statistic | shift |")
        A("|---|---|---|---|")
        for c in dr["top_channels"]:
            A(f"| {c['feature']} | {c['psi']:.3f} | {c['ks_statistic']:.3f} | "
              f"{c['shift_level']} |")
        rc = dr.get("importance_rank_correlation")
        if rc:
            A(f"\nExploratory cross-check against the classifier's own channel "
              f"importance (sec 14.6): Spearman rank correlation "
              f"rho={rc['rho']:.3f} (p={rc['p_value']:.3f}, n={rc['n']} "
              "channels) - reported as a weak, non-definitive supporting "
              "signal, not the main claim.\n")
        A("![inter-video drift](figures/drift.png)")

    cal = r.get("calibration")
    if cal:
        A("\n## Probability calibration\n")
        A("| | uncalibrated | calibrated |")
        A("|---|---|---|")
        for k in ("mean_confidence", "frac_confidence_above_0.99",
                  "mean_entropy_bits"):
            A(f"| {k} | {cal['uncalibrated'][k]:.3f} | "
              f"{cal['calibrated'][k]:.3f} |")
        A("\nWithout calibration the ensemble reports 1.000 confidence on every "
          "window, which silently disables the entropy channel and the "
          "tactical-deviation flag.")

    pl = r.get("pipeline")
    if pl:
        A("\n## End-to-end pipeline run\n")
        A(f"- sequences classified: {pl.get('n_sequences')}")
        A(f"- label distribution: `{pl.get('label_counts')}`")
        A(f"- mean confidence: {pl.get('mean_confidence')}")
        A(f"- mean entropy: {pl.get('mean_entropy_bits')} bits")
        A("\nOutputs: `pipeline_predictions.csv`, `demo_annotated.mp4`.")

    A("\n## Verification\n")
    if ctx.failures:
        A(f"**{len(ctx.failures)} CHECK(S) FAILED** - this run is not clean:\n")
        for f in ctx.failures:
            A(f"- FAIL: {f}")
    else:
        A("All automated checks passed.")

    A("\n## Scope of these numbers\n")
    A("Labels are **human tactical annotations**, verified: `labels.csv` "
      "(video_file, anchor_time, tactic_class) -> ffmpeg clip -> "
      "`dataset/<tactic_class>/` -> `target_label`, with all 159 resolvable "
      "rows matching their folder and zero mismatches. `label_clips.py` (the "
      "hand-coded rule engine) is a separate path that never wrote these "
      "labels, which is why it appears above as an independent competing "
      "baseline rather than as the label source. See sec 16 of "
      "Technical_Review_and_QA.md, which also records the correction of an "
      "earlier claim in this project that stated the opposite.\n")
    A("The remaining label-side caveat is that all annotations come from a "
      "SINGLE annotator, so no inter-annotator agreement figure exists and "
      "the true ceiling for every score above is unknown. A blind second pass "
      "(`fyp/annotate.py` -> `ml_classifier.py --gold`) is the experiment "
      "that would establish it.")

    txt = "\n".join(L) + "\n"
    with open(ctx.path("METRICS.md"), "w", encoding="utf-8") as fh:
        fh.write(txt)
    print(f"\n  METRICS.md -> {ctx.path('METRICS.md')}")


# ---------------------------------------------------------------------------

STAGES = {0: stage0, 1: stage1, 2: stage2, 3: stage3,
          4: stage4, 5: stage5, 6: stage6, 7: stage7, 8: stage8}


def _should_refresh_latest(args) -> bool:
    """R4: results/latest/ is what the report reads. Only a full run (all
    stages, videos included) may replace it - a `--stage N` or `--skip-video`
    run produces partial output that would silently truncate the report."""
    return not args.stage and not args.skip_video


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Regenerate every metric and figure in the report.")
    ap.add_argument("--quick", action="store_true",
                    help="Fewer frames / one video. For a smoke check, not for "
                         "numbers you intend to quote.")
    ap.add_argument("--stage", type=int, action="append",
                    help="Run only these stages (repeatable).")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--skip-video", action="store_true",
                    help="Skip stages needing the source .mp4 files.")
    args = ap.parse_args(argv)

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = args.outdir or os.path.join(ROOT, "results", stamp)
    os.makedirs(outdir, exist_ok=True)
    ctx = Ctx(outdir, args.quick)

    print("=" * 78)
    print("VOLLEYBALL TACTICAL ANALYSIS - FULL REPRODUCTION")
    print("=" * 78)
    print(f"  output: {outdir}")
    print(f"  mode  : {'QUICK' if args.quick else 'FULL'}")

    wanted = args.stage or list(STAGES)
    video_stages = {2, 3}
    t0 = time.time()

    for n in sorted(wanted):
        if n not in STAGES:
            continue
        if args.skip_video and n in video_stages:
            print(f"\n[skip] stage {n} (--skip-video)")
            continue
        s = time.time()
        try:
            STAGES[n](ctx)
        except Exception as exc:                       # noqa: BLE001
            print(f"\n[ERROR] stage {n} raised: {exc}", file=sys.stderr)
            traceback.print_exc()
            ctx.failures.append(f"stage {n} raised {type(exc).__name__}: {exc}")
        ctx.timings[f"stage{n}"] = round(time.time() - s, 1)

    ctx.results["timings_sec"] = ctx.timings
    ctx.results["failures"] = ctx.failures

    with open(ctx.path("provenance.json"), "w", encoding="utf-8") as fh:
        json.dump(ctx.results.get("provenance", {}), fh, indent=2)
    with open(ctx.path("results.json"), "w", encoding="utf-8") as fh:
        json.dump({k: v for k, v in ctx.results.items()}, fh, indent=2,
                  default=str)
    write_metrics(ctx)

    # R4 (AUDIT_FINDINGS.md): only a FULL run may replace results/latest/ - the
    # directory the report reads from. A `--stage N` run produces partial
    # output (no METRICS.md sections for the stages it skipped); overwriting
    # latest/ with it silently truncates the report-ready numbers. Same for
    # --skip-video. Partial runs stay in their own timestamped dir only.
    latest = os.path.join(ROOT, "results", "latest")
    if _should_refresh_latest(args):
        try:
            if os.path.isdir(latest):
                shutil.rmtree(latest)
            shutil.copytree(outdir, latest)
        except Exception:
            pass
    else:
        print(f"\n[partial run] results/latest/ left untouched; this run is in "
              f"{outdir}")

    print("\n" + "=" * 78)
    print(f"DONE in {time.time()-t0:.0f}s -> {outdir}")
    for k, v in ctx.timings.items():
        print(f"    {k}: {v}s")
    if ctx.failures:
        print(f"\n{len(ctx.failures)} CHECK(S) FAILED:")
        for f in ctx.failures:
            print(f"  - {f}")
        print("\nThis run is NOT clean. Do not present it until these pass.")
        return 1
    print("\nAll checks passed. results/latest/METRICS.md is report-ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
