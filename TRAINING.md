# Training runbook

Exact commands, in order, with the constraints that make each one correct.
Read §0 first — one mistake there invalidates every number downstream.

---

## 0. The one constraint that matters most

**Feature-shaping flags must be IDENTICAL in extraction and inference.**

The classifier consumes an 18-channel × 29-frame tensor built from detected
player positions. Any flag that changes *what gets detected* or *where it lands
on the court* changes that tensor. If extraction and inference disagree on even
one of them, the model is served a distribution it never trained on — and it
fails silently, producing confident-looking wrong labels rather than an error.

This has bitten the project three times (§12.8, §13.7, §16 of
`Technical_Review_and_QA.md`). The flags that must match:

| flag | value | why it changes the features |
|---|---|---|
| `--yolo-model` | `yolo11n.pt` | which players are found at all |
| `--ball-model` | `fyp/volleyball_best.pt` | ball channel presence |
| `--imgsz` | `1280` | a ~15px ball vanishes at 640 |
| `--team-split` | `colour` | *which six players* enter the six slots |
| `--auto-court` | on | court mask; without it the crowd enters the population |
| `--court-coords` | `linear` | the pixel→court mapping itself |
| `--tracker` | `botsort` | identity stability → velocity features |

`run_shards.sh` and `MAKE_DEMO_VIDEO.bat` already carry the matching set. If you
change one, change both.

---

## 1. Extract features from the clips

508 labelled clips → 508 feature CSVs.

```bash
bash run_shards.sh 4
```

`4` = parallel processes. Use ~⅓ of your logical cores; more causes thrashing,
not speedup (measured: 38 s/clip serial → ~20 s/clip aggregate at 4 shards on a
12-thread i5-1335U, i.e. ~1.9×, not 4×).

**Constraints**

- **Time**: ~2–3 h on CPU for all 508. This is the long pole. On a GPU
  (Kaggle T4) it is 20–45 min — use `kaggle_retrain_after_detector_fix.ipynb`.
- **Interruptible and resumable-ish**: the clip list is shuffled with a fixed
  seed before sharding, so a partial run is a class- and video-representative
  sample you can still train on. It does *not* skip already-done clips — to
  resume, re-run and let it overwrite.
- **All shards must share `--shuffle-seed`** (default 0). Different seeds across
  shards ⇒ clips duplicated and dropped. `run_shards.sh` handles this.
- **Do not add `--clean-output` to a sharded run** — each shard would wipe the
  others' work. The launcher pre-cleans the directory once instead.

**Verify before training on it:**

```bash
python -c "import pandas as pd,numpy as np,glob,os; fs=sorted(glob.glob('training_csv_v3/*.csv')); pc=[f'p{i}_{a}' for i in range(1,7) for a in 'xy']; pres=[pd.read_csv(f)[[c for c in pc if c in pd.read_csv(f).columns]].replace(0.0,np.nan).notna().sum(axis=1).mean()/2 for f in fs]; import collections; print(f'{len(fs)} CSVs, players/frame mean {np.mean(pres):.2f}, empty {sum(1 for p in pres if p<1)}'); print(collections.Counter(os.path.basename(f).split('__')[0] for f in fs))"
```

Expect **~5.8 players/frame** and **0 empty clips**. If you see ~4.0 and many
empties, the detector flags are wrong — stop, do not train.

---

## 2. Train and evaluate the tactical classifier

```bash
python fyp/ml_classifier.py training_csv_v3 --features invariant --save-model tactical_ml.joblib
```

This trains 9 candidate models, evaluates each under two protocols, picks the
winner by the honest one, calibrates it, and saves it.

**Constraints**

- **Runtime**: ~10–25 min on CPU (leave-one-video-out refits every model once
  per source video, and permutation importance refits again per fold).
- **`--features invariant` is the default and the right choice.** `absolute`
  scores higher on k-fold and *worse* cross-video — it buys within-video
  performance with camera identity. Use `--ablate-features` to reproduce that
  comparison rather than assuming it.
- **Selection metric is `lovo_macro_f1`**, not accuracy. With a 5.8:1 imbalance,
  accuracy rewards always predicting Coordinated Defense.
- **Do not tune anything on the LOVO scores** and then report those same scores.
  The prior-corrected decision rule is used precisely because it is
  parameter-free and cannot overfit a 41-example class.

**What to read in the output**

| block | what it tells you |
|---|---|
| baseline ladder | majority < rule engine < model. The rule-engine row is the one that matters — same input, same human labels. |
| k-fold vs LOVO | the gap is the per-video signature, not a mistake to hide |
| ROC-AUC / PR-AUC | whether probabilities rank correctly even where argmax fails |
| overfit gap | flagged >0.05; with few source videos expect all models flagged |
| decision rule | argmax vs prior-corrected; the latter trades accuracy for minority recall |

---

## 3. Regenerate every reported number and figure

```bash
python reproduce.py
```

Nine stages → `results/latest/` with `METRICS.md`, `provenance.json` (package
versions + SHA-256 of every weight file), CSV tables and five figures. **Exits
non-zero if any asserted property fails.**

**Constraints**

- **Runtime**: ~28 min full, ~8 min with `--quick`. Quick mode uses fewer
  frames/videos — fine for a smoke test, **not** for numbers you quote.
- Stage 7 (detector mAP) needs `volleyball-detection.yolov11/`, which is **not
  in git** (239 MB). It skips gracefully if absent; the rest still runs.
- Stages 2–3 need the source `.mp4` files, also not in git. Use `--skip-video`
  on a machine without them.
- `reproduce.py` reads `training_csv` by default. To evaluate the new
  extraction, either point `CSV_DIR` at `training_csv_v3` or rename it.

---

## 4. Render the demo video

```bash
MAKE_DEMO_VIDEO.bat
```

Or explicitly:

```bash
python fyp/pipeline.py "videoplayback (4).mp4" tactical_ml.joblib --yolo-model yolo11n.pt --ball-model fyp/volleyball_best.pt --imgsz 1280 --team-split colour --tracker botsort --auto-court --court-coords linear --start-frame 11200 --max-frames 1800 --output-video demo.mp4 --output-csv preds.csv
```

**Constraints**

- ~20–40 min per 60 s of footage on CPU.
- The **preflight gate runs first** and aborts with **exit code 2** if the
  surviving player population is degenerate. That is the gate working, not a
  bug — read the `[FATAL]` line, it names the fix. Override only for debugging
  with `--skip-preflight`.
- Pick a start frame inside an actual rally. Broadcast footage opens on
  replays, graphics and crowd shots.

---

## 5. Establish the performance ceiling (not yet done)

The labels are human (verified, §16) but come from a **single annotator**, so
the ceiling for every score above is unknown.

```bash
python fyp/annotate.py dataset/dataset --out gold2_labels.csv --per-class 40
python fyp/ml_classifier.py training_csv_v3 --gold gold2_labels.csv
```

A **second person** must do the annotation — ~1.5–2 h for 160 clips. It is
deliberately blind and class-balanced. This yields annotator-vs-annotator κ,
which is the number no model should be presented as exceeding.

---

## 6. Optional: retrain the ball detector

Only if you have the Roboflow export.

```bash
python fyp/train_detector.py --verify-only fyp/volleyball_best.pt
```

The player detector is **stock `yolo11n.pt` and should stay that way** — the
416-image fine-tune collapses to 0.3 players/frame on unseen footage (§13.1).
Fine-tune the ball only.

---

## Full sequence, copy-paste

```bash
bash run_shards.sh 4
python fyp/ml_classifier.py training_csv_v3 --features invariant --save-model tactical_ml.joblib
python -m pytest fyp/tests -q
python reproduce.py
```

Total ~3–4 h on CPU, ~1 h on a GPU box.

---

## Known constraints, summarised

| constraint | value | consequence |
|---|---|---|
| Source videos | 4 | LOVO trains on 3; all models overfit (§15.3). More *matches* is the binding need. |
| Clips | 508 | small for 132-dim features |
| Class imbalance | 237 : 41 (5.8:1) | report macro-F1 and κ, never accuracy |
| Annotators | 1 | no inter-annotator κ ⇒ ceiling unknown |
| Court calibration | linear fallback | distances are relative, not metric, without per-camera homography |
| Compute | CPU only | extraction is the bottleneck; use Kaggle for speed |
| `labels.csv` | 160 of 508 rows preserved | annotation file for 68% of clips was overwritten; labels survive only in the folder layout (§16.5d) |
