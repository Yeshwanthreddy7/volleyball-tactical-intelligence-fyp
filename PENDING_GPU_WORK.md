# Pending work — run on the GPU machine

Everything in this repo up to this commit is **static code + unit tests only**
(no GPU available on the authoring machine). `pytest fyp/tests` is green
(256 passed, 1 skipped). The items below need real compute and could not be
run or verified here. Do them **in order** — later steps depend on earlier ones.

---

## 0. Sanity check (fast, ~1 min)

```bash
.venv/Scripts/python.exe -m pytest fyp/tests -q
```
Expect `256 passed, 1 skipped`. If not, stop and report before running anything heavy.

---

## 1. Phase 4 — finish the feature re-extraction  (~2–3 h CPU/GPU)

**State:** `training_csv_v3/` has **375 / 378** CSVs. The 4 that were missing
were corrupt source clips (`moov atom not found` — AUDIT_FINDINGS.md **D8**);
they have been **re-cut from the in-repo source videos** and are valid on disk
now, just not yet extracted to CSV.

**Do a clean full run** (recommended — guarantees every CSV came from the same
fixed pipeline):

```bash
CLEAN=1 bash run_shards.sh 4
```

- `run_shards.sh` already passes every train/serve-parity flag and
  `--exclude-pattern '^videoplayback_clip_[0-9]+\.mp4$'` (drops the 130
  `(plain)` clips — likely a re-encode of source `(3)`, AUDIT_FINDINGS.md **D5**).
- Without `CLEAN=1` it resumes with `--skip-existing` (skips the 375 already
  done). Use that only if you trust the existing 375 were all made post-fix.
- Expected result: **`ALL SHARDS DONE: 378/378 CSVs`** + `training_csv_v3/extraction_provenance.json`.
- `N=4` shards; raise if the box has more cores + RAM headroom.

**Verify:** `ls training_csv_v3/*.csv | wc -l` → 378.

---

## 2. Phase 5 — S3 / S9: Mamba comparison arm  (needs GPU; ~30–60 min)

`train_mamba.py` now prints an `[S3 WARNING]` / `[S3]` notice every run. The
**comparison is not apples-to-apples** until all of these are true at once
(checklist is also in a comment at the top of `train_mamba.py::train()`):

| Axis | sklearn headline | Mamba path now | Fix needed |
|---|---|---|---|
| features | invariant block only | all 18 raw channels incl. absolute `hull_area`/`centroid`/`ball` | feed Mamba the invariant block |
| normalisation | per-clip z-score | global train mean/std | switch to per-clip |
| classes | 4 | 5 (`NUM_CLASSES`) | 4-class (or the Q3 collapse) |
| split | LOVO | random unless `--test-video` | always `--test-video` |
| corpus | `training_csv_v3` (step 1) | whatever is passed | `training_csv_v3` |

**S3:** apply the 5 fixes above in `fyp/train_mamba.py` / `fyp/mamba_model.py`,
then:
```bash
.venv/Scripts/python.exe fyp/train_mamba.py training_csv_v3 --augment --epochs 80 --test-video "(1)" --checkpoint mamba_lovo_1.pt
.venv/Scripts/python.exe fyp/train_mamba.py training_csv_v3 --augment --epochs 80 --test-video "(3)" --checkpoint mamba_lovo_3.pt
```
Report the LOVO macro-F1 per held-out video. **Do not quote a random-split number.**

**S9:** run the disclosed small baseline and report the param:sample ratio:
```bash
.venv/Scripts/python.exe fyp/train_mamba.py training_csv_v3 --augment --epochs 80 --test-video "(1)" --d_model 32 --n_layers 2 --checkpoint mamba_small_1.pt
```

Frame it as **trajectory inductive bias / long-rally scalability**, not
performance (~70k params over ~200 clips = ~300:1; memorisation is structural).

---

## 3. Phase 4 — R3: regenerate all headline numbers on the clean corpus

The primary (deployed) model is scikit-learn HistGradientBoosting, not Mamba.
`RETRAIN_AFTER_DETECTOR_FIX.bat` is the full wrapper (retrains the primary +
the Mamba comparison arm + LOVO eval). Or directly:

```bash
.venv/Scripts/python.exe fyp/ml_classifier.py training_csv_v3 --save-model tactical_ml.joblib --output-dir ml_results
```

This single command now also produces (all added this session, unit-tested,
never run on real data):
- **S1** pre-declared model (`--declared-model hist_gradient_boosting`, the default) — headline is not max-of-9
- **S2** bootstrap 95% CI on macro-F1 + κ; McNemar model-vs-rule-engine
- **S4** PCA(0.95) inside the kNN/SVM/MLP pipelines
- **S5** per-fold macro-F1 mean±SD alongside the pooled number
- **S6** prior-corrected decision rule with the **per-fold training prior** (no leak)
- **S7** rule-engine baseline + decision-rule comparison (were never regenerated)

Then update every doc that quotes a number (README, METHODOLOGY,
Technical_Review) from the new `ml_results/ml_results.json` — the **2026-03-28
`training_csv` numbers are stale** after re-extraction.

---

## 4. Phase 5 — R2: corrected gold-set numbers  (needs a gold CSV)

The R2 fix is **in the code** (`ml_classifier.py` fits a separate model with
the gold clips excluded before scoring — `gold_clip_names()` + a mask), but
there is **no `gold_labels.csv` in the repo** (it was never committed). Make one:

```bash
.venv/Scripts/python.exe fyp/annotate.py dataset/dataset --out gold_labels.csv   # blind, ~30 min of clicking
.venv/Scripts/python.exe fyp/ml_classifier.py training_csv_v3 --gold gold_labels.csv --output-dir ml_results
```
The console will print `fitting a held-out model for this comparison (excluding N/… clips …)` — that line means the fix engaged. Commit `gold_labels.csv` this time.

---

## 5. reproduce.py full run  (Phase 7)

```bash
.venv/Scripts/python.exe reproduce.py            # full, ~30–60 min
```
- Must exit **0**. It now also regenerates the rule baseline, decision rule,
  bootstrap CI, McNemar, and an **out-of-fold reliability curve + ECE**
  (`results/latest/figures/reliability_curve.png`) — all **S7 / S2 / S8**, none
  run before.
- Mamba is **hashed for provenance only, not scored** (S3) — this is intentional.
- A `--stage N` or `--skip-video` run no longer overwrites `results/latest/`
  (**R4**), so use a full run for the report-ready numbers.

---

## 6. Real broadcast-video pipeline run  (Phase 7 + confirm H3)

```bash
.venv/Scripts/python.exe fyp/pipeline.py "videoplayback (4).mp4" tactical_ml.joblib \
  --yolo-model yolo11n.pt --ball-model fyp/volleyball_best.pt --imgsz 1280 \
  --team-split colour --tracker botsort --auto-court --court-coords linear \
  --start-frame 15000 --max-frames 1800 \
  --output-video final_annotated_v3.mp4 --output-csv predictions_seq_v3.csv
```
- Use **`--start-frame 15000`**, not 11200 — frame 11200 is a genuinely hard
  segment and is **no longer a known-good demo point** (AUDIT_FINDINGS.md **H7**).
- Watch the console for **`[scene cut]`** lines — that's the new **H3**
  scene-cut gate. It is unit-tested on synthetic frames only; this is the first
  real-video confirmation that it fires on actual cuts (replays / graphics) and
  not on normal play. Also watch the new **`[WARN] risky default`** lines
  (**R13**) — with the flags above there should be none.
- Read the output MP4 + the per-sequence CSV. The CSV now has
  `window_ball_recall` / `window_nan_frac` / `window_id_bridges` per row (**M17**)
  and a `Scene cuts detected` line in the final report.

---

## 7. Close-out (Phase 7 doc work — no compute)

- Fill the `AUDIT_FINDINGS.md` status column for every remaining row
  (D4, and anything reclassified after the runs above).
- Panel-defence doc: each likely objection → its answer tied to a regenerable
  number. `DEFENSE_MASTER.md` already has the reasoning; wire in the new numbers.
- `panel_review_1.tex` slide numbers → update from the fresh `ml_results.json`.

---

## Quick status table

| Phase / item | State | Blocker |
|---|---|---|
| Phase 2 (C1 C2 H8 M16 M17 L18–L20) | **done + verified** | — |
| Phase 3a (H4 H5 H6 H7 M10 M11 M12 M15) | **done + verified** | — |
| Phase 3b (H3 M9 M13 M14 L21 L22) | **done**; H3 needs a real-video confirm (step 6) | — |
| Phase 4 D5 (exclude `(plain)`) | **done** (code + smoke) | — |
| Phase 4 D3 (re-extract) | **375/378** | step 1 |
| Phase 4 D8 (corrupt clips) | **re-cut, valid on disk** | step 1 extracts them |
| Phase 4 Q4 (linear coords) | **decided** (kept linear) | — |
| Phase 4 R3 (new numbers) | code ready | steps 1 → 3 |
| Phase 4 R13 (unsafe defaults) | **done** (warnings) | — |
| Phase 5 S1 S2 S4 S5 S6 S7 S8 | **code done + unit-tested** | run on step 3 |
| Phase 5 S3 S9 (Mamba) | **flagged in code**; refactor + run | step 2 |
| Phase 5 R2 (gold eval) | **code done** | step 4 (needs gold CSV) |
| Phase 5 R4 (reproduce rmtree) | **done + tested** | — |
| Phase 6 (roadmap methods) | not started | optional for Panel Review 1 |
| Phase 7 (verify + defend) | — | steps 5–7 |
