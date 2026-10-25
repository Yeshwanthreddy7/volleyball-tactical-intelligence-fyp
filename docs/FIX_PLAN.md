# Fix plan — phased

Sequencing layer on top of `AUDIT_FINDINGS.md`. Every finding ID lands in a phase;
IDs not repeated here (`D6 D7 S0 R0`) are "verified good / no action".

Order logic: decisions and doc edits cost nothing and unblock everything → make
silent failures loud → fix the CV stages that change extracted features →
re-extract once → redo ML methodology on clean data → new methods are roadmap.

Guide decision already taken: **Mamba is the single classifier** (kills S1; changes S3/S4).

---

## Phase 0 — Decisions, no code (~1–2 h)

Four questions in `AUDIT_FINDINGS.md` §5 gate the expensive work. Answer before Phase 4.

| # | Question | How to settle | Blocks |
|---|---|---|---|
| Q1 | ~~Is `(plain)` a distinct match or a duplicate of (1)/(3)/(4)? (D5)~~ **PARTIALLY ANSWERED.** fps/resolution fingerprint: 12/12 sampled `(plain)` clips = 50.0fps/1920×1080, exact match to source (3); sources (1)/(4) = 30.0fps/1280×720, categorical mismatch. Pixel-hash confirmation attempted twice, both interrupted by decode cost on the ~90k-frame source, not completed. **Decision: treat as a likely duplicate of (3)**, i.e. 3 source videos / 3 LOVO folds, not 4 — see [AUDIT_FINDINGS.md](AUDIT_FINDINGS.md) D5. | frame-hash sample clips vs the 3 source videos | re-extraction, any 4-fold LOVO |
| Q2 | Is the `gold` block in `ml_results/ml_results.json` real or a smoke artefact? (R2) | trace which script wrote it; check against `labels.csv` | headline story (κ 0.161 vs 0.599) |
| Q3 | 4-class or collapsed *Coordinated vs Breakdown*? (D1) | decide from per-video table — 4-class is not defensible with 6 Breakdown train examples | extraction target, all ML |
| Q4 | Homography on by default, or keep linear fallback? (§4.3) | decide now — it changes extracted coordinates | re-extraction, H6 |

---

## Phase 1 — Doc + number fixes, no re-extraction (~2–3 h, fully parallel)

**STATUS: DONE** (this session). All rows below closed: test counts, ablation
numbers, label-provenance root cause (`reproduce.py` generator was already
correct; `results/latest/METRICS.md` patched to match; `fyp/AGENT.md` fixed;
3 historical false-claim locations in `Technical_Review_and_QA.md` got
correction pointers to §16, not deleted — preserves the "we found our own
error" record). `fyp/README.md` replaced with a redirect stub (was a second,
drifted copy of root `README.md` — deleted the duplication instead of
resyncing it, so it can't drift again). `pipeline.py` docstring/`--help`
fixed. `MAKE_DEMO_VIDEO.bat` now defaults to the deployed primary model
instead of hard-requiring Mamba. `TRAIN_AFTER_COLAB.bat` /
`TRAIN_NOW_STOCK_DETECTOR.bat` deleted (`git rm`, recoverable from history) —
confirmed missing 6 of 7 required flags each, would silently poison a
training run. `pytest fyp/tests`: 191 passed, 1 skipped throughout, no
regressions.

Pure edits. Each doc currently disagrees with itself.

| ID | Action | Files |
|---|---|---|
| R5 | test count 144 → **191 passed, 1 skipped** | `fyp/README.md:340,351`, `Technical_Review §15/16` |
| R7 | feature-ablation: k-fold 0.436→0.440, LOVO F1 0.297→0.299, κ 0.157→0.161 | `README.md:198`, `METHODOLOGY.md:253`, `Technical_Review §14.3/14.4` |
| R1 | label provenance: rewrite rule-engine claims → HUMAN (cite the `labels.csv` typos as evidence, D6) | `fyp/AGENT.md:34`, `Technical_Review §14.1:1339` + `§12.7:1005`, `fyp/README.md:42,180`, regen/delete `ml_results/ml_results.json` |
| R6 | disambiguate "3 vs 4 source videos" everywhere; state the 4th (`(plain)`) is not in-repo | `METHODOLOGY §8:380`, `Technical_Review §12.10` |
| D1 D2 | add per-video × per-class table + note the label-prior shift confound | `METHODOLOGY`, `Technical_Review` |
| R9 | mark `fyp/README.md` historical (YOLOv8/ByteTrack/Mamba/4-class/(29,14)) or rewrite | `fyp/README.md` |
| R12 | fix module docstring / `--help` ("YOLOv8 → ByteTrack → Mamba") | `fyp/pipeline.py` |
| R10 | `.bat` files still point at `mamba_checkpoint_v2.pt` — repoint to `tactical_ml.joblib` or delete | `MAKE_DEMO_VIDEO.bat`, `RETRAIN_AFTER_DETECTOR_FIX.bat` |
| R11 | quarantine/delete 2 legacy scripts carrying 1 of 7 flags | `TRAIN_AFTER_COLAB.bat`, `TRAIN_NOW_STOCK_DETECTOR.bat` |
| R8 | reconcile minor numeric drift (AP50 0.542/0.543, PR-AUC, mAP 0.681/0.688, §14.4 counts) | METRICS vs docs |

---

## Phase 2 — Make silent failures loud (~4–5 h, code)

**STATUS: DONE, all 8 items.** 195/195 tests (191 original + 4 new: 3 determinism,
1 preflight regression). Every fix verified beyond "tests still pass" — either a
real end-to-end pipeline run against `videoplayback (4).mp4` (C1, C2, H8, M17),
a synthetic case with hand-computed ground truth (M17's 4 metrics), or a direct
mismatch-injection test (C2's `check_provenance`). No claim below rests on
"should work."

The "every past failure was silent" class. Independent of the data question.

| ID | Action | File |
|---|---|---|
| C1 | **DONE.** `create_tracker()` now raises `TrackerFallbackError` on a failed BoT-SORT construction unless `allow_fallback=True`; both `pipeline.py` and `prepare_training_data.py` wire this to a new `--allow-tracker-fallback` flag (default off). Effective tracker (`tracker.name`) is now a column in every per-window CSV row and printed in the final report block. Smoke-tested: BoT-SORT still constructs cleanly in this env; 191/191 tests still pass. | `fyp/tracking.py`, `fyp/pipeline.py`, `fyp/prepare_training_data.py` |
| C2 | **DONE.** `prepare_training_data.py` writes `<output-dir>/extraction_provenance.json` (imgsz, ball_model, team_split, court_coords, effective tracker, feature_version) once per run. `ml_classifier.py` and `train_mamba.py` both read it and embed it in their saved bundle/checkpoint. New `BaseTemporalClassifier.check_provenance()` (interfaces.py) compares it against the LIVE run's flags, built in `pipeline.py` right after the tracker is known; raises `ValueError` on a real mismatch unless `--allow-provenance-mismatch`. Missing data on either side (every bundle saved before this fix, including the currently-deployed `tactical_ml.joblib`) WARNS, never crashes — verified directly: real bundle load, missing-data case, injected real mismatch (raises under strict, warns under lenient), matching case (silent). Also fixed while in this code: `ml_classifier.py`'s own stale "labels come from rule engine" string baked into every `ml_results.json` it writes (same bug as the METRICS.md one fixed last turn, different generator — this is the one that would have kept reproducing it). | `fyp/prepare_training_data.py`, `fyp/train_mamba.py`, `fyp/ml_classifier.py`, `fyp/interfaces.py`, `fyp/pipeline.py` |
| H8 | **DONE.** MAP found `teams._kmeans` was ALREADY correctly self-seeded (own `np.random.default_rng`, not global state) — did not "fix" a non-bug, re-verified it instead (now a regression test). Real gap was everything else: `run_pipeline` and `prepare_training_data.run` now seed `random`/`numpy`/`cv2.setRNGSeed`/`torch` (pipeline.py, new `--seed`) at the top, before any model/detector work. **Verified empirically, not asserted:** two independent `pipeline.py` process runs, identical args, real video — sha256-identical output CSVs. Added `fyp/tests/test_determinism.py` (3 tests, unit-speed) so this can't silently regress without a full video re-run every time. | `fyp/pipeline.py`, `fyp/prepare_training_data.py`, `fyp/tests/test_determinism.py` |
| M16 | **DONE.** `_SIDE_FILTER_WARNED` reset to `False` right after preflight finishes (the exact point the calibrator is already rebuilt for the real run), so a trip during preflight's own sampling no longer spends the real run's one-time warning budget. | `fyp/pipeline.py` |
| M17 | **DONE.** `_sequence_metrics` returns `window_ball_recall`, `window_mean_present`, `window_nan_frac`, `window_id_bridges` (contiguous-run count), merged into every CSV row via the existing `**metrics` spread. Verified against a synthetic sequence with a planted 5-frame gap + ball missing in 9/29 frames — all 4 values matched hand-computed expectations, and confirmed present in a real pipeline-run CSV. | `fyp/pipeline.py` |
| L18 | **DONE.** New `_safe_fps`/`_safe_frame_count` helpers (shared by `run_pipeline` and `_preflight_sample`, which read these independently) validate finite-and-positive / positive-int instead of relying on `... or default` (NaN and negative ints are both truthy in Python, so the old pattern didn't catch either). | `fyp/pipeline.py` |
| L19 | **DONE.** Main frame loop now checks `frame is None` after a successful `cap.read()` and skips-with-counter (`none_frame_count`, reported in the final block) instead of passing `None` into `calibrator.update`/`detector.predict`. | `fyp/pipeline.py` |
| L20 | **DONE.** `preflight.assess()` now checks `n_frames == 0` as its own Stage 0, before the detection-quality check, with an accurate remedy (check `--start-frame`/file integrity) instead of recommending a detector-weights swap for a segment that was never read. New regression test `test_l20_zero_frames_read_is_diagnosed_as_a_read_failure_not_detector`; confirmed every existing test already defaults `n=24` so none spuriously trip the new branch — 195/195 pass. | `fyp/preflight.py`, `fyp/tests/test_preflight.py` |

---

## Phase 3 — CV correctness (~8 h, code)

### 3a — changes extracted features, MUST land before Phase 4

| ID | Action | File |
|---|---|---|
| H4 | **DONE — 3 design iterations, documented.** (1) First tried a between/within CENTROID-distance ratio: rejected after direct testing showed k-means's own within-cluster-variance-minimisation manufactures apparent "separation" even fitting pure noise into `n_clusters=4` (measured: 4 identical-colour players still scored ~3, indistinguishable from a genuinely-similar-color case). (2) Restricted the same ratio to near-cluster-vs-far-cluster pairs only: better but still didn't discriminate reliably. (3) **Landed on:** per-sample agreement between the colour-derived NEAR/FAR label and the GEOMETRIC (position) label for that same warm-up sample — has no k-means self-bias since it checks against raw positions, not derived centroids. `degenerate` now also fires below `MIN_GEOMETRIC_AGREEMENT=0.75` or outside a plausible per-side sample-count band (`MIN_SIDE_FRACTION=0.15`), in addition to the original all-one-side check. **Verified with realistic per-frame colour noise** (an exact-repeated synthetic frame has ~zero within-cluster spread and breaks any distance-based measure in ways real video never does): identical colour -> agreement 0.55 (chance) -> correctly degenerate; barely-different navies -> 0.55-0.6 -> correctly degenerate; stark contrast control -> 1.0 -> correctly NOT degenerate. 3 new permanent regression tests. 202/202. | `fyp/teams.py` |
| H5 | **DONE.** `TeamClassifier` now: (a) rejects a warm-up/re-fit frame as evidence unless `len(xyxy) >= min_active_players` (new param, default 6) — a dead-ball/bench/substitution frame no longer poisons the fit; (b) once `ready`, keeps accumulating fresh evidence and re-fits every `refit_every` frames (new param, default 900 ≈ 30s), with a rollback safety net — if the re-fit evidence itself turns out degenerate, the previous good fit is kept, not silently replaced. Scoped to the live-serving path (`pipeline.py`) only — `prepare_training_data.py`'s per-clip batch fit bypasses `update()` entirely (direct `_desc`/`_ys` assignment + manual `_fit()`), and the failure mode (a long warm-up window landing on a bad moment) doesn't really apply to a 29-frame pre-selected clip. Every existing test using a small synthetic fixture (all of them — 1-5 boxes, well below the new default) now explicitly passes `min_active_players=1`, since population-size gating isn't what those tests are about; this is what surfaced the change (7 tests broke on the first run, all fixed with the explicit override, not by weakening the new default). **Verified deterministically**, not by chance noise draws: activity-gate accepts/rejects on exact population counts; periodic re-fit confirmed to actually change centres on fresh evidence; rollback confirmed by directly seeding known-degenerate re-fit evidence (identical descriptor regardless of position → agreement forced to exactly 0.5) and checking the previous good centres survive. 3 new permanent tests. 205/205. | `fyp/teams.py`, `fyp/tests/test_teams.py` |
| H6 | **DONE.** New `CourtCalibrator.net_y_cm_current()`: derives the net's court-Y from the detected quad's own top edge via the SAME `pixel_to_court` call every other distance uses (so homography mode needs no special-casing — a quad's own top edge maps back through its own H to ~450 by construction; only `--court-coords linear` actually needed correcting). Wired into `_filter_players_by_team_side` (zero call-site changes — `calibrator` was already a parameter), `_team_labels_geometric` (new optional param, one call site updated), and `TeamClassifier.net_y_cm` (updated live each frame in `pipeline.py`'s main loop, since `_fit()` only reads it once ~45 frames in; updated once post-Phase-A in `prepare_training_data.py`'s per-clip batch fit). **Verified against the exact numbers the finding cited**: a quad with top edge at pixel-row 289 (video (4)'s real measurement) under `force_linear` → 361.2 cm, matching the cited "~361" almost exactly (constant was 450); homography mode on the same quad → 450.0 (confirms no correction needed there, as designed); a court_y=400 player flips FAR→NEAR exactly as described; a court_y=380 player goes from confidently-wrong-FAR to correctly-UNKNOWN (inside the now-accurate net margin) rather than a forced flip — arguably the more correct outcome. 199/199, no regressions. | `fyp/court.py`, `fyp/pipeline.py`, `fyp/prepare_training_data.py` |
| H7 | **DONE — and it caught a real, previously-invisible problem while being built.** `_preflight_sample` now test-fits a batch `TeamClassifier` (when `--team-split colour`) and `assess()` uses ITS verdict — not the crude geometric proxy — for the team-split FATAL check, plus a new independent `degenerate`-colour-fit FATAL (catches "right headcount, wrong six" — the case a person-count alone can't see). **First implementation was wrong and I caught it before shipping it**: fit the colour test on the SAME sparse, whole-segment-spread samples used for population counting → false-negative "degenerate" (0.667 agreement) on a clip that classifies cleanly via the real dense 45-frame warm-up. Fixed: colour test-fit now reads a fresh CONSECUTIVE run from the segment's actual start, matching exactly what the live warm-up sees. **After that fix, the SAME verdict (0.667, degenerate) still appeared** — at that point stopped assuming a bug and tested the actual hypothesis: ran the identical command at a different start-frame (15000) on the same video. **Result: clean end-to-end run, exit 0, 3 windows classified, healthy team-split (8/frame).** This confirms frame 11200 (the start-frame every smoke test this session had been using) is a genuinely harder segment — not a bug in today's fixes. Every earlier "confirmed working" smoke-run claim at frame 11200 this session predates H4's `geometric_agreement` check and M9's FATAL tightening, which is exactly why it went uncaught until today. **Action item, not closed by this fix:** frame 11200 on `videoplayback (4).mp4` should not be treated as a known-good demo point going forward; frame 15000 is confirmed clean. 6 new tests (pure stats-dict, no video). 211/211. | `fyp/pipeline.py`, `fyp/preflight.py`, `fyp/tests/test_preflight.py` |
| M10 | **DONE.** `MIN_RAW_MEDIAN` raised 4.0→4.5 (FATAL floor); new `LOW_RAW_MEDIAN=8.0` WARN tier for the 4.5-8 band that used to pass with zero comment. `_activity_gate`'s `min_players` (3.0) checked against `MIN_TEAM_MEDIAN` (3.0) — **already aligned**, not changed speculatively (they measure different populations — raw detector output vs. our-team slot occupancy — so "align" means conceptually consistent, not numerically forced equal; confirmed they already agree where it matters). New regression test for the WARN tier. 199/199. | `fyp/preflight.py`, `fyp/tests/test_preflight.py` |
| M11 | **DONE.** `order_corners` rewritten: angle-sort around the centroid first (structurally guarantees a simple, non-self-intersecting polygon — a bowtie becomes provably impossible, not just checked-for), then rotated so the smallest-(x+y) point is first (preserves the existing TL,TR,BR,BL semantic every caller depends on). Shoelace-vs-hull area check kept as a belt-and-braces assertion. Manual `--court-corners` now **fails loud** (`ValueError`) on a degenerate input instead of silently degrading to uncalibrated — this project's own recurring failure shape. New temporal-consistency bound in `update()`: a refresh candidate whose centroid has moved >25% of the frame diagonal from the current quad is rejected (kept previous) instead of adopted, since CMC already tracks real smooth motion between refreshes. **Verified directly** (not just "tests pass"): clean/scrambled/degenerate `order_corners` inputs (caught and fixed my own flawed first test case — a "duplicate point" input I assumed was degenerate turned out to have real triangle area, not a bug); manual-corners `ValueError` fires on a genuinely collinear input; the displacement bound correctly rejects a wild jump and accepts a small plausible drift (mocked `detect_court_quad`, corrected a `refresh_every=1` test-setup bug where the modulo condition could never fire before trusting the result). **Permanent test added** (`fyp/tests/test_court_m11.py`, 7 tests: order_corners TL-first / any-permutation-stays-simple / collinear-rejected, manual-corners ValueError vs valid quad, auto-refresh far-jump rejected vs small-drift accepted). 233/233, zero regressions. | `fyp/court.py`, `fyp/tests/test_court_m11.py` |
| M12 | **DONE.** `CameraMotionEstimator._is_plausible()` rejects a frame-to-frame affine outside documented bounds (scale within ±15%, rotation within 5°, translation within 15% of the smaller frame dimension) or below a 60% RANSAC inlier ratio (previously discarded — `estimateAffinePartial2D`'s second return value was thrown away). `--cmc` without `--auto-court` now refused at startup (was previously permanently uncorrectable if a bad estimate ever landed). **Verified directly**: 6 constructed cases (healthy pass, each of the 4 bounds individually violated, a composed just-inside-all-bounds case) plus 2 more added while writing the permanent test (first-frame / blank-frame). Live-verified the `--cmc requires --auto-court` refusal and the flag-combination logic (only the true positive combination triggers it). New `fyp/tests/test_camera_motion.py` (8 tests). (M11 now also has its own permanent test — `fyp/tests/test_court_m11.py`, added same session.) 219/219 at the time; 233/233 now. | `fyp/court.py`, `fyp/pipeline.py`, `fyp/tests/test_camera_motion.py` |
| M15 | **DONE.** Feature rows are no longer appended to the window buffer while `calibrator.auto and calibrator.corners is None` (the exact documented no-op-mask state — `filter_detections`'s own docstring: "no calibration polygon -> everything passes"). Detection/tracking still runs during pre-lock so identity has continuity once locked; only the feature-row contribution is gated. `frames_before_lock` counted and reported. Manual-corners mode (`--auto-court` off) is untouched — confirmed by design, not by omission. Verified directly against the real `CourtCalibrator` class (3 cases: fresh/pre-lock, post-lock, manual-corners-mode) plus a real pipeline run, no regressions. | `fyp/pipeline.py` |

### 3b — serving only, can follow Phase 4

| ID | Action | File |
|---|---|---|
| H3 | **DONE.** `_frame_hsv_hist` + `_is_scene_cut` in `pipeline.py`: per-frame 32×32 HSV histogram vs previous (CORREL) OR raw-tracker-ID carry-over fraction; either dropping below `--scene-cut-corr` (0.5) / `--scene-cut-id-churn` (0.2) discards the partial window so no 29-frame sequence spans a cut. Count + discarded frames in the final report. 5 unit tests (synthetic frames). Serving-only as scoped. Live broadcast-video check left for the user's GPU machine — code-only + unit-tested here. | `fyp/pipeline.py`, `fyp/tests/test_upgrades.py` |
| M9 | **DONE.** `_preflight_sample` now records `team_side` in its stats; `assess()` makes the `MAX_TEAM_MEDIAN` breach FATAL unless `team_side == "all"` (previously WARN unconditionally). Updated the one existing test whose docstring literally described the ambiguity this closes, split it into the two cases it was conflating. 196/196. | `fyp/pipeline.py`, `fyp/preflight.py`, `fyp/tests/test_preflight.py` |
| M13 | **DONE.** (File was actually `fyp/teams.py`, not `identity.py` — corrected.) `TeamVoter`'s tie-break no longer defaults to NEAR on zero evidence *or* a genuine 50/50 split — both now return `UNKNOWN`, which `keep = voted == want` already excludes correctly. Confirmed the bug was genuinely reachable, not theoretical: geometric net-margin players feed real `UNKNOWN` into the voter during colour-model warm-up (the first ~45 frames of *every* colour-split run), and the old default (NEAR) matched `want` for the default `--team-side bottom` — an opponent tie-breaking the "our team" way would have silently filled a slot. 2 new tests (zero-evidence, genuine-tie); traced all 5 existing `TeamVoter` tests by hand first, confirmed none exercised this path (why the bug was invisible). 198/198. | `fyp/teams.py`, `fyp/tests/test_teams.py` |
| M14 | **DONE.** Load-time warning when any checkpoint channel's std sits at the `MIN_STD` training-time floor (near-constant channel — common cause named: `--ball-model` omitted at training). Serve-time tensor clamped to `[-8, 8]` regardless of cause, as defense in depth. **Verified against the real deployed `mamba_checkpoint_v2.pt`** (min std 0.34 — healthy, no false-positive warning) **and a constructed synthetic checkpoint with a genuinely constant channel** (warning fires, names the exact channel index). Extreme-input test: a `1e6`-valued sequence produced no crash, no NaN, and correctly flipped `is_anomaly=True` instead of the failure mode's signature confident-wrong/`is_anomaly=False`. | `fyp/interfaces.py` |
| L21 | **INVESTIGATED, NARROWED.** MAP found `label_clips.py`'s rules were themselves tuned against the `(0,0)`-means-missing convention (rule2's own docstring documents it) — the conversion is correct for its actual consumer, not a bug. Real residual risk was narrower and silent: a high-missing-data window degrading this fallback's reliability with zero indication. Added a `>25%` NaN-fraction warning instead of changing the sentinel (which would mean re-tuning 4 hand-calibrated rule thresholds with no ground truth to validate against, for a path nothing currently deployed reaches). | `fyp/pipeline.py` |
| L22 | **DONE.** `--max-frames` below `SEQ_LEN` (and above 0, the "unlimited" sentinel) now fails immediately with a clear error, before any detector/tracker/video setup. Verified live: exits code 1 in ~14.5s — which is entirely module-import overhead (confirmed separately: `import pipeline` alone takes ~18s on this machine), not any of the old wasted work. | `fyp/pipeline.py` |

---

## Phase 4 — Re-extraction (~2–3 h compute)

Gate: Phase 0 answered + Phase 2 + Phase 3a done.

| ID | Action |
|---|---|
| D5 | **DONE — excluded.** `_iter_clip_files` gained `--exclude-pattern` (regex on bare filename, `fyp/prepare_training_data.py`); `run_shards.sh` passes `^videoplayback_clip_[0-9]+\.mp4$`, dropping the 130 plain clips (fps/resolution fingerprint match to (3) — see AUDIT_FINDINGS.md D5). 508 target → 378 clips. Smoke-tested on 3 real clips (2 kept, 1 dropped) before the full run — exit 0, correct CSV shape, provenance embedded. 3 new tests (`fyp/tests/test_prepare_training_data.py`), 222/222 full suite. |
| D3 | **RUNNING (resumed).** First launch (`bb5hlyy4j`) was killed by the user mid-run at ~299/378 clips. Added `--skip-existing` to `prepare_training_data.py` (skips a clip whose output CSV already exists — live-verified: re-ran on a 2-clip fixture with one CSV deleted + one clip added, correctly re-extracted only the missing/new ones) so the resume doesn't redo ~300 already-completed clips. `run_shards.sh` now defaults to resume mode (`--skip-existing`, no wipe); `CLEAN=1 ./run_shards.sh` still available for a full from-zero run. Resumed as background task `bg0i4zzm3`. |
| Q4 | **Kept `--court-coords linear`** (already the flag `run_shards.sh` used) — deliberate, not re-litigated: keeps continuity with the existing CSV corpus and avoids confounding this run (which already changes 16+ things) with a second, independent coordinate-mapping change. Homography-on is a documented future option, not a Phase-4 blocker. |
| R3 | new headline numbers computed on `training_csv_v3`, not the 2026-03-28 `training_csv` |
| R13 | **DONE.** `_risky_serving_defaults` (`pipeline.py`) and `_risky_extraction_defaults` (`prepare_training_data.py`) — pure functions, called at startup, print `[WARN] risky default: …` per omission (no ball model / no court mask / homography-coords-vs-linear-corpus / bytetrack-vs-botsort-corpus). Warning not error: rule-only / `--team-side all` / ablation runs are valid, and C2 already hard-fails a real train/serve provenance mismatch. 4 unit tests, 237/237. Independent of Phase 4 completion. |

---

## Phase 5 — ML methodology on clean data (~6 h, code)

Gate: Phase 4 done. **The user has no GPU — nothing in this phase can be
*validated* here; S2/S4/S5/S6/S7/S8 can be *implemented* as code-only changes
and run later on the user's other device. S3/S9 need a training run. R2/R4 done.**

| ID | Action | File |
|---|---|---|
| S1 | **DONE (code).** `--declared-model` (default `hist_gradient_boosting`, the deployed model) — pre-registered headline, not max-of-9. Zoo table kept but relabelled "exploratory"; declared-vs-zoo-best both shown. `reproduce.py` hard-codes `DECLARED`. `ml_results.json` records `model_selection`/`zoo_best`. 1 test. (Note: the tree model is the declared one, not Mamba — matches the deployed reality / the newer pptx data.) | `fyp/ml_classifier.py`, `reproduce.py` |
| S3 | **BLOCKED (needs GPU run).** Confounds now flagged in `train_mamba.py` (`[S3 WARNING]` / `[S3]` notices + checklist comment). The real refactor — invariant block, per-clip norm, 4 classes, LOVO, on `training_csv_v3` — deferred to the user's device (unverifiable here). | `fyp/train_mamba.py`, `fyp/mamba_model.py` |
| S9 | **BLOCKED (needs GPU run).** `--d_model 32 --n_layers 2` is already a valid flag combo; just needs running + the param:sample ratio reported. Bundled with S3. | `fyp/train_mamba.py` |
| S2 | **DONE (code).** `bootstrap_metric_ci` (clip-level, 10k resamples, 95% CI on macro-F1 + κ) and `mcnemar_test` (exact binomial on discordant pairs). `reproduce.py` stage 4 prints/stores the winner's CI + McNemar-vs-rule-engine. Mamba comparison deferred to S3. 4 tests. | `fyp/ml_classifier.py`, `reproduce.py`, `fyp/tests/test_ml_classifier.py` |
| S4 | **DONE (code).** `reduced()` = `impute→scale→PCA(0.95)→clf` for `knn`/`rbf_svm`/`mlp` (Pipeline step → refits per fold); trees keep raw features (self-regularise + preserve channel importance). `both` dropped from `--features` choices + ablation loop (legacy-load path kept). 3 tests. Run on user's box for new numbers. | `fyp/ml_classifier.py`, `fyp/tests/test_ml_classifier.py` |
| S5 | **DONE (code).** `_overfit_gap` returns per-fold macro-F1 + balanced-acc + mean±SD from its existing single `cross_validate` call (no added cost); `main()` prints a labelled "Macro-F1 aggregation" block (pooled headline vs mean±SD vs per-fold list). Fields auto-flow to `ml_results.json`. 2 tests. | `fyp/ml_classifier.py`, `fyp/tests/test_ml_classifier.py` |
| S6 | **DONE (code).** `_prior_corrected_predict_per_fold` (per-fold training prior, no leak); `compare_decision_rules(groups=)` reports it + labels the prior; `reproduce.py` regenerates it; serving stays argmax (deliberate — validate on re-extracted data first) but joblib carries `train_class_counts`. 2 tests. | `fyp/ml_classifier.py`, `reproduce.py`, `fyp/tests/test_ml_classifier.py` |
| S7 | **DONE (code).** `reproduce.py` stage 4 now runs `rule_engine_baseline` + `compare_decision_rules` (were implemented, never called from the harness). Mamba stays hashed-only with an explicit printed reason (S3: stale checkpoint, LOVO needs a retrain). README / docstring claims corrected to "every sklearn metric". `both` dropped from stage-5 ablation (S4). | `reproduce.py`, `README.md` |
| S8 | **DONE (code).** `reliability_curve()` (per-bin conf vs empirical acc + ECE); `reproduce.py` stage 6 runs it on pooled LOVO out-of-fold probs, prints + plots (`reliability_curve.png`) + stores. 2 tests. | `fyp/ml_classifier.py`, `reproduce.py`, `fyp/tests/test_ml_classifier.py` |
| R2 | **DONE.** Gold set is real, not fabricated (root cause: `ml_classifier.py:1244-1248` — model scored against gold was fit on ALL 285 clips including gold, so the old `model_vs_rules`/`model_vs_human` were in-sample, not held-out; `rules_vs_human` was always unaffected). **Fix applied:** new `gold_clip_names()` + a mask against `files` now fit a SEPARATE model with gold rows excluded before scoring it against gold; deployment model elsewhere untouched. `gold_report` carries `model_held_out_of_fit`/`n_excluded_from_fit`. 4 new tests, 226/226 suite. Live CLI smoke-test (synthetic 24-clip/2-video set) confirmed the exclusion fires: `excluding 6/24 clips that overlap gold_csv`. **Not yet done: re-running this on the real 285-clip set** — no `gold_labels.csv` exists in the working tree (never committed, per R2's own finding above) to run it against; the corrected real numbers can only be produced once that file is available again. | `fyp/ml_classifier.py`, `fyp/tests/test_ml_classifier.py` |
| R4 | **DONE.** `_should_refresh_latest(args)` — only a full run (no `--stage`, no `--skip-video`) replaces `results/latest/`; a partial run stays in its timestamped dir. 4-case unit test. Code-only, no run needed. | `reproduce.py`, `fyp/tests/test_reproduce.py` |

---

## Phase 6 — New methods / roadmap (big, optional for Panel Review 1)

`AUDIT_FINDINGS.md §4`. Do the ones that buy the most defensibility first.

| # | Method | Buys |
|---|---|---|
| 1 | class collapse (binary Coordinated vs Breakdown) as headline, 4-class as honest negative | a number that survives (D1) |
| 4 | `GroupKFold` by rally + per-video prior tables | kills D4 near-duplicate leak |
| 2 | abstention as a first-class output (promote `is_anomaly`); report decline rate | "zero error tolerance" story |
| 9 | blind second annotator → inter-annotator κ | the ceiling; interacts with R2 |
| 3 | homography on by default (validate: reprojection error px, court 18×9 m ±2%) | metric distances (bundle into Phase 4) |
| 5 | domain-adversarial branch on Mamba (gradient reversal predicting `video_id`) | the one place neural beats trees |
| 6 | ball-channel ablation (train with/without) | is AP50 0.543 ball worth it |
| 7 | detector mAP on the 3 source videos, not only the 70-image Roboflow split | CV examiner will ask |
| 8 | sub-component metrics: tracking IDF1/MOTA, team-assignment accuracy, reprojection error, replay-detection PR | "I assert" → "I measured" |
| 10 | cite team-sport collective-behaviour literature for the taxonomy | taxonomy principled not invented |

---

## Phase 7 — Verify + defend (~2 h)

| Action |
|---|
| `pytest fyp/tests` green |
| clean `reproduce.py` full run, exit 0 |
| real pipeline run on `videoplayback (4).mp4` — read the MP4 + CSV |
| fill `AUDIT_FINDINGS.md` status column (closed / justified / deferred) per item |
| panel-defence doc: each likely objection → its answer, tied to a regenerable number |

---

## Coverage check

- **§0 dataset:** D1→P1/P6, D2→P1, D3→P4, D4→P6, D5→P0/P4. (D6 D7 = good)
- **§1 ML:** S1→P5, S2→P5, S3→P5, S4→P5, S5→P5, S6→P5, S7→P5, S8→P5, S9→P5. (S0 = good)
- **§2 CV:** C1 C2→P2 · H3→P3b · H4 H5 H6 H7 H8→P2/P3a · M9 M13 M14→P3b · M10 M11 M12 M15→P3a · M16 M17→P2 · L18 L19 L20→P2 · L21 L22→P3b.
- **§3 repro:** R1 R5 R6 R7 R8 R9 R10 R11 R12→P1 · R2 R4→P5 · R3 R13→P4. (R0 = good)
- **§4 methods:** all 10 → P6.
- **§5 questions:** Q1–Q4 → P0.

Critical path to a defensible Panel Review 1: **P0 → P1 → P2 → P3a → P4 → P5 → P7** (~20–25 h). P3b and P6 are follow-on.
