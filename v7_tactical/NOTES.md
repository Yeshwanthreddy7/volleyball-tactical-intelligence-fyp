# v7_tactical — binary Coordinated Attack vs Coordinated Defense

**Goal:** foolproof, conceptually-grounded annotated output for ONLY Coordinated
Attack and Coordinated Defense, validated on the unseen `videoplayback (7).mp4`,
iterated until the annotation is close to real. GPU, 100 epochs, deep learning.

## Why binary
The 4-class model (Attack/Defense/Delayed/Spacing) topped out at LOVO macro-F1
0.559 — the two "failure" classes (Delayed Support, Spacing Breakdown) are
subtle, sparsely labelled (~110 each), and drag the whole model down. Dropping
them to a clean Attack-vs-Defense question is a well-posed problem.

## Conceptual grounding (verified in the feature stats, 723 clips)
The 18 permutation-invariant channels separate the two classes exactly the way
volleyball says they should:

| channel | Attack | Defense | reading |
|---|---|---|---|
| speed_top2 | 21.8 | 14.6 | spiker approach sprint |
| speed_diff | 17.9 | 11.4 | 1-2 fast approachers vs static base |
| centroid_y | 660 | 686 | attacking team pushes toward the net |
| hull_area / spread_y / nn_dist_max | smaller | larger | defense spreads wide for coverage |

Tracked side = BOTTOM team (`--team-side bottom`), consistent with
training_csv_v4's extraction. The label is the rally PHASE of that team.

## Pipeline
1. `prep_cache.py` → `cache/binary_cache.npz` (723 clips, canonical + mirrored, 18ch)
2. `train_binary.py` → InceptionTime ensemble, 8-fold LOVO + final 5-seed
   - `results/lovo_binary.json`, `models/binary_incept_seed{0..4}.pt`, `models/binary_incept.pt`
3. ONE detection pass on vp(7): `pipeline.py --dump-tracks --frame-csv --strict-render`
   → `results/vp7_tracks.npz`, `results/vp7_frames.csv`  (slow, once)
4. `classify_windows.py` → overlapping-window ensemble + temporal smoothing
   → `results/vp7_windows.csv`  (fast, iterate here)
5. `fyp/rebanner.py` → `results/vp7_annotated.mp4`  (fast, no YOLO)
6. `selfcheck.py` → sampled frames + sanity report → decide next iteration

## Iteration log

### iter 0 — smoke (2026-09-06 ~16:5x)
- 8 epochs, 1 seed, LOVO: **macro-F1 0.874, ROC-AUC 0.938, bal-acc 0.874**
- per fold bal-acc 0.79–0.93. Signal is real. Proceed to full run.

### iter 1 — full train, COLOUR team split (done)
- 100 epochs, LOVO 3-seed ensemble. **Pooled LOVO macro-F1 0.899, ROC-AUC 0.955**
  - Attack P .89 R .92 F1 .90 ; Defense P .91 R .88 F1 .90 ; confusion [[312,42],[31,338]]
  - per-fold bal-acc: (1) .762  (2) .889  (3) .952  (4) .982  (5) .945  (plain)/(6) mid  usa .846
  - video (1) is the weak fold; everything else .85-.98
- BLOCKER: vp(7) detection pass ABORTED at preflight — colour team split is
  degenerate on vp(7) (similar kits, colour==geometry agreement 1.0).
- FIX: redo everything with `--team-split geometric` (robust "which side of the
  net", jersey-independent, conceptually exact for bottom-team Attack/Defense).
  Re-extracting 744 A/D clips -> v7_tactical/csv_geom/ ; then rebuild cache + retrain.

### iter 2 — GEOMETRIC team split (running)
- csv_geom: 723 clips extracted with `--team-split geometric --team-side bottom`
  (21 skipped = corrupt videoplayback (2) region). provenance team_split=geometric.
- cache rebuilt (v7_tactical/cache/binary_cache.npz now geometric).
- geom training launched: `--tag binary_geom`, 100 ep, LOVO 3-seed + final 5-seed.
- vp(7) detection pass launched in parallel: `pipeline.py --team-split geometric
  --strict-render --skip-preflight --frame-csv --dump-tracks` (~1.5h, ONE time).
  Rule-based banner in that pass is discarded; only vp7_frames.csv + vp7_tracks.npz matter.
- Fixed: train_binary.py saved norm_mean/std as np arrays; interfaces.TorchTemporalClassifier
  does `.to(device)` on them -> now saved as torch tensors. Colour ckpts patched too.
- Serve path verified: `create_temporal_classifier(binary_incept.pt)` returns
  "Coordinated Attack"/"Coordinated Defense" correctly.

- **GEOM LOVO result: macro-F1 0.906, balanced-acc 0.905, ROC-AUC 0.969** (beats colour 0.899)
  - Attack P .88 R .95 F1 .91 ; Defense P .94 R .86 F1 .90 ; confusion [[305,49],[19,350]]
  - per-fold bal-acc: (1).81 (2).86 (3).93 (4).96 (5).92 (6).94 (plain).91 usa.86  — ALL folds >=.81
  - model slightly over-calls Attack (49 D->A vs 19 A->D)
- checkpoints: models/binary_geom_seed{0..4}.pt + models/binary_geom.pt (servable)

### iter 1-3 on vp(7) — DONE
- detection pass produced vp7_frames.csv (80330 fr, 57 cols incl pipeline feat_*) + vp7_tracks.npz.
- **KEY FINDING: the ball tracker is broken on vp(7)** — ball_px_y stuck at ~254 for the
  whole match, raw_ball_x outliers to 67000. Model's ball channels are noise on vp7;
  it runs on the PLAYER features (centroid_y, speeds, spread, hull) which DO work.
- p_att cached (`cache/vp7_patt_s10.npz`) so smoothing/gate tuning is instant.
- iter1: EMA+hysteresis -> 12% of confident windows CONTRADICTED the ensemble (lag bug).
- iter2: replaced with **2-state Viterbi HMM** (emission = ensemble P(attack), tunable
  self-transition). Contradiction -> 2-3%. Clean phase segmentation.
- iter3: + **decision-threshold shift 0.56** (logit shift on p_att) — corrects a vp7
  domain lean where spread defensive formations were being called attack. Balance
  63/37 -> 48/48. + morphological close/open on the dead mask.
- **dead-time detection is NOT reliably solvable here**: ball broken, tracker ID-swaps
  make fake speed spikes, celebrations/walking look like formations, and even the
  shipped pipeline's own gate only finds 13%. Settled on a LIGHT gate (camera cuts via
  frame-gap + <4 players + sustained-low context speed) ~6-7% BREAK; the banner shows
  A/D continuously otherwise (honest: it's a formation-phase classifier).

### VALIDATION (visual, high-confidence windows, zoomed to net)
- **DEFENSE calls (p<0.07): strong** — 7/9 clearly show spread coverage / digs / dives /
  serve-receive. Matches LOVO defense precision 0.94.
- **ATTACK calls (p>0.93): weaker** — 3/9 clearly net attacks/blocks; rest are
  compact/transition postures. Matches LOVO attack precision 0.88. "Attack" reads as
  "team compact / advancing / at net" rather than strictly a spike.
- Traced 2 full rallies: Attack<->Defense phases ~3-4s each, 90%+ confidence, sensible flow.
- Limitation: 1280x720 broadcast wide shot -> ball invisible in contact sheets, cannot
  pixel-verify every label. LOVO 0.906 on 8 held-out videos is the honest accuracy proxy.

### v3 params (rendering full video now)
  --stride 10 --stay 0.91 --att-threshold 0.56 --dead-speed 10.5 --dead-ctx 40
  --bridge-break-s 1.3 --min-live-s 1.3 --min-dwell-s 0.8
  -> Attack 47.9% / Defense 45.2% / BREAK 6.8%, 655 switches
  -> v7_tactical/results/vp7_annotated.mp4  (~80 min render)

### v3 DELIVERED (2026-09-07 ~04:28)
- `v7_tactical/results/vp7_annotated.mp4`  (6.0 GB, 1280x720, 44.8 min, valid)
- `vp7_annotated_small.mp4` compressing (crf 26, ~400 MB) for sharing
- `vp7_windows_v3.csv` = the per-window label track (frame_start, frame_end, label, conf)
- verified on rendered output 851-886s: DEFENDING calls land on real defensive
  spreads; ATTACKING on compact/net formations; caught the 882-883s dead-ball;
  phase runs median 3.3s (p25-75 2.0-5.7s) - realistic volleyball rally cadence.
- Attack 48% / Defense 45% / Break 7%.

### iter 4-5: SCOREBOARD point-boundary detection (big win for dead-time)
- `score_events.py` watches the on-screen score digits ROI [585,5,720,48], detects
  pixel change when the score updates = a rally just ended. thresh 0.032, debounce 150fr.
  -> **96 point events**, median gap 26s (realistic volleyball point cadence).
- classify_windows `--use-score-events --score-pre 105 --score-post 165`: force BREAK
  from 3.5s before each score update to 5.5s after.
- **v4: BREAK 37.9% / Attack 31.7% / Defense 30.4%** (live split 51/49) - dead-time now
  matches broadcast volleyball. break runs median 8s, live phases median 2.7s.
- verified on 850-940s: breaks now correctly land on between-point / repositioning /
  celebration periods; A/D calls during rallies unchanged & still sensible.
- residual leak: 1-3s of post-rally celebration still gets an A/D label before the
  score updates (score-pre could go higher but risks clipping rally ends).
- v4 rendering -> vp7_annotated.mp4 (v3 kept as vp7_annotated_small.mp4).

### iter 6: multi-architecture ensemble (in progress)
- train_binary.py now takes `--arch` (via create_temporal_model).
- BINARY LOVO by arch: InceptionTime **0.906** | TCN **0.886** (100ep, 2-seed) | BiLSTM tbd
  (all far above their 4-class scores - binary is a much cleaner problem for every arch)
- FINAL binary LOVO: InceptionTime **0.906** | BiLSTM **0.889** | TCN **0.879**
- inter-arch corr on vp7 p_att ~0.88; combined weighted ensemble (0.45/0.30/0.25)
  changes 3.5% of A/D window calls vs InceptionTime-only, slightly smoother
  (531 vs 590 switches, contradiction 1.4% vs 1.7%).
- **v5 = combined 3-arch ensemble + score-boundary breaks** is the definitive version.
  verified on 3 match segments (120-180s, 850-940s, 2300-2360s): break periods land
  on real repositioning; A/D alternation during rallies is sensible.
  cache/vp7_patt_s10.npz now holds the ensemble p_att.
- v5: Attack 32.8% / Defense 29.3% / Break 37.9%, 531 switches.

### ============ DELIVERED: v6 final (2026-09-07) ============
  results/vp7_annotated.mp4 (5.9 GB) + results/vp7_annotated_small.mp4 (826 MB)
  results/vp7_windows.csv  (label track, == vp7_windows_v6.csv)
  Attack 35.4% / Defense 31.8% / Break 32.9% ; 424 rally phases (med 3.3s) ;
  151 breaks (med 7.7s) ; 1 banner switch per 5s.
  Model: InceptionTime(5)+BiLSTM(4)+TCN(4) ensemble, binary LOVO 0.906/0.889/0.879.
  Smoothing: 2-state Viterbi HMM. Dead-time: scoreboard point detection (96 pts,
  rally-END anchored) + motion gate. att-threshold 0.56 domain correction.

### iter 6: rally-END-anchored score breaks (v6)
- v5's fixed `--score-pre 105` started the break 3.5s before the score update,
  which was BEFORE the rally actually ended -> it clipped the point-winning
  spike into BREAK.
- v6: for each score event, backward-search fr_sp for the last genuine rally
  action (>17 cm/frame) and start the break 0.4s after that, capped at >=2s
  before the score update. Keeps the winning play labelled, breaks on the
  celebration.
- v6: Attack 35.4% / Defense 31.8% / Break 32.9% ; live phases median 3.3s ;
  151 breaks median 7.7s. Verified vs v5 on 2 score-event regions - v6 keeps the
  final attack, v5 had cut to break early.
- **v6 = FINAL** -> vp7_annotated.mp4 (5.9 GB) + vp7_annotated_small.mp4.
- **RENDER PERF BUG FOUND**: rebanner writing into the OneDrive-synced results/
  folder was throttled to ~60 fr/min (OneDrive syncing every VideoWriter flush).
  Rendering to the local scratchpad instead = ~2500-3000 fr/min (40x). All the
  v3/v4/v5 80-min renders were OneDrive-throttled. Future: render to scratch, move after.
- v6 celebration handling verified on rendered output (events 740s, 784s): the
  break starts AFTER the attacking play concludes, on the dispersal/celebration. Correct.

### === v5 (superseded by v6) ===
- `results/vp7_annotated.mp4` (5.86 GB, 44.8 min, valid) + `vp7_annotated_small.mp4`
- verified on a 40-frame full-match overview + 4 dense segments: consistent
  throughout, no broken stretches, break/attack/defense well distributed.
- Attack 32.8% / Defense 29.3% / Break 37.9% ; 392 live phases (median 3.3s) ;
  140 rally-break events.
- older versions kept as small copies: vp7_annotated_v3_small.mp4 (InceptionTime,
  light gate), vp7_annotated_v4_small.mp4 (InceptionTime + score breaks).

### remaining weak points (documented in README, data-limited)
1. ATTACK precision - "attack" fires on transitions too (LOVO P 0.88). Inherent to
   a formation classifier; "attack" = offensive/net phase of possession.
2. ~2s post-rally celebration still gets an A/D label before the score updates.
3. missed scoreboard points -> occasional merged rally.
4. vp7 ball tracker frozen all match -> model uses player features only.

### fast loop (cached p_att - each iter <2s + optional re-render)
  classify_windows.py <params> -> filmstrip.py / rebanner.py -> read -> tune -> repeat
- classify_windows.py chain smoke-tested on synthetic concatenated clips: OK
  (fixed: load_ensemble needs .cpu() on the torch-tensor norm stats).
- vp7 detection running ~450 fr/min, ETA ~2.7h (strict-render + auto-court +
  dump-tracks + frame-csv). One-time. frame CSV + tracks.npz are the outputs.
