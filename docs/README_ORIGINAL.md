# Automated Volleyball Tactical Analysis from Broadcast Video

Final Year Project — research-based. Converts raw broadcast match video into a
per-rally tactical coordination report, using detection → tracking → court
projection → identity-stable formation features → a learned sequence classifier.

**Everything in this README is reproducible with one command:**

```bash
python reproduce.py
```

It regenerates every sklearn metric and figure quoted below into
`results/latest/` — the model comparison, the feature ablation, the rule-engine
baseline and the prior-corrected decision rule — records package versions and
SHA-256 hashes of every model file used, asserts the properties the report
claims, and **exits non-zero if any of them fail.** The Mamba checkpoint is
hashed for provenance but not scored (it is a stale comparison-arm artefact;
see §5 / Technical Review S3).

---

## 1. Problem and research gap

Coaching staff need consistent, fast feedback on team coordination. Manual
notation is slow and subjective; existing automated sports analytics focus on
*individual* actions (spike, block, serve) or on ball trajectory, not on
**whole-team spatial coordination**.

The gap this project addresses: classifying a rally window by the *collective*
tactical pattern of six players — how the formation is shaped, how it moves, and
whether it moves together — from a single uncalibrated broadcast camera.

**Four tactical classes**, plus a class-0 noise sink for dead-ball / transition
windows:

| Index | Class | Definition |
|---|---|---|
| 1 | Coordinated Attack | fast approach by front-court attackers with base players held near the net |
| 2 | Coordinated Defense | the team shifts as a rigid unit |
| 3 | Delayed Support | the nearest player reacts late after ball impact |
| 4 | Spacing Breakdown | structural failure — formation too spread or too clustered |
| 0 | Unclassified / Transition | dead ball, too few players, out of play |

---

## 2. Architecture

```
broadcast video (1280x720, 30 fps)
   │
   ├─ PREFLIGHT GATE ─────────── samples the segment, measures the surviving
   │                             player population per stage, ABORTS (exit 2)
   │                             if it is degenerate                fyp/preflight.py
   ▼
DETECTION            dual backend                                  fyp/interfaces.py
   ├─ players: stock YOLOv11 (COCO person) @ imgsz 1280
   └─ ball   : volleyball fine-tune        @ imgsz 1280
   ▼
COURT CALIBRATION    auto court-quad detection, camera-motion            fyp/court.py
                     compensation, binary in/out-of-court mask
   ▼
TRACKING             BoT-SORT / ByteTrack + Kalman tuned for          fyp/tracking.py
                     explosive kinetics; occlusion-bridged slots      fyp/identity.py
   ▼
TEAM SEPARATION      jersey-colour clustering, clusters labelled by       fyp/teams.py
                     median court-side, per-track majority vote
   ▼
FEATURES             18-dim permutation-invariant per-frame           fyp/features.py
                     descriptor (`perm_invariant_v2`)
   ▼
WINDOWING            non-overlapping 29-frame (≈1 s) windows
   ▼
CLASSIFIER           learned — either backend, chosen by file extension
   ├─ scikit-learn (default)                                 fyp/ml_classifier.py
   └─ Mamba SSM / Transformer                                 fyp/mamba_model.py
   ▼
REPORT               per-sequence CSV + annotated MP4 + coordination report
```

Both classifier backends sit behind one `BaseTemporalClassifier` interface and
consume the identical `(29, 18)` tensor, so swapping them is a filename change.

---

## 3. Method — the parts that required a decision

### 3.1 Dual detector

A single detector cannot serve both classes. Measured on this project's own
footage (`results/latest/detector_ablation.csv`):

- **Player** is a canonical COCO object. Stock `yolo11n.pt` returns 14–23
  people/frame on all three source videos. The 416-image Roboflow fine-tune
  collapses to **0.3 players/frame** on unseen footage — max class confidence
  0.06 on a frame containing twelve clearly visible players.
- **Ball** is a ~15 px motion-blurred object. Here the fine-tune wins decisively:
  **67% recall vs stock's 16%** on the same video.

So players come from stock weights and the ball from the fine-tune.

**Inference resolution is not a detail.** A volleyball spans ~15–20 px on a 720p
frame; the Ultralytics default of 640 downscales it to ~9 px, below the smallest
detection stride. Ball recall with *identical weights*: **15% at 640 → 77% at
1280**.

### 3.2 Team separation by jersey colour

The court mask removes coaches, referees and crowd, but still passes **both
teams** — ten players into a six-slot system. No foot-point-vs-net rule can fix
this: at a block, both front rows stand within a metre of each other, on opposite
sides of a line a few pixels wide in a foreshortened view.

Two signals, each used only where it is reliable:

- **Geometry is reliable in aggregate.** Players may not cross the net, so a
  colour cluster's *median* court-side over a warm-up window is a solid team label.
- **Colour is reliable instantaneously**, but has no intrinsic team meaning.

Cluster torso colour → label each cluster by its members' median court-side →
decide membership by colour thereafter, majority-voted per track.
`n_clusters=4` by default because FIVB rules require a contrasting libero
jersey, so a team is *two* colour populations, not one.

Measured: **10.7 → 5.8 players/frame** (team size is 6).

### 3.3 Scale-invariant features

Without a per-camera homography the pipeline uses a **linear** pixel→court
scaling, so "cm" means something different in every video. Absolute features
(`hull_area`, `centroid_x`, `max_pair_dist`) are therefore partly *camera
identity*, and a learner will use them to recognise which video it is looking at.

The invariant block feeds only quantities that survive an unknown scale factor:
dimensionless ratios (compactness, tightest/loosest pair, speed ratio) and the
per-clip z-scored temporal *shape* of every channel.

### 3.4 Evaluation protocol

Two protocols are always reported together:

- **Stratified k-fold** — clips from one video land on both sides. Measures
  *within-video* generalisation. Optimistic.
- **Leave-one-video-out (LOVO)** — every clip of a held-out video is unseen.
  Measures *cross-video* generalisation. This is the honest number.

**The gap between them is a result, not an embarrassment** — it quantifies how
much the model relies on camera-specific cues.

---

## 4. Results

Regenerate with `python reproduce.py`; full tables in `results/latest/METRICS.md`.

### 4.1 Label provenance — the labels are human, and this was verified

The training labels are **human tactical annotations**, not machine-generated.
The chain (`fyp/extract_clips.py`) is:

```
labels.csv  ← HUMAN: (video_file, anchor_time, tactic_class)
   → ffmpeg cuts a 1-second clip at each annotated timestamp
   → clip written into dataset/<tactic_class>/
   → prepare_training_data.py takes the label from that folder name
```

Verified directly: all 159 resolvable `labels.csv` rows land in the folder
their `tactic_class` names — **zero mismatches**.

`fyp/label_clips.py` (a hand-coded rule engine) is a *separate, unused*
labelling path — it never populates those folders and is not in this chain.
So the common objection *"the model just learned your own rules"* does not
apply here, and the rule engine is instead available as an **independent
competing baseline**. See §16 of `Technical_Review_and_QA.md` for the full
verification, including a correction of an earlier version of this README that
stated the opposite.

### 4.2 The baseline ladder

Every predictor sees the identical 29-frame observation window and is scored
against the identical human labels:

| predictor | accuracy | macro-F1 | Cohen κ |
|---|---|---|---|
| majority class (no information) | 0.470 | 0.160 | 0.000 |
| hand-coded rule engine | 0.432 | 0.193 | 0.097 |
| **learned model** (LOVO, cross-video) | **0.484** | **0.299** | **0.161** |

**The learned model beats the hand-coded rule engine by +55% relative macro-F1
and +66% relative κ** — while being scored under the harder cross-video
protocol. That is the quantitative answer to "what does the ML add over your
rules?". The rule engine, for its part, barely clears the majority class
(κ=0.097), abstains on 8% of clips, and *never once* correctly identifies
Delayed Support (recall 0.000).

### 4.3 Feature ablation

| | k-fold macro-F1 | LOVO macro-F1 | LOVO kappa |
|---|---|---|---|
| absolute features | **0.520** | 0.265 | 0.081 |
| scale-invariant features | 0.440 | **0.299** | **0.161** |

Read the *direction*: dimensionless features **lower** the within-video score and
**raise** the cross-video score, nearly doubling kappa. That is the signature of
removing a camera-identity shortcut.

**Formal detector mAP** (§15.1, on the leak-free labelled validation split):
mAP@50 **0.681**, player AP50 **0.819** vs ball AP50 **0.543** — the first
formal metric confirming the player/ball detector split is the right call,
not just the raw detection-count argument in §3.1.

**Two sharper findings from a follow-up audit** (§15.2–15.4, full detail in
`Technical_Review_and_QA.md`):

- **Cross-video ROC-AUC is 0.47–0.54 for every one of the nine candidate
  models — statistically indistinguishable from chance (0.50).** Not "the
  argmax is sometimes wrong" — the predicted probabilities carry almost no
  cross-video ranking information at all.
- **All nine models are flagged overfitting under leave-one-video-out**
  (train-vs-held-out macro-F1 gap 0.71–0.78). Direct measurement of
  inter-video feature drift (`fyp/drift.py`, KS-test + PSI) confirms why:
  mean PSI 0.54–1.51 between every pair of the three source videos — several
  times past the conventional "major shift" threshold of 0.25 — even on
  features specifically engineered to be camera-scale-invariant.

### Honest limitations — state these, do not hide them

1. **Single annotator.** Labels are human tactical annotations (verified — see
   §4.1), but all from one person, so no inter-annotator agreement figure
   exists and the true ceiling for every score below is unknown. Section 6 is
   the protocol for a blind second pass.
2. **Cross-video accuracy (0.484) barely exceeds the majority baseline (0.470),
   and cross-video ROC-AUC is at chance.** The classifier is closer to
   *memorising two videos* than *learning a generalisable tactic classifier* —
   see the overfit-gap and drift findings above. Two of four classes are
   effectively not learned.
3. **16 examples in the smallest class** (Delayed Support), spread over LOVO
   folds, is ~4 per fold. No learner recovers a class from that.
4. **Only 3 source videos** means LOVO trains on 2 of them — not enough for
   any model class evaluated to avoid overfitting to camera-specific cues.
   More *source videos*, not just more clips from the same 3, is the priority.
5. **The current CSVs predate the detector fix** — 3.97 players/frame against the
   5.8 the fixed pipeline now produces. Re-extraction is the first improvement.
6. **Distances are metric only with a homography.** Under the linear fallback
   they are relative, and should be reported as such.

---

## 5. Install and run

> **Training from scratch?** See **[TRAINING.md](TRAINING.md)** — the ordered
> runbook with every command, runtime, and the flag-parity constraint that
> invalidates results if broken.

```bash
python -m venv .venv
```

```bash
pip install -r fyp/requirements.txt
```

Reproduce every metric and figure:

```bash
python reproduce.py
```

Analyse a video end-to-end:

```bash
python fyp/pipeline.py "match.mp4" tactical_ml.joblib --yolo-model yolo11n.pt --ball-model fyp/volleyball_best.pt --imgsz 1280 --team-split colour --tracker botsort --auto-court --court-coords linear --output-video annotated.mp4 --output-csv predictions.csv
```

Train the tactical classifier:

```bash
python fyp/ml_classifier.py training_csv --features invariant --save-model tactical_ml.joblib
```

Re-extract features after a detector change (**flags must match the pipeline**):

```bash
python fyp/prepare_training_data.py "dataset/dataset" --output-dir training_csv_v3 --yolo-model yolo11n.pt --ball-model fyp/volleyball_best.pt --imgsz 1280 --team-split colour --auto-court --court-coords linear --tracker botsort --clean-output
```

One-click runners: `MAKE_DEMO_VIDEO.bat`, `RETRAIN_AFTER_DETECTOR_FIX.bat`.
GPU retraining: `kaggle_retrain_after_detector_fix.ipynb`.

---

## 6. Establishing the performance ceiling (the next step)

The labels are human (§4.1), but they come from a **single annotator**, so the
true ceiling for every score in §4.2 is unknown — no model can be expected to
agree with one person's tactical judgement better than a second qualified
person would.

The protocol that measures it: have a second annotator independently label a
sample of the same clips, blind.

```bash
python fyp/annotate.py dataset/dataset --out gold2_labels.csv --per-class 40
```

```bash
python fyp/ml_classifier.py training_csv --gold gold2_labels.csv
```

`annotate.py` is deliberately **blind** (clips are shuffled across classes and
the existing label is never shown — otherwise the second annotator is anchored
to the first and the agreement becomes self-fulfilling) and **balanced** (equal
clips per class, because proportional sampling would give ~4 Delayed Support
examples).

This reports **annotator-1-vs-annotator-2 κ** — the ceiling — alongside
**model-vs-annotator-2**. If the model's κ approaches inter-annotator κ, the
model is performing as well as the task's own label noise permits, and further
modelling effort is better spent on data collection. **No automated score
should be presented as exceeding the inter-annotator ceiling.**

---

## 7. Repository map

| Path | Role |
|---|---|
| `reproduce.py` | one-command regeneration of every sklearn metric and figure (rule baseline + decision rule included; Mamba hashed only) |
| `TRAINING.md` | ordered training runbook + constraints |
| `run_shards.sh` | parallel feature extraction across all 508 clips |
| `fyp/pipeline.py` | end-to-end video → tactical report |
| `fyp/preflight.py` | pre-run population gate; aborts rather than emit a meaningless report |
| `fyp/interfaces.py` | detector + classifier abstractions, dual detector, backend factory |
| `fyp/features.py` | 18-dim `perm_invariant_v2` contract (single source of truth) |
| `fyp/teams.py` | jersey-colour team separation + per-track voting |
| `fyp/court.py` | homography, auto court detection, camera-motion compensation, mask |
| `fyp/tracking.py`, `fyp/identity.py` | tracker backends; persistent player slots |
| `fyp/ml_classifier.py` | learned tactical classifier, CV protocols, ablations |
| `fyp/annotate.py` | blind human annotation tool for the gold set |
| `fyp/drift.py` | KS-test + PSI inter-video feature drift; live reference-drift scoring |
| `fyp/mamba_model.py`, `fyp/train_mamba.py` | Mamba SSM comparison arm |
| `fyp/analytics.py`, `fyp/roles.py` | measured-vs-literature biomechanics, role inference |
| `fyp/tests/` | 243 tests (1 skipped), most runnable with no GPU |
| `Technical_Review_and_QA.md` | full audit log: every bug, its root cause, its fix |

---

## 8. Testing

```bash
python -m pytest fyp/tests -q
```

243 passed, 1 skipped. The suite encodes the project's own regressions: each
historical failure (§12.8 half-court homography, §12.9 slot starvation, §13.1
detector blindness, §13.7 crowd-captured colour clusters) has a test that
replays the real measured values. Every one of those failures ran to completion
and printed a formatted report — none raised an error. That is why the preflight
gate exists.

---

## 9. References

- Gu, A. & Dao, T. (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* arXiv:2312.00752
- Zhang, Y. et al. (2022). *ByteTrack: Multi-Object Tracking by Associating Every Detection Box.* ECCV 2022
- Aharon, N. et al. (2022). *BoT-SORT: Robust Associations Multi-Pedestrian Tracking.* arXiv:2206.14651
- Ratner, A. et al. (2017). *Snorkel: Rapid Training Data Creation with Weak Supervision.* VLDB 2017
- Jocher, G. et al. *Ultralytics YOLO* (v8, v11). https://docs.ultralytics.com
- Sports-science sources for the biomechanical validation are listed in
  `Technical_Review_and_QA.md` §10.5.
