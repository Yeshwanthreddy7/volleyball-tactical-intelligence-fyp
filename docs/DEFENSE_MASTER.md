# Defense master — not for slides, for your head

Everything cut from `panel_review_1.tex` lives here. The deck states facts;
this document is where the reasoning behind each fact lives, for verbal
answers only. Never read from this file at the podium — know it, don't show it.

---

## 1. If asked "why HGB and not RBF SVM?"

The table (Appendix A1) shows RBF SVM narrowly ahead on macro-F1 (45.1 vs
44.2, a 0.9-point gap) and balanced accuracy (44.9 vs 44.7, effectively
tied). HGB leads accuracy (61.7 vs 58.4) and kappa (0.404 vs 0.367).

**Say:** "Macro-F1 and balanced accuracy were the metrics we pre-declared as
imbalance-robust before ranking. RBF SVM edges HGB there, but by under a
point — not distinguishable from fold noise without a significance test,
which we haven't run yet. HGB was deployed for three operational reasons
independent of that margin: it needs no feature scaling, handles missing
values natively (a real occurrence — occluded players), and is faster at
inference. If pressed on whether the margin is real: 'we haven't computed a
paired confidence interval, that's the next thing we'd run.'"

**Do not say:** anything implying HGB was picked because it "won" — the
table shows a split decision and a panelist who read it will know that.

---

## 2. The feature-count arithmetic

The pipeline states "132 engineered features" from 6 statistics per channel
over a 29-frame window. **18 channels × 6 statistics = 108, not 132.**
132 = 22 × 6.

**Before presenting:** confirm on the training device whether the real
channel count is 22 (most likely — the base 18 tactical channels plus 4
more, e.g. presence flags or window-level meta-features) or whether the
statistic count per channel isn't uniformly 6. Whichever it is, know the
real breakdown cold.

**If asked "how do you get 132 features?":** "132 comes from [N] channels
times 6 window statistics — mean, std, min, max, IQR, slope." Fill in the
real N before you say this out loud. Never let "18 × 6" leave your mouth on
stage; a panelist does that multiplication in under two seconds.

**Why this matters more than it looks:** one caught arithmetic error makes
every other number in the deck suspect to that panelist for the rest of the
session. It's the cheapest possible finding for them and the most expensive
for you. Fix it before the number is ever spoken aloud.

---

## 3. "Is this actually reproducible?"

The repo's own `reproduce.py` regenerates the pipeline's metrics from
source, stage by stage, with provenance (package versions + weight hashes)
and an assertion per stage. It does **not** currently include the
HGB/Mamba training run on the 1,094-clip dataset — that was trained on a
separate device and the results were reported, not regenerated, from this
repo.

**Say, if asked directly:** "The pipeline code and its own test suite are
reproducible end to end from this repo. The classifier results on the
current dataset were produced on a separate training run; folding that
training script and its outputs into `reproduce.py` so the headline numbers
regenerate from one command here is the next infrastructure task."

**Do not claim** the 61.7% figure regenerates from a command in this repo
today. It doesn't yet. Saying it does and having it fail live is much worse
than stating the honest scope up front if asked.

---

## 4. Significance / variance gaps — if asked "how confident are you in this gap?"

Two things are missing and both are cheap to add before Guide Review 2:

- **Paired significance test** (bootstrap CI or McNemar) on the per-clip
  predictions, for HGB vs RBF SVM and HGB vs Mamba. Nothing currently
  distinguishes "real gap" from "fold noise" for the closer comparisons.
- **Per-fold (per-source-video) breakdown.** Only the pooled 7-fold number
  is reported. If one or two of the seven source videos are unusually easy
  or hard, the pooled number can hide that. A per-video table would show it.

**Say, if pushed:** "That's exactly the gap in our own audit — we have the
pooled number but not yet the per-fold spread or a significance test. Both
are on the immediate roadmap; neither changes the reported numbers, they
change how much weight we can put on the ordering between the top few
models."

---

## 5. `gold_labels.csv` provenance — if asked "how were these labels made?"

The label file is named `gold_labels.csv` but no inter-annotator agreement
figure or annotation protocol was stated in the material available for this
deck. "Gold" implies verified/adjudicated labels, not just a filename.

**Before presenting:** get the real answer from whoever ran the labeling —
single annotator, multiple annotators with adjudication, or a semi-automated
process reviewed by a human? Have the actual protocol ready, one sentence.

**If you don't have it in time:** "The labeling protocol is documented on
the training side; I'd want to pull the exact inter-annotator numbers before
quoting them rather than guess here." Better than inventing a number.

---

## 6. Full result table (already on the deck as Appendix A1 — reasoning here)

| Model | Accuracy | Balanced Acc. | Macro-F1 | Kappa |
|---|---|---|---|---|
| **HGB (primary)** | **61.70%** | 44.74% | 44.22% | **0.404** |
| RBF SVM | 58.41% | **44.88%** | **45.14%** | 0.367 |
| Random Forest | 60.42% | 42.89% | 41.50% | 0.381 |
| Gradient Boosting | 60.60% | 42.98% | 41.54% | 0.389 |
| Mamba SSM (comparison) | 48.45% | 40.08% | 39.63% | 0.267 |

Protocol: 7-fold leave-one-video-out, pooled out-of-fold predictions as the
primary headline (pre-declared over per-fold averaging).

Source: 1,094 labelled clips, 7 source-video groups, `gold_labels.csv`,
4 classes — Coordinated Attack 418 / Coordinated Defense 415 / Spacing
Breakdown 152 / Delayed Support 109 (3.8:1 imbalance). Trained on a separate
device from this repo (see §3).

**Why Mamba trails HGB by this much (48.5% vs 61.7%) — the real mechanism,
not an excuse.** This is not asserted, it is a named, well-studied pattern:

- **Why HGB wins now.** Grinsztajn, Oyallon & Varoquaux, *"Why do
  tree-based models still outperform deep learning on tabular data?"*
  (NeurIPS 2022), and Shwartz-Ziv & Armon, *"Tabular Data: Deep Learning is
  Not All You Need"* (Information Fusion, 2022) both show gradient-boosted
  trees beating deep nets consistently on datasets in the thousands-of-rows
  range. The mechanism: HGB's 132 features (mean/std/min/max/IQR/slope per
  channel) are a strong hand-built inductive bias — the model is told in
  advance which shapes of the trajectory matter. Mamba gets none of that;
  it must discover temporal structure from the raw sequence with far more
  free parameters. Standard bias-variance trade-off: the low-bias/
  high-variance model (Mamba) needs more data to match the high-bias/
  low-variance model (HGB + engineered features) at the same $N$.

- **Why the gap is expected to close, not stay fixed, as $N$ grows.** The
  132 summary statistics have a hard information ceiling: if the
  discriminating signal is in exact temporal ordering or a specific
  mid-window inflection, no amount of data lets a tree recover what
  mean/std/slope already smoothed away. Mamba has access to the full
  signal, so its ceiling is higher — reaching it just costs more data.

- **Direct precedent, not analogy from a different field.** This task
  (group tactical behaviour from multi-player position sequences) is a
  studied CV subfield: \emph{group activity recognition}. Ibrahim,
  Muralidharan, Deng, Vahdat \& Mori, \emph{"A Hierarchical Deep Temporal
  Model for Group Activity Recognition,"} CVPR 2016, introduced a
  volleyball dataset for exactly this --- player trajectories $\rightarrow$
  group-activity label --- and found a deep hierarchical LSTM beat
  non-deep baselines there. That dataset is roughly 4--5$\times$ the
  current 1{,}094-clip set. \textbf{Verify the paper's exact numbers before
  quoting them to the panel} --- cited here for the mechanism and as a
  starting point to check, not as a number to parrot.

**If asked "which one actually works, can both work?":** both are
legitimate, standard approaches to this exact class of problem; the field
has not settled on one. The right choice is a function of dataset size, not
a right-vs-wrong call. State the crossover as an empirical question about
*this* dataset, not a settled fact: "we'd expect Mamba to close and likely
cross the gap somewhere in the low thousands of clips, based on the closest
comparable published result; we haven't hit that scale yet." That is a
solid, literature-grounded position — not a guess and not an excuse.

---

## 7. Standing answers for common panel angles

- **"Why not accuracy alone?"** 3.8:1 class imbalance — a model that always
  predicts the majority class scores well on accuracy but zero on kappa.
  Macro-F1/balanced accuracy weight every class equally.
- **"Is LOVO the same as external validation?"** No, and the deck says so.
  LOVO holds out an entire source video's clips — it protects against
  leaking camera angle, team identity, and lighting between train and test,
  but every video is still from the same labelled collection. A genuinely
  new venue/camera is untested.
- **"Why permutation-invariant features?"** The six players are a set, not
  a sequence with a canonical order. If swapping which column holds player
  3 and player 5 changes the predicted tactic, the model has learned an
  artifact of the data pipeline, not volleyball.
- **"Why two classifiers instead of one?"** Not a zoo — one classical
  baseline (deployed) and one deep comparison arm (documents the
  alternative), evaluated identically. If asked why not just ship whichever
  wins: because the margin between them is close enough on some metrics
  that the honest answer is operational preference, not a clean win — see §1.
- **"What happens on a video you've never seen?"** The safety check runs
  first: if too few players, no ball, or no court is detected, the pipeline
  aborts and emits no report rather than a confident wrong one. That's a
  design decision, not an accident — worth stating proactively if the
  live-demo question comes up.

---

## 8. Roadmap (same as the deck's 3.4, expanded)

1. Fix the feature-count claim (§2) — same day, no dependencies.
2. State the label-provenance protocol (§5) — needs one conversation with
   whoever ran the labeling.
3. Fold the HGB/Mamba training script and its outputs into `reproduce.py`
   (§3) — makes every number in this deck regenerate from one command in
   this repo.
4. Compute per-fold variance + a paired significance test (§4) — turns the
   "HGB vs RBF SVM" split decision into a defensible statement either way.
5. More source-video matches — the single largest lever on closing the
   Mamba/HGB gap and on giving the significance test something to work with.
