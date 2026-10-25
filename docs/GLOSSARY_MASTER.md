# Glossary master — every metric, every component, explained

Reference document, not slide content. Read this to actually understand the
project; `panel_review_1.tex` states facts, `DEFENSE_MASTER.md` has Q&A
talking points, this document is *why those facts are true* — the mechanism
behind every number and every term in the deck, in full depth.

No `CONTEXT.md` / `docs/adr/` exist in this repo (it's an ML/CV pipeline,
not a DDD-style business-domain codebase), so this is written as a
standalone technical + metrics reference rather than forced into that
format.

---

## Part 1 — Evaluation Metrics

### 1.1 Accuracy

$$\text{Accuracy} = \frac{\text{correct predictions}}{\text{total predictions}}$$

The simplest metric, and the most misleading one here. With a 3.8:1 class
imbalance (Attack 418 / Defense 415 / Breakdown 152 / Delayed 109), a model
that *only ever predicts* the majority class still scores well on accuracy
while being useless — it never gets a minority class right. Accuracy is
reported (61.7% for HGB) because panels expect it, but it is never the
metric a decision is made on.

### 1.2 Balanced Accuracy

$$\text{BAcc} = \frac{1}{K}\sum_{k=1}^{K} \text{Recall}_k$$

Average the *per-class* recall (of all true Attack clips, what fraction did
the model catch?), then average those $K$ numbers with equal weight —
regardless of how many clips each class actually has. A model that nails
the big classes and misses every minority-class clip gets punished here in
a way plain accuracy can't see. This is why HGB's balanced accuracy
(44.7%) is so much lower than its raw accuracy (61.7%): the gap *is* the
imbalance problem, made visible.

### 1.3 Macro-F1

First, per class $k$:

$$P_k = \frac{TP_k}{TP_k+FP_k}, \qquad R_k = \frac{TP_k}{TP_k+FN_k}, \qquad
F1_k = \frac{2 P_k R_k}{P_k + R_k}$$

Then average unweighted across classes:

$$\text{F1}_{\text{macro}} = \frac{1}{K}\sum_k F1_k$$

Precision asks "of the clips I called Attack, how many really were?";
recall asks "of the real Attack clips, how many did I catch?" F1 is their
harmonic mean — it punishes a model that trades one off cheaply for the
other (e.g. predicting Attack constantly to inflate recall while cratering
precision). Macro-F1 is the single number this project treats as the
primary headline, because — like balanced accuracy — it weights Delayed
Support (109 clips) exactly as much as Coordinated Attack (418 clips).

### 1.4 Cohen's Kappa ($\kappa$)

$$\kappa = \frac{p_o - p_e}{1 - p_e}$$

$p_o$ = observed agreement (= accuracy). $p_e$ = the agreement you'd expect
from two random guessers who happen to know the class frequencies but
nothing else. Kappa asks: *how much better than chance is this, given how
easy chance is on an imbalanced problem?* A model can post 60%+ accuracy
almost "for free" here just by leaning on the two big classes; $\kappa$
strips that out. Standard interpretation bands (Landis & Koch, 1977):
$<0$ worse than chance, $0$–$0.20$ slight, $0.21$–$0.40$ fair, $0.41$–$0.60$
moderate, $0.61$–$0.80$ substantial, $0.81$–$1.0$ almost perfect. HGB's
$\kappa=0.404$ sits right at the fair/moderate boundary — genuinely better
than chance, not yet a strong classifier. Say that number in exactly those
terms if asked; don't oversell it as "high."

### 1.5 Leave-One-Video-Out (LOVO)

Standard $k$-fold cross-validation shuffles clips randomly into folds. If
two clips from the same source video land in different folds — one
training, one testing — the model can partly "cheat" by recognizing the
camera angle, lighting, or a specific team's jersey colour rather than the
tactic itself. LOVO instead holds out *one entire source video* per fold:
every clip from that video is test-only, every other video is train-only.
With 7 source-video groups here, that's 7 folds, and a clip's video is
never split across train/test.

**What LOVO does and does not prove.** It proves the model generalizes
across the 7 videos in hand — a real, non-trivial claim. It does **not**
prove generalization to a venue, camera, or team never seen in *any* fold
(a true external test set). Say this distinction proactively if it's not
asked; it's the single most common thing an ML panelist probes for and
finding it stated up front reads as rigor, not a gap.

**Pooled vs per-fold.** "Pooled out-of-fold" means: run all 7 folds,
collect every clip's prediction from whichever fold it was in the test set
for, then compute one metric over all 1,094 predictions at once. The
alternative — "per-fold average" — computes the metric separately in each
of the 7 folds, then averages those 7 numbers. Pooled weights every *clip*
equally; per-fold weights every *video* equally regardless of how many
clips it contributed. This project reports pooled as the headline (stated
as a pre-declared rule, in 2.4) because clip-level weighting is the more
natural read for "how good is this classifier on a client's data," and
because per-fold variance is not yet computed (see `DEFENSE_MASTER.md` §4).

### 1.6 Confidence, calibration, entropy (the "anomaly" flag)

A classifier's raw softmax/probability output is not automatically a
trustworthy confidence. Tree ensembles in particular tend to output
extreme probabilities (near 1.0) even when they're wrong, because a
majority vote across many trees saturates fast. *Calibration* (e.g. Platt
scaling — fitting a logistic curve from raw score to true probability on
held-out data) corrects this so "70% confident" really does mean "right
about 70% of the time" in aggregate.

Once calibrated, Shannon entropy over the class-probability vector
$\mathbf{p}$:

$$H(\mathbf{p}) = -\sum_k p_k \log_2 p_k$$

is a genuine uncertainty signal: $H=0$ means the model put all its mass on
one class; $H$ near $\log_2 4 = 2$ bits means it's near-uniform across all
4 classes. The "tactical-deviation" / anomaly flag fires when the top
class probability drops below a threshold $\tau$ (0.5 here) — i.e. the
model itself is saying "this window doesn't look confidently like any one
tactic," which is itself a potentially useful signal to a coach (a
formation breaking down mid-transition, for instance) rather than a
failure to hide.

---

## Part 2 — Pipeline Components

### 2.1 YOLO (You Only Look Once)

A single-stage object detector: one forward pass through a CNN divides the
image into a grid, and each grid cell directly predicts bounding boxes,
objectness confidence, and class probabilities — no separate
"propose regions, then classify them" step like older two-stage detectors
(e.g. Faster R-CNN). That single-pass design is why it's fast enough to run
frame-by-frame on video. Here, players and the ball use **independent** YOLO
backends/weights rather than one shared model, because a canonical,
large, slow-moving object (a player) and a small, fast, frequently
occluded object (a volleyball) are not well served by one detector tuned
to one operating point (see `DEFENSE_MASTER.md` for why this specific
choice, not just that it was made).

### 2.2 ByteTrack / BoT-SORT (tracking)

Detection alone gives you boxes *per frame*, with no notion that "the box
at frame 10" and "the box at frame 11" are the same person. Tracking-by-
detection algorithms solve the association problem frame-to-frame:

- **ByteTrack** associates *every* detection box, including low-confidence
  ones, via IoU (intersection-over-union) matching against predicted next
  positions — the insight being that a real object with a temporarily
  low-confidence detection (partial occlusion, motion blur) is still worth
  keeping in the track rather than discarding.
- **BoT-SORT** extends this with an appearance embedding (what does this
  player look like, not just where are they) and **camera-motion
  compensation** (estimating and correcting for the camera panning/zooming
  between frames, so a moving camera doesn't look like every player moved).

The output either way is a stable *identity* per player across the clip,
not just a per-frame box.

### 2.3 BallTracker

The ball is small, fast, and frequently fully occluded (behind a player,
above the frame, motion-blurred into invisibility for a frame or two).
Rather than treating a missed ball detection as "no ball this frame,"
BallTracker bridges short gaps with a constant-velocity prediction: given
the ball's last known position and velocity, extrapolate forward for a
bounded number of frames. This is deliberately simple (not a learned
model) — appropriate because the gaps it's meant to cover are short.

### 2.4 Homography (court projection)

$$\begin{bmatrix} X' \\ Y' \\ W' \end{bmatrix} = \mathbf{H}
\begin{bmatrix} u \\ v \\ 1 \end{bmatrix}, \qquad
(X,Y) = \left(\frac{X'}{W'}, \frac{Y'}{W'}\right)$$

A homography is a $3\times3$ matrix (8 independent degrees of freedom —
the 9th is a scale factor) that maps points on one plane to points on
another *under perspective*. Here it maps a pixel coordinate $(u,v)$ in
the camera image to a real-world coordinate $(X,Y)$ on the flat court
surface. It is estimated from known correspondences — typically the 4
court corners, whose real-world positions are fixed by the court's known
dimensions — via a direct linear transform, refined with RANSAC to reject
noisy corner detections.

**Why it's necessary, not cosmetic:** a broadcast camera views the court
at an angle, so equal *pixel* distances represent very different *real*
distances depending on how far from the camera they are (near the camera,
a few pixels covers centimeters; far from it, the same few pixels covers
meters). Every spacing/formation feature (hull area, nearest-neighbour
distance, max-pair distance) is meaningless without this correction —
only foot-points (where the player touches the floor) are valid inputs,
since only the floor is the plane $\mathbf{H}$ was calibrated against; a
head or torso point sits above the plane and projects incorrectly.

**The documented fallback:** when the 4 court corners cannot be reliably
detected in a given clip, the pipeline falls back to a simpler, known-
approximate coordinate mapping rather than silently producing wrong
metric distances — a deliberate, logged degradation rather than a hidden
failure.

### 2.5 Team assignment (jersey clustering + slots)

Detection and tracking tell you *where* the players are, not *which team*
each belongs to — a detector has no concept of "team," only "person."
Team assignment clusters players by jersey-colour appearance (the two
teams wear visually distinct kits) and assigns each track to one of 6
persistent "own-team" slots — 6 because that's a full volleyball team on
court under standard 6-player rotation rules. This is what turns "here are
12+ detected people, some on court, some sidelines, both teams, maybe a
referee" into "here are our 6 players, tracked."

---

## Part 3 — Feature Engineering (29-frame window, 18 channels)

### 3.1 Why permutation-invariant, precisely

At any frame $t$, the six players are not a *sequence* with a canonical
order (there is no "first" or "second" player in a formation) — they are a
**set** $\mathcal{P}_t = \{\mathbf{p}_{t,1}, \ldots, \mathbf{p}_{t,6}\}$.
If swapping which detected box got labelled "player 3" vs "player 5"
changes the predicted tactic, the model has learned an artifact of
detection/tracking order, not of volleyball. Every feature below is
therefore built as a **symmetric function** of the set — a function whose
output does not depend on the order its arguments are listed in (sum,
mean, max, pairwise-distance-based statistics — never "player 3's
position" as a raw, order-dependent input).

### 3.2 The six feature categories

- **Formation** — centroid $\bar{\mathbf{p}}_t$ (the team's average
  position), spread (how tightly clustered), convex-hull area $A_t$ (the
  area of the smallest polygon containing all six players — a direct
  proxy for how much court the formation is covering).
- **Spacing** — nearest-neighbour distances (how close is each player to
  their nearest teammate) and $D_t=\max_{i\ne j}\lVert
  \mathbf{p}_{t,i}-\mathbf{p}_{t,j}\rVert$, the maximum pairwise distance
  (how stretched the formation is end-to-end).
- **Motion** — centroid speed (is the whole team moving, and how fast),
  spatial variance of speed (are some players moving much faster than
  others), speed differential between the fastest and slowest subgroups.
- **Synchrony** — mean pairwise cosine similarity of velocity vectors:
  $$\text{sync}_t = \frac{2}{n(n-1)}\sum_{i<j}
  \frac{\mathbf{v}_{t,i}\cdot\mathbf{v}_{t,j}}
  {\lVert\mathbf{v}_{t,i}\rVert\,\lVert\mathbf{v}_{t,j}\rVert}$$
  Cosine similarity of two vectors is $+1$ when they point the same
  direction, $-1$ when opposite, $0$ when perpendicular — averaged over
  every player pair, this is a single number in $[-1,1]$ measuring whether
  the team is moving *as a coordinated unit* (near $+1$) or at cross
  purposes (near $0$ or negative). This is the feature most directly
  targeting "coordination," the project's actual subject.
- **Ball relation** — ball position, whether it's currently detected
  (presence), and its offset from the team centroid (is the team
  positioned relative to where the play is happening).
- **Robustness channels** — explicit handling for missing values, gap
  interpolation, and identity recovery, so a track that briefly drops out
  doesn't silently corrupt every downstream feature for that window.

### 3.3 From raw window to model input: the full dimensional pipeline

1. **Raw window:** 29 frames ($\approx$ 1 second at $\sim$30 fps) $\times$
   14 values per frame (6 players $\times$ ($x,y$) = 12, plus ball
   ($x,y$) = 2).
2. **Engineered channels:** the raw positions/velocities are turned into
   18 permutation-invariant channels per frame (the categories above),
   giving a $29\times18$ tensor $Z$ — this tensor is the shared contract
   both classifiers read.
3. **Aggregated features (HGB path only):** each of the 18 channels is
   collapsed across the 29-frame window into 6 summary statistics — mean,
   std, min, max, IQR, and a least-squares slope (the trend over time,
   the one statistic a snapshot-based rule could never see) — giving a
   flat feature vector. *(The reported total is 132; confirm the exact
   channel count behind that number before quoting a specific "N × 6"
   breakdown out loud — see `DEFENSE_MASTER.md` §2.)*
4. **Mamba path:** skips step 3 entirely and consumes the full
   $29\times18$ tensor directly, preserving exact temporal order — the
   central architectural difference between the two models.

---

## Part 4 — Models

### 4.1 Histogram Gradient Boosting (HGB) — the primary model

**Gradient boosting**, in general: build an ensemble of shallow decision
trees *sequentially*, where each new tree is trained to predict the
*residual error* of the ensemble so far (in classification, the gradient
of the loss with respect to the current predictions) — so each tree
corrects what the previous ones got wrong, rather than each tree
independently voting (as in a Random Forest).

**"Histogram"** specifically: instead of considering every possible
numeric split threshold during tree-building (slow), continuous features
are pre-binned into a fixed number of histogram buckets, and splits are
chosen among bucket boundaries — much faster, and as a side effect this
binning is what lets the algorithm handle missing values natively (a
missing value is simply routed to whichever branch handles it best during
training, rather than needing imputation beforehand).

**Why it's a strong fit here specifically:** the 132-dimensional
aggregated feature vector is exactly the kind of medium-sized, tabular,
hand-engineered input this model class is empirically best-in-class on
(see `DEFENSE_MASTER.md` §6 for the literature). It needs no feature
scaling (unlike SVM/logistic regression, which are sensitive to feature
magnitude) and handles occasional missing detections gracefully.

### 4.2 Mamba (selective state-space model) — the comparison model

A **state-space model** maintains a hidden state $h_t$ that evolves over
time as a (here, discretized) linear dynamical system:

$$h_t = \bar{\mathbf{A}}\,h_{t-1} + \bar{\mathbf{B}}(x_t)\,x_t, \qquad
y_t = \mathbf{C}(x_t)\,h_t$$

This resembles a recurrent neural network's hidden-state update, but with
a more structured, more parallelizable mathematical form. **Mamba's**
specific contribution is making $\mathbf{B}$, $\mathbf{C}$, and the
discretization step $\Delta$ **input-dependent** — i.e. the model can
choose, based on what it's currently seeing, how much to write into and
read from its memory ("this frame's motion matters a lot, update the
state heavily" vs "this frame is unremarkable, mostly pass the state
through unchanged"). This *selective* gating is what distinguishes it
from a plain (non-selective) SSM, and is analogous in spirit to what
gates do in an LSTM, but scales more efficiently to long sequences.

Final prediction pools the per-frame outputs and classifies:
$\hat{y}=\mathrm{softmax}(\mathbf{W}\cdot\frac{1}{29}\sum_t y_t)$.

**Why it's the comparison arm, not (yet) primary:** it has no hand-built
inductive bias toward "these 6 statistics matter" — it must learn
everything from the raw $29\times18$ sequence, which needs more data to
reach a given accuracy than a model handed pre-digested summary
statistics. See `DEFENSE_MASTER.md` §6 for the full reasoning and the
literature this claim rests on.

---

## Part 5 — Dataset & Labels

### 5.1 The four tactical classes

Working definitions (confirm the exact operational rubric used by
whoever labelled `gold_labels.csv` before stating these as fixed
definitions to a panel — see `DEFENSE_MASTER.md` §5):

- **Coordinated Attack** — the team executes an organized offensive
  structure (e.g. setter distributes to an attacker with blockers/
  coverage positioned in support).
- **Coordinated Defense** — the team is organized in a recognized
  defensive shape (block + back-row coverage) responding to an opposing
  attack.
- **Delayed Support** — a teammate who should be backing up the
  play (covering a tip, backing up a block) arrives late or is out of
  position when needed.
- **Spacing Breakdown** — the formation's spacing itself has failed —
  players bunched together leaving court uncovered, or overextended with
  gaps a coordinated team wouldn't leave.

The features in Part 3 are direct, quantifiable proxies for these:
Spacing Breakdown should correlate with unusual hull-area/NN-distance
values; Delayed Support with desynchronized motion (low `sync`); the two
Coordinated classes with organized, moderate spacing and higher `sync`.

### 5.2 "Gold" labels

In ML, "gold" or "gold-standard" labels conventionally mean human-verified
ground truth of stated reliability — ideally with a reported
inter-annotator agreement figure (e.g. a Cohen's $\kappa$ *between two
human annotators*, using the exact same formula as §1.4, applied to
label-vs-label instead of prediction-vs-label). A filename containing
"gold" is not itself evidence of that reliability; see
`DEFENSE_MASTER.md` §5 for what to confirm before presenting this claim.

### 5.3 Class imbalance, restated numerically

Coordinated Attack (418) and Coordinated Defense (415) together are 76%
of the dataset; Delayed Support (109, 10%) is the rarest class by a wide
margin. A model can get 3 of 4 classes essentially right while never
correctly identifying Delayed Support and still post a deceptively
reasonable accuracy — which is exactly why §1.2–1.4's metrics exist.
