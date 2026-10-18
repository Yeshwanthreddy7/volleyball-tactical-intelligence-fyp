# Methodology and Mathematical Formulation

Formal specification of the pipeline. Every symbol here maps to named code, so a
reviewer can check the implementation against the definition.

---

## 1. Problem formulation

Let a rally window be a sequence of $T = 29$ frames (≈1 s at 30 fps). At frame
$t$ the system observes a set of player court positions

$$\mathcal{P}_t = \{\mathbf{p}_{t,1},\dots,\mathbf{p}_{t,n_t}\},\quad \mathbf{p}_{t,i}\in\mathbb{R}^2,\ n_t\le 6$$

and an optional ball position $\mathbf{b}_t\in\mathbb{R}^2\cup\{\varnothing\}$.

The task is a mapping

$$f:\ (\mathcal{P}_1,\mathbf{b}_1),\dots,(\mathcal{P}_T,\mathbf{b}_T)\ \longmapsto\ y\in\{0,1,2,3,4\}$$

**Key constraint:** $\mathcal{P}_t$ is a **set**, not a vector. Two orderings of
the same six players describe the same formation and must give the same $y$.
Any representation indexed by player slot violates this. Section 4 constructs a
permutation-invariant descriptor instead. *(`fyp/features.py`)*

---

## 2. Pixel → court projection

### 2.1 Homography

The court floor is a plane, so the pixel↔court map is a **homography**
$\mathbf{H}\in\mathbb{R}^{3\times3}$ (8 DOF) acting on homogeneous coordinates:

$$\begin{bmatrix}u'\\v'\\w'\end{bmatrix} = \mathbf{H}\begin{bmatrix}X\\Y\\1\end{bmatrix},\qquad (x_{\text{court}},y_{\text{court}}) = \left(\frac{u'}{w'},\frac{v'}{w'}\right)$$

Estimated from four court-corner correspondences via `cv2.findHomography` with
RANSAC. The court frame is $1800 \times 900$ cm with the net at $y = 450$.
*(`fyp/court.py: build_homography`)*

**Why it matters:** without $\mathbf{H}$, equal pixel distances at different
depths correspond to different real distances. All four tactical classes are
defined by spacing and velocity, so the homography is what makes the features
mean what the labels claim.

### 2.2 Foot point

A player's court contact is the **bottom-centre** of the bounding box
$(x_1,y_1,x_2,y_2)$:

$$\mathbf{q} = \left(\tfrac{x_1+x_2}{2},\ y_2\right)$$

not the centroid — only the feet lie on the floor plane, and the box centroid
floats ~half a body up-court, biasing distance with camera depth.
*(`fyp/features.py: foot_point`)*

### 2.3 Camera-motion compensation

Between calibration refreshes, a partial-affine camera model
$\mathbf{A}\in\mathbb{R}^{2\times3}$ is estimated from sparse Lucas–Kanade
optical flow with RANSAC, and applied to the stored corners:

$$\mathbf{c}_{t} = \mathbf{A}\,[\mathbf{c}_{t-1};\,1]$$

*(`fyp/court.py: CameraMotionEstimator`)*

**Stated limitation:** this is incremental and 4-DOF. It cannot represent the
perspective change of a tilting camera, and composing it over hundreds of frames
accumulates drift. A drift-free formulation registers each frame directly to a
reference keyframe by homography — the correct model for a camera rotating about
its optical centre. Documented as future work in `Technical_Review_and_QA.md` §13.3.

---

## 3. Identity and team assignment

### 3.1 Persistent slots

Tracker IDs are mapped to six persistent slots $s:\ \text{id}\to\{1..6\}$, held
across occlusion gaps. Without this, $\Delta\mathbf{p}$ over a slot measures the
jump between *different people*, corrupting every velocity feature.
*(`fyp/identity.py`)*

### 3.2 Occlusion-gap interpolation

An interior NaN run of length $L$ bounded on both sides is bridged by linear
interpolation iff $L \le 15$ frames. Longer gaps and leading/trailing gaps are
left missing, so a player who genuinely left the frame is never fabricated.
*(`fyp/features.py: interpolate_gaps`)*

### 3.3 Team separation

Torso colour descriptor for a detection, from the median HSV over the torso
patch:

$$\mathbf{d} = \big(s\cos h,\ s\sin h,\ v\big)$$

Encoding hue as a vector scaled by saturation keeps the hue metric circular, and
collapses desaturated (white/black) kit toward the origin where it separates on
$v$.

$k$-means partitions $\{\mathbf{d}\}$ into $k = 4$ clusters. Cluster $j$ is
assigned to the near team iff

$$\operatorname{median}\{\,y_{\text{court}}(i)\ :\ \text{cluster}(i)=j\,\}\ \ge\ 450$$

Then team membership is decided by colour alone and majority-voted over each
track's lifetime.

**Justification.** Geometry is reliable *in aggregate* (players may not cross the
net) but ambiguous *instantaneously* (both front rows stand within a metre at the
net). Colour is reliable instantaneously but carries no team identity. Each
signal is used only where it is valid. $k=4$ rather than $2$ because FIVB rules
require a contrasting libero jersey, so a team is two colour populations.
*(`fyp/teams.py`)*

A fit placing **every** cluster on one side is flagged degenerate and rejected in
favour of geometry — it indicates the crowd, not two teams.

---

## 4. Feature construction

### 4.1 Permutation-invariant per-frame descriptor

For frame $t$ with present players $\mathcal{P}_t$ and centroid
$\bar{\mathbf{p}}_t = \frac{1}{n_t}\sum_i \mathbf{p}_{t,i}$:

| Symbol | Definition |
|---|---|
| $\bar{\mathbf{p}}_t$ | team centroid |
| $\sigma_x,\sigma_y$ | positional spread per axis |
| $A_t$ | convex-hull area of $\mathcal{P}_t$ |
| $d^{\min}_t,\ \bar d_t,\ d^{\max}_t$ | min / mean / max nearest-neighbour distance |
| $D_t = \max_{i\ne j}\lVert\mathbf{p}_{t,i}-\mathbf{p}_{t,j}\rVert$ | maximum pair distance |
| $n_t$ | players present |
| $\lVert \mathbf{b}_t - \bar{\mathbf{p}}_t\rVert$ | ball-to-centroid distance |

Each is a **symmetric function** of the player set, hence invariant to ordering.

### 4.2 Kinematic channels

Velocity $\mathbf{v}_{t,i} = \mathbf{p}_{t,i}-\mathbf{p}_{t-1,i}$ on
identity-consistent tracks.

**Synchronisation** — mean pairwise cosine similarity of velocity vectors:

$$\text{sync}_t = \frac{2}{n(n-1)}\sum_{i<j} \frac{\mathbf{v}_{t,i}\cdot\mathbf{v}_{t,j}}{\lVert\mathbf{v}_{t,i}\rVert\,\lVert\mathbf{v}_{t,j}\rVert} \in [-1,1]$$

$\text{sync}_t \to 1$ is a rigid unit shift (Coordinated Defense);
$\text{sync}_t \to 0$ is uncoordinated motion.

**Speed differential** — separates frontline attackers from base players:

$$\Delta v_t = \underbrace{\tfrac{1}{2}\textstyle\sum_{i\in\text{top-2}}\lVert\mathbf{v}_{t,i}\rVert}_{\text{attackers}} - \underbrace{\tfrac{1}{4}\textstyle\sum_{i\in\text{bottom-4}}\lVert\mathbf{v}_{t,i}\rVert}_{\text{base}}$$

Together: $\mathbf{z}_t \in \mathbb{R}^{18}$, the `perm_invariant_v2` contract.
A window is $\mathbf{Z} \in \mathbb{R}^{29\times18}$.

### 4.3 Window-level aggregation

For each channel $c$, six statistics: mean, std, min, max, IQR, and least-squares
slope

$$\beta_c = \frac{\sum_t (t-\bar t)(z_{t,c}-\bar z_c)}{\sum_t (t-\bar t)^2}$$

The slope is the channel the rule engine cannot see: it reads scalars at fixed
cut-points, so "spacing widening" and "spacing already wide" are
indistinguishable to it and separable here.

### 4.4 Scale-invariant block

Under the linear pixel→court fallback the scale factor $\lambda$ is unknown and
**camera-specific**: $\mathbf{p} \mapsto \lambda\mathbf{p}$. Any absolute length
feature is then partly a camera identifier. Two constructions are invariant to
$\lambda$:

**(a) Ratios of like quantities** — $\lambda$ cancels:

$$\rho_1=\frac{\bar d_t}{D_t},\quad \rho_2=\frac{d^{\min}_t}{\bar d_t},\quad \rho_3=\frac{d^{\max}_t}{\bar d_t},\quad \rho_4=\frac{\sigma_x}{\sigma_y},\quad \rho_5=\frac{v^{\text{top2}}}{v^{\text{bot4}}}$$

**(b) Per-clip z-scoring** — retains temporal *shape*, discards level and scale:

$$\tilde z_{t,c} = \frac{z_{t,c}-\mu_c}{\sigma_c}$$

**This is the project's principal experimental finding** (§6.2).

---

## 5. Classification

Learned classifier $f_\theta:\mathbb{R}^{d}\to\Delta^4$ fitted by empirical risk
minimisation with inverse-frequency class weights $w_k \propto 1/n_k$ (absent
classes get $w_k = 0$ exactly — a nonzero weight on an empty class caused total
model collapse, `Technical_Review_and_QA.md` §12.6).

Nine candidates are evaluated: logistic regression, linear/RBF SVM, kNN, MLP,
random forest, extra trees, gradient boosting, histogram gradient boosting.
All preprocessing lives **inside** the cross-validation pipeline so it is refit
per fold — fitting a scaler before splitting leaks test statistics into training.

**Deep-learning comparison arm:** a Mamba selective state-space model
$h_t = \bar{\mathbf{A}}h_{t-1} + \bar{\mathbf{B}}x_t$, $y_t = \mathbf{C}h_t$ with
input-dependent $(\mathbf{B},\mathbf{C},\Delta)$, over the same $29\times18$
tensor. *(`fyp/mamba_model.py`)*

### 5.1 Calibration

Tree ensembles emit $\max_k p_k = 1.000$ on nearly every window, which silently
disables the entropy channel and the anomaly flag. Cross-validated Platt scaling

$$p(y=k\mid s) = \sigma(a_k s + b_k)$$

restores a usable range. Sigmoid rather than isotonic: isotonic is
non-parametric and would overfit a 16-example class.

### 5.2 Anomaly / tactical deviation

$$H(\mathbf{p}) = -\sum_k p_k\log_2 p_k,\qquad \text{anomaly} \iff \max_k p_k < \tau\ (\tau = 0.5)$$

High entropy = the window matches no tactical template the model knows. This is
an exploratory novelty indicator, **not** a coordinated-vs-breakdown classifier.

---

## 6. Evaluation

### 6.1 Metrics

With confusion matrix $C$, per class $k$: $P_k = \frac{C_{kk}}{\sum_i C_{ik}}$,
$R_k = \frac{C_{kk}}{\sum_j C_{kj}}$.

$$\text{macro-F1} = \frac{1}{K}\sum_k \frac{2P_kR_k}{P_k+R_k},\qquad \text{balanced acc} = \frac{1}{K}\sum_k R_k$$

$$\kappa = \frac{p_o - p_e}{1 - p_e}$$

**Macro-F1 and $\kappa$ are the headline metrics, never accuracy.** With an
8.4:1 imbalance, always predicting the majority class scores 0.470 accuracy
while carrying zero information; $\kappa$ is 0 for exactly that predictor.

### 6.2 Two protocols, always reported together

| Protocol | Split | Measures |
|---|---|---|
| Stratified $k$-fold ($k=5$) | clips from one video on both sides | within-video generalisation (optimistic) |
| **Leave-one-video-out** | all clips of one video held out | **cross-video generalisation (honest)** |

The **gap** between them quantifies reliance on camera-specific cues. Measured:

| features | $d$ | k-fold macro-F1 | LOVO macro-F1 | $\kappa$ |
|---|---|---|---|---|
| absolute | 108 | **0.520** | 0.265 | 0.081 |
| scale-invariant | 132 | 0.440 | **0.299** | **0.161** |

Making features dimensionless **lowers** within-video score and **raises**
cross-video score, nearly doubling $\kappa$. That opposing movement is the
signature of removing a shortcut: the absolute block was buying k-fold
performance with camera identity, which transfers to no new match.

### 6.3 Interpretability

Channel-level permutation importance, measured **out of sample**:

$$I_c = \mathbb{E}\big[\text{F1}_{\text{macro}}(y,\hat y) - \text{F1}_{\text{macro}}(y,\hat y^{\pi(c)})\big]$$

where $\pi(c)$ permutes **all statistics of channel $c$ jointly**, across LOVO
folds with the model refit per fold.

Two corrections over the textbook call, both of which changed the answer:
single-column permutation gave every importance $<0.0005$ (correlated columns
substitute for each other), and in-sample measurement gave baseline macro-F1
$1.000$ (the model has memorised, so nothing looks important).

### 6.4 Probability-ranking metrics (senior-data-scientist checklist)

Every metric in 6.2 is a **hard-label** metric — a function only of
$\arg\max_k p_k$. On an 8.4:1-imbalanced dataset this can look like a class
carries zero signal purely because the argmax loses to a majority class. The
one-vs-rest ranking metrics ask a different, more forgiving question:

$$\text{ROC-AUC}_k = P(p_k(x^+) > p_k(x^-)),\qquad
\text{PR-AUC}_k = \int_0^1 \text{Precision}_k(r)\, dr$$

macro-averaged over $k$, computed on the pooled out-of-fold probabilities
under both protocols (pooling avoids the degenerate case of a fold with zero
examples of the 16-example Delayed Support class). PR-AUC is the more
imbalance-sensitive of the two: ROC is insensitive to the negative:positive
ratio, so a classifier can score well on ROC while being useless once
negatives vastly outnumber positives.

Measured, LOVO protocol, winning model: **ROC-AUC macro $= 0.543$** — chance
is $0.500$ — and **PR-AUC macro $= 0.309$** against a macro-average class
prevalence of $\approx 0.25$. Cross-video ranking quality is close to chance
for every one of the nine candidates evaluated, not only the winner.

### 6.5 Overfit gap

For cross-validation splitter $\mathcal{S}$ and model $f_\theta$, refit per
fold with `return_train_score=True`:

$$\text{gap} = \mathbb{E}_{\text{fold}}\big[\text{F1}_{\text{macro}}^{\text{train}}\big] - \mathbb{E}_{\text{fold}}\big[\text{F1}_{\text{macro}}^{\text{test}}\big],\qquad \text{flag iff gap} > 0.05$$

This is independent of the k-fold-vs-LOVO protocol gap in 6.2 — it measures
overfitting *within a single fold*, refitting the same model on training rows
versus held-out rows of the *same* split. Measured under LOVO: **every one of
the nine candidates is flagged**, with gaps of $0.71$–$0.78$ (train
macro-F1 $\approx 1.0$, held-out $\approx 0.26$–$0.30$). With three source
videos, LOVO trains on two of them; $\sim$130–190 rows in $d=132$ dimensions
is insufficient for any model class evaluated to avoid substantially
memorising the training videos.

### 6.7 The baseline ladder

Three predictors, identical input window $\mathbf{Z}\in\mathbb{R}^{29\times18}$,
identical human ground truth $y$:

$$\hat y_{\text{maj}} = \arg\max_k n_k \qquad
\hat y_{\text{rule}} = \texttt{label\_clip}(\mathbf{Z}) \qquad
\hat y_{\text{ml}} = \arg\max_k f_\theta(\mathbf{Z})_k$$

$\hat y_{\text{rule}}$ is the hand-coded rule engine (§5) evaluated on the
*observation* window. Because $y$ is human annotation and the rules did not
produce it (see §7), this comparison is well-posed: the rules are an
independent competing predictor, not the label source.

| predictor | accuracy | macro-F1 | $\kappa$ |
|---|---|---|---|
| majority class | 0.470 | 0.160 | 0.000 |
| rule engine | 0.432 | 0.193 | 0.097 |
| learned model (LOVO) | 0.484 | **0.299** | **0.161** |

The rule engine abstains ("Unclassified") on 8% of clips; abstentions are
scored as misses, never dropped — a baseline that declines the hard cases must
not be credited for skipping them.

### 6.6 Distribution drift between source videos

Direct measurement of the "per-video signature" (6.2), rather than only the
indirect performance-gap argument. For feature $c$ and videos $A,B$:

$$D_{\text{KS}}(c) = \sup_x \big| F_A^c(x) - F_B^c(x) \big|,\qquad
\text{PSI}(c) = \sum_i (q_i^B - q_i^A)\,\ln\!\frac{q_i^B}{q_i^A}$$

with $q_i$ the fraction of samples in decile bin $i$ of $A$'s own quantiles
(PSI is therefore asymmetric by construction — $A\!\to\!B \ne B\!\to\!A$ — which
is the correct behaviour for a live monitor scoring new data against a fixed
training reference). Conventional reading: $\text{PSI}<0.1$ no shift,
$0.1$–$0.25$ moderate, $>0.25$ major.

Measured across all six ordered pairs of the three source videos, on the
132-dimensional scale-invariant block: mean PSI $0.54$–$1.51$, with
$64$–$73\%$ of individual features clearing the *major*-shift threshold. The
scale-invariant construction (6.2) measurably reduces but does not eliminate
this — the most-shifted channels include already-dimensionless quantities
($n_{\text{present}}$, a count; $\text{sync}_t$, a cosine similarity), so the
residual shift reflects genuine between-match differences, not only a units
artefact. *(`fyp/drift.py`)*

---

## 7. Validity threats

| Threat | Status |
|---|---|
| ~~Label circularity~~ — *withdrawn.* Labels are HUMAN annotations (`labels.csv` → clip folder → `target_label`), verified 159/159 with zero mismatches; `label_clips.py` is a separate, unused path | **Closed.** The rule engine is instead scored as an independent competing baseline (§6.7) |
| **Single annotator** — no inter-annotator agreement, so the true performance ceiling is unknown | **Open.** Blind second-pass protocol implemented (`fyp/annotate.py`); requires a second human |
| Small minority class ($n=16$) | Open. Balanced gold sampling + class weights mitigate, do not solve |
| Per-video signature | **Quantified directly** — mean inter-video PSI $0.54$–$1.51$ (§6.6), not only the indirect LOVO/k-fold gap; partially addressed by §4.4 |
| **Cross-video overfitting** — every candidate model memorises the training videos under LOVO | **Quantified.** All 9 models flagged (train/held-out gap $0.71$–$0.78$, §6.5); cross-video ROC-AUC $\approx$ chance (§6.4). Root cause: 3 source videos is too few |
| Non-metric distances without homography | Documented; scale-invariant features are the workaround |
| Silent pipeline failure | **Closed.** Preflight gate aborts (exit 2) on a degenerate population |
| Train/serve feature mismatch | **Closed.** One extractor shared by training and live inference, with a version tag |

---

## 8. Datasets

| Dataset | Size | Use |
|---|---|---|
| Broadcast match video | 3 sources confirmed in-repo, ~1280×720 @30 fps (a 4th, `(plain)`, appears only as pre-cut clips — no source file present, unverified whether it duplicates one of the 3) | detection, tracking, demo |
| Roboflow detection set | 416 images, 4,672 player + 233 ball instances | ball detector fine-tune (temporal block split — a random split puts near-identical consecutive frames on both sides) |
| Tactical clip set | 285 clips currently extracted and evaluated (3-video LOVO); 508-clip collection including the unverified 4th source is not yet re-extracted | classifier training/evaluation |

Class distribution 134 / 90 / 45 / 16 — an 8.4:1 imbalance that drives the choice
of macro-F1 and $\kappa$ throughout.
