"""
make_data_atlas.py  --  one-shot regeneration of the project Data Atlas.

Reads every scattered data artefact in the repo (raw per-clip tracking CSVs,
the full broadcast per-frame feature trace, window-level predictions, the
model-comparison JSON/MD, the deep training histories) and renders a labelled
figure set into figures_atlas/, grouped by the audience each figure speaks to:

  1_dataset      what the corpus is         (all reviewers)
  2_cv           detection / tracking       (computer-vision reviewers)
  3_volleyball   tactical / biomechanical   (volleyball / domain reviewers)
  4_ml           model comparison           (classical-ML reviewers)
  5_dl           deep temporal models       (deep-learning reviewers)

Run from the finalyearproject/ directory:  python make_data_atlas.py
"""

import os
import re
import json
import glob
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.normpath(os.path.join(HERE, "..", "temporal_bench"))
OUT = os.path.join(HERE, "figures_atlas")

# ---------------------------------------------------------------- palette
# dataviz reference categorical palette, assigned in fixed order.
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
       "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
CLASSES = ["Coordinated Attack", "Coordinated Defense",
           "Delayed Support", "Spacing Breakdown"]
CCOL = {"Coordinated Attack": "#2a78d6", "Coordinated Defense": "#eb6834",
        "Delayed Support": "#1baf7a", "Spacing Breakdown": "#eda100",
        "Unclassified": "#9aa0a6"}
CMARK = {"Coordinated Attack": "o", "Coordinated Defense": "s",
         "Delayed Support": "^", "Spacing Breakdown": "D"}
SHORT = {"Coordinated Attack": "Attack", "Coordinated Defense": "Defense",
         "Delayed Support": "Delayed", "Spacing Breakdown": "Spacing",
         "Unclassified": "Uncl."}
FAMCOL = {"ML": "#2a78d6", "DL": "#eb6834", "baseline": "#4a3aa7"}

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.labelsize": 10, "axes.edgecolor": "#666666",
    "axes.linewidth": 0.8, "xtick.color": "#333333", "ytick.color": "#333333",
    "axes.grid": True, "grid.color": "#dddddd", "grid.linewidth": 0.6,
    "legend.frameon": False, "figure.dpi": 130,
})


def ax_clean(ax, grid_axis="y"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis=grid_axis, alpha=0.7)
    ax.set_axisbelow(True)




def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, name)
    fig.savefig(p, bbox_inches="tight", dpi=130)
    plt.close(fig)
    print("  wrote", os.path.relpath(p, HERE))


def load_json(path):
    s = open(path).read()
    s = re.sub(r"\bNaN\b", "null", s)
    return json.loads(s)


def video_of(fname):
    m = re.search(r"\(([^)]+)\)", fname)
    if m:
        return m.group(1)
    low = fname.lower()
    for t in ("plain", "usa"):
        if t in low:
            return t
    return "?"


def class_of(fname):
    return fname.split("__")[0].replace("_", " ")


def corpus_frame(d):
    rows = []
    for f in glob.glob(os.path.join(d, "*.csv")):
        b = os.path.basename(f)
        if b in ("extraction_provenance.json",):
            continue
        rows.append({"file": b, "cls": class_of(b), "video": video_of(b)})
    return pd.DataFrame(rows)


# ======================================================================
# 1  DATASET
# ======================================================================
def fig_inventory():
    corpora = [("training_csv", "v1  (pre-detector-fix)"),
               ("training_csv_v3", "v3  (fixed pipeline, 3 videos)"),
               ("training_csv_v4", "v4  (6+ videos, rebalanced)")]
    counts, labels, vids = [], [], []
    for d, lab in corpora:
        df = corpus_frame(os.path.join(HERE, d))
        counts.append(len(df))
        labels.append(lab)
        vids.append(df.video.nunique())
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2), gridspec_kw={"wspace": 0.55})
    y = np.arange(len(counts))
    axes[0].barh(y, counts, color=PAL[0], height=0.6)
    for i, (c, v) in enumerate(zip(counts, vids)):
        axes[0].text(c + 12, i, f"{c} clips · {v} videos", va="center", fontsize=9)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)
    axes[0].set_xlim(0, max(counts) * 1.4)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("1-second labelled tracking clips  (29 frames each)")
    axes[0].set_title("Training corpora — every clip is a 6-player + ball track")
    ax_clean(axes[0], "x")

    other = [
        ("Broadcast feature trace", 80330, "frames · 2 685 s match"),
        ("13-model prediction matrix", 309 * 13, "model x window cells"),
        ("Output tactical windows", 309, "1 s windows · 4-class + reject"),
        ("Biomechanical summary", 285, "clips x 13 measured features"),
    ]
    y2 = np.arange(len(other))
    vals = [o[1] for o in other]
    axes[1].barh(y2, vals, color=PAL[2], height=0.55)
    for i, o in enumerate(other):
        axes[1].text(o[1] * 1.15, i, f"{o[1]:,}", va="center", fontsize=9, fontweight="bold")
        axes[1].text(o[1] * 1.15, i - 0.32, o[2], va="center", fontsize=7.5, color="#666")
    axes[1].set_yticks(y2)
    axes[1].set_yticklabels([o[0] for o in other], fontsize=9)
    axes[1].set_xscale("log")
    axes[1].set_xlim(100, max(vals) * 40)
    axes[1].set_ylim(len(other) - 0.5, -0.7)
    axes[1].set_xlabel("row / cell count  (log scale)")
    axes[1].set_title("Other raw data assets fed into the atlas")
    ax_clean(axes[1], "x")
    fig.suptitle("1 · What data exists in this project", fontsize=13, fontweight="bold", y=1.06)
    save(fig, "1_dataset/01_corpus_inventory.png")


def fig_class_distribution():
    corpora = [("training_csv", "v1  (285)"), ("training_csv_v3", "v3  (375)"),
               ("training_csv_v4", "v4  (945)")]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    w = 0.26
    x = np.arange(len(CLASSES))
    for j, (d, lab) in enumerate(corpora):
        df = corpus_frame(os.path.join(HERE, d))
        n = len(df)
        share = [100 * (df.cls == c).sum() / n for c in CLASSES]
        b = ax.bar(x + (j - 1) * w, share, w, label=lab, color=PAL[j])
        for rect, s, c in zip(b, share, CLASSES):
            cnt = int(round(s / 100 * n))
            ax.text(rect.get_x() + rect.get_width() / 2, s + 0.6, f"{cnt}",
                    ha="center", fontsize=8, color="#333")
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[c] for c in CLASSES])
    ax.set_ylabel("share of corpus  (%)")
    ax.set_title("Class balance across corpus versions — the minority classes\n"
                 "(Delayed Support, Spacing Breakdown) stay rare even after rebalancing")
    ax.legend(title="corpus (n clips)", loc="upper right")
    ax_clean(ax)
    save(fig, "1_dataset/02_class_distribution.png")


def fig_class_by_video():
    df = corpus_frame(os.path.join(HERE, "training_csv_v3"))
    vids = sorted(df.video.unique(), key=lambda v: (v.isdigit() is False, v))
    fig, ax = plt.subplots(figsize=(9, 4.6))
    bottoms = np.zeros(len(vids))
    for ci, c in enumerate(CLASSES):
        vals = np.array([100 * ((df.video == v) & (df.cls == c)).sum() / (df.video == v).sum()
                         for v in vids])
        ax.bar(range(len(vids)), vals, 0.6, bottom=bottoms, label=SHORT[c], color=CCOL[c])
        for i, (val, b) in enumerate(zip(vals, bottoms)):
            if val > 4:
                ax.text(i, b + val / 2, f"{val:.0f}%", ha="center", va="center",
                        fontsize=8, color="white", fontweight="bold")
        bottoms += vals
    ax.set_xticks(range(len(vids)))
    ax.set_xticklabels([f"video ({v})\nn={ (df.video==v).sum() }" for v in vids])
    ax.set_ylabel("class share within the video  (%)")
    ax.set_title("Label-prior shift between source videos (corpus v3)\n"
                 "Spacing Breakdown is 31% of video (1) but ~4% of video (3) — "
                 "a confound for leave-one-video-out")
    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.32))
    ax_clean(ax)
    save(fig, "1_dataset/03_class_by_video.png")


def fig_feature_pca():
    c = pd.read_csv(os.path.join(HERE, "coordination_analysis.csv"))
    feats = ["top2_speed_cms", "rest_speed_cms", "speed_ratio", "mean_spacing_cm",
             "centroid_vel_cms", "sync_score", "mean_hull_area", "reaction_lag_frames"]
    X = c[feats].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median())
    Xs = (X - X.mean()) / X.std(ddof=0)
    U, S, Vt = np.linalg.svd(Xs.values, full_matrices=False)
    pc = U[:, :2] * S[:2]
    var = (S ** 2 / (S ** 2).sum())[:2] * 100
    fig, ax = plt.subplots(figsize=(7.6, 6))
    for cls in CLASSES:
        m = (c.label == cls).values
        ax.scatter(pc[m, 0], pc[m, 1], s=34, c=CCOL[cls], marker=CMARK[cls],
                   edgecolors="white", linewidths=0.5, alpha=0.85, label=SHORT[cls])
    ax.set_xlabel(f"PC1  ({var[0]:.0f}% var)")
    ax.set_ylabel(f"PC2  ({var[1]:.0f}% var)")
    ax.set_title("1 · PCA of the 8 measured biomechanical features (285 clips)\n"
                 "the four classes overlap heavily in feature space — the raw signal for the task is weak",
                 pad=12)
    ax.legend(title="tactical class")
    ax_clean(ax, "both")
    save(fig, "1_dataset/04_feature_space_pca.png")


# ======================================================================
# 2  COMPUTER VISION
# ======================================================================
def fig_player_funnel():
    stages = ["Raw YOLOv11\n(COCO person)", "After court\nmask", "After jersey-colour\nteam split",
              "Regulation\nteam size"]
    vals = [18.0, 10.7, 5.8, 6.0]
    lo = [14, np.nan, np.nan, np.nan]
    hi = [23, np.nan, np.nan, np.nan]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    colors = [PAL[0], PAL[0], PAL[2], "#888888"]
    b = ax.bar(range(4), vals, 0.6, color=colors)
    ax.errorbar(0, 18, yerr=[[18 - 14], [23 - 18]], fmt="none", ecolor="#333", capsize=5, lw=1.2)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.4, f"{v:g}", ha="center", fontsize=10, fontweight="bold")
    ax.axhline(6, color="#888888", ls="--", lw=1)
    ax.set_xticks(range(4))
    ax.set_xticklabels(stages)
    ax.set_ylabel("players per frame")
    ax.set_title("2 · Players-per-frame through the pre-classifier pipeline\n"
                 "the court mask and colour split turn a 10-player broadcast crop into a 6-slot system",
                 pad=16)
    ax.set_ylim(0, 26)
    ax_clean(ax)
    fig.text(0.5, -0.06,
             "measured on this project's footage — README §3.2 ; per-corpus feature extraction: "
             "4.40 players/frame (v1) -> 5.74 (v3)", ha="center", fontsize=8, color="#555")
    save(fig, "2_cv/05_player_funnel.png")


def fig_ball_recall():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    fig.subplots_adjust(top=0.72, wspace=0.28)
    axes[0].bar(["640 px\n(Ultralytics default)", "1280 px\n(this project)"], [15, 77],
                color=[PAL[7], PAL[2]], width=0.55)
    for i, v in enumerate([15, 77]):
        axes[0].text(i, v + 2, f"{v}%", ha="center", fontweight="bold")
    axes[0].set_ylabel("ball recall  (%)")
    axes[0].set_ylim(0, 100)
    axes[0].set_title("identical weights — resolution only", fontsize=10.5)
    ax_clean(axes[0])

    axes[1].bar(["stock yolo11n", "volleyball\nfine-tune"], [16, 67],
                color=[PAL[7], PAL[2]], width=0.55)
    for i, v in enumerate([16, 67]):
        axes[1].text(i, v + 2, f"{v}%", ha="center", fontweight="bold")
    axes[1].set_ylabel("ball recall  (%)")
    axes[1].set_ylim(0, 100)
    axes[1].set_title("identical resolution (1280) — weights only", fontsize=10.5)
    ax_clean(axes[1])
    fig.suptitle("2 · Why the detector is dual (stock players + fine-tuned ball) and runs at 1280\n"
                 "a ~15 px ball drops below the detection stride at 640; the fine-tune then wins on the ball",
                 fontsize=11.5, fontweight="bold", y=1.04)
    save(fig, "2_cv/06_ball_recall.png")


def fig_detector_ap():
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    names = ["player AP@50", "ball AP@50", "overall mAP@50"]
    vals = [0.819, 0.543, 0.681]
    b = ax.bar(names, vals, 0.55, color=[PAL[0], PAL[1], PAL[6]])
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontweight="bold")
    ax.set_ylim(0, 1)
    ax.set_ylabel("average precision")
    ax.set_title("Formal detector metrics on the leak-free labelled validation split\n"
                 "the player/ball split is confirmed by AP, not just raw detection counts")
    ax_clean(ax)
    save(fig, "2_cv/07_detector_ap.png")


def _one_clip(path):
    d = pd.read_csv(path)
    P = [(f"p{i}_x", f"p{i}_y") for i in range(1, 7)]
    return d


def fig_court_trajectories():
    base = os.path.join(HERE, "training_csv_v3")
    picks = {}
    for c in CLASSES:
        tag = c.replace(" ", "_")
        cands = sorted(glob.glob(os.path.join(base, f"{tag}__*(1)*.csv"))) or \
                sorted(glob.glob(os.path.join(base, f"{tag}__*.csv")))
        if cands:
            picks[c] = cands[0]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.4))
    for ax, c in zip(axes.flat, CLASSES):
        d = pd.read_csv(picks[c])
        n = len(d)
        for i in range(1, 7):
            xs, ys = d[f"p{i}_x"].values, d[f"p{i}_y"].values
            ax.plot(xs, ys, "-", color=CCOL[c], alpha=0.35, lw=1)
            sc = ax.scatter(xs, ys, c=np.arange(n), cmap="Blues", s=14, vmin=0, vmax=n)
        if d["ball_x"].notna().any():
            ax.plot(d["ball_x"], d["ball_y"], "-", color="#e34948", lw=1.4, label="ball")
            ax.scatter(d["ball_x"], d["ball_y"], c="#e34948", s=10)
        ax.invert_yaxis()
        ax.set_title(f"{SHORT[c]}   ·   {os.path.basename(picks[c])[:38]}…", fontsize=10)
        ax.set_xlabel("image x (px)")
        ax.set_ylabel("image y (px)")
        ax.grid(alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("2 · Raw 6-player + ball tracks, one representative clip per tactical class\n"
                 "(corpus v3 · 29 frames ≈ 1 s · marker colour = time, light→dark · red = ball)",
                 fontsize=12, fontweight="bold")
    save(fig, "2_cv/08_court_trajectories.png")


def fig_broadcast_timeline():
    f = pd.read_csv(os.path.join(HERE, "v7_tactical", "results", "vp7_frames.csv"))
    f = f.sort_values("time_s")
    win = 300
    ball = f["feat_ball_present"].rolling(win, min_periods=30).mean()
    npl = f["n_players_tracked"].rolling(win, min_periods=30).mean()
    ev = load_json(os.path.join(HERE, "v7_tactical", "results", "vp7_score_events.json"))
    ev_t = np.array(ev["events"]) / 30.0
    fig, axes = plt.subplots(2, 1, figsize=(12, 5.4), sharex=True)
    for t in ev_t:
        axes[0].axvline(t, color="#e34948", alpha=0.14, lw=0.8)
        axes[1].axvline(t, color="#e34948", alpha=0.14, lw=0.8)
    axes[0].plot(f["time_s"], ball, color=PAL[0], lw=1.4)
    axes[0].set_ylabel("ball-present\nfraction (30 s roll)")
    axes[0].set_ylim(0.9, 1.005)
    axes[0].set_title("2 · Detection reliability across a full 45-minute broadcast (vp7 · 80 330 frames)\n"
                      "ball detected in 99.8% of frames · mean 7.2 tracked players (regulation = 6, both teams in crop)",
                      pad=10)
    axes[1].plot(f["time_s"], npl, color=PAL[2], lw=1.4)
    axes[1].axhline(6, color="#333", ls="--", lw=1)
    axes[1].set_ylabel("tracked players\n(30 s roll)")
    axes[1].set_xlabel("match time  (s)")
    for a in axes:
        ax_clean(a)
    axes[0].legend([Line2D([0], [0], color="#e34948", alpha=0.5)],
                   ["scoreboard-change event (n=96)"], loc="lower right", fontsize=8)
    axes[1].legend([Line2D([0], [0], color="#333", ls="--")],
                   ["regulation team size = 6"], loc="upper right", fontsize=8)
    save(fig, "2_cv/09_broadcast_detection_timeline.png")


def fig_raw_vs_smooth():
    f = pd.read_csv(os.path.join(HERE, "v7_tactical", "results", "vp7_frames.csv"))
    # find a calm ~12 s window where slot p3 barely moves (a back-court player at rest)
    f = f.sort_values("time_s").reset_index(drop=True)
    seg = f[(f.time_s >= 610) & (f.time_s <= 624)]
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    for ax, coord, lab in [(axes[0], "x", "image x  (px)"), (axes[1], "y", "image y  (px)")]:
        ax.plot(seg["time_s"], seg[f"raw_p3_{coord}"], "-", color="#c4c4c4", lw=1.6,
                label="raw per-frame detection")
        ax.plot(seg["time_s"], seg[f"smooth_p3_{coord}"], "-", color=PAL[0], lw=2.2,
                label="BoT-SORT + Kalman track")
        ax.set_ylabel(lab)
        ax_clean(ax)
    axes[0].set_title("2 · Tracking filter — one player slot, position vs time over a 14 s window\n"
                      "the Kalman track absorbs detector jitter and rides through dropped-frame gaps", pad=12)
    axes[1].set_xlabel("match time  (s)")
    axes[0].legend(loc="upper right", fontsize=8)
    save(fig, "2_cv/10_tracking_raw_vs_smooth.png")


# ======================================================================
# 3  VOLLEYBALL / DOMAIN
# ======================================================================
def fig_biomech_by_class():
    c = pd.read_csv(os.path.join(HERE, "coordination_analysis.csv"))
    specs = [("top2_speed_cms", "front-court attacker speed (cm/s)"),
             ("mean_spacing_cm", "mean inter-player spacing (cm)"),
             ("sync_score", "movement synchrony score"),
             ("speed_ratio", "front / back speed ratio"),
             ("reaction_lag_frames", "nearest-player reaction lag (frames)"),
             ("mean_hull_area", "formation convex-hull area (px²)")]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    fig.subplots_adjust(hspace=0.5, wspace=0.28)
    for ax, (col, lab) in zip(axes.flat, specs):
        data = [pd.to_numeric(c[c.label == cls][col], errors="coerce").dropna().values
                for cls in CLASSES]
        bp = ax.boxplot(data, patch_artist=True, widths=0.6, showfliers=False)
        for patch, cls in zip(bp["boxes"], CLASSES):
            patch.set_facecolor(CCOL[cls])
            patch.set_alpha(0.75)
        for med in bp["medians"]:
            med.set_color("#111")
        for i, d in enumerate(data):
            x = np.random.normal(i + 1, 0.05, len(d))
            ax.scatter(x, d, s=6, c="#333", alpha=0.25)
        ax.set_xticks(range(1, 5))
        ax.set_xticklabels([SHORT[c] for c in CLASSES], fontsize=8.5)
        ax.set_title(lab, fontsize=10)
        ax_clean(ax)
    fig.suptitle("3 · Measured biomechanical / coordination features by tactical class (285 clips)\n"
                 "these are the human-interpretable quantities behind the learned model",
                 fontsize=12, fontweight="bold", y=1.0)
    save(fig, "3_volleyball/11_biomech_by_class.png")


def fig_role_velocities():
    r = pd.read_csv(os.path.join(HERE, "role_analysis.csv"))
    specs = [("attacker_vmax", "attacker peak speed (cm/s)"),
             ("setter_vmax", "setter peak speed (cm/s)"),
             ("libero_vmax", "libero peak speed (cm/s)"),
             ("attacker_approach_cms", "attacker approach speed (cm/s)")]
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    for ax, (col, lab) in zip(axes.flat, specs):
        data = [pd.to_numeric(r[r.target_label == cls][col], errors="coerce").dropna().values
                for cls in CLASSES]
        bp = ax.boxplot(data, patch_artist=True, widths=0.6, showfliers=False)
        for patch, cls in zip(bp["boxes"], CLASSES):
            patch.set_facecolor(CCOL[cls]); patch.set_alpha(0.75)
        for med in bp["medians"]:
            med.set_color("#111")
        ax.set_xticks(range(1, 5))
        ax.set_xticklabels([SHORT[c] for c in CLASSES], rotation=25, fontsize=8.5)
        ax.set_title(lab, fontsize=9)
        ax_clean(ax)
    fig.subplots_adjust(wspace=0.3, bottom=0.18)
    fig.suptitle("3 · Inferred player-role kinematics by tactical class "
                 "(roles assigned by court position + motion; role_analysis.csv)",
                 fontsize=12, fontweight="bold")
    save(fig, "3_volleyball/12_role_velocities.png")


def fig_tactical_fingerprint():
    c = pd.read_csv(os.path.join(HERE, "coordination_analysis.csv"))
    feats = ["top2_speed_cms", "rest_speed_cms", "speed_ratio", "centroid_vel_cms",
             "mean_spacing_cm", "mean_hull_area", "sync_score", "reaction_lag_frames",
             "ball_present_frac", "tactical_deviation"]
    X = c[feats].apply(pd.to_numeric, errors="coerce")
    Xs = (X - X.mean()) / X.std(ddof=0)
    M = np.vstack([Xs[c.label == cls].mean().values for cls in CLASSES])
    fig, ax = plt.subplots(figsize=(11, 3.9))
    vmax = np.nanmax(np.abs(M))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.grid(False)
    ax.set_xticks(range(len(feats)))
    ax.set_xticklabels(feats, rotation=40, ha="right", fontsize=9)
    ax.set_yticks(range(4))
    ax.set_yticklabels([SHORT[c] for c in CLASSES])
    for i in range(4):
        for j in range(len(feats)):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:+.1f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(M[i, j]) > vmax * 0.55 else "#222")
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label("mean feature value  (per-clip z-score)")
    ax.set_title("3 · Tactical fingerprint — mean standardised feature profile of each class\n"
                 "Attack = fast + spread ; Defense = slow + synchronised ; the minority classes have weak signatures",
                 pad=12)
    save(fig, "3_volleyball/13_tactical_fingerprint.png")


def fig_attack_defense_scatter():
    c = pd.read_csv(os.path.join(HERE, "coordination_analysis.csv"))
    sub = c[c.label.isin(["Coordinated Attack", "Coordinated Defense"])].copy()
    sub["spd_gap"] = pd.to_numeric(sub.top2_speed_cms, errors="coerce") - \
        pd.to_numeric(sub.rest_speed_cms, errors="coerce")
    fig, ax = plt.subplots(figsize=(7.8, 6))
    for cls in ["Coordinated Attack", "Coordinated Defense"]:
        m = sub.label == cls
        ax.scatter(sub[m]["spd_gap"], pd.to_numeric(sub[m]["sync_score"], errors="coerce"),
                   s=40, c=CCOL[cls], marker=CMARK[cls], edgecolors="white",
                   linewidths=0.5, alpha=0.85, label=SHORT[cls])
    ax.set_xlabel("front-minus-back speed gap  (cm/s)")
    ax.set_ylabel("movement synchrony score")
    ax.set_title("3 · Attack vs Defense is the separable sub-problem\n"
                 "a dedicated binary model reaches ROC-AUC 0.955 / macro-F1 0.90 under 8-fold LOVO")
    ax.legend()
    ax_clean(ax, "both")
    save(fig, "3_volleyball/14_attack_vs_defense.png")


def fig_rally_timeline():
    w = pd.read_csv(os.path.join(HERE, "vp7_3to8_results", "vp7_3to8_HGB_windows.csv"))
    t = w["frame_start"] / 30.0
    t0, t1 = w["frame_start"].min() / 30.0, w["frame_end"].max() / 30.0
    fig, axes = plt.subplots(2, 1, figsize=(13, 5.2), sharex=True,
                             gridspec_kw={"height_ratios": [0.8, 1]})
    for _, row in w.iterrows():
        axes[0].axvspan(row["frame_start"] / 30, row["frame_end"] / 30,
                        color=CCOL.get(row["label"], "#ccc"), alpha=0.9, lw=0)
    axes[0].set_yticks([])
    axes[0].set_xlim(t0, t1)
    axes[0].grid(False)
    for s in axes[0].spines.values():
        s.set_visible(False)
    axes[0].set_title("3 · Deployed model output — per-second tactical label, vp7 3:00–8:00 segment "
                      "(what a coach receives)", pad=10)
    axes[1].plot(t, w["confidence"], color=PAL[0], lw=1.4, label="calibrated confidence")
    axes[1].plot(t, w["entropy_norm"], color=PAL[1], lw=1.1, label="normalised entropy")
    axes[1].axhline(0.5, color="#999", ls=":", lw=1)
    axes[1].set_xlabel("match time  (s)")
    axes[1].set_ylabel("0–1")
    axes[1].set_ylim(-0.03, 1.05)
    axes[1].legend(loc="lower left", fontsize=8, ncol=2)
    ax_clean(axes[1])
    leg = [Patch(facecolor=CCOL[c], label=SHORT[c]) for c in CLASSES] + \
          [Patch(facecolor=CCOL["Unclassified"], label="Unclassified")]
    fig.legend(handles=leg, ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.04), fontsize=9)
    save(fig, "3_volleyball/15_rally_timeline_vp7.png")


def fig_match_budget():
    f = pd.read_csv(os.path.join(HERE, "v7_tactical", "results", "vp7_frames.csv"))
    vc = f.pred_label.value_counts()
    order = ["Coordinated Defense", "Coordinated Attack", "Unclassified",
             "Delayed Support", "Spacing Breakdown"]
    vals = [100 * vc.get(k, 0) / len(f) for k in order]
    fig, ax = plt.subplots(figsize=(9, 3.4))
    left = 0
    for k, v in zip(order, vals):
        ax.barh(0, v, left=left, color=CCOL[k], edgecolor="white")
        if v > 3:
            ax.text(left + v / 2, 0, f"{SHORT[k]}\n{v:.0f}%", ha="center", va="center",
                    color="white", fontweight="bold", fontsize=9)
        left += v
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.grid(False)
    ax.set_xlabel("share of the 2 685-second match")
    ax.set_title("3 · Match-level tactical budget — per-frame predicted state over the full broadcast\n"
                 "the court spends most of its time in Defense; Spacing Breakdown is rare (1.4%)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    save(fig, "3_volleyball/16_match_tactical_budget.png")


# ======================================================================
# 4  MACHINE LEARNING
# ======================================================================
def fig_kfold_vs_lovo():
    d = load_json(os.path.join(HERE, "ml_results", "ml_results.json"))
    m = d["models"]
    rows = sorted(m.items(), key=lambda kv: kv[1]["lovo_macro_f1"])
    names = [k.replace("_", " ") for k, _ in rows]
    kf = [v["kfold_macro_f1"] for _, v in rows]
    lo = [v["lovo_macro_f1"] for _, v in rows]
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(9.8, 5.4))
    for i in range(len(names)):
        ax.plot([lo[i], kf[i]], [i, i], color="#bbbbbb", lw=2, zorder=1)
    ax.scatter(kf, y, s=70, color=PAL[3], label="stratified k-fold (within-video)", zorder=2)
    ax.scatter(lo, y, s=70, color=PAL[0], label="leave-one-video-out (cross-video)", zorder=2)
    mb = d["majority_baseline"]["macro_f1"]
    ax.axvline(mb, color="#999", ls="--", lw=1)
    ax.text(mb - 0.006, -0.7, "majority\nbaseline", fontsize=7.5, color="#777", ha="right", va="bottom")
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel("macro-F1")
    ax.set_xlim(0.12, 0.50)
    ax.set_ylim(-0.9, len(names) - 0.3)
    ax.set_title("4 · The optimism gap — same 9 classical models, two evaluation protocols\n"
                 "corpus v3 · 285 clips · every model loses 0.10–0.20 macro-F1 when the test video is unseen",
                 pad=14)
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.02))
    ax_clean(ax, "x")
    save(fig, "4_ml/17_kfold_vs_lovo.png")


def fig_train_vs_test():
    d = pd.read_csv(os.path.join(HERE, "model_comparison", "trainvstest_metrics.csv"))
    d = d.sort_values("test_macro_f1_LOVO")
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    for i, (_, r) in enumerate(d.iterrows()):
        ax.plot([r.test_macro_f1_LOVO, r.train_macro_f1], [i, i], color="#cccccc", lw=2, zorder=1)
    ax.scatter(d.train_macro_f1, y, s=65, color="#c22", label="train", zorder=2)
    cols = [FAMCOL[f] for f in d.family]
    ax.scatter(d.test_macro_f1_LOVO, y, s=75, c=cols, label="test (LOVO)", zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(d.model)
    ax.set_xlabel("macro-F1")
    ax.set_xlim(0, 1.05)
    ax.set_title("4 · Train vs cross-video (LOVO) test for all 13 models · corpus v4 · 945 clips\n"
                 "every tree ensemble memorises the training set (train F1 = 1.00); "
                 "the DL models overfit less but score no higher", pad=12)
    h = [Line2D([0], [0], marker="o", color="w", markerfacecolor="#c22", markersize=9, label="train F1"),
         Line2D([0], [0], marker="o", color="w", markerfacecolor=FAMCOL["ML"], markersize=9, label="test F1 · classical ML"),
         Line2D([0], [0], marker="o", color="w", markerfacecolor=FAMCOL["DL"], markersize=9, label="test F1 · deep learning")]
    ax.legend(handles=h, loc="lower right")
    ax_clean(ax, "x")
    save(fig, "4_ml/18_train_vs_test_all.png")


PERCLASS_V3 = {
    "sklearn HGB (deployed)": [0.645, 0.596, 0.000, 0.080],
    "mamba (SSM)": [0.522, 0.496, 0.052, 0.043],
    "transformer": [0.531, 0.385, 0.085, 0.092],
    "tcn": [0.556, 0.571, 0.093, 0.000],
    "bilstm": [0.547, 0.507, 0.098, 0.072],
    "inceptiontime": [0.485, 0.297, 0.000, 0.095],
}


def fig_perclass_heatmap():
    M = np.array([PERCLASS_V3[k] for k in PERCLASS_V3])
    names = list(PERCLASS_V3)
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=0.8, aspect="auto")
    ax.grid(False)
    ax.set_xticks(range(4))
    ax.set_xticklabels([SHORT[c] for c in CLASSES])
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    for i in range(len(names)):
        for j in range(4):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if M[i, j] > 0.45 else "#222")
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label("per-class F1  (pooled LOVO)")
    ax.set_title("4 · Per-class F1 — every model, classical and deep (corpus v3, cross-video)\n"
                 "Delayed Support is never learned; Spacing Breakdown barely — a data problem, not a model problem")
    save(fig, "4_ml/19_perclass_f1_heatmap.png")


def fig_confusions():
    hgb = load_json(os.path.join(BENCH, "bench_out", "sklearn_hgb.json"))["pooled_confusion_4x4"]
    dl = load_json(os.path.join(HERE, "deep_results_v4", "deep_results.json"))
    inc = np.array(dl["architectures"]["inceptiontime"]["pooled_confusion"], float)
    # deep arm label order is [Spacing, Delayed, Attack, Defense] (verified vs class counts /
    # per_class_f1); permute to canonical [Attack, Defense, Delayed, Spacing]
    p = [2, 3, 1, 0]
    inc = inc[np.ix_(p, p)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, M, title in [(axes[0], np.array(hgb, float),
                          "HistGradientBoosting (deployed)\ncorpus v3 · 375 clips · pooled LOVO"),
                         (axes[1], inc,
                          "InceptionTime (best deep arm)\ncorpus v4 · 945 clips · pooled LOVO")]:
        Mn = M / M.sum(1, keepdims=True)
        im = ax.imshow(Mn, cmap="Blues", vmin=0, vmax=1)
        ax.grid(False)
        ax.set_xticks(range(4)); ax.set_yticks(range(4))
        ax.set_xticklabels([SHORT[c] for c in CLASSES], rotation=20)
        ax.set_yticklabels([SHORT[c] for c in CLASSES])
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{Mn[i,j]*100:.0f}%\n{int(M[i,j])}", ha="center", va="center",
                        fontsize=8, color="white" if Mn[i, j] > 0.5 else "#222")
        ax.set_title(title, fontsize=10)
    fig.suptitle("4 · Confusion matrices (row-normalised) — the minority classes leak into Attack / Defense",
                 fontsize=12, fontweight="bold")
    save(fig, "4_ml/20_confusion_hgb_inception.png")


def fig_channel_importance():
    d = load_json(os.path.join(HERE, "ml_results", "ml_results.json"))["channel_importance"]
    ch = d["channels"][:12][::-1]
    names = [c["channel"] for c in ch]
    mean = [c["importance_mean"] for c in ch]
    std = [c["importance_std"] for c in ch]
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(y, mean, xerr=std, color=PAL[0], height=0.62,
            error_kw=dict(ecolor="#888", lw=1, capsize=3))
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel("drop in cross-video macro-F1 when this channel is permuted")
    ax.set_title("4 · Permutation feature importance (leave-one-video-out)\n"
                 f"baseline macro-F1 {d['baseline_macro_f1']:.3f} — no single channel dominates; "
                 "formation shape (centroid_y, hull_area) leads")
    ax_clean(ax, "x")
    save(fig, "4_ml/21_channel_importance.png")


def fig_calibration():
    d = load_json(os.path.join(HERE, "ml_results", "ml_results.json"))
    raw, cal = d["confidence_uncalibrated"], d["confidence_calibrated"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    fig.subplots_adjust(top=0.74, wspace=0.3)
    axes[0].bar(["raw model", "sigmoid-calibrated"],
                [raw["mean_confidence"], cal["mean_confidence"]],
                color=[PAL[7], PAL[2]], width=0.5)
    for i, v in enumerate([raw["mean_confidence"], cal["mean_confidence"]]):
        axes[0].text(i, v + 0.02, f"{v:.3f}", ha="center", fontweight="bold")
    axes[0].set_ylim(0, 1.08)
    axes[0].set_ylabel("mean predicted confidence")
    axes[0].set_title("Mean confidence of the top class")
    ax_clean(axes[0])
    axes[1].bar(["raw model", "sigmoid-calibrated"],
                [raw["mean_entropy_bits"], cal["mean_entropy_bits"]],
                color=[PAL[7], PAL[2]], width=0.5)
    for i, v in enumerate([raw["mean_entropy_bits"], cal["mean_entropy_bits"]]):
        axes[1].text(i, v + 0.03, f"{v:.3f}", ha="center", fontweight="bold")
    axes[1].set_ylabel("mean predictive entropy (bits)")
    axes[1].set_title("Predictive entropy  (max = 2.0 bits for 4 classes)")
    ax_clean(axes[1])
    fig.suptitle("4 · Calibration matters — the raw model is 99.998% confident on every clip "
                 "(100% of clips > 0.99);\nsigmoid calibration restores a usable confidence signal",
                 fontsize=11, fontweight="bold", y=1.06)
    save(fig, "4_ml/22_calibration.png")


def fig_minority_perfold():
    corpora = [("training_csv", "v1"), ("training_csv_v3", "v3"), ("training_csv_v4", "v4")]
    fig, ax = plt.subplots(figsize=(9, 4.4))
    x = np.arange(len(CLASSES))
    w = 0.26
    for j, (d, lab) in enumerate(corpora):
        df = corpus_frame(os.path.join(HERE, d))
        nv = df.video.nunique()
        per_fold = [((df.cls == c).sum() * (nv - 1) / nv) / (nv - 1) for c in CLASSES]
        # simpler: total in class / n_videos  = avg held-out count per LOVO fold
        per_fold = [(df.cls == c).sum() / nv for c in CLASSES]
        b = ax.bar(x + (j - 1) * w, per_fold, w, color=PAL[j], label=f"{lab}  ({nv} videos)")
        for rect, v in zip(b, per_fold):
            ax.text(rect.get_x() + rect.get_width() / 2, v + 0.4, f"{v:.0f}", ha="center", fontsize=8)
    ax.axhline(10, color="#e34948", ls="--", lw=1.2)
    ax.text(-0.55, 10.6, "≈ floor to learn a class", color="#e34948", fontsize=8.5, ha="left")
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[c] for c in CLASSES])
    ax.set_ylim(0, 66)
    ax.set_ylabel("avg clips of this class\nper leave-one-video-out fold")
    ax.set_title("4 · Why two classes never train — examples available per cross-video fold\n"
                 "Delayed Support gives a learner ~5–13 examples per fold, spread over 3–6 videos",
                 pad=12)
    ax.legend()
    ax_clean(ax)
    save(fig, "4_ml/23_minority_class_per_fold.png")


# ======================================================================
# 5  DEEP LEARNING
# ======================================================================
ARCHS = ["mamba", "transformer", "tcn", "bilstm", "inceptiontime"]
ARCHCOL = dict(zip(ARCHS, [PAL[0], PAL[1], PAL[2], PAL[4], PAL[6]]))


def _hist_all(arch):
    out = []
    for f in glob.glob(os.path.join(BENCH, "bench_out", "hist", f"{arch}__*.csv")):
        try:
            out.append(pd.read_csv(f))
        except Exception:
            pass
    return out


def fig_training_curves():
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.8), sharey=True)
    for ax, arch in zip(axes, ARCHS):
        hs = _hist_all(arch)
        for h in hs:
            ax.plot(h["epoch"], h["val_macro_f1"], color=ARCHCOL[arch], alpha=0.30, lw=1)
        if hs:
            me = min(len(h) for h in hs)
            stack = np.vstack([h["val_macro_f1"].values[:me] for h in hs])
            ax.plot(range(1, me + 1), stack.mean(0), color=ARCHCOL[arch], lw=2.6)
        ax.axhline(0.330, color="#e34948", ls="--", lw=1.2)
        ax.text(0.5, 0.06, arch, transform=ax.transAxes, ha="center", fontsize=11,
                fontweight="bold", color=ARCHCOL[arch])
        ax.set_xlabel("epoch")
        ax.set_ylim(0, 0.62)
        ax_clean(ax, "y")
    axes[0].set_ylabel("validation macro-F1")
    h = [Line2D([0], [0], color="#333", lw=2.6, label="mean across folds & seeds"),
         Line2D([0], [0], color="#333", lw=1, alpha=0.35, label="individual fold / seed run"),
         Line2D([0], [0], color="#e34948", ls="--", label="sklearn HGB baseline (0.330)")]
    fig.legend(handles=h, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.10))
    fig.suptitle("5 · Deep temporal training histories — validation macro-F1 per epoch\n"
                 "corpus v3 · 3 LOVO folds × 2 seeds per arch · every arm plateaus at or below the baseline",
                 fontsize=12, fontweight="bold", y=1.13)
    save(fig, "5_dl/24_training_curves.png")


PARAMS = {"mamba": 69381, "transformer": 207621, "tcn": 135685,
          "bilstm": 177798, "inceptiontime": 54341}
POOLED_F1 = {"mamba": 0.274, "transformer": 0.232, "tcn": 0.291,
             "bilstm": 0.327, "inceptiontime": 0.250}   # seed-averaged, v3


def fig_params_vs_f1():
    fig, ax = plt.subplots(figsize=(8.4, 5))
    for arch in ARCHS:
        ax.scatter(PARAMS[arch], POOLED_F1[arch], s=120, color=ARCHCOL[arch], zorder=3,
                   edgecolors="white", linewidths=1)
        ax.annotate(arch, (PARAMS[arch], POOLED_F1[arch]),
                    xytext=(6, 6), textcoords="offset points", fontsize=9)
    ax.axhline(0.330, color="#e34948", ls="--", lw=1.4)
    ax.text(6e4, 0.335, "sklearn HGB (7 900 fitted params)  macro-F1 0.330", color="#e34948", fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("trainable parameters  (log)")
    ax.set_ylabel("pooled LOVO macro-F1  (seed-averaged)")
    ax.set_ylim(0.18, 0.38)
    ax.set_title("5 · Parameters vs cross-video accuracy — every deep arm carries 55k–208k parameters\n"
                 "for ~250 training clips per fold (par:sample 270:1 to 1033:1). None beat the baseline.")
    ax_clean(ax, "both")
    save(fig, "5_dl/25_params_vs_f1.png")


PERFOLD = {  # arch -> seed -> [f1 on (1),(3),(4)]
    "mamba": {42: [0.181, 0.286, 0.293], 7: [0.158, 0.301, 0.252]},
    "transformer": {42: [0.178, 0.303, 0.207], 7: [0.240, 0.068, 0.279]},
    "tcn": {42: [0.186, 0.343, 0.296], 7: [0.191, 0.260, 0.329]},
    "bilstm": {42: [0.191, 0.298, 0.359], 7: [0.364, 0.286, 0.337]},
    "inceptiontime": {42: [0.145, 0.227, 0.192], 7: [0.200, 0.293, 0.345]},
}


def fig_deep_vs_baseline():
    fig, ax = plt.subplots(figsize=(9.5, 5))
    for i, arch in enumerate(ARCHS):
        for seed, dx in [(42, -0.13), (7, 0.13)]:
            vals = PERFOLD[arch][seed]
            ax.scatter([i + dx] * 3, vals, s=45, color=ARCHCOL[arch],
                       alpha=0.55, zorder=2)
            mean = np.mean(vals)
            sd = np.std(vals)
            ax.plot([i + dx - 0.06, i + dx + 0.06], [mean, mean], color="#111", lw=2, zorder=3)
            ax.errorbar(i + dx, mean, yerr=sd, color="#111", capsize=3, lw=1, zorder=3)
            ax.text(i + dx, 0.03, f"s{seed}", ha="center", fontsize=7, color="#666")
    ax.axhline(0.330, color="#e34948", ls="--", lw=1.6)
    ax.text(-0.35, 0.345, "sklearn HGB baseline 0.330", color="#e34948", ha="left", fontsize=9)
    ax.set_xticks(range(5))
    ax.set_xticklabels(ARCHS)
    ax.set_ylabel("per-fold LOVO macro-F1")
    ax.set_ylim(0, 0.45)
    ax.set_title("5 · Pre-registered bar: a deep arm wins only if its per-fold mean beats 0.330\n"
                 "by more than its own SD, at BOTH seeds. Dots = folds, bar = mean ± SD. No arm clears it.")
    ax_clean(ax, "y")
    save(fig, "5_dl/26_deep_vs_baseline.png")


def fig_seed_sensitivity():
    pooled = {"mamba": (0.278, 0.269), "transformer": (0.273, 0.191), "tcn": (0.305, 0.276),
              "bilstm": (0.306, 0.349), "inceptiontime": (0.219, 0.281)}
    y = np.arange(len(pooled))
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for i, (arch, (s42, s7)) in enumerate(pooled.items()):
        ax.plot([s42, s7], [i, i], color="#ccc", lw=2, zorder=1)
        ax.scatter(s42, i, s=80, color=PAL[0], zorder=2)
        ax.scatter(s7, i, s=80, color=PAL[1], zorder=2)
    ax.axvline(0.330, color="#e34948", ls="--", lw=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels(list(pooled))
    ax.set_xlabel("pooled LOVO macro-F1")
    ax.set_title("5 · Seed sensitivity (42 vs 7) — the transformer collapses (routes 42% of clips to the\n"
                 "reject head at seed 7); BiLSTM is the most seed-stable deep arm")
    ax.legend(handles=[Line2D([0], [0], marker="o", color="w", markerfacecolor=PAL[0], markersize=9, label="seed 42"),
                       Line2D([0], [0], marker="o", color="w", markerfacecolor=PAL[1], markersize=9, label="seed 7"),
                       Line2D([0], [0], color="#e34948", ls="--", label="baseline 0.330")],
              loc="lower right")
    ax_clean(ax, "x")
    save(fig, "5_dl/27_seed_sensitivity.png")


def fig_model_agreement():
    a = pd.read_csv(os.path.join(HERE, "vp7_3to8_results", "vp7_3to8_all_models.csv"))
    models = ["extra_trees", "gradient_boosting", "hist_gradient_boosting", "knn", "linear_svm",
              "logistic_regression", "mlp", "random_forest", "rbf_svm",
              "bilstm", "inceptiontime", "mamba", "tcn"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), gridspec_kw={"width_ratios": [1, 2.4]})
    axes[0].hist(a["n_agree_majority"], bins=np.arange(3.5, 14.5), color=PAL[0], rwidth=0.9)
    axes[0].set_xlabel("models agreeing with the majority vote  (of 13)")
    axes[0].set_ylabel("windows")
    axes[0].set_title("Ensemble consensus per window")
    ax_clean(axes[0])

    idx = {c: i for i, c in enumerate(CLASSES + ["Unclassified"])}
    M = np.array([[idx.get(v, 4) for v in a[m].values] for m in models], float)
    cmap = matplotlib.colors.ListedColormap([CCOL[c] for c in CLASSES] + [CCOL["Unclassified"]])
    axes[1].imshow(M, aspect="auto", cmap=cmap, vmin=0, vmax=4, interpolation="nearest")
    axes[1].grid(False)
    axes[1].set_yticks(range(len(models)))
    axes[1].set_yticklabels(models, fontsize=8)
    axes[1].set_xlabel("window index  (309 consecutive 1-second windows, vp7 3–8 min)")
    axes[1].set_title("Every model's per-window prediction — classical ML rows agree; deep rows are noisier")
    axes[1].legend(handles=[Patch(facecolor=CCOL[c], label=SHORT[c]) for c in CLASSES] +
                   [Patch(facecolor=CCOL["Unclassified"], label="Uncl.")],
                   ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.5), fontsize=8)
    fig.suptitle("5 · 13-model prediction matrix on the same held-out broadcast segment",
                 fontsize=12, fontweight="bold")
    save(fig, "5_dl/28_model_agreement_vp7.png")


# ======================================================================
FIGS = [
    fig_inventory, fig_class_distribution, fig_class_by_video, fig_feature_pca,
    fig_player_funnel, fig_ball_recall, fig_detector_ap, fig_court_trajectories,
    fig_broadcast_timeline,
    fig_biomech_by_class, fig_role_velocities, fig_tactical_fingerprint,
    fig_attack_defense_scatter, fig_rally_timeline, fig_match_budget,
    fig_kfold_vs_lovo, fig_train_vs_test, fig_perclass_heatmap, fig_confusions,
    fig_channel_importance, fig_calibration, fig_minority_perfold,
    fig_training_curves, fig_params_vs_f1, fig_deep_vs_baseline,
    fig_seed_sensitivity, fig_model_agreement,
]

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    ok = 0
    for fn in FIGS:
        try:
            print("::", fn.__name__)
            fn()
            ok += 1
        except Exception as e:
            import traceback
            print("  FAILED:", fn.__name__, "->", e)
            traceback.print_exc()
    print(f"\n{ok}/{len(FIGS)} figures generated into {OUT}")
