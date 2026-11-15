"""Charts for the post-review deck, drawn from the evidence JSON files.
Out: FINAL_REVIEW/figures/rv_*.png
Run: python src/make_review_charts.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EV, FIG = ROOT / "evidence", ROOT / "figures"
MAROON, BLUE, INK, MUTED, GRID = "#A4123F", "#3A6EA5", "#1f1f1f", "#5f5f5f", "#e3e3e3"
plt.rcParams.update({"font.family": "Arial", "font.size": 15, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150})
R = json.loads((EV / "review_response_eval.json").read_text())
SHORT = ["Attack", "Defense", "Delayed\nSupport", "Spacing\nBreakdown"]

# 1. confusion matrix (deployed model, row-normalised, counts shown)
cm = np.array(R["deployed_prior_corrected"]["confusion_rows_true_cols_pred"])
pct = cm / cm.sum(1, keepdims=True)
fig, ax = plt.subplots(figsize=(7.2, 6.2))
ax.imshow(pct, cmap="Reds", vmin=0, vmax=1)
for i in range(4):
    for j in range(4):
        ax.text(j, i, f"{cm[i, j]}\n{pct[i, j]:.0%}", ha="center", va="center", fontsize=14,
                color="white" if pct[i, j] > 0.5 else INK)
ax.set_xticks(range(4), SHORT, fontsize=13)
ax.set_yticks(range(4), SHORT, fontsize=13)
ax.set_xlabel("Predicted tactic")
ax.set_ylabel("True tactic (human label)")
for s in ax.spines.values():
    s.set_visible(False)
fig.tight_layout()
fig.savefig(FIG / "rv_confusion.png", transparent=False, facecolor="white")
plt.close(fig)

# 2. fold-wise macro-F1, HGB vs InceptionTime, per held-out video
folds = R["deployed_prior_corrected"]["folds"]
inc = R["deep_vs_deployed_foldwise"]["inceptiontime"]["fold_macro_f1"]
vids = list(folds)
names = ["vp" + v.strip("()") if v.strip("()").isdigit() else ("plain" if "plain" in v else "USA") for v in vids]
x = np.arange(len(vids))
fig, ax = plt.subplots(figsize=(10, 4.6))
a = [folds[v]["macro_f1"] for v in vids]
b = [inc[v] for v in vids]
ax.bar(x - 0.21, a, 0.4, color=MAROON, label="HGB (our model)", edgecolor="white", linewidth=2)
ax.bar(x + 0.21, b, 0.4, color=BLUE, label="InceptionTime (deep)", edgecolor="white", linewidth=2)
ax.set_xticks(x, [f"{n}\n({folds[v]['n']})" for n, v in zip(names, vids)], fontsize=13)
ax.set_ylabel("Macro-F1 on held-out video")
ax.set_ylim(0, 0.8)
ax.yaxis.grid(True, color=GRID)
ax.set_axisbelow(True)
ax.legend(frameon=False, loc="upper left", ncol=2)
fig.tight_layout()
fig.savefig(FIG / "rv_folds.png", facecolor="white")
plt.close(fig)

# 3. window-length study
w = R["window_study_hgb"]
n = [int(k) for k in w]
fig, ax = plt.subplots(figsize=(7.2, 4.6))
acc = [w[str(k)]["accuracy"] * 100 for k in n]
f1 = [w[str(k)]["macro_f1"] * 100 for k in n]
ax.plot(n, acc, "-o", color=MAROON, lw=2, ms=8, label="Accuracy %")
ax.plot(n, f1, "-s", color=BLUE, lw=2, ms=8, label="Macro-F1 x100")
ax.annotate(f"{acc[-1]:.1f}", (n[-1], acc[-1]), textcoords="offset points", xytext=(-8, 10), color=INK)
ax.annotate(f"{f1[-1]:.1f}", (n[-1], f1[-1]), textcoords="offset points", xytext=(-8, 10), color=INK)
ax.set_xticks(n, [f"{k}\n{k / 30:.2f}s" for k in n], fontsize=12)
ax.set_xlabel("Frames in the window (seconds at 30 fps)")
ax.set_ylim(30, 75)
ax.yaxis.grid(True, color=GRID)
ax.set_axisbelow(True)
ax.legend(frameon=False, loc="lower right")
fig.tight_layout()
fig.savefig(FIG / "rv_window.png", facecolor="white")
plt.close(fig)

# 4. calibration: leave-one-line-out miss per video (cm)
C = json.loads((EV / "calibration_eval.json").read_text())["per_video"]
vv = [k for k in C if C[k]["leave_one_line_out"]]
med = [np.median([z["miss_cm"] for z in C[k]["leave_one_line_out"]]) for k in vv]
lab = ["plain" if k == "videoplayback" else k.replace("videoplayback (", "vp").replace(")", "").replace("usa", "USA") for k in vv]
fig, ax = plt.subplots(figsize=(8, 4.4))
ax.bar(lab, med, color=MAROON, edgecolor="white", linewidth=2)
for i, m in enumerate(med):
    ax.text(i, m + 0.3, f"{m:.0f}", ha="center", color=INK, fontsize=13)
ax.set_ylabel("Median miss of a hidden line (cm)")
ax.set_ylim(0, 15)
ax.yaxis.grid(True, color=GRID)
ax.set_axisbelow(True)
plt.xticks(fontsize=12)
fig.tight_layout()
fig.savefig(FIG / "rv_calibration.png", facecolor="white")
plt.close(fig)
print("charts written")
