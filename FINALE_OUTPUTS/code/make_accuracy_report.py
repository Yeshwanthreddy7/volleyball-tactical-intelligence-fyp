"""Accuracy table + chart for the 13 tactical models (9 ML + 4 DL).

Accuracy = leave-one-video-out (LOVO): each model is trained on the clips of
7 source videos and tested on the 8th, for all 8 videos, so every test clip
comes from a match the model never saw. Source: model_comparison/
trainvstest_metrics.csv, deep_results_v4/deep_results.json, ml_results_v4/report.txt.

videoplayback (7) has no hand labels, so there is no accuracy on it; the table
adds what can be measured there: each model's class split and how often it
agrees with the 13-model majority vote.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "FINALE_OUTPUTS" / "accuracy"
CLASSES = ["Coordinated Attack", "Coordinated Defense", "Delayed Support", "Spacing Breakdown"]

from render_13_models import MODELS  # noqa: E402

ML_RGB, DL_RGB = "#2a78d6", "#eb6834"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#1f1f1e", "#5f5e58", "#e6e5df"

# per-class F1 of the declared ML model (report.txt, LOVO)
HGB_PER_CLASS = {"Coordinated Attack": 0.785, "Coordinated Defense": 0.755,
                 "Delayed Support": 0.206, "Spacing Breakdown": 0.294}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    met = pd.read_csv(ROOT / "model_comparison" / "trainvstest_metrics.csv").set_index("model")
    deep = json.load(open(ROOT / "deep_results_v4" / "deep_results.json"))["architectures"]
    win = pd.read_csv(ROOT / "vp7_3to8_results" / "vp7_3to8_all_models.csv")

    rows = []
    for col, name, fam, mname in MODELS:
        m = met.loc[mname]
        per = deep[col]["pooled_per_class_f1"] if col in deep else (
            HGB_PER_CLASS if col == "hist_gradient_boosting" else {})
        split = win[col].value_counts(normalize=True)
        row = {
            "model": name, "family": "Machine learning" if fam == "ML" else "Deep learning",
            "test_accuracy_LOVO_%": round(m.test_accuracy_LOVO * 100, 1),
            "test_macro_F1_LOVO": round(m.test_macro_f1_LOVO, 3),
            "train_accuracy_%": round(m.train_accuracy * 100, 1),
            "train_macro_F1": round(m.train_macro_f1, 3),
            "overfit_gap_accuracy": round(m.acc_gap, 3),
        }
        for c in CLASSES:
            row[f"F1_{c.replace(' ', '_')}"] = per.get(c, "")
        for c in CLASSES:
            row[f"vp7_share_{c.replace(' ', '_')}_%"] = round(split.get(c, 0) * 100, 1)
        row["vp7_agree_with_13_model_majority_%"] = round((win[col] == win["majority_label"]).mean() * 100, 1)
        row["vp7_mean_confidence"] = round(win[col + "__conf"].mean(), 3)
        row["_stem"] = f"{name.replace(' ', '_').replace('(', '').replace(')', '').replace('+', '_')}_{fam}"
        rows.append(row)
    df = pd.DataFrame(rows).sort_values(["test_accuracy_LOVO_%", "test_macro_F1_LOVO"], ascending=False)
    df.insert(0, "rank", range(1, len(df) + 1))
    df["video_file"] = [f"videos/{r:02d}_{st}.mp4" for r, st in zip(df["rank"], df["_stem"])]
    df = df.drop(columns="_stem")
    df.to_csv(OUT / "model_accuracy.csv", index=False)
    print(df[["rank", "model", "family", "test_accuracy_LOVO_%", "test_macro_F1_LOVO",
              "train_accuracy_%", "vp7_agree_with_13_model_majority_%"]].to_string(index=False))
    chart(df)
    return df


def chart(df):
    d = df.iloc[::-1]
    colors = [ML_RGB if f == "Machine learning" else DL_RGB for f in d["family"]]
    plt.rcParams.update({"font.family": "Arial", "font.size": 11})
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.4), sharey=True, facecolor=SURFACE)
    panels = [("test_accuracy_LOVO_%", "Test accuracy (%)", 100, "{:.1f}%"),
              ("test_macro_F1_LOVO", "Test macro-F1 (4 classes)", 1.0, "{:.3f}")]
    for ax, (col, title, xmax, fmt) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        y = range(len(d))
        ax.barh(y, d[col], color=colors, height=0.62, edgecolor=SURFACE, linewidth=2)
        for yi, v in zip(y, d[col]):
            ax.text(v + xmax * 0.012, yi, fmt.format(v), va="center", fontsize=10, color=INK)
        ax.set_xlim(0, xmax * 1.1)
        ax.set_title(title, loc="left", fontsize=13, color=INK, fontweight="bold")
        ax.xaxis.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=INK2, length=0)
    axes[0].set_yticks(range(len(d)))
    axes[0].set_yticklabels(d["model"], color=INK)
    axes[0].axvline(39.0, color=INK2, lw=1, ls="--")
    axes[0].text(39.5, len(d) - 0.4, "majority-class baseline 39%", fontsize=9, color=INK2)
    handles = [plt.Rectangle((0, 0), 1, 1, color=ML_RGB), plt.Rectangle((0, 0), 1, 1, color=DL_RGB)]
    fig.legend(handles, ["Machine learning (9)", "Deep learning (4)"], loc="upper right",
               frameon=False, ncol=2, fontsize=11)
    fig.suptitle("Tactical classifiers - accuracy on unseen matches (leave-one-video-out, 945 clips)",
                 x=0.01, ha="left", fontsize=15, fontweight="bold", color=INK)
    fig.text(0.01, 0.005, "Each model is tested on a video it never saw during training, for all 8 source videos. "
             "Classes: Coordinated Attack, Coordinated Defense, Delayed Support, Spacing Breakdown.",
             fontsize=9, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    fig.savefig(OUT / "model_accuracy_chart.png", dpi=160, facecolor=SURFACE)


if __name__ == "__main__":
    main()
