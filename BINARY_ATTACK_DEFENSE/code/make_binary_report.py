"""Combine the 9 ML + 4 DL Attack-vs-Defense results into one table and chart.

Reads results/ml_binary_results.json and results/lovo_<arch>.json (+ oof_<arch>.npz
for accuracy), writes results/binary_model_accuracy.csv and
results/binary_model_accuracy_chart.png.
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

OUT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(OUT, "results")
ML_NAMES = {"hist_gradient_boosting": "HistGradientBoosting", "gradient_boosting": "GradientBoosting",
            "random_forest": "RandomForest", "extra_trees": "ExtraTrees", "rbf_svm": "RBF-SVM",
            "linear_svm": "Linear SVM", "logistic_regression": "LogisticRegression", "knn": "k-NN",
            "mlp": "MLP"}
DL_NAMES = {"inceptiontime": "InceptionTime", "tcn": "TCN", "bilstm": "BiLSTM+Attention",
            "mamba": "Mamba (SSM)"}
ML_RGB, DL_RGB = "#2a78d6", "#eb6834"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#1f1f1e", "#5f5e58", "#e6e5df"


def main():
    rows = []
    ml = json.load(open(os.path.join(RES, "ml_binary_results.json")))["models"]
    for k, r in ml.items():
        p = r["lovo"]
        rows.append(dict(model=ML_NAMES[k], family="Machine learning", file=f"ML_models/{k}.joblib",
                         test_accuracy=p["accuracy"], test_macro_f1=p["macro_f1"],
                         balanced_accuracy=p["balanced_accuracy"], roc_auc=p["roc_auc"],
                         attack_f1=p["attack_f1"], defense_f1=p["defense_f1"],
                         train_accuracy=r["train_accuracy"]))
    for k, name in DL_NAMES.items():
        j = json.load(open(os.path.join(RES, f"lovo_{k}.json")))["pooled"]
        o = np.load(os.path.join(RES, f"oof_{k}.npz"))
        pred = o["prob"].argmax(1)
        rows.append(dict(model=name, family="Deep learning", file=f"DL_models/{k}.pt",
                         test_accuracy=round(float(accuracy_score(o["y"], pred)), 4),
                         test_macro_f1=j["macro_f1"], balanced_accuracy=j["balanced_accuracy"],
                         roc_auc=j["roc_auc"], attack_f1=j["attack_f1"], defense_f1=j["defense_f1"],
                         train_accuracy=None))
    df = pd.DataFrame(rows).sort_values(["test_accuracy", "test_macro_f1"], ascending=False)
    df.insert(0, "rank", range(1, len(df) + 1))
    df.to_csv(os.path.join(RES, "binary_model_accuracy.csv"), index=False)
    print(df[["rank", "model", "family", "test_accuracy", "test_macro_f1", "roc_auc",
              "train_accuracy"]].to_string(index=False))

    d = df.iloc[::-1]
    colors = [ML_RGB if f == "Machine learning" else DL_RGB for f in d["family"]]
    plt.rcParams.update({"font.family": "Arial", "font.size": 11})
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.4), sharey=True, facecolor=SURFACE)
    for ax, (col, title, xmax, fmt, mul) in zip(axes, [
            ("test_accuracy", "Test accuracy (%)", 100, "{:.1f}%", 100),
            ("roc_auc", "ROC-AUC", 1.0, "{:.3f}", 1)]):
        ax.set_facecolor(SURFACE)
        vals = d[col] * mul
        ax.barh(range(len(d)), vals, color=colors, height=0.62, edgecolor=SURFACE, linewidth=2)
        for yi, v in enumerate(vals):
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
    base = 369 / 723 * 100
    axes[0].axvline(base, color=INK2, lw=1, ls="--")
    axes[0].text(base + 0.5, len(d) - 0.4, f"always 'Attack' {base:.0f}%", fontsize=9, color=INK2)
    h = [plt.Rectangle((0, 0), 1, 1, color=ML_RGB), plt.Rectangle((0, 0), 1, 1, color=DL_RGB)]
    fig.legend(h, ["Machine learning (9)", "Deep learning (4)"], loc="upper left", bbox_to_anchor=(0.005, 0.93), frameon=False, ncol=2)
    fig.suptitle("Coordinated Attack vs Coordinated Defense - accuracy on unseen matches (LOVO, 723 clips)",
                 x=0.01, ha="left", fontsize=15, fontweight="bold", color=INK)
    fig.text(0.01, 0.005, "Each model is tested on a video it never saw during training, for all 8 source videos.",
             fontsize=9, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 0.88))
    fig.savefig(os.path.join(RES, "binary_model_accuracy_chart.png"), dpi=160, facecolor=SURFACE)


if __name__ == "__main__":
    main()
