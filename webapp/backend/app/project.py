"""Read-only project data for the documentation portal, read live from the
research repo so the website always shows the current results and docs."""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

from . import config

DOCS = [
    ("project-documentation", "Project documentation (full)", "docs/PROJECT_DOCUMENTATION.md"),
    ("readme", "Project README", "README.md"),
    ("installation", "Installation guide", "docs/INSTALLATION.md"),
    ("methodology", "Methodology", "docs/METHODOLOGY.md"),
    ("training", "Training guide", "docs/TRAINING.md"),
    ("all-13-models", "All 13 models", "ALL_13_MODELS/README.md"),
    ("binary-attack-defense", "Binary attack / defense models", "BINARY_ATTACK_DEFENSE/README.md"),
    ("final-outputs", "Final outputs", "FINALE_OUTPUTS/README.md"),
    ("panel-review", "Panel review: two-stream model", "PANEL_REVIEW/README.md"),
    ("data-atlas", "Data atlas (figures)", "figures_atlas/README.md"),
    ("glossary", "Glossary", "docs/GLOSSARY_MASTER.md"),
    ("defense", "Viva / defense notes", "docs/DEFENSE_MASTER.md"),
    ("technical-review", "Technical review and Q&A", "docs/Technical_Review_and_QA.md"),
    ("audit", "Audit findings", "docs/AUDIT_FINDINGS.md"),
    ("fix-plan", "Fix plan", "docs/FIX_PLAN.md"),
]

FIGURES = [
    ("Architecture diagram", "PANEL_REVIEW/architecture_diagram.svg"),
    ("HGB workflow", "figures_workflow/workflow_hgb.png"),
    ("InceptionTime workflow", "figures_workflow/workflow_inceptiontime.png"),
    ("Train vs test accuracy (all models)", "model_comparison/trainvstest_accuracy.png"),
    ("Train vs test macro F1 (all models)", "model_comparison/trainvstest_macrof1.png"),
    ("Per-class F1, all four tactics", "figures_perclass_f1/perclass_f1_all4.png"),
    ("Leave-one-video-out metrics", "lovo_metrics_chart.png"),
    ("Train / test split", "train_test_split_chart.png"),
    ("Court corner detection, all base frames", "BASEFRAMES/_contact_sheet.png"),
]

ACTION_EXAMPLES = ["SERVE", "RECEPTION", "SET", "SPIKE", "BLOCK", "DIG", "PASS", "FREE_BALL"]

REPO_FILE_TYPES = {".png", ".jpg", ".jpeg", ".svg", ".gif", ".webp"}

PRETTY = {
    "hist_gradient_boosting": "HistGradientBoosting (deployed)", "gradient_boosting": "Gradient Boosting",
    "xgboost": "XGBoost", "lightgbm": "LightGBM", "catboost": "CatBoost", "random_forest": "Random Forest",
    "extra_trees": "Extra Trees", "rbf_svm": "SVM (RBF)", "linear_svm": "Linear SVM",
    "logistic_regression": "Logistic Regression", "knn": "k-NN", "mlp": "MLP",
    "inceptiontime": "InceptionTime", "tcn": "TCN", "bilstm": "BiLSTM", "mamba": "Mamba SSM",
}


def _load(rel: str):
    p = config.REPO_ROOT / rel
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


@lru_cache(maxsize=1)
def metrics() -> dict:
    ml = _load("ml_results_v6/ml_results.json") or {}
    dl = _load("deep_results_v6/deep_results.json") or {}
    rows = []
    for key, m in (ml.get("models") or {}).items():
        rows.append({"key": key, "name": PRETTY.get(key, key), "family": "Machine learning",
                     "lovo_accuracy": m.get("lovo_accuracy"), "lovo_macro_f1": m.get("lovo_macro_f1"),
                     "lovo_kappa": m.get("lovo_cohen_kappa"), "kfold_accuracy": m.get("kfold_accuracy"),
                     "kfold_macro_f1": m.get("kfold_macro_f1"), "deployed": key == ml.get("winner")})
    for key, m in (dl.get("architectures") or {}).items():
        rows.append({"key": key, "name": PRETTY.get(key, key), "family": "Deep learning",
                     "lovo_accuracy": m.get("pooled_lovo_accuracy"),
                     "lovo_macro_f1": m.get("pooled_lovo_macro_f1_4class"),
                     "lovo_kappa": None, "kfold_accuracy": None, "kfold_macro_f1": None, "deployed": False,
                     "per_class_f1": m.get("pooled_per_class_f1")})
    rows.sort(key=lambda r: -(r["lovo_macro_f1"] or 0))
    return {
        "n_clips": ml.get("n_clips"), "class_counts": ml.get("class_counts"),
        "winner": ml.get("winner"), "selected_by": ml.get("selected_by"),
        "majority_baseline": ml.get("majority_baseline"), "models": rows,
    }


def _doc_path(rel: str) -> Path:
    # on Hugging Face the root README.md is the Space card; the project README is kept as PROJECT_README.md
    alt = config.REPO_ROOT / "PROJECT_README.md"
    return alt if rel == "README.md" and alt.exists() else config.REPO_ROOT / rel


def docs_index() -> list[dict]:
    out = []
    for slug, title, rel in DOCS:
        p = _doc_path(rel)
        if p.exists():
            out.append({"slug": slug, "title": title, "path": rel, "bytes": p.stat().st_size})
    return out


def doc(slug: str) -> dict | None:
    for s, title, rel in DOCS:
        if s == slug:
            p = _doc_path(rel)
            if not p.exists():
                return None
            return {"slug": s, "title": title, "path": rel,
                    "base": str(Path(rel).parent.as_posix()), "markdown": p.read_text(encoding="utf-8")}
    return None


def figures() -> list[dict]:
    out = [{"title": t, "path": rel} for t, rel in FIGURES if (config.REPO_ROOT / rel).exists()]
    for a in ACTION_EXAMPLES:
        rel = f"PAPER_OUTPUT/action_examples/{a}.png"
        if (config.REPO_ROOT / rel).exists():
            out.append({"title": f"Action example: {a.replace('_', ' ')}", "path": rel, "group": "actions"})
    return out


def repo_file(rel: str) -> Path | None:
    """A repo image, only if it is an image inside REPO_ROOT (no traversal)."""
    rel = re.sub(r"^(\./)+", "", rel.replace("\\", "/"))
    p = (config.REPO_ROOT / rel).resolve()
    try:
        p.relative_to(config.REPO_ROOT)
    except ValueError:
        return None
    if p.suffix.lower() not in REPO_FILE_TYPES or not p.is_file():
        return None
    return p


def demos() -> list[dict]:
    meta = config.DEMOS_DIR / "demos.json"
    if not meta.exists():
        return []
    items = json.loads(meta.read_text(encoding="utf-8"))
    return [d for d in items if (config.DEMOS_DIR / d["file"]).exists()]
