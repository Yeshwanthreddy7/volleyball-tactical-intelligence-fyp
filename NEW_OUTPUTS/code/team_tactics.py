"""Per-frame tactical call for the focus team (near / bottom side).

The four classes are the project's tactical labels:

    Coordinated Attack | Coordinated Defense | Delayed Support | Spacing Breakdown

They come from the deployed classifier (tactical_ml_v6.joblib, the calibrated
HistGradientBoosting model, see METHODOLOGY.md / DEFENSE_MASTER.md). The input
is built exactly as at training time:

  per frame   ball + up to 6 near-team foot points in the training court plane
              (1800 cm across x 900 cm deep, net at y = 450 cm, near half = y > 450)
  per window  29 samples (~1 s at 30 fps; 50 fps sources are resampled) ->
              features.build_model_sequence -> (29, 18) -> classifier

Windows slide with a ~1/3 s stride. Each frame gets the mean probability of the
windows that cover it, then a short temporal smoothing and a minimum dwell.
Window probabilities are prior-corrected (divided by the training class
frequencies, 39/37/11/12 %), i.e. the balanced decision rule; without it the
two rare classes almost never win the argmax after smoothing.
Frames outside live play (camera change / replay, fewer than 3 near players,
no ball touch nearby) get no call.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]          # finalyearproject/
sys.path.insert(0, str(ROOT / "fyp"))

from features import build_model_sequence           # noqa: E402
import ml_classifier                                 # noqa: E402

CLASSES = ["Coordinated Attack", "Coordinated Defense", "Delayed Support", "Spacing Breakdown"]
MODEL_PATH = ROOT / "tactical_ml_v6.joblib"
WIN = 29                 # samples per window (training contract)
TRAIN_FPS = 30.0
# class counts of the 948 training clips (ml_results_v6/ml_results.json)
TRAIN_PRIOR = np.array([370, 355, 108, 115], np.float64) / 948.0


def court_m_to_train_cm(c):
    """bicausal_render court metres (X 0..9 across, Y 0..18 far->near) ->
    the training plane (x 0..1800 across, y 0..900 far->near)."""
    c = np.asarray(c, np.float64)
    return np.stack([c[..., 0] * 200.0, c[..., 1] * 50.0], axis=-1)


def raw_rows(players, ball, to_court, team):
    """(n, 14) raw rows [ball_x, ball_y, p1_x, p1_y, ..., p6_x, p6_y]; missing = 0
    (the training CSV sentinel, turned into NaN by features.raw_frame_to_arrays)."""
    n = len(players)
    rows = np.zeros((n, 14), np.float64)
    for fi in range(n):
        b = ball[fi]
        if not np.isnan(b[0]):
            rows[fi, 0:2] = court_m_to_train_cm(to_court(b)[0])
        tq = sorted((q for q in players[fi] if q["team"] == team), key=lambda q: q["id"])[:6]
        for k, q in enumerate(tq):
            rows[fi, 2 + 2 * k: 4 + 2 * k] = court_m_to_train_cm(q["court"])
    return rows


def window_probs(rows, fps, stride_s=1 / 3):
    """Slide 29-sample windows over the rows; returns (starts, ends, probs)."""
    import joblib
    bundle = joblib.load(MODEL_PATH)
    assert list(bundle["class_names"]) == CLASSES, bundle["class_names"]
    pipe, mode = bundle["pipeline"], bundle.get("feature_mode", "invariant")

    span = int(round(WIN * fps / TRAIN_FPS))         # source frames per window
    stride = max(1, int(round(stride_s * fps)))
    starts = np.arange(0, len(rows) - span + 1, stride)
    feats = []
    for s in starts:
        idx = np.round(np.linspace(s, s + span - 1, WIN)).astype(int)
        seq = build_model_sequence(rows[idx], target_len=WIN)
        feats.append(ml_classifier.features_from_model_sequence(seq, mode))
    X = np.nan_to_num(np.asarray(feats, np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    P = pipe.predict_proba(X) if len(X) else np.zeros((0, 4))
    # balanced decision rule: divide by the training class frequencies and
    # renormalise, so the two rare classes are not lost to the 39/37 % priors
    P = P / TRAIN_PRIOR
    P /= P.sum(1, keepdims=True)
    return starts, starts + span - 1, P


def frame_calls(n, fps, starts, ends, P, live, min_dwell_s=0.8, smooth_s=0.5):
    """Per-frame label index (-1 = no call) and per-frame probability vector."""
    acc = np.zeros((n, 4))
    cnt = np.zeros(n)
    for s, e, p in zip(starts, ends, P):
        acc[s:e + 1] += p
        cnt[s:e + 1] += 1
    prob = np.where(cnt[:, None] > 0, acc / np.maximum(cnt, 1)[:, None], 0.0)
    # gaussian smoothing along time
    sig = max(1.0, smooth_s * fps / 2)
    k = np.exp(-0.5 * (np.arange(-3 * sig, 3 * sig + 1) / sig) ** 2)
    k /= k.sum()
    prob = np.stack([np.convolve(prob[:, c], k, "same") for c in range(4)], 1)
    lab = np.where(live & (cnt > 0), prob.argmax(1), -1)

    # minimum dwell: a short run inside live play takes the label of its
    # longer neighbour whose summed probability over the run is higher
    min_len = int(min_dwell_s * fps)
    # (shortest run first; every merge removes a run, so this terminates)
    while True:
        runs = _runs(lab)
        short = []
        for i, (a, b, v) in enumerate(runs):
            if v < 0 or b - a + 1 >= min_len:
                continue
            nb = [runs[j][2] for j in (i - 1, i + 1) if 0 <= j < len(runs) and runs[j][2] >= 0]
            if nb:
                short.append((b - a, a, b, nb))
        if not short:
            break
        _, a, b, nb = min(short)
        lab[a:b + 1] = max(nb, key=lambda c: prob[a:b + 1, c].sum())
    # isolated short calls (no labelled neighbour) are dropped
    for a, b, v in _runs(lab):
        if v >= 0 and b - a + 1 < min_len:
            lab[a:b + 1] = -1
    return lab, prob


def _runs(lab):
    out, a = [], 0
    for i in range(1, len(lab) + 1):
        if i == len(lab) or lab[i] != lab[a]:
            out.append((a, i - 1, int(lab[a])))
            a = i
    return out


def segments(lab, prob, fps, start):
    """Run-length segments of the per-frame calls, for the CSV."""
    out = []
    for a, b, v in _runs(lab):
        if v < 0:
            continue
        out.append(dict(start_frame=start + a, end_frame=start + b,
                        start_s=round((start + a) / fps, 2), end_s=round((start + b) / fps, 2),
                        tactic=CLASSES[v], mean_confidence=round(float(prob[a:b + 1, v].mean()), 3)))
    return out
