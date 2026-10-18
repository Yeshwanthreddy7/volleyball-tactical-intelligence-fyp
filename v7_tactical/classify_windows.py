"""
Stage 4 - vp(7) per-frame tracked coords -> smoothed Attack/Defense banner track.

Reads pipeline.py --frame-csv, slides a 29-frame window at --stride, builds the
SAME 18-channel model input the trainer used, runs the deep ensemble, then:

  * dead-ball gate  - a window is a rally BREAK (no banner) when the ball is
    absent/coasting most of the window, too few / too many players are tracked,
    the players barely move, or the 29 rows span a camera cut.
  * Viterbi smoothing - the live windows are decoded as a 2-state HMM
    (Attack / Defense) with emission = ensemble P(attack) and a tunable
    self-transition prior.  Unlike an EMA this never overrides sustained
    confident evidence, only isolated single-window noise.
  * min-dwell - runs shorter than --min-dwell-s are re-decoded by a
    confidence-weighted vote of their own windows (not blindly filled).

Out: results/vp7_windows.csv (frame_start, frame_end, label, confidence, ...)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "fyp"))

from features import build_model_sequence                        # noqa: E402
from mamba_model import InceptionTimeClassifier                  # noqa: E402

RAW = ["ball_x", "ball_y"] + [f"p{i}_{a}" for i in range(1, 7) for a in "xy"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_ensemble(pattern: str):
    cks = sorted(glob.glob(pattern))
    if not cks:
        sys.exit(f"no checkpoints match {pattern}")
    models, mean, std = [], None, None
    for c in cks:
        d = torch.load(c, map_location=DEVICE, weights_only=False)
        a = d["args"]
        m = InceptionTimeClassifier(input_dim=18, d_model=a["d_model"],
                                    n_layers=a["n_layers"], num_classes=2,
                                    dropout=0.0).to(DEVICE)
        m.load_state_dict(d["model_state"])
        m.eval()
        models.append(m)
        if mean is None:
            mean = np.asarray(d["norm_mean"].cpu() if hasattr(d["norm_mean"], "cpu")
                              else d["norm_mean"], np.float32)
            std = np.asarray(d["norm_std"].cpu() if hasattr(d["norm_std"], "cpu")
                             else d["norm_std"], np.float32)
    print(f"loaded {len(models)} ensemble members from {os.path.dirname(pattern)}")
    return models, mean, std


@torch.no_grad()
def predict(models, mean, std, seqs: np.ndarray) -> np.ndarray:
    x = torch.from_numpy(((seqs - mean) / std).clip(-8, 8).astype(np.float32)).to(DEVICE)
    acc = torch.zeros(len(x), 2, device=DEVICE)
    for m in models:
        for i in range(0, len(x), 256):
            acc[i:i + 256] += torch.softmax(m(x[i:i + 256]), 1)
    return (acc[:, 1] / len(models)).cpu().numpy()


def _close_open(dead: np.ndarray, close_w: int, open_w: int) -> np.ndarray:
    """Morphological cleanup of the dead/live mask over the window sequence.
      close: a BREAK run shorter than close_w flanked by live on BOTH sides
             -> live  (bridges ball-detector misses inside a rally)
      open : a LIVE run shorter than open_w flanked by break on BOTH sides
             -> break (kills 1-2 window blips between points)"""
    d = dead.copy()
    n = len(d)
    for target, width in ((True, close_w), (False, open_w)):
        i = 0
        while i < n:
            j = i
            while j < n and d[j] == d[i]:
                j += 1
            if d[i] == target and (j - i) < width and i > 0 and j < n \
               and d[i - 1] != target and d[j] != target:
                d[i:j] = not target
            i = j
    return d


def viterbi(p_att: np.ndarray, live: np.ndarray, stay: float) -> np.ndarray:
    """2-state (0=Defense,1=Attack) HMM decode over the LIVE windows only.
    Returns an int array over ALL windows: 0/1 for live, -1 for break."""
    out = np.full(len(p_att), -1, int)
    trans = np.log([[stay, 1 - stay], [1 - stay, stay]])
    # decode each maximal run of live windows independently
    i = 0
    while i < len(live):
        if not live[i]:
            i += 1
            continue
        j = i
        while j < len(live) and live[j]:
            j += 1
        seg = p_att[i:j]
        eps = 1e-4
        em = np.log(np.clip(np.stack([1 - seg, seg], 1), eps, 1 - eps))  # (T,2)
        T = len(seg)
        dp = np.full((T, 2), -np.inf)
        bp = np.zeros((T, 2), int)
        dp[0] = np.log([0.5, 0.5]) + em[0]
        for t in range(1, T):
            for s in (0, 1):
                cand = dp[t - 1] + trans[:, s]
                bp[t, s] = int(np.argmax(cand))
                dp[t, s] = cand[bp[t, s]] + em[t, s]
        path = np.zeros(T, int)
        path[-1] = int(np.argmax(dp[-1]))
        for t in range(T - 2, -1, -1):
            path[t] = bp[t + 1, path[t + 1]]
        out[i:j] = path
        i = j
    return out


def min_dwell(state: np.ndarray, p_att: np.ndarray, min_w: int) -> np.ndarray:
    s = state.copy()
    n = len(s)
    i = 0
    while i < n:
        j = i
        while j < n and s[j] == s[i]:
            j += 1
        if s[i] != -1 and (j - i) < min_w:
            # re-vote this short run by its own windows' confidence
            seg = p_att[i:j]
            s[i:j] = 1 if seg.mean() >= 0.5 else 0
            # if that just recreates the same short run vs both neighbours, merge
            left = s[i - 1] if i > 0 else -1
            right = s[j] if j < n else -1
            if left == right and left != -1 and s[i] != left \
               and abs(seg.mean() - 0.5) < 0.25:
                s[i:j] = left
        i = j
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default=os.path.join(HERE, "results", "vp7_frames.csv"))
    ap.add_argument("--out", default=os.path.join(HERE, "results", "vp7_windows.csv"))
    ap.add_argument("--ckpt-glob", default=os.path.join(HERE, "models", "binary_geom_seed*.pt"))
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--stay", type=float, default=0.92,
                    help="HMM self-transition prob per window (higher = smoother)")
    ap.add_argument("--min-dwell-s", type=float, default=1.0)
    ap.add_argument("--dead-speed", type=float, default=9.0,
                    help="context 90th-pctile of the 95th-pctile per-frame player "
                         "speed (cm/frame) below which the stretch is a rally break")
    ap.add_argument("--dead-ctx", type=int, default=45,
                    help="+/- frames of context for the break-speed test")
    ap.add_argument("--min-players", type=float, default=4.0)
    ap.add_argument("--ball-present-min", type=float, default=0.15,
                    help="min mean feat_ball_present over the window for a live rally")
    ap.add_argument("--gap-tol", type=int, default=6)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--att-threshold", type=float, default=0.56,
                    help="P(attack) decision boundary; >0.5 corrects the vp7 attack lean")
    ap.add_argument("--bridge-break-s", type=float, default=1.3,
                    help="BREAK runs shorter than this, inside live play, are bridged")
    ap.add_argument("--min-live-s", type=float, default=1.3,
                    help="LIVE runs shorter than this, between breaks, are dropped")
    ap.add_argument("--use-score-events", action="store_true",
                    help="force a BREAK around each detected on-screen score change")
    ap.add_argument("--score-pre", type=int, default=75,
                    help="fallback frames-before-score-update when no rally action is found")
    ap.add_argument("--score-post", type=int, default=150)
    ap.add_argument("--rally-action-speed", type=float, default=17.0,
                    help="fr_sp above this = genuine rally action; the last one before "
                         "a score update anchors the break start")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.frames)
    cols = {c.lower(): c for c in df.columns}
    raw_cols = [cols.get(f"raw_{c}", cols.get(c)) for c in RAW]
    if any(c is None for c in raw_cols):
        sys.exit(f"frame CSV missing raw coords; have {list(df.columns)[:25]}")
    fcol = cols.get("frame", cols.get("frame_id", "frame"))
    frames = df[fcol].values.astype(int)
    coord = np.nan_to_num(df[raw_cols].values.astype(np.float32), nan=0.0)
    ball_pred = df[cols["ball_predicted"]].values.astype(float) if "ball_predicted" in cols else np.zeros(len(df))
    npl = df[cols["n_players_tracked"]].values.astype(float) if "n_players_tracked" in cols else np.full(len(df), 6.0)
    bpres = df[cols["feat_ball_present"]].values.astype(float) if "feat_ball_present" in cols else np.ones(len(df))
    n = len(df)
    print(f"{n} frames  ({frames[0]}..{frames[-1]})")

    starts = list(range(0, n - 29 + 1, args.stride))
    # cache the expensive part (window build + ensemble predict); keyed on
    # frames file + stride + checkpoint set, so only smoothing/gate tuning reruns.
    ck_key = "|".join(sorted(os.path.basename(x) for x in glob.glob(args.ckpt_glob)))
    cache_p = os.path.join(HERE, "cache", f"vp7_patt_s{args.stride}.npz")
    p_att = None
    if not args.no_cache and os.path.isfile(cache_p):
        cz = np.load(cache_p, allow_pickle=True)
        _k = str(cz["key"])
        if (_k == ck_key or _k.startswith("ENSEMBLE")) and len(cz["p_att"]) == len(starts):
            p_att = cz["p_att"]
            print(f"p_att from cache ({cache_p})")
    if p_att is None:
        models, mean, std = load_ensemble(args.ckpt_glob)
        seqs = np.stack([build_model_sequence(coord[s:s + 29], 29) for s in starts])
        p_att = predict(models, mean, std, seqs)
        os.makedirs(os.path.dirname(cache_p), exist_ok=True)
        np.savez(cache_p, p_att=p_att, key=ck_key)
        print(f"p_att computed + cached ({cache_p})")

    # ---- rally-BREAK gate ----------------------------------------------- #
    # PRIMARY: a rally always has a fast movement (spike approach / dive) so
    # the 90th-pctile player speed spikes; between points nobody sprints.
    # Measured over a wide context so a brief mid-rally lull is not a "break".
    ctx = args.dead_ctx
    P_all = coord[:, 2:].reshape(n, 6, 2)
    pres_all = ~(np.all(P_all == 0, axis=2) | np.any(~np.isfinite(P_all), axis=2))
    dsp = np.linalg.norm(np.diff(P_all, axis=0), axis=2)
    vld = pres_all[1:] & pres_all[:-1]
    fr_sp = np.array([np.percentile(dsp[t][vld[t]], 95) if vld[t].any() else 0.0
                      for t in range(n - 1)] + [0.0])
    npres = pres_all.sum(1).astype(float)

    # point-scored events (on-screen score digits change) -> a real rally break.
    # The score updates 1-3s AFTER the rally ends, so anchor the break START at
    # the last genuine rally action (fr_sp spike) before the score change, not a
    # fixed offset - this stops the 2s of post-point celebration getting an A/D
    # label. Break END is a fixed offset after the score updates.
    score_ev = []
    sef = os.path.join(HERE, "results", "vp7_score_events.json")
    if args.use_score_events and os.path.isfile(sef):
        score_ev = json.load(open(sef)).get("events", [])
    ev_lo, ev_hi = [], []
    for e in score_ev:
        lo_search = max(0, e - args.score_pre - 90)
        seg = fr_sp[lo_search:e]
        hot = np.where(seg > args.rally_action_speed)[0]
        anchor = (lo_search + hot[-1] + 12) if len(hot) else (e - args.score_pre)
        ev_lo.append(min(anchor, e - 60))     # always break >= 2s before the update
        ev_hi.append(e + args.score_post)

    dead = np.zeros(len(starts), bool)
    for k, s in enumerate(starts):
        lo, hi = max(0, s - ctx), min(n, s + 29 + ctx)
        peak = np.percentile(fr_sp[lo:hi], 90)
        contiguous = (frames[min(s + 28, n - 1)] - frames[s]) <= 28 + args.gap_tol
        fmid = frames[s] + 14
        near_point = any(a <= fmid <= b for a, b in zip(ev_lo, ev_hi))
        dead[k] = (
            peak < args.dead_speed
            or npres[s:s + 29].mean() < args.min_players
            or bpres[s:s + 29].mean() < args.ball_present_min
            or not contiguous
            or near_point
        )
    w_per_s = args.fps / args.stride
    dead = _close_open(dead, int(round(args.bridge_break_s * w_per_s)),
                       int(round(args.min_live_s * w_per_s)))
    live = ~dead

    # apply the decision-threshold correction as a logit shift on p_att so the
    # Viterbi emission still sees a smooth probability
    shift = np.log(args.att_threshold / (1 - args.att_threshold))
    p_shift = 1 / (1 + np.exp(-(np.log(np.clip(p_att, 1e-4, 1 - 1e-4) /
                                       (1 - np.clip(p_att, 1e-4, 1 - 1e-4))) - shift)))

    state = viterbi(p_shift, live, args.stay)
    min_w = max(1, int(round(args.min_dwell_s * args.fps / args.stride)))
    state = min_dwell(state, p_shift, min_w)

    name = {0: "Coordinated Defense", 1: "Coordinated Attack", -1: "Unclassified"}
    rows = []
    for k, s in enumerate(starts):
        st = int(state[k])
        fs, fe = int(frames[s]), int(frames[min(s + 28, n - 1)])
        conf = float(abs(p_att[k] - 0.5) * 2) if st != -1 else float("nan")
        rows.append({"frame_start": fs, "frame_end": fe, "label": name[st],
                     "confidence": round(conf, 4),
                     "p_attack_raw": round(float(p_att[k]), 4),
                     "dead": bool(dead[k])})
    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_csv(args.out, index=False)

    vc = out["label"].value_counts()
    sw = int((out["label"].values[1:] != out["label"].values[:-1]).sum())
    print(f"\nwrote {args.out}  ({len(out)} windows, stride {args.stride}, stay {args.stay})")
    for k2, v in vc.items():
        print(f"  {k2:22s} {v:5d}  ({v/len(out)*100:5.1f}%)")
    live_lab = out.loc[out.label != "Unclassified", "label"]
    contra = ((out["label"] == "Coordinated Attack") != (out["p_attack_raw"] > 0.5))
    contra = (contra & (out["label"] != "Unclassified") & (out["confidence"] > 0.5)).mean()
    print(f"  state switches: {sw}   dead: {dead.mean()*100:.1f}%   "
          f"confident-contradiction: {contra*100:.1f}%")


if __name__ == "__main__":
    main()
