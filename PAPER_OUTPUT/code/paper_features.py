"""Computes every feature named in Zhang et al., "Bi-Causal: Group Activity
Recognition via Bidirectional Causality" (CVPR 2024), over the FULL length of
one of this project's own videos, using this project's own detector/tracker
output (the bicausal_extract.py cache) as the source of player/ball geometry.

IMPORTANT — read this before trusting any number this script prints:
The paper's Interaction Module, Relation Module, fusion encoders and the
classifier "delta" used for the KL-divergence diagnostic are all TRAINED
neural network layers. The paper trains them for 80 epochs on 3,493 labelled
clips of the public VOLLEYBALL dataset. We do not have that dataset
downloaded, and we have no group-activity labels for this project's own
continuous footage (videoplayback (2).mp4 is not in fyp/labels.csv). There is
also no public pretrained checkpoint released for this paper.

So every nn.Linear / nn.TransformerEncoderLayer / attention weight below is
RANDOMLY INITIALISED ONCE (fixed seed=42) and never trained. This script is
architecturally faithful to every equation in the paper (O, P, K, E, I, R,
cls_inter, cls_rela, exchange tokens, L_KL, final fusion Z) and really runs
real tensors through real self-attention / Transformer layers shaped exactly
as the paper specifies -- but the NUMBERS are feature-extraction-with-random-
weights, not a trained classifier's accurate output. Treat this as "the
paper's machinery, executed end-to-end on our own footage" for inspection
and methodology-report purposes, not as a tactic classifier you'd deploy.

Pipeline:
  1. Load the bicausal_extract.py cache (people+ball per raw frame).
  2. Reuse bicausal_render.py's own geometry/team-split/ball-tracking code
     (same homography, same player roster logic) so our player/ball positions
     match the annotated video exactly.
  3. Slide a non-overlapping T=10-frame window across every frame of the
     video (paper section 4.2: T=10 is the paper's own evaluation setting).
  4. Per window, compute every quantity in section 3 of the paper, save to
     one compressed .npz per 500-window shard under PAPER_OUTPUT/features/.

Usage:
  python paper_features.py --cache ../cache/cache_vp2_full.pkl \
      --video-name "videoplayback (2).mp4" --out ../features
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "OUTPUTS"))
import bicausal_render as bc  # noqa: E402  (reuse its geometry/team/ball code)

D = 256          # feature dimension D, fixed in the paper (section 4.2)
T = 10            # sequence window T, fixed in the paper (section 4.2)
NP_MAX = 12       # paper's Np: up to 6 players per side x 2 sides
NM = 2            # paper's Nm: 2 subgroups (far team, near team)
NO = 1            # paper's No: 1 object (the ball)
NV = NP_MAX + NO + NM   # total graph nodes per frame used for E/I
M_CLASSES = 8     # paper's M: VOLLEYBALL dataset has 8 group-activity classes
RAW_P, RAW_O, RAW_K = 38, 5, 6   # see _raw_person/_raw_ball/_raw_kinematic


# --------------------------------------------------------------------- model
class PaperModules(nn.Module):
    """Every learnable piece the paper defines (section 3), fixed-seed random
    init, never trained (see module docstring)."""

    def __init__(self):
        super().__init__()
        self.person_proj = nn.Sequential(nn.Linear(RAW_P, D), nn.LayerNorm(D))
        self.ball_proj = nn.Sequential(nn.Linear(RAW_O, D), nn.LayerNorm(D))
        self.kin_proj = nn.Sequential(nn.Linear(RAW_K, D), nn.LayerNorm(D))

        # Interaction Module (eq. 1): W in R^{D x D}, Self-Attention(E), Norm
        self.W = nn.Linear(D, D, bias=False)
        self.norm_v = nn.LayerNorm(D)
        self.attn_E = nn.MultiheadAttention(embed_dim=NV * NV, num_heads=5, batch_first=True)

        # Relation Module (eq. 3-5): S-encoder / T-encoder, applied in both orders
        self.s_encoder = nn.TransformerEncoderLayer(d_model=D, nhead=8, batch_first=True)
        self.t_encoder = nn.TransformerEncoderLayer(d_model=D, nhead=8, batch_first=True)

        # fuse kinematic+interaction -> cls_inter, kinematic+relation -> cls_rela
        self.fuse_inter = nn.TransformerEncoderLayer(d_model=D, nhead=8, batch_first=True)
        self.fuse_rela = nn.TransformerEncoderLayer(d_model=D, nhead=8, batch_first=True)

        # final fusion of cls_inter + cls_rela -> Z (eq. 10's Lgroup input)
        self.fuse_final = nn.TransformerEncoderLayer(d_model=D, nhead=8, batch_first=True)

        # classifier delta used only for the L_KL diagnostic (eq. 6-8)
        self.delta_inter = nn.Linear(D, M_CLASSES)
        self.delta_rela = nn.Linear(D, M_CLASSES)

    def interaction_module(self, V, E):
        # V: (T, Nv, D)   E: (T, Nv, Nv)      -- eq. 1: I = sigma(SelfAttn(E) . Norm(VW))
        Tn, Nv, _ = E.shape
        E_flat = E.reshape(1, Tn, Nv * Nv)
        sa, _ = self.attn_E(E_flat, E_flat, E_flat)
        sa = sa.reshape(Tn, Nv, Nv)
        VW = self.norm_v(self.W(V))              # (T, Nv, D)
        I = F.relu(torch.bmm(sa, VW))             # (T, Nv, D)
        return I

    def relation_module(self, P):
        # P: (T, Np, D)
        # S-encoder = spatial (across players within a frame)
        # T-encoder = temporal (across frames for one player)
        s_first = self.s_encoder(P)                       # spatial per frame (batch=T, seq=Np)
        r_st = self.t_encoder(s_first.transpose(0, 1)).transpose(0, 1)   # then temporal
        t_first = self.t_encoder(P.transpose(0, 1)).transpose(0, 1)      # temporal per player (batch=Np, seq=T)
        r_ts = self.s_encoder(t_first)                     # then spatial
        return r_st + r_ts                                  # eq. 5

    def fuse_cls(self, K, X, encoder):
        # concat kinematic tokens with X (I or R), mean-pool through a transformer encoder -> cls token
        seq = torch.cat([K.reshape(1, -1, D), X.reshape(1, -1, D)], dim=1)
        out = encoder(seq)
        return out.mean(dim=1).squeeze(0)   # (D,)

    def kl_diagnostic(self, cls_inter, cls_rela):
        # eq. 6-8, computed (not trained against labels -- diagnostic only)
        p_inter = F.log_softmax(self.delta_inter(cls_inter), dim=-1)
        p_rela = F.log_softmax(self.delta_rela(cls_rela), dim=-1)
        d_ir = F.kl_div(p_rela, p_inter, log_target=True, reduction="sum")
        d_ri = F.kl_div(p_inter, p_rela, log_target=True, reduction="sum")
        return float(d_ir + d_ri)


# --------------------------------------------------------------- raw features
def _raw_person(q, prev_court, fps):
    kp = q.get("kp")
    node = q["node"]
    vec = np.zeros(34, np.float32)
    if kp is not None:
        for i in range(17):
            if kp[i, 2] >= 0.2:
                vec[2 * i] = (kp[i, 0] - node[0]) / 100.0
                vec[2 * i + 1] = (kp[i, 1] - node[1]) / 100.0
    X, Y = q["court"]
    vx, vy = 0.0, 0.0
    if prev_court is not None:
        vx, vy = (X - prev_court[0]) * fps, (Y - prev_court[1]) * fps
    return np.concatenate([[X, Y, vx, vy], vec]).astype(np.float32)   # 4+34=38


def _raw_ball(court_xy, prev_xy, detected, fps):
    X, Y = court_xy
    vx, vy = 0.0, 0.0
    if prev_xy is not None and detected:
        vx, vy = (X - prev_xy[0]) * fps, (Y - prev_xy[1]) * fps
    return np.array([X, Y, vx, vy, float(detected)], np.float32)   # 5


def _raw_kinematic(court_pts, prev_mean, fps):
    if len(court_pts) == 0:
        return np.zeros(RAW_K, np.float32), prev_mean
    pts = np.array(court_pts, np.float32)
    mean = pts.mean(axis=0)
    std = pts.std(axis=0)
    vx, vy = 0.0, 0.0
    if prev_mean is not None:
        vx, vy = (mean[0] - prev_mean[0]) * fps, (mean[1] - prev_mean[1]) * fps
    return np.concatenate([mean, std, [vx, vy]]).astype(np.float32), mean   # 6


def window_roster(frames_players, team):
    cnt = Counter()
    for g in frames_players:
        for q in g:
            if q["team"] == team and q["id"] >= 0:
                cnt[q["id"]] += 1
    return [i for i, _ in cnt.most_common(6)]


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--video-name", required=True, help='e.g. "videoplayback (2).mp4"')
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard-windows", type=int, default=500)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("loading cache ...", flush=True)
    with open(args.cache, "rb") as f:
        cache = pickle.load(f)
    frames, fps = cache["frames"], cache["fps"]
    n = len(frames)
    print(f"{n} frames @ {fps:.2f} fps", flush=True)

    bc.set_court(args.video_name)
    players = bc.build_players(frames)
    ball_px, segs, detected = bc.build_ball(frames, players)
    ball_court = np.full((n, 2), np.nan, np.float32)
    valid = ~np.isnan(ball_px[:, 0])
    ball_court[valid] = bc.to_court(ball_px[valid])

    torch.manual_seed(42)
    model = PaperModules().eval()

    n_windows = n // T
    print(f"{n_windows} non-overlapping windows of T={T} frames", flush=True)

    shard, shard_idx, t0 = [], 0, time.time()

    def flush_shard():
        nonlocal shard, shard_idx
        if not shard:
            return
        out = {k: np.stack([s[k] for s in shard]) for k in shard[0]}
        path = out_dir / f"paper_features_shard{shard_idx:04d}.npz"
        np.savez_compressed(path, **out)
        print(f"  wrote {path.name} ({len(shard)} windows)", flush=True)
        shard, shard_idx = [], shard_idx + 1

    with torch.no_grad():
        for w in range(n_windows):
            a, b = w * T, w * T + T
            win_players = players[a:b]
            roster = {bc.FAR: window_roster(win_players, bc.FAR),
                      bc.NEAR: window_roster(win_players, bc.NEAR)}
            slots = roster[bc.FAR] + roster[bc.NEAR]          # up to 12 ids, fixed order this window
            slots += [-999] * (NP_MAX - len(slots))            # pad

            P_raw = np.zeros((T, NP_MAX, RAW_P), np.float32)
            P_mask = np.zeros((T, NP_MAX), bool)
            O_raw = np.zeros((T, NO, RAW_O), np.float32)
            K_raw = np.zeros((T, NM, RAW_K), np.float32)

            prev_court = {sid: None for sid in slots}
            prev_ball = None
            prev_kin = {bc.FAR: None, bc.NEAR: None}
            for ti in range(T):
                fi = a + ti
                g = win_players[ti]
                by_id = {q["id"]: q for q in g if q["id"] >= 0}
                for si, sid in enumerate(slots):
                    q = by_id.get(sid)
                    if q is not None:
                        P_raw[ti, si] = _raw_person(q, prev_court[sid], fps)
                        P_mask[ti, si] = True
                        prev_court[sid] = q["court"]
                det = bool(valid[fi])
                cxy = ball_court[fi] if det else np.zeros(2, np.float32)
                O_raw[ti, 0] = _raw_ball(cxy, prev_ball, det, fps)
                prev_ball = cxy if det else prev_ball
                for team in (bc.FAR, bc.NEAR):
                    pts = [q["court"] for q in g if q["team"] == team]
                    kvec, prev_kin[team] = _raw_kinematic(pts, prev_kin[team], fps)
                    K_raw[ti, team] = kvec

            P_t = torch.from_numpy(P_raw)
            O_t = torch.from_numpy(O_raw)
            K_t = torch.from_numpy(K_raw)

            P_proj = model.person_proj(P_t)     # (T, NP_MAX, D)
            O_proj = model.ball_proj(O_t)        # (T, NO, D)
            K_proj = model.kin_proj(K_t)         # (T, NM, D)

            V = torch.cat([P_proj, O_proj, K_proj], dim=1)      # (T, Nv, D)  -- preprocessed O/P/K, fig.3
            E = torch.bmm(V, V.transpose(1, 2))                   # eq.: e_t = v_t . v_t^T, stacked -> (T,Nv,Nv)

            I = model.interaction_module(V, E)                    # eq. 1          (T, Nv, D)
            R = model.relation_module(P_proj)                      # eq. 3-5        (T, NP_MAX, D)

            cls_inter = model.fuse_cls(K_proj, I, model.fuse_inter)
            cls_rela = model.fuse_cls(K_proj, R, model.fuse_rela)

            # causality communication channel: exchange tokens = graph-mean "message"
            exch_inter = I.mean(dim=(0, 1))      # message FROM interaction graph
            exch_rela = R.mean(dim=(0, 1))       # message FROM relation graph
            cls_inter_final = cls_inter + exch_rela     # interaction graph guided by relation's prior
            cls_rela_final = cls_rela + exch_inter       # relation graph guided by interaction's prior

            l_kl = model.kl_diagnostic(cls_inter_final, cls_rela_final)

            fused_seq = torch.stack([cls_inter_final, cls_rela_final]).unsqueeze(0)
            Z = model.fuse_final(fused_seq).mean(dim=1).squeeze(0)    # final GAR representation

            shard.append({
                "window": w, "frame_start": a, "frame_end": b,
                "O": O_raw.astype(np.float16), "P": P_raw.astype(np.float16),
                "K": K_raw.astype(np.float16), "P_mask": P_mask,
                "E": E.numpy().astype(np.float16),
                "I": I.numpy().astype(np.float16), "R": R.numpy().astype(np.float16),
                "cls_inter": cls_inter_final.numpy().astype(np.float16),
                "cls_rela": cls_rela_final.numpy().astype(np.float16),
                "exchange_token_inter": exch_inter.numpy().astype(np.float16),
                "exchange_token_rela": exch_rela.numpy().astype(np.float16),
                "L_KL": np.float32(l_kl),
                "Z_final": Z.numpy().astype(np.float16),
                "player_track_ids": np.array(slots, np.int32),
            })
            if len(shard) >= args.shard_windows:
                flush_shard()
            if w % 200 == 0:
                el = time.time() - t0
                print(f"window {w}/{n_windows}  {el:.0f}s elapsed", flush=True)

    flush_shard()

    manifest = {
        "video_name": args.video_name, "fps": fps, "n_frames": n,
        "window_T": T, "n_windows": n_windows, "D": D,
        "NP_MAX": NP_MAX, "NO": NO, "NM": NM, "NV": NV, "M_CLASSES": M_CLASSES,
        "n_shards": shard_idx,
        "note": "Attention/Transformer weights are fixed-seed random init, NOT trained. "
                "See docstring at top of paper_features.py.",
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print("done:", manifest, flush=True)


if __name__ == "__main__":
    main()
