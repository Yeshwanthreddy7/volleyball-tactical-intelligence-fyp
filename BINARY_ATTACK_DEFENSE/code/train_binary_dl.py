"""
Stage 2 - train the binary Coordinated-Attack vs Coordinated-Defense model.

Architecture: InceptionTime (won the 4-class LOVO bench at 0.559).
Regularisation stack (the 4-class models all hit train-F1 ~1.0 and lost ~0.07
to cross-video generalisation - this is the fix):
  * mirror augmentation (baked in the cache)
  * per-step Gaussian feature jitter
  * mixup (alpha 0.2)
  * dropout 0.3, weight decay 3e-4, label smoothing 0.05
  * deep ensemble - E seeds, logits averaged  (InceptionTime's intended use)

Protocol: 8-fold leave-one-video-out, held-out probabilities pooled across
folds -> one honest binary macro-F1 / balanced-accuracy / ROC-AUC.
Then E seeds are retrained on ALL clips for deployment.

Usage:
  python v7_tactical/train_binary.py --epochs 100 --ensemble 5
  python v7_tactical/train_binary.py --epochs 100 --ensemble 3 --lovo-only
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.join(ROOT, "v7_tactical")

sys.path.insert(0, os.path.join(ROOT, "fyp"))

from mamba_model import create_temporal_model                  # noqa: E402
from features import FEATURE_VERSION, MODEL_FEATURE_COLS       # noqa: E402

CACHE = os.path.join(HERE, "cache", "binary_cache.npz")
MODEL_DIR = os.path.join(ROOT, "BINARY_ATTACK_DEFENSE", "DL_models")
RES_DIR = os.path.join(ROOT, "BINARY_ATTACK_DEFENSE", "results")
LABEL_NAMES = ["Coordinated Defense", "Coordinated Attack"]    # index 0, 1
FOLDS = ["(1)", "(2)", "(3)", "(4)", "(5)", "(6)", "(plain)", "?"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
class SeqSet(Dataset):
    """Serves normalised (29,18) sequences.  Training mode: random mirror +
    Gaussian jitter in feature space (fast - the expensive rebuild is cached)."""

    def __init__(self, seq, seq_mir, y, mean, std, train: bool, jitter: float = 0.10):
        self.seq = seq
        self.seq_mir = seq_mir
        self.y = y
        self.mean = mean
        self.std = std
        self.train = train
        self.jitter = jitter

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        s = self.seq_mir[i] if (self.train and np.random.rand() < 0.5) else self.seq[i]
        s = (s - self.mean) / self.std
        if self.train and self.jitter > 0:
            s = s + np.random.normal(0, self.jitter, s.shape).astype(np.float32)
        return torch.from_numpy(s.astype(np.float32)), int(self.y[i])


def _norm_stats(seq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = seq.reshape(-1, seq.shape[-1]).mean(0)
    s = seq.reshape(-1, seq.shape[-1]).std(0)
    s[s < 1e-6] = 1.0
    return m.astype(np.float32), s.astype(np.float32)


# --------------------------------------------------------------------------- #
# train / eval one network
# --------------------------------------------------------------------------- #
ARCH = "inceptiontime"   # set from args in main()


def _make(seed: int, d_model: int, n_layers: int, dropout: float) -> nn.Module:
    torch.manual_seed(seed)
    np.random.seed(seed)
    return create_temporal_model(ARCH, input_dim=18, d_model=d_model,
                                 n_layers=n_layers, num_classes=2,
                                 dropout=dropout).to(DEVICE)


def _run_epoch(model, loader, opt, crit, mixup_alpha):
    model.train()
    tot = 0.0
    n = 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        if mixup_alpha > 0 and np.random.rand() < 0.5:
            lam = float(np.random.beta(mixup_alpha, mixup_alpha))
            perm = torch.randperm(xb.size(0), device=DEVICE)
            xb = lam * xb + (1 - lam) * xb[perm]
            out = model(xb)
            loss = lam * crit(out, yb) + (1 - lam) * crit(out, yb[perm])
        else:
            loss = crit(model(xb), yb)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()
        tot += float(loss) * xb.size(0)
        n += xb.size(0)
    return tot / max(n, 1)


@torch.no_grad()
def _probs(model, seq, mean, std) -> np.ndarray:
    model.eval()
    x = torch.from_numpy(((seq - mean) / std).astype(np.float32)).to(DEVICE)
    out = []
    for i in range(0, len(x), 512):
        out.append(torch.softmax(model(x[i:i + 512]), dim=1).cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def _val_bal_acc(model, seq, y, mean, std) -> float:
    p = _probs(model, seq, mean, std).argmax(1)
    accs = [(p[y == c] == c).mean() for c in (0, 1) if (y == c).any()]
    return float(np.mean(accs))


def train_seed(seq_tr, seqm_tr, y_tr, seq_va, y_va, mean, std, seed, args):
    model = _make(seed, args.d_model, args.n_layers, args.dropout)
    ds = SeqSet(seq_tr, seqm_tr, y_tr, mean, std, train=True, jitter=args.jitter)
    # class-balanced sampling
    cw = np.array([1.0 / max((y_tr == c).sum(), 1) for c in (0, 1)])
    w = cw[y_tr]
    sampler = torch.utils.data.WeightedRandomSampler(w, len(w), replacement=True)
    dl = DataLoader(ds, batch_size=args.batch, sampler=sampler, drop_last=False)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=3e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.02)

    best_state, best_va, best_ep, bad = None, -1.0, 0, 0
    for ep in range(1, args.epochs + 1):
        _run_epoch(model, dl, opt, crit, args.mixup)
        sched.step()
        va = _val_bal_acc(model, seq_va, y_va, mean, std) if len(seq_va) else 0.0
        if va >= best_va:
            best_va, best_ep = va, ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if args.patience and bad >= args.patience and ep >= 40:
                break
    model.load_state_dict(best_state)
    return model, best_va, best_ep


def _stratified_val(y, groups, frac, seed):
    """hold out `frac` of each (video,class) cell for early-stopping val."""
    rng = np.random.default_rng(seed)
    va = np.zeros(len(y), bool)
    for g in set(groups):
        for c in (0, 1):
            idx = np.where((groups == g) & (y == c))[0]
            if len(idx) < 4:
                continue
            k = max(1, int(round(len(idx) * frac)))
            va[rng.choice(idx, k, replace=False)] = True
    return ~va, va


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def binary_report(y, prob) -> dict:
    pred = prob.argmax(1)
    p1 = prob[:, 1]
    cm = np.zeros((2, 2), int)
    for t, q in zip(y, pred):
        cm[t, q] += 1
    tp, fp, fn, tn = cm[1, 1], cm[0, 1], cm[1, 0], cm[0, 0]
    prec1 = tp / max(tp + fp, 1)
    rec1 = tp / max(tp + fn, 1)
    prec0 = tn / max(tn + fn, 1)
    rec0 = tn / max(tn + fp, 1)
    f1_1 = 2 * prec1 * rec1 / max(prec1 + rec1, 1e-9)
    f1_0 = 2 * prec0 * rec0 / max(prec0 + rec0, 1e-9)
    order = np.argsort(p1)
    ys = y[order]
    n_pos, n_neg = ys.sum(), (1 - ys).sum()
    auc = (np.cumsum(1 - ys)[ys == 1].sum()) / max(n_pos * n_neg, 1)
    return {
        "accuracy": round(float((pred == y).mean()), 4),
        "balanced_accuracy": round(float((rec0 + rec1) / 2), 4),
        "macro_f1": round(float((f1_0 + f1_1) / 2), 4),
        "attack_f1": round(float(f1_1), 4), "attack_precision": round(float(prec1), 4),
        "attack_recall": round(float(rec1), 4),
        "defense_f1": round(float(f1_0), 4), "defense_precision": round(float(prec0), 4),
        "defense_recall": round(float(rec0), 4),
        "roc_auc": round(float(auc), 4),
        "confusion_rows_true_DA": cm.tolist(),
        "n": int(len(y)),
    }


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--ensemble", type=int, default=5)
    ap.add_argument("--lovo-ensemble", type=int, default=3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--d_model", type=int, default=64)
    ap.add_argument("--n_layers", type=int, default=6)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--jitter", type=float, default=0.10)
    ap.add_argument("--mixup", type=float, default=0.2)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--lovo-only", action="store_true")
    ap.add_argument("--final-only", action="store_true")
    ap.add_argument("--tag", default="binary_incept")
    ap.add_argument("--arch", default="inceptiontime")
    args = ap.parse_args()
    global ARCH; ARCH = args.arch

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RES_DIR, exist_ok=True)
    d = np.load(CACHE, allow_pickle=True)
    seq, seq_mir, y, groups = d["seq"], d["seq_mir"], d["y"], d["groups"]
    print(f"device={DEVICE}  N={len(y)}  channels={seq.shape[-1]}  "
          f"attack={int(y.sum())} defense={int((y==0).sum())}")
    t0 = time.time()

    if not args.final_only:
        oof = np.full((len(y), 2), np.nan)
        fold_rows = []
        for vid in FOLDS:
            te = groups == vid
            tr = ~te
            if te.sum() == 0:
                continue
            mean, std = _norm_stats(seq[tr])
            tr_idx = np.where(tr)[0]
            seed_probs = []
            for k in range(args.lovo_ensemble):
                m_tr, m_va = _stratified_val(y[tr], groups[tr], args.val_frac, 100 + k)
                gi_tr, gi_va = tr_idx[m_tr], tr_idx[m_va]
                model, bva, bep = train_seed(
                    seq[gi_tr], seq_mir[gi_tr], y[gi_tr],
                    seq[gi_va], y[gi_va], mean, std, 1000 + k, args)
                seed_probs.append(_probs(model, seq[te], mean, std))
                del model
                torch.cuda.empty_cache()
            pr = np.mean(seed_probs, 0)
            oof[te] = pr
            r = binary_report(y[te], pr)
            r["video"] = vid
            fold_rows.append(r)
            print(f"  {vid:8s} n={te.sum():4d}  bal_acc={r['balanced_accuracy']:.3f}  "
                  f"macroF1={r['macro_f1']:.3f}  A_f1={r['attack_f1']:.3f} "
                  f"D_f1={r['defense_f1']:.3f}  ({time.time()-t0:.0f}s)")

        pooled = binary_report(y, oof)
        print("\n=== POOLED LOVO (binary) ===")
        for k, v in pooled.items():
            if not isinstance(v, list):
                print(f"  {k:20s} {v}")
        print(f"  confusion [true D/A x pred D/A]: {pooled['confusion_rows_true_DA']}")
        out = {"protocol": "8-fold leave-one-video-out, pooled OOF probs",
               "pooled": pooled, "folds": fold_rows,
               "config": vars(args), "feature_version": FEATURE_VERSION,
               "channels": MODEL_FEATURE_COLS}
        with open(os.path.join(RES_DIR, f"lovo_{args.tag}.json"), "w") as fh:
            json.dump(out, fh, indent=2)
        np.savez(os.path.join(RES_DIR, f"oof_{args.tag}.npz"), y=y, prob=oof, groups=groups)
        print(f"  -> {os.path.join('results', f'lovo_{args.tag}.json')}")

    if args.lovo_only:
        return

    # ---- final ensemble on ALL clips -------------------------------------- #
    print("\n=== FINAL ENSEMBLE (all clips) ===")
    mean, std = _norm_stats(seq)
    m_tr, m_va = _stratified_val(y, groups, args.val_frac, 7)
    saved = []
    for k in range(args.ensemble):
        model, bva, bep = train_seed(
            seq[m_tr], seq_mir[m_tr], y[m_tr], seq[m_va], y[m_va],
            mean, std, 2000 + k, args)
        ck = os.path.join(MODEL_DIR, f"{args.tag}_seed{k}.pt")
        torch.save({
            "epoch": bep, "model_state": model.state_dict(),
            # interfaces.TorchTemporalClassifier does .to(device) on these
            "norm_mean": torch.as_tensor(mean, dtype=torch.float32),
            "norm_std": torch.as_tensor(std, dtype=torch.float32),
            "args": {"arch": args.arch, "d_model": args.d_model,
                     "n_layers": args.n_layers, "dropout": args.dropout,
                     "input_dim": 18},
            "label_names": LABEL_NAMES, "best_val_macro_f1": bva,
            "feature_version": FEATURE_VERSION, "extraction_provenance": None,
        }, ck)
        saved.append(ck)
        print(f"  seed{k}: val_bal_acc={bva:.3f} @ep{bep}  -> {os.path.basename(ck)}")
        del model
        torch.cuda.empty_cache()

    # servable single checkpoint = first seed (pipeline.py loads one .pt)
    import shutil
    shutil.copy(saved[0], os.path.join(MODEL_DIR, f"{args.tag}.pt"))
    print(f"  servable: DL_models/{args.tag}.pt  (+{len(saved)} ensemble seeds)")
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
