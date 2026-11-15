"""Player-tracking accuracy against HAND-LABELLED ground truth (added 2026-11-15).

Ground truth: the public Volleyball Dataset (Ibrahim et al., CVPR 2016) tracking
annotation - every player box, with a fixed player id, hand-checked per frame.
Frames: the 4 sample clips in finalyearproject/videos_sample (41 frames each).

Our tracker is run exactly as in the project (OUTPUTS/bicausal_extract.py):
YOLO11m-pose + BoT-SORT, imgsz 1280, conf 0.25, people only.

Metrics (standard CLEAR-MOT and ID measures, IoU >= 0.5 match):
  MOTA  = 1 - (misses + false boxes + ID switches) / ground-truth boxes
  IDF1  = how often a player keeps the SAME id across the clip (0..1)
  ID switches, precision, recall
"False boxes" counts referees, coaches, line judges and crowd, which the
Volleyball Dataset does not label; the project removes most of them with the
court gate, so a second MOTA is given with people outside the players' area
(no overlap with any labelled player box in that frame and foot point outside
the box around all labelled players' feet, 5 % margin) ignored.

Run:  python tracking_eval.py      Out: tracking_eval.json
"""
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

ROOT = Path(r"C:\Users\yashwanth\OneDrive\Desktop\GROUP_56\finalyearproject")
FR = ROOT / "videos_sample" / "videos_sample"
GT = ROOT / "volleyball_tracking_annotation" / "volleyball_tracking_annotation" / "_"
OUT = Path(__file__).resolve().parent / "tracking_eval.json"
CLIPS = [("10", "18360"), ("10", "20525"), ("7", "38025"), ("7", "51725")]


def iou(a, b):
    x1, y1 = np.maximum(a[:, None, 0], b[None, :, 0]), np.maximum(a[:, None, 1], b[None, :, 1])
    x2, y2 = np.minimum(a[:, None, 2], b[None, :, 2]), np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    ar = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])  # noqa: E731
    return inter / (ar(a)[:, None] + ar(b)[None, :] - inter + 1e-9)


def load_gt(v, c):
    g = defaultdict(list)
    for line in open(GT / v / c / f"{c}.txt"):
        p = line.split()
        pid, x1, y1, x2, y2, fr, lost = int(p[0]), *map(float, p[1:5]), int(p[5]), int(p[6])
        if not lost:
            g[fr].append((pid, x1, y1, x2, y2))
    return g


def in_player_area(box, gts):
    if not len(gts):
        return False
    if iou(np.array([box]), gts).max() > 0:
        return True
    fx, fy = (box[0] + box[2]) / 2, box[3]
    x0, x1 = gts[:, 0].min(), gts[:, 2].max()
    y0, y1 = gts[:, 1].min(), gts[:, 3].max()
    mx, my = 0.05 * (x1 - x0), 0.05 * (y1 - y0)
    return x0 - mx <= fx <= x1 + mx and y0 - my <= fy <= y1 + my


def evaluate(seq, gated, thr=0.5):
    """seq: list of (gt list, pred list); returns CLEAR-MOT + IDF1 counts."""
    n_gt = fp = fn = sw = tp = 0
    last = {}
    pair = defaultdict(int)          # (gt id, pred id) -> frames matched, for IDF1
    gt_cnt, pr_cnt = defaultdict(int), defaultdict(int)
    for gts, prs in seq:
        G = np.array([g[1:] for g in gts]) if gts else np.zeros((0, 4))
        if gated:
            prs = [p for p in prs if in_player_area(p[1:], G)]
        P = np.array([p[1:] for p in prs]) if prs else np.zeros((0, 4))
        n_gt += len(G)
        for g in gts:
            gt_cnt[g[0]] += 1
        for p in prs:
            pr_cnt[p[0]] += 1
        m = []
        if len(G) and len(P):
            M = iou(G, P)
            r, c = linear_sum_assignment(-M)
            m = [(i, j) for i, j in zip(r, c) if M[i, j] >= thr]
        tp += len(m)
        fn += len(G) - len(m)
        fp += len(P) - len(m)
        for i, j in m:
            gid, pid = gts[i][0], prs[j][0]
            if gid in last and last[gid] != pid:
                sw += 1
            last[gid] = pid
            pair[(gid, pid)] += 1
    # IDF1: best one-to-one assignment of gt ids to predicted ids over the whole clip
    gids, pids = sorted(gt_cnt), sorted(pr_cnt)
    idtp = 0
    if gids and pids:
        W = np.array([[pair.get((g, p), 0) for p in pids] for g in gids])
        r, c = linear_sum_assignment(-W)
        idtp = int(W[r, c].sum())
    idf1 = 2 * idtp / (sum(gt_cnt.values()) + sum(pr_cnt.values()))
    return {"gt_boxes": n_gt, "matched": tp, "misses": fn, "false_boxes": fp, "id_switches": sw,
            "mota": 1 - (fn + fp + sw) / n_gt, "idf1": idf1,
            "precision": tp / max(tp + fp, 1), "recall": tp / max(n_gt, 1),
            "gt_players": len(gids), "track_ids_used": len(pids)}


res = {"clips": {}}
seqs = []
for v, c in CLIPS:
    pose = YOLO(str(ROOT / "OUTPUTS" / "yolo11m-pose.pt"))      # fresh tracker per clip
    gt = load_gt(v, c)
    seq = []
    for f in sorted(FR.glob(f"{v}/{c}/*.jpg"), key=lambda p: int(p.stem)):
        fr = int(f.stem)
        r = pose.track(cv2.imread(str(f)), persist=True, tracker="botsort.yaml", imgsz=1280,
                       conf=0.25, classes=[0], verbose=False)[0]
        prs = []
        if r.boxes is not None and r.boxes.id is not None:
            for pid, b in zip(r.boxes.id.cpu().numpy().astype(int), r.boxes.xyxy.cpu().numpy()):
                prs.append((int(pid), *map(float, b)))
        if fr in gt:
            seq.append((gt[fr], prs))
    seqs.append(seq)
    res["clips"][f"{v}/{c}"] = {"frames_scored": len(seq)}
    for thr in (0.5, 0.3):
        for gated in (False, True):
            res["clips"][f"{v}/{c}"][f"iou{thr}_{'gated' if gated else 'raw'}"] = evaluate(seq, gated, thr)

for thr in (0.5, 0.3):
    for gated in (False, True):
        agg = {m: 0.0 for m in ("gt_boxes", "matched", "misses", "false_boxes", "id_switches", "idf1_w")}
        for s in seqs:
            e = evaluate(s, gated, thr)
            for m in ("gt_boxes", "matched", "misses", "false_boxes", "id_switches"):
                agg[m] += e[m]
            agg["idf1_w"] += e["idf1"] * e["gt_boxes"]
        n = agg["gt_boxes"]
        res[f"overall_iou{thr}_{'gated' if gated else 'raw'}"] = {
            "gt_boxes": int(n), "id_switches": int(agg["id_switches"]),
            "mota": 1 - (agg["misses"] + agg["false_boxes"] + agg["id_switches"]) / n,
            "idf1": agg["idf1_w"] / n, "recall": agg["matched"] / n,
            "precision": agg["matched"] / max(agg["matched"] + agg["false_boxes"], 1)}
OUT.write_text(json.dumps(res, indent=1))
print(json.dumps({k: res[k] for k in res if k.startswith("overall")}, indent=1))
