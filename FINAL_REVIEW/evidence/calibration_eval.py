"""Court-calibration accuracy (added 2026-11-15).

The court calibration is a homography H (3x3) that maps a court point in metres
(X across 0..9, Y along 0..18) to a pixel (u, v):
    [u*w, v*w, w]^T = H [X, Y, 1]^T
H has 8 unknowns, so 4 point pairs fix it; we use every detected court line
(2 end points each, 3-5 lines = 6-10 points) and least squares (cv2.findHomography).

Three checks, all on the 11 base frames in BASEFRAMES/corner_court_detection:
  1. fit error     - pixel distance between detected line ends and the fitted model
  2. leave-one-line-out - fit H without one painted line, predict where that line
     should be, measure the miss in pixels AND in centimetres on the court
  3. hand-clicked check - for videoplayback (1) and (7) a person clicked court-line
     crossings by hand (OUTPUTS/bicausal_render.py COURTS); the automatic H maps
     those pixels to metres and we compare with the true court position.

Run:  python calibration_eval.py      Out: calibration_eval.json
"""
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(r"C:\Users\yashwanth\OneDrive\Desktop\GROUP_56\finalyearproject")
LAB = ROOT / "BASEFRAMES" / "corner_court_detection" / "labels"
OUT = Path(__file__).resolve().parent / "calibration_eval.json"

HAND = {  # copied from OUTPUTS/bicausal_render.py (hand-measured, independent of the detector)
    "videoplayback (1)": ([[385, 275], [895, 275], [268, 392], [1012, 392], [200, 460], [1080, 460]],
                          [[0, 0], [9, 0], [0, 9], [9, 9], [0, 12], [9, 12]]),
    "videoplayback (7)": ([[290, 410], [1015, 410], [234, 480], [1070, 480], [200, 522], [1103, 522],
                           [160, 572], [1142, 572], [55, 703], [1245, 703]],
                          [[0, 0], [9, 0], [0, 6], [9, 6], [0, 9], [9, 9], [0, 12], [9, 12], [0, 18], [9, 18]]),
}


def proj(H, pts):
    p = cv2.perspectiveTransform(np.float32(pts).reshape(-1, 1, 2), H)
    return p.reshape(-1, 2)


def line_pts(lines):
    img, crt = [], []
    for ln in lines:
        img += [ln["left_px"], ln["right_px"]]
        crt += [[0, ln["court_y_m"]], [9, ln["court_y_m"]]]
    return np.float32(img), np.float32(crt)


res = {"per_video": {}}
loo_px, loo_cm = [], []
for jf in sorted(LAB.glob("*.json")):
    d = json.loads(jf.read_text())
    name = Path(d["image"]).stem
    lines = d["lines_used"]
    img, crt = line_pts(lines)
    H, _ = cv2.findHomography(crt, img)              # metres -> pixels
    fit = np.linalg.norm(proj(H, crt) - img, axis=1)
    r = {"lines_used": len(lines), "fit_rmse_px": float(np.sqrt(np.mean(fit ** 2))),
         "image_width": d["width"]}
    lo = []
    if len(lines) >= 4:                               # need >= 3 lines left to refit
        for k in range(len(lines)):
            rest = [ln for i, ln in enumerate(lines) if i != k]
            i2, c2 = line_pts(rest)
            Hk, _ = cv2.findHomography(c2, i2)
            ik, ck = line_pts([lines[k]])
            px = np.linalg.norm(proj(Hk, ck) - ik, axis=1)
            cm = np.linalg.norm(proj(np.linalg.inv(Hk), ik) - ck, axis=1) * 100
            lo.append({"line": lines[k]["name"], "miss_px": float(px.mean()), "miss_cm": float(cm.mean())})
            loo_px += list(px)
            loo_cm += list(cm)
    r["leave_one_line_out"] = lo
    if name in HAND:
        hp, hc = HAND[name]
        m = proj(np.linalg.inv(H), hp)
        err = np.linalg.norm(m - np.float32(hc), axis=1) * 100
        r["hand_clicked_check_cm"] = {"points": len(hp), "mean": float(err.mean()), "max": float(err.max())}
    res["per_video"][name] = r

fits = [v["fit_rmse_px"] for v in res["per_video"].values()]
hand = [v["hand_clicked_check_cm"]["mean"] for v in res["per_video"].values() if "hand_clicked_check_cm" in v]
res["summary"] = {
    "videos": len(fits), "videos_calibrated": len(fits),
    "fit_rmse_px_mean": float(np.mean(fits)), "fit_rmse_px_max": float(np.max(fits)),
    "leave_one_line_out_points": len(loo_px),
    "leave_one_line_out_px_median": float(np.median(loo_px)), "leave_one_line_out_cm_median": float(np.median(loo_cm)),
    "leave_one_line_out_cm_mean": float(np.mean(loo_cm)), "leave_one_line_out_cm_p90": float(np.percentile(loo_cm, 90)),
    "hand_clicked_mean_cm": hand,
}
OUT.write_text(json.dumps(res, indent=1))
print(json.dumps(res["summary"], indent=1))
for k, v in res["per_video"].items():
    print(k, round(v["fit_rmse_px"], 2), [round(x["miss_cm"], 1) for x in v["leave_one_line_out"]], v.get("hand_clicked_check_cm"))
