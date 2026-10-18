"""
One folder per video with its full-court corners and the court boundary path.

Reads  : BASEFRAMES/corner_court_detection/labels/<video>.json
Writes : BASEFRAMES/corner_court_detection/individual/<video>/
           <video>_court_path.png   base frame + numbered corners + boundary path
                                    (canvas grows when a corner is off-frame)
           <video>_court_mask.png   white = inside the court, black = outside
           <video>_court_path.svg   path over the image, opens in any browser
           <video>_corners.txt      corners and path in plain text
         BASEFRAMES/corner_court_detection/court_paths.csv   one row per video

Path order: 1 far_left -> 2 far_right -> 3 near_right -> 4 near_left -> back to 1
(clockwise as seen in the picture).

Run (from finalyearproject/):  python BASEFRAMES/make_court_paths.py
"""
from __future__ import annotations

import base64
import csv
import json
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
DET = HERE / "corner_court_detection"
ORDER = ["far_left", "far_right", "near_right", "near_left"]
VIDEO_DIR = HERE.parent / "fyp" / "Datasets"


def svg_path(pts):
    return "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in pts) + " Z"


def main():
    out_root = DET / "individual"
    out_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for jp in sorted((DET / "labels").glob("*.json")):
        L = json.loads(jp.read_text())
        stem = jp.stem
        video = stem + ".mp4"
        img = cv2.imread(str(HERE / "base_frame" / L["image"]))
        h, w = img.shape[:2]
        pts = np.array([L["corners_px"][n] for n in ORDER], float)
        vis = [L["corners_visible"][n] for n in ORDER]
        sc = w / 1280

        # canvas large enough for every corner (+ margin) when one is off-frame
        m = int(40 * sc)
        x0 = int(min(0, pts[:, 0].min() - m)); y0 = int(min(0, pts[:, 1].min() - m))
        x1 = int(max(w, pts[:, 0].max() + m)); y1 = int(max(h, pts[:, 1].max() + m))
        canvas = np.full((y1 - y0, x1 - x0, 3), 60, np.uint8)
        canvas[-y0:-y0 + h, -x0:-x0 + w] = img
        P = pts - [x0, y0]
        if (x0, y0, x1, y1) != (0, 0, w, h):
            cv2.rectangle(canvas, (-x0, -y0), (-x0 + w - 1, -y0 + h - 1), (255, 255, 255), max(1, int(2 * sc)))

        over = canvas.copy()
        cv2.fillPoly(over, [P.round().astype(np.int32)], (0, 200, 255))
        cv2.addWeighted(over, 0.22, canvas, 0.78, 0, canvas)
        t = max(2, int(3 * sc))
        for i in range(4):
            a, b = P[i], P[(i + 1) % 4]
            cv2.arrowedLine(canvas, tuple(a.round().astype(int)), tuple(b.round().astype(int)),
                            (0, 255, 0), t, cv2.LINE_AA, tipLength=min(0.08, 30 * sc / max(np.linalg.norm(b - a), 1)))
        fs = 0.7 * sc
        for i, (n, (x, y), v) in enumerate(zip(ORDER, P, vis)):
            c = (int(round(x)), int(round(y)))
            col = (0, 0, 255) if v else (0, 140, 255)
            cv2.circle(canvas, c, int(11 * sc), (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(canvas, c, int(8 * sc), col, -1, cv2.LINE_AA)
            cv2.putText(canvas, str(i + 1), (c[0] - int(5 * sc), c[1] + int(5 * sc)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5 * sc, (255, 255, 255), max(1, int(sc)), cv2.LINE_AA)
            ox, oy = pts[i]
            label = f"{i + 1} {n} ({ox:.0f}, {oy:.0f})" + ("" if v else " off-frame")
            tx = int(np.clip(x + 16 * sc, 5, canvas.shape[1] - 330 * sc))
            ty = int(np.clip(y - 14 * sc if i < 2 else y + 34 * sc, 30 * sc, canvas.shape[0] - 10))
            for th, cc in ((int(5 * sc), (0, 0, 0)), (max(1, int(2 * sc)), col)):
                cv2.putText(canvas, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, fs, cc, th, cv2.LINE_AA)
        title = f"{video}  |  full court path 1-2-3-4  |  fit {L['fit_rmse_px']:.2f} px"
        for th, cc in ((int(6 * sc), (0, 0, 0)), (max(1, int(2 * sc)), (255, 255, 255))):
            cv2.putText(canvas, title, (int(12 * sc), int(34 * sc)), cv2.FONT_HERSHEY_SIMPLEX, 0.9 * sc, cc, th, cv2.LINE_AA)

        d = out_root / stem
        d.mkdir(exist_ok=True)
        cv2.imwrite(str(d / f"{stem}_court_path.png"), canvas)

        mask = np.zeros((h, w), np.uint8)
        cv2.fillPoly(mask, [pts.round().astype(np.int32)], 255)
        cv2.imwrite(str(d / f"{stem}_court_mask.png"), mask)

        path = svg_path(pts)
        ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        b64 = base64.b64encode(jpg.tobytes()).decode()
        vb = f"{x0} {y0} {x1 - x0} {y1 - y0}"
        circles = "".join(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{8 * sc:.0f}" fill="{"red" if v else "orange"}" stroke="black" stroke-width="2"/>'
            f'<text x="{x + 14 * sc:.1f}" y="{y - 10 * sc:.1f}" font-size="{20 * sc:.0f}" fill="yellow" '
            f'stroke="black" stroke-width="0.6">{i + 1} {n}</text>'
            for i, (n, (x, y), v) in enumerate(zip(ORDER, pts, vis)))
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{vb}" width="{x1 - x0}" height="{y1 - y0}">'
               f'<title>{video} full court path</title>'
               f'<rect x="{x0}" y="{y0}" width="{x1 - x0}" height="{y1 - y0}" fill="#3c3c3c"/>'
               f'<image href="data:image/jpeg;base64,{b64}" x="0" y="0" width="{w}" height="{h}"/>'
               f'<path d="{path}" fill="rgba(255,200,0,0.2)" stroke="lime" stroke-width="{3 * sc:.0f}"/>'
               f'{circles}</svg>')
        (d / f"{stem}_court_path.svg").write_text(svg)

        lines = [f"video        : {video}",
                 f"video file   : fyp/Datasets/{video}" + ("" if (VIDEO_DIR / video).exists() else "  (not found)"),
                 f"base frame   : BASEFRAMES/base_frame/{L['image']}  ({w}x{h})",
                 f"fit error    : {L['fit_rmse_px']:.2f} px RMSE",
                 "",
                 "corners (pixel x, y)   court position (m)   in picture"]
        for i, n in enumerate(ORDER):
            x, y = pts[i]
            cx, cy = L["corner_court_m"][n]
            lines.append(f"  {i + 1} {n:<11s} ({x:8.2f}, {y:8.2f})   ({cx:g}, {cy:g})      {'yes' if vis[i] else 'NO (off-frame)'}")
        lines += ["", "path (clockwise, closed): 1 far_left -> 2 far_right -> 3 near_right -> 4 near_left -> 1",
                  "  " + " -> ".join(f"({x:.1f}, {y:.1f})" for x, y in list(pts) + [pts[0]]),
                  "", f"SVG path: {path}"]
        (d / f"{stem}_corners.txt").write_text("\n".join(lines) + "\n")

        rows.append([video, f"fyp/Datasets/{video}", f"BASEFRAMES/base_frame/{L['image']}"] +
                    [round(v, 2) for p in pts for v in p] + [str(v).lower() for v in vis] +
                    [" -> ".join(f"({x:.1f}, {y:.1f})" for x, y in list(pts) + [pts[0]]), path,
                     f"BASEFRAMES/corner_court_detection/individual/{stem}/"])
        print(f"{video:24s} {path}")

    with open(DET / "court_paths.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["video", "video_file", "base_frame"] +
                    [f"{i + 1}_{n}_{a}" for i, n in enumerate(ORDER) for a in ("x", "y")] +
                    [f"{i + 1}_{n}_in_picture" for i, n in enumerate(ORDER)] +
                    ["path_1_2_3_4_1", "svg_path", "folder"])
        wr.writerows(rows)

    # overview: every video's path in one sheet
    tiles = []
    for r in rows:
        stem = Path(r[0]).stem
        t = cv2.imread(str(out_root / stem / f"{stem}_court_path.png"))
        hh = 360
        t = cv2.resize(t, (int(t.shape[1] * hh / t.shape[0]), hh), interpolation=cv2.INTER_AREA)
        t = cv2.copyMakeBorder(t, 0, 0, 0, max(0, 700 - t.shape[1]), cv2.BORDER_CONSTANT)[:, :700]
        tiles.append(t)
    while len(tiles) % 3:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(str(out_root / "_all_court_paths.png"),
                np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]))
    print("->", out_root)


if __name__ == "__main__":
    main()
