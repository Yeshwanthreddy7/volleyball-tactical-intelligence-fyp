"""Generate the system architecture and model overview diagrams as SVG.

Run:  python docs/assets/diagrams/make_diagrams.py
Then render PNGs with render_png.ps1 (headless Chrome/Edge).

Every label in these figures is taken from the repository:
  docs/ARCHITECTURE.md, webapp/backend/app/*.py, OUTPUTS/bicausal_extract.py,
  NEW_OUTPUTS/code/new_render.py, fyp/features.py, fyp/ml_classifier.py,
  tactical_ml_v6.joblib (bundle metadata), ml_results_v6/ml_results.json,
  deep_results_v6/deep_results.json.
"""
from pathlib import Path
from html import escape

OUT = Path(__file__).resolve().parent

FONT = "Arial, Helvetica, sans-serif"
INK = "#1f2933"
MUTED = "#52606d"
LINE = "#3e4c59"
ACCENT = "#7a1f2b"        # maroon, matches the panel deck
ACCENT_FILL = "#f6eaec"
BOX = "#ffffff"
LANE = {
    "client": "#f4f6f8",
    "app": "#eef3f8",
    "engine": "#f3f7f1",
    "train": "#f8f5ee",
}


class Svg:
    def __init__(self, w, h):
        self.w, self.h = w, h
        self.parts = []

    def add(self, s):
        self.parts.append(s)

    # -- primitives -------------------------------------------------------
    def rect(self, x, y, w, h, fill=BOX, stroke=LINE, sw=1.2, rx=4, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
                 f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')

    def text(self, x, y, s, size=13, weight="normal", fill=INK, anchor="middle",
             style="normal", family=FONT):
        self.add(f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" '
                 f'font-weight="{weight}" font-style="{style}" fill="{fill}" '
                 f'text-anchor="{anchor}">{escape(s)}</text>')

    def lines(self, x, y, rows, size=12, gap=None, **kw):
        gap = gap or size + 4
        for i, r in enumerate(rows):
            if isinstance(r, tuple):
                s, extra = r
                self.text(x, y + i * gap, s, size=size, **{**kw, **extra})
            else:
                self.text(x, y + i * gap, r, size=size, **kw)

    def box(self, x, y, w, h, title, rows=(), fill=BOX, stroke=LINE, sw=1.2,
            title_size=13.5, row_size=11.5, dash=None, title_fill=INK, mono_rows=False):
        self.rect(x, y, w, h, fill=fill, stroke=stroke, sw=sw, dash=dash)
        n = 1 + len(rows)
        block = title_size + (len(rows) * (row_size + 4)) + (4 if rows else 0)
        ty = y + (h - block) / 2 + title_size - 2
        self.text(x + w / 2, ty, title, size=title_size, weight="bold", fill=title_fill)
        fam = "Consolas, 'Courier New', monospace" if mono_rows else FONT
        for i, r in enumerate(rows):
            self.text(x + w / 2, ty + 6 + (i + 1) * (row_size + 4), r, size=row_size,
                      fill=MUTED, family=fam)

    def cylinder(self, x, y, w, h, title, rows=()):
        e = 7
        self.add(f'<path d="M{x},{y+e} L{x},{y+h-e} A{w/2},{e} 0 0 0 {x+w},{y+h-e} '
                 f'L{x+w},{y+e}" fill="{BOX}" stroke="{LINE}" stroke-width="1.2"/>')
        self.add(f'<ellipse cx="{x+w/2}" cy="{y+e}" rx="{w/2}" ry="{e}" fill="{BOX}" '
                 f'stroke="{LINE}" stroke-width="1.2"/>')
        ty = y + e + (h - e - 13 - len(rows) * 15) / 2 + 11
        self.text(x + w / 2, ty, title, size=13, weight="bold")
        for i, r in enumerate(rows):
            self.text(x + w / 2, ty + 4 + (i + 1) * 15, r, size=11.5, fill=MUTED)

    def lane(self, x, y, w, h, label, fill, sub=None):
        self.rect(x, y, w, h, fill=fill, stroke="#9aa5b1", sw=1, rx=6, dash="5 4")
        self.text(x + 14, y + 22, label, size=14, weight="bold", anchor="start", fill=INK)
        if sub:
            self.text(x + 14 + 9.2 * len(label) + 24, y + 22, sub, size=11.5, anchor="start", fill=MUTED,
                      family="Consolas, 'Courier New', monospace")

    def arrow(self, pts, color=LINE, sw=1.4, dash=None, label=None, lpos=None,
              lsize=11, head=True):
        d = "M" + " L".join(f"{x},{y}" for x, y in pts)
        da = f' stroke-dasharray="{dash}"' if dash else ""
        mk = ' marker-end="url(#ah)"' if head and color == LINE else (
            ' marker-end="url(#ahA)"' if head else "")
        self.add(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{sw}"{da}{mk}/>')
        if label:
            lx, ly = lpos
            self.text(lx, ly, label, size=lsize, fill=MUTED, style="italic")

    def save(self, name):
        head = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" '
                f'viewBox="0 0 {self.w} {self.h}">\n'
                '<defs>'
                f'<marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" '
                f'markerHeight="8" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{LINE}"/></marker>'
                f'<marker id="ahA" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" '
                f'markerHeight="8" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{ACCENT}"/></marker>'
                '</defs>\n'
                f'<rect width="{self.w}" height="{self.h}" fill="#ffffff"/>\n')
        (OUT / name).write_text(head + "\n".join(self.parts) + "\n</svg>\n", encoding="utf-8")


# ===========================================================================
# Figure 1 - System architecture
# ===========================================================================
def architecture():
    s = Svg(1800, 1240)
    s.text(40, 42, "System Architecture — Volleyball Tactical Analysis", size=22,
           weight="bold", anchor="start")
    s.text(40, 66, "Client, application, analysis engine and offline training layers. "
           "Solid arrows: request / data flow. Dashed arrows: model artefacts and offline steps.",
           size=13, anchor="start", fill=MUTED)

    # ---------------- Lane 1: clients -----------------------------------
    s.lane(30, 90, 1740, 140, "1  Client tier", LANE["client"])
    s.box(250, 115, 300, 90, "Web browser / PWA",
          ["React 18 + Vite single-page app", "upload · progress · results · history"])
    s.box(600, 115, 300, 90, "Desktop application",
          ["pywebview window (desktop_app.py)", "Windows installer (Inno Setup)"])
    s.box(950, 115, 340, 90, "Static project portal",
          ["Vercel · volleyballanalyzer.dpdns.org", "docs, metrics, demos (export_static.py)"])
    s.box(1340, 115, 300, 90, "Interactive API docs",
          ["FastAPI OpenAPI at /docs", "for testing endpoints"], dash="4 3")

    # ---------------- Lane 2: application ------------------------------
    s.lane(30, 260, 1740, 300, "2  Application tier", LANE["app"],
           "webapp/backend/app  —  Docker Compose on Azure VM (CPU) or local GPU laptop")
    s.box(70, 320, 220, 90, "Caddy reverse proxy", ["automatic HTTPS (TLS)", "port 443 → app"])
    s.box(350, 305, 330, 120, "FastAPI REST API", [
        "main.py", "POST /api/uploads (chunked)", "POST /api/analyses · GET /api/analyses/{id}",
        "GET /api/analyses/{id}/files, /clips"], row_size=11)
    s.box(740, 305, 290, 120, "Upload validation", [
        "validate.py", "magic bytes: MP4/MOV/AVI/MKV/WebM", "8 sampled frames: court fit +",
        "≥ 4 YOLO11n-pose people (3 of 8)"], row_size=11)
    s.cylinder(1090, 312, 220, 108, "MongoDB 7", ["db.py · analysis records", "JSON-file fallback"])
    s.box(1370, 305, 360, 120, "Job queue + worker", [
        "worker.py", "one job at a time, survives restart", "retention clean-up of old jobs"])

    s.box(350, 455, 330, 85, "Reports & clips", [
        "reports.py: tactical + opponent report", "TXT / CSV / JSON · clips.py per tactic"])
    s.cylinder(740, 452, 290, 92, "File storage", ["MP4 renditions ultra/720/480/360p", "reports, clips, court.png"])
    s.box(1370, 455, 360, 85, "Analysis engine", [
        "engine.py — one subprocess per job", "calibrate → extract → render → encode"],
        stroke=ACCENT, sw=1.8, title_fill=ACCENT)
    s.box(1090, 455, 220, 85, "ffmpeg encoder", ["H.264 + original AAC", "4 renditions + poster"])

    # client -> caddy -> api
    s.arrow([(400, 205), (400, 240), (180, 240), (180, 318)])
    s.arrow([(750, 205), (750, 240), (400, 240)], head=False)
    s.arrow([(1120, 205), (1120, 240), (750, 240)], head=False)
    s.text(300, 254, "HTTPS", size=11, fill=MUTED, style="italic")
    s.arrow([(290, 365), (348, 365)])
    s.arrow([(680, 365), (738, 365)], label="validate", lpos=(709, 357))
    s.arrow([(1030, 365), (1088, 365)])
    s.arrow([(1310, 365), (1368, 365)], label="enqueue", lpos=(1339, 357))
    s.arrow([(1550, 425), (1550, 453)])
    s.arrow([(1370, 497), (1312, 497)])
    s.arrow([(1090, 497), (1032, 497)])
    s.arrow([(1450, 540), (1450, 548), (515, 548), (515, 542)], head=True)
    s.text(980, 560, "", size=1)
    s.arrow([(680, 497), (738, 497)])
    s.arrow([(515, 455), (515, 427)], label="serve files", lpos=(560, 445))

    # ---------------- Lane 3: analysis engine -------------------------------
    s.lane(30, 590, 1740, 300, "3  Analysis engine (research code)", LANE["engine"],
           "BASEFRAMES/  ·  OUTPUTS/bicausal_extract.py  ·  NEW_OUTPUTS/code/  ·  fyp/")
    y0, h0 = 660, 150
    cols = [
        (55, 205, "Court calibration", ["base frame (median", "of dominant shot)", "detect_court_corners.py",
                                         "4 corners → homography"]),
        (285, 225, "Detection + tracking", ["YOLO11m-pose (players)", "fine-tuned YOLO (ball)",
                                             "volleyball_best.pt", "BoT-SORT · imgsz 1280"]),
        (535, 150, None, None),  # cache cylinder
        (710, 215, "Team separation", ["jersey colour (Lab a/b)", "+ median court side", "near team = focus team",
                                        "image → court metres"]),
        (950, 215, "Feature extraction", ["18 channels per frame", "geometry + kinematics",
                                           "29-frame window", "tensor (29 × 18)"]),
        (1190, 230, "Tactic classifier", ["HistGradientBoosting", "tactical_ml_v6.joblib",
                                           "132 scale-invariant", "features → 4 tactics"]),
        (1445, 300, "Renderer", ["new_render.py", "near team: relation graph + hull", "+ tactic label",
                                 "far team: SERVE / SPIKE / BLOCK"]),
    ]
    for x, w, t, rows in cols:
        if t is None:
            s.cylinder(x, y0 + 20, w, 110, "Cache", ["cache_<video>.pkl", "pose, tracks, ball"])
            continue
        hl = t == "Tactic classifier"
        s.box(x, y0, w, h0, t, rows, stroke=ACCENT if hl else LINE, sw=2 if hl else 1.2,
              fill=ACCENT_FILL if hl else BOX, title_fill=ACCENT if hl else INK)
    for (x1, w1, *_), (x2, *_r) in zip(cols, cols[1:]):
        s.arrow([(x1 + w1, y0 + h0 / 2), (x2 - 2, y0 + h0 / 2)])
    # step numbers
    for i, (x, w, *_) in enumerate(cols, 1):
        s.add(f'<circle cx="{x + 14}" cy="{y0 + 14}" r="10" fill="{INK}"/>')
        s.text(x + 14, y0 + 18.5, str(i), size=11.5, weight="bold", fill="#ffffff")
    s.text(700, 850, "Two-pass design: the GPU extraction pass (steps 1–2) runs once and is cached; "
           "steps 4–7 replay the cache, so re-rendering with another model or style is cheap.",
           size=12, fill=MUTED, style="italic")

    # engine <-> pipeline
    s.arrow([(1650, 540), (1650, 588)])
    s.text(1658, 578, "runs steps 1–7", size=11, fill=MUTED, style="italic", anchor="start")
    s.arrow([(1745, 735), (1760, 735), (1760, 497), (1732, 497)])
    s.text(1764, 690, "frames", size=11, fill=MUTED, style="italic", anchor="start")

    # ---------------- Lane 4: offline training -----------------------------
    s.lane(30, 920, 1740, 290, "4  Offline training and evaluation", LANE["train"], "fyp/")
    ty, th = 990, 120
    tcols = [
        (55, 200, "Labels", ["labels.csv", "video, anchor time,", "tactic class"]),
        (300, 220, "Clip extraction", ["extract_clips.py", "1 s clips per label", "dataset/<Class>/"]),
        (565, 240, "Training data", ["prepare_training_data.py", "29 × 18 CSV per clip",
                                      "+ extraction provenance"]),
        (850, 300, "Classical model zoo", ["ml_classifier.py · 12 models", "leave-one-video-out (LOVO)",
                                            "Platt calibration (sigmoid, 5-fold)"]),
        (1195, 250, "Deployed model", ["tactical_ml_v6.joblib", "HGB · 948 clips · 8 videos",
                                        "provenance checked at load"]),
    ]
    for x, w, t, rows in tcols:
        hl = t == "Deployed model"
        s.box(x, ty, w, th, t, rows, stroke=ACCENT if hl else LINE, sw=2 if hl else 1.2,
              fill=ACCENT_FILL if hl else BOX, title_fill=ACCENT if hl else INK)
    for (x1, w1, *_), (x2, *_r) in zip(tcols, tcols[1:]):
        s.arrow([(x1 + w1, ty + th / 2), (x2 - 2, ty + th / 2)])
    s.box(850, 1130, 300, 64, "Deep comparison arms",
          ["train_deep_lovo.py: InceptionTime, TCN, BiLSTM, Mamba"], dash="5 4", row_size=10.5)
    s.arrow([(800, 1110), (800, 1162), (848, 1162)], dash="5 4")
    s.box(1500, 990, 245, 120, "Evaluation outputs", ["ml_results_v6.json", "deep_results_v6.json",
                                                      "per-class LOVO accuracy"], dash="5 4")
    s.arrow([(1445, 1050), (1498, 1050)], dash="5 4")
    # deployed model -> classifier (up)
    s.arrow([(1320, 990), (1320, 812)], color=ACCENT, sw=1.8, dash="6 4")
    s.text(1328, 905, "load model", size=11.5, fill=ACCENT, style="italic", anchor="start")
    s.save("architecture_diagram.svg")


# ===========================================================================
# Figure 2 - Model overview
# ===========================================================================
def model_overview():
    s = Svg(1800, 1000)
    s.text(40, 42, "Model Overview — Tactical Coordination Classifier", size=22,
           weight="bold", anchor="start")
    s.text(40, 66, "From tracked players to one of four team tactics. Deployed path: "
           "HistGradientBoosting (solid, maroon). Comparison path: deep sequence models (dashed).",
           size=13, anchor="start", fill=MUTED)

    # ---- Stage A: input --------------------------------------------------
    s.text(150, 112, "A. Per-frame input", size=14, weight="bold")
    s.rect(40, 125, 220, 300, fill=LANE["engine"], stroke="#9aa5b1", sw=1, rx=6)
    # mini court
    cx, cy, cw, ch = 70, 150, 160, 170
    s.rect(cx, cy, cw, ch, fill="#ffffff", stroke=LINE, sw=1.2, rx=0)
    s.add(f'<line x1="{cx}" y1="{cy+ch/2}" x2="{cx+cw}" y2="{cy+ch/2}" stroke="{LINE}" stroke-width="2"/>')
    s.add(f'<line x1="{cx}" y1="{cy+ch/2+28}" x2="{cx+cw}" y2="{cy+ch/2+28}" stroke="#9aa5b1" stroke-width="1"/>')
    s.add(f'<line x1="{cx}" y1="{cy+ch/2-28}" x2="{cx+cw}" y2="{cy+ch/2-28}" stroke="#9aa5b1" stroke-width="1"/>')
    near = [(95, 262), (150, 255), (205, 265), (100, 300), (155, 305), (205, 298)]
    for a in range(len(near)):
        for b in range(a + 1, len(near)):
            pass
    hull = [(95, 262), (150, 255), (205, 265), (205, 298), (155, 305), (100, 300)]
    s.add('<polygon points="' + " ".join(f"{x},{y}" for x, y in hull) +
          f'" fill="{ACCENT_FILL}" stroke="{ACCENT}" stroke-width="1" stroke-dasharray="3 2"/>')
    for x, y in near:
        s.add(f'<circle cx="{x}" cy="{y}" r="5" fill="{ACCENT}"/>')
    for x, y in [(95, 175), (150, 170), (205, 178), (100, 210), (150, 215), (200, 208)]:
        s.add(f'<circle cx="{x}" cy="{y}" r="4" fill="none" stroke="{MUTED}" stroke-width="1.2"/>')
    s.add(f'<circle cx="168" cy="242" r="3.5" fill="#d9a400" stroke="{INK}" stroke-width="0.8"/>')
    s.lines(150, 345, ["≤ 6 tracked players of the", "near (focus) team + ball,",
                       "in court coordinates", "(homography from corners)"], size=11.5, fill=MUTED)

    # ---- Stage B: 18 channels -----------------------------------------
    s.text(450, 112, "B. 18 channels per frame", size=14, weight="bold")
    s.rect(300, 125, 300, 300, fill="#ffffff", stroke=LINE, sw=1.2)
    s.text(316, 148, "Permutation-invariant geometry (14)", size=12, weight="bold", anchor="start")
    geo = ["ball_present, ball_x, ball_y", "centroid_x, centroid_y", "spread_x, spread_y, hull_area",
           "nn_dist_mean, nn_dist_min, nn_dist_max", "max_pair_dist, n_present", "ball_to_centroid"]
    s.lines(316, 170, geo, size=11.5, gap=18, anchor="start", fill=MUTED,
            family="Consolas, 'Courier New', monospace")
    s.add(f'<line x1="316" y1="285" x2="584" y2="285" stroke="#cbd2d9"/>')
    s.text(316, 306, "Kinematic (4)", size=12, weight="bold", anchor="start")
    s.lines(316, 328, ["speed_top2, speed_bot4", "speed_diff, sync_inst"], size=11.5, gap=18,
            anchor="start", fill=MUTED, family="Consolas, 'Courier New', monospace")
    s.lines(316, 380, ["Order-independent: a formation is a set,", "not an ordered list of players."],
            size=11, gap=15, anchor="start", fill=MUTED, style="italic")
    s.text(450, 440, "fyp/features.py", size=11, fill=MUTED, family="Consolas, monospace")

    # ---- Stage C: window tensor -----------------------------------------
    s.text(745, 112, "C. Window tensor", size=14, weight="bold")
    gx, gy, cwid, chei = 650, 140, 190, 230
    rows_, cols_ = 29, 18
    s.rect(gx, gy, cwid, chei, fill="#ffffff", stroke=LINE, sw=1.2, rx=0)
    for i in range(1, cols_):
        x = gx + cwid * i / cols_
        s.add(f'<line x1="{x:.1f}" y1="{gy}" x2="{x:.1f}" y2="{gy+chei}" stroke="#d5dbe1" stroke-width="0.6"/>')
    for j in range(1, rows_):
        y = gy + chei * j / rows_
        s.add(f'<line x1="{gx}" y1="{y:.1f}" x2="{gx+cwid}" y2="{y:.1f}" stroke="#d5dbe1" stroke-width="0.6"/>')
    s.text(gx + cwid / 2, gy + chei + 20, "18 channels →", size=11.5, fill=MUTED)
    s.add(f'<text x="{gx-10}" y="{gy+chei/2}" font-family="{FONT}" font-size="11.5" fill="{MUTED}" '
          f'text-anchor="middle" transform="rotate(-90 {gx-10} {gy+chei/2})">29 frames (≈ 1 s) →</text>')
    s.text(gx + cwid / 2, gy + chei + 42, "float32 (29, 18)", size=12, weight="bold",
           family="Consolas, monospace")
    s.text(gx + cwid / 2, gy + chei + 60, "gaps interpolated, NaN-free", size=11, fill=MUTED)

    s.arrow([(260, 275), (298, 275)])
    s.arrow([(600, 360), (648, 360)])

    # ---- Branch split ----------------------------------------------------
    bx = 870
    s.arrow([(840, 255), (bx, 255), (bx, 205), (898, 205)], color=ACCENT, sw=1.8)
    s.arrow([(bx, 255), (bx, 585), (898, 585)], dash="5 4")

    # ---- Deployed path ---------------------------------------------------
    s.text(1300, 112, "D. Deployed path — scale-invariant aggregation + gradient boosting",
           size=14, weight="bold", fill=ACCENT)
    s.rect(900, 125, 470, 300, fill="#ffffff", stroke=ACCENT, sw=1.6)
    s.text(1135, 148, "132 window features (ml_classifier.py, feature_mode = invariant)",
           size=12, weight="bold")
    blocks = [
        ("7 ratios × 6 stats", "42", ["compactness, tightest_pair, loosest_pair,", "spread_aspect, ball_offset,",
                                            "speed_ratio, speed_diff_share"]),
        ("3 dimensionless × 6 stats", "18", ["n_present, sync_inst, ball_present"]),
        ("18 z-scored channels × 4", "72", ["temporal shape per channel:", "min, max, slope, IQR"]),
    ]
    yy = 165
    for title, n, rows in blocks:
        h = 30 + 15 * len(rows)
        s.rect(915, yy, 380, h, fill=ACCENT_FILL, stroke="#c9a3a9", sw=1)
        s.text(927, yy + 19, title, size=12, weight="bold", anchor="start")
        s.lines(927, yy + 36, rows, size=11, gap=15, anchor="start", fill=MUTED,
                family="Consolas, 'Courier New', monospace")
        s.rect(1305, yy, 52, h, fill="#ffffff", stroke="#c9a3a9", sw=1)
        s.text(1331, yy + h / 2 + 6, n, size=16, weight="bold", fill=ACCENT)
        yy += h + 8
    s.text(1135, yy + 14, "6 stats = mean, std, min, max, slope, IQR   │   42 + 18 + 72 = 132",
           size=11.5, fill=INK)
    s.text(1135, yy + 32, "Ratios cancel per-camera scale, so distances do not leak camera identity.",
           size=11, fill=MUTED, style="italic")

    # HGB + calibration + prior
    s.box(1410, 125, 350, 105, "HistGradientBoostingClassifier",
          ["max_iter = 300, learning_rate = 0.06", "median imputation, no feature scaling",
           "winner of the 12-model LOVO zoo"], stroke=ACCENT, sw=1.8, fill=ACCENT_FILL, title_fill=ACCENT,
          row_size=11)
    s.box(1410, 250, 350, 70, "Probability calibration",
          ["Platt sigmoid, 5-fold CalibratedClassifierCV"], row_size=11)
    s.box(1410, 340, 350, 85, "Prior correction (renderer)",
          ["divide by training class share", "370 / 355 / 108 / 115 clips"], row_size=11)
    s.arrow([(1370, 177), (1408, 177)], color=ACCENT, sw=1.8)
    s.arrow([(1585, 230), (1585, 248)], color=ACCENT, sw=1.6)
    s.arrow([(1585, 320), (1585, 338)], color=ACCENT, sw=1.6)

    # ---- Comparison path -----------------------------------------------
    s.text(1135, 470, "E. Comparison path — deep models on raw (29 × 18)",
           size=14, weight="bold")
    s.rect(900, 485, 470, 200, fill="#ffffff", stroke=LINE, sw=1.2, dash="5 4")
    dl = [("InceptionTime", "multi-scale 1-D conv"), ("TCN", "dilated causal conv"),
          ("BiLSTM + Attention", "recurrent"), ("Mamba SSM", "selective state space")]
    for i, (t_, r) in enumerate(dl):
        x = 920 + (i % 2) * 220
        y = 502 + (i // 2) * 76
        s.box(x, y, 210, 62, t_, [r], row_size=11)
    s.text(1135, 670, "train_deep_lovo.py → *.pt (TorchTemporalClassifier)",
           size=11.5, fill=MUTED, family="Consolas, monospace")

    # ---- Output -----------------------------------------------------------
    s.text(1585, 470, "F. Output", size=14, weight="bold")
    s.rect(1410, 485, 350, 200, fill="#ffffff", stroke=LINE, sw=1.4)
    s.text(1585, 508, "ClassificationResult", size=13, weight="bold", family="Consolas, monospace")
    s.text(1585, 526, "label, confidence, entropy, probs", size=11, fill=MUTED,
           family="Consolas, monospace")
    tac = ["Coordinated Attack", "Coordinated Defense", "Delayed Support", "Spacing Breakdown"]
    for i, t_ in enumerate(tac):
        y = 543 + i * 34
        s.rect(1430, y, 310, 26, fill=ACCENT_FILL if i < 2 else "#f4f6f8", stroke="#c9ced6", sw=1)
        s.text(1585, y + 18, t_, size=12.5, weight="bold")
    s.arrow([(1585, 425), (1585, 455), (1700, 455), (1700, 483)], color=ACCENT, sw=1.8)
    s.arrow([(1370, 600), (1408, 600)], dash="5 4")

    # common interface note
    s.rect(40, 485, 800, 150, fill=LANE["app"], stroke="#9aa5b1", sw=1, rx=6, dash="5 4")
    s.text(56, 509, "Shared model interface (fyp/interfaces.py)", size=13, weight="bold", anchor="start")
    s.lines(56, 533, [
        "BaseTemporalClassifier.classify(window: float32[29, 18]) → ClassificationResult",
        "create_temporal_classifier(path):   .joblib → SklearnTemporalClassifier   |   .pt → TorchTemporalClassifier",
        "Extraction provenance (detector, imgsz 1280, tracker, team split, feature_version",
        "perm_invariant_v2) is stored in every model and checked at load: a mismatch aborts the run.",
    ], size=11.5, gap=21, anchor="start", fill=INK, family="Consolas, 'Courier New', monospace")

    # ---- Results table ----------------------------------------------------
    s.text(40, 690, "G. Leave-one-video-out results (948 labelled 1 s clips, 8 source videos, 4 classes)",
           size=14, weight="bold", anchor="start")
    hdr = ["Model", "Path", "LOVO accuracy", "Macro-F1"]
    rows = [
        ("HistGradientBoosting", "deployed", "67.0 %", "0.510"),
        ("InceptionTime", "comparison", "61.9 %", "0.515"),
        ("TCN", "comparison", "51.2 %", "0.441"),
        ("BiLSTM + Attention", "comparison", "45.9 %", "0.398"),
        ("Mamba SSM", "comparison", "46.1 %", "0.390"),
        ("Majority class (baseline)", "—", "39.0 %", "0.140"),
    ]
    xs = [40, 330, 490, 650, 790]
    ty = 705
    s.rect(40, ty, 750, 30, fill="#e4e7eb", stroke=LINE, sw=1, rx=0)
    for i, h in enumerate(hdr):
        s.text(xs[i] + 12, ty + 20, h, size=12.5, weight="bold", anchor="start")
    for r, row in enumerate(rows):
        y = ty + 30 + r * 27
        hl = r == 0
        s.rect(40, y, 750, 27, fill=ACCENT_FILL if hl else ("#ffffff" if r % 2 else "#fafbfc"),
               stroke="#cbd2d9", sw=0.8, rx=0)
        for i, c in enumerate(row):
            s.text(xs[i] + 12, y + 18, c, size=12, anchor="start",
                   weight="bold" if hl else "normal", fill=ACCENT if hl and i == 0 else INK)
    s.text(40, ty + 30 + len(rows) * 27 + 22,
           "Source: ml_results_v6/ml_results.json, deep_results_v6/deep_results.json (pooled out-of-fold predictions).",
           size=11, fill=MUTED, anchor="start", style="italic")
    s.text(40, ty + 30 + len(rows) * 27 + 40,
           "Tree models are preferred on small tabular-like data (Grinsztajn et al., 2022); deep arms kept for comparison.",
           size=11, fill=MUTED, anchor="start", style="italic")

    # class split mini note
    s.rect(900, 730, 470, 200, fill="#ffffff", stroke=LINE, sw=1.2)
    s.text(1135, 753, "Training class distribution (948 clips)", size=13, weight="bold")
    counts = [("Coordinated Attack", 370), ("Coordinated Defense", 355),
              ("Spacing Breakdown", 115), ("Delayed Support", 108)]
    for i, (n, c) in enumerate(counts):
        y = 772 + i * 36
        s.text(1060, y + 17, n, size=12, anchor="end")
        w = c / 370 * 240
        s.rect(1070, y + 3, w, 20, fill=ACCENT if i < 2 else "#9aa5b1", stroke="none", rx=0)
        s.text(1070 + w + 6, y + 18, str(c), size=12, anchor="start", weight="bold")
    s.text(1135, 920, "3.4 : 1 imbalance → macro-F1 used for model selection", size=11, fill=MUTED,
           style="italic")
    s.rect(1410, 730, 350, 200, fill="#ffffff", stroke=LINE, sw=1.2)
    s.text(1585, 753, "Why HGB is the deployed model", size=13, weight="bold")
    s.lines(1428, 782, [
        "• top LOVO accuracy (67.0 %, tied",
        "   with LightGBM) in 12-model zoo",
        "• no feature scaling needed; gaps from",
        "   occluded players filled by median",
        "• fast CPU inference, small model file",
        "• trees beat deep nets on small",
        "   tabular-like data (948 clips)",
            ], size=11.5, gap=19, anchor="start", fill=INK)
    s.save("model_overview_diagram.svg")


if __name__ == "__main__":
    architecture()
    model_overview()
    print("wrote", OUT)
