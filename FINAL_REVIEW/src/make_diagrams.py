"""Draw the two report diagrams:
  figures/web_flow_loop.png     (report Figure 1: end-to-end system architecture, user/server loop)
  figures/app_architecture.png  (7-step pipeline + outputs; used in the presentation)
Figure 2 (figures/analysis_pipeline.png) is a supplied image, not drawn here.
Run: python src/make_diagrams.py
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FIG = Path(__file__).resolve().parent.parent / "figures"

STYLE = {
    "blue":  dict(edge="#3b6fc4", face="#eaf1fb", title="#2457a8"),
    "red":   dict(edge="#c0395a", face="#f9e9ee", title="#a3203f"),
    "green": dict(edge="#3f9a4a", face="#eaf5ea", title="#2b7a34"),
}
TXT = "#222222"
ARROW = "#444444"
FONT = "DejaVu Sans"


def box(ax, x, y, w, h, title, body, color, tsize=13, bsize=10.5):
    s = STYLE[color]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.12",
                                lw=1.6, ec=s["edge"], fc=s["face"]))
    ax.text(x + w / 2, y + h - 0.28, title, ha="center", va="top", fontsize=tsize,
            fontweight="bold", color=s["title"], family=FONT)
    ax.text(x + w / 2, y + h - 0.72, body, ha="center", va="top", fontsize=bsize,
            color=TXT, family=FONT, linespacing=1.35)


def arrow(ax, p, q, color=ARROW, lw=1.8, style="-|>", ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=18, lw=lw,
                                 color=color, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


def label(ax, x, y, text, size=9.5, color="#333333", ha="center", va="center", bold=False):
    ax.text(x, y, text, ha=ha, va=va, fontsize=size, color=color, family=FONT,
            fontweight="bold" if bold else "normal",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none"))


# --------------------------------------------------------------------------- #
# Figure 2: analysis pipeline
# --------------------------------------------------------------------------- #
def pipeline():
    W, H = 16.0, 5.6
    fig, ax = plt.subplots(figsize=(W, H), dpi=150)
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

    steps = [
        ("1. Video", "match video,\none camera", "blue"),
        ("2. Check", "court found?\npeople on\ncourt?", "red"),
        ("3. Detect", "players\n(YOLO11)\nball (fine-tuned\nYOLO)", "red"),
        ("4. Track", "same ID per\nplayer (BoT-\nSORT) + court\nmap", "red"),
        ("5. Teams", "shirt colour +\ncourt side", "red"),
        ("6. Features", "18 team\nmeasures per\nframe, 1-s\nwindows", "red"),
        ("7. Tactic", "HGB model\n→ 1 of 4\ntactics", "green"),
    ]
    bw, bh, gap, x0, ytop = 1.75, 2.15, 0.52, 0.45, 3.1
    xs = [x0 + i * (bw + gap) for i in range(len(steps))]
    for x, (t, b, c) in zip(xs, steps):
        box(ax, x, ytop, bw, bh, t, b, c)
    for i in range(len(xs) - 1):
        arrow(ax, (xs[i] + bw + 0.06, ytop + bh / 2), (xs[i + 1] - 0.06, ytop + bh / 2))

    # Output row, laid out right-to-left so the flow is
    # Tactic -> Tactic clips -> Reports -> Annotated video
    ow, oh, oy = 4.55, 1.55, 0.35
    outs = [
        ("Annotated video", "player links, ball, current tactic, opponent\nSERVE / SPIKE / BLOCK"),
        ("Reports", "tactic timeline + opponent actions\n(TXT, CSV, JSON)"),
        ("Tactic clips", "best examples of each tactic from the\nuser's own video"),
    ]
    right_edge = xs[-1] + bw
    ogap = (right_edge - x0 - 3 * ow) / 2
    oxs = [x0 + i * (ow + ogap) for i in range(3)]
    for x, (t, b) in zip(oxs, outs):
        box(ax, x, oy, ow, oh, t, b, "green", tsize=12.5, bsize=10.5)

    # Tactic (7) down to Tactic clips
    tx = xs[-1] + bw / 2
    arrow(ax, (tx, ytop - 0.06), (tx, oy + oh + 0.06))
    # Tactic clips -> Reports -> Annotated video
    my = oy + oh / 2
    arrow(ax, (oxs[2] - 0.06, my), (oxs[1] + ow + 0.06, my))
    arrow(ax, (oxs[1] - 0.06, my), (oxs[0] + ow + 0.06, my))

    fig.savefig(FIG / "app_architecture.png", dpi=150, bbox_inches="tight",
                pad_inches=0.15, facecolor="white")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Figure 1: end-to-end architecture, one-way numbered loop
# --------------------------------------------------------------------------- #
def architecture():
    W, H = 16.0, 7.4
    fig, ax = plt.subplots(figsize=(W, H), dpi=150)
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

    bw, bh = 4.1, 1.85
    cx = [0.6, 5.95, 11.3]           # column x positions
    yt, yb = 4.55, 0.75              # top / bottom row y

    # lane bands
    for y0, h, name in [(yt - 0.35, bh + 1.05, "USER SIDE  (browser)"),
                        (yb - 0.35, bh + 1.05, "SERVER SIDE")]:
        ax.add_patch(FancyBboxPatch((0.15, y0), W - 0.3, h,
                                    boxstyle="round,pad=0,rounding_size=0.1",
                                    lw=1.0, ec="#cfcfcf", fc="#fafafa", ls="--"))
        ax.text(0.35, y0 + h - 0.12, name, ha="left", va="top", fontsize=10.5,
                fontweight="bold", color="#666666", family=FONT)

    # top row: user side, left to right
    box(ax, cx[0], yt, bw, bh, "1. Coach / Analyst",
        "opens the site and uploads\na match video", "blue")
    box(ax, cx[1], yt, bw, bh, "2. Web App (React)",
        "Upload page sends video in chunks\nProgress bar · Results page\n(video player, tactic timeline, reports)", "blue")
    box(ax, cx[2], yt, bw, bh, "3. API Server (FastAPI)",
        "receives chunks · file & court check\nadds job to queue\nserves results for download", "red")

    # bottom row: server side, right to left
    box(ax, cx[2], yb, bw, bh, "4. Storage",
        "Disk: uploaded video\nMongoDB: job record = queued\n(JSON file store if MongoDB is off)", "green")
    box(ax, cx[1], yb, bw, bh, "5. Worker",
        "takes next queued job, one at a time\nruns it in its own process\nwrites progress.json", "red")
    box(ax, cx[0], yb, bw, bh, "6. Analysis Engine (Python)",
        "detect & track players + ball\n(YOLO11, GPU) · team features\nHGB tactic model → annotated\nvideo, reports, clips", "red")

    ym_t, ym_b = yt + bh / 2, yb + bh / 2
    # 1 -> 2 -> 3
    arrow(ax, (cx[0] + bw + 0.06, ym_t), (cx[1] - 0.06, ym_t))
    arrow(ax, (cx[1] + bw + 0.06, ym_t), (cx[2] - 0.06, ym_t))
    label(ax, (cx[0] + bw + cx[1]) / 2, ym_t + 0.32, "upload", size=9)
    label(ax, (cx[1] + bw + cx[2]) / 2, ym_t + 0.32, "chunks", size=9)
    # 3 down to 4
    ax_x = cx[2] + bw / 2
    arrow(ax, (ax_x, yt - 0.06), (ax_x, yb + bh + 0.06))
    label(ax, ax_x + 0.12, (yt + yb + bh) / 2, "save video + job", size=9, ha="left")
    # 4 -> 5 -> 6 (right to left)
    arrow(ax, (cx[2] - 0.06, ym_b), (cx[1] + bw + 0.06, ym_b))
    arrow(ax, (cx[1] - 0.06, ym_b), (cx[0] + bw + 0.06, ym_b))
    label(ax, (cx[1] + bw + cx[2]) / 2, ym_b + 0.32, "next job", size=9)
    label(ax, (cx[0] + bw + cx[1]) / 2, ym_b + 0.32, "run", size=9)
    # 6 back up to 1: results returned to the user
    g = STYLE["green"]["edge"]
    ux = cx[0] + bw / 2
    arrow(ax, (ux, yb + bh + 0.06), (ux, yt - 0.06), color=g, lw=2.2)
    label(ax, ux + 0.12, (yt + yb + bh) / 2,
          "7. Results saved to disk, served by API,\nshown on Results page",
          size=9, color=g, ha="left", bold=True)

    fig.savefig(FIG / "web_flow_loop.png", dpi=150, bbox_inches="tight",
                pad_inches=0.15, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    pipeline()
    architecture()
    print("wrote", FIG / "app_architecture.png", FIG / "web_flow_loop.png")
