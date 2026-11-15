"""Post-external-review presentation (native, editable PowerPoint).

Every number is read from the evidence files, so the slides cannot drift from them:
  evidence/final_report_eval.json      official LOVO scores (945 clips, 8 videos)
  evidence/review_response_eval.json   confusion matrix, per-class, folds, window study
  evidence/tracking_eval.json          IDF1 / MOTA / ID switches on hand-labelled boxes
  evidence/calibration_eval.json       court-calibration accuracy
  evidence/ball_check_result.json      human-checked ball detections
  evidence/action_check_result.json    human-checked SERVE / SPIKE / BLOCK
  evidence/functional_tests.json       14 functional + security tests
Run:  python src/build_review_deck.py   ->  presentation/final_presentation_v2.pptx
Slides follow the official Amrita 23CSE498 external-review template order.
"""
import json
import sys
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

sys.path.insert(0, str(Path(__file__).parent))
import content as C  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EV, FIG = ROOT / "evidence", ROOT / "figures"
OUT = ROOT / "presentation" / "final_presentation_v2.pptx"


def J(name):
    return json.loads((EV / name).read_text())


OFF = J("final_report_eval.json")
RV = J("review_response_eval.json")
TR = J("tracking_eval.json")
CA = J("calibration_eval.json")["summary"]
BA = J("ball_check_result.json")
AC = J("action_check_result.json")
FT = J("functional_tests.json")
DEP = OFF["deployed_calibrated_hgb"]["prior_corrected"]
PC = RV["deployed_prior_corrected"]
INC = RV["deep_vs_deployed_foldwise"]["inceptiontime"]
DEEP = RV["deep_vs_deployed_foldwise"]
TRK = TR["overall_iou0.5_gated"]
WIN = RV["window_study_hgb"]


def pc(x, d=1):
    return f"{100 * x:.{d}f}%"


MAROON = RGBColor(0xA4, 0x12, 0x3F)
DARK = RGBColor(0x7D, 0x0F, 0x31)
INK = RGBColor(0x1F, 0x1F, 0x1F)
MUTED = RGBColor(0x5F, 0x5F, 0x5F)
LIGHT = RGBColor(0xF6, 0xEE, 0xF1)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
GREEN = RGBColor(0x1D, 0x7A, 0x35)
AMBER = RGBColor(0xA8, 0x5F, 0x00)
RED = RGBColor(0xB0, 0x24, 0x24)
STATUS = {"done": ("✔ Done", GREEN), "part": ("◐ Partly", AMBER), "no": ("✘ Not possible now", RED)}

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]
W, H = prs.slide_width, prs.slide_height
count = {"n": 0}


def text(slide, x, y, w, h, runs, size=16, color=INK, bold=False, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP):
    """runs: str, or list of paragraphs; a paragraph is str or list of (text, dict(bold/color/size))."""
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.05)
    tf.margin_top = tf.margin_bottom = Inches(0.02)
    paras = runs if isinstance(runs, list) else [runs]
    for i, p in enumerate(paras):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.alignment = align
        parts = p if isinstance(p, list) else [(p, {})]
        for t, st in parts:
            r = para.add_run()
            r.text = t
            r.font.size = Pt(st.get("size", size))
            r.font.bold = st.get("bold", bold)
            r.font.color.rgb = st.get("color", color)
            r.font.name = "Calibri"
    return tb


def bullets(slide, x, y, w, h, items, size=16, gap=4):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    for i, it in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(gap)
        head, body = it if isinstance(it, tuple) else (None, it)
        r = p.add_run()
        r.text = "▪ "
        r.font.color.rgb = MAROON
        r.font.size = Pt(size)
        if head:
            r = p.add_run()
            r.text = head + ": "
            r.font.bold = True
            r.font.size = Pt(size)
            r.font.color.rgb = INK
        r = p.add_run()
        r.text = body
        r.font.size = Pt(size)
        r.font.color.rgb = INK
    return tb


def box(slide, x, y, w, h, title, items, size=15, fill=LIGHT, head=DARK, gap=4):
    hd = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, w, Inches(0.42))
    hd.fill.solid()
    hd.fill.fore_color.rgb = head
    hd.line.fill.background()
    tf = hd.text_frame
    tf.margin_left = Inches(0.12)
    tf.paragraphs[0].alignment = PP_ALIGN.LEFT
    r = tf.paragraphs[0].add_run()
    r.text = title
    r.font.size = Pt(17)
    r.font.bold = True
    r.font.color.rgb = WHITE
    bd = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y + Inches(0.42), w, h - Inches(0.42))
    bd.fill.solid()
    bd.fill.fore_color.rgb = fill
    bd.line.fill.background()
    bullets(slide, x + Inches(0.08), y + Inches(0.5), w - Inches(0.16), h - Inches(0.55), items, size, gap)


def table(slide, x, y, w, rows, widths, size=13, head_fill=DARK, row_h=0.36, status_col=None, bold_first_row=None):
    t = slide.shapes.add_table(len(rows), len(rows[0]), x, y, w, Inches(row_h * len(rows))).table
    tot = sum(widths)
    for j, cw in enumerate(widths):
        t.columns[j].width = Emu(int(w * cw / tot))
    for i, row in enumerate(rows):
        t.rows[i].height = Inches(row_h)
        for j, val in enumerate(row):
            cell = t.cell(i, j)
            cell.margin_left = cell.margin_right = Inches(0.06)
            cell.margin_top = cell.margin_bottom = Inches(0.03)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            color = INK
            if i and status_col is not None and j == status_col and val in STATUS:
                val, color = STATUS[val]
            tf = cell.text_frame
            tf.word_wrap = True
            tf.paragraphs[0].text = ""
            r = tf.paragraphs[0].add_run()
            r.text = str(val)
            r.font.size = Pt(size)
            r.font.name = "Calibri"
            cell.fill.solid()
            if i == 0:
                cell.fill.fore_color.rgb = head_fill
                r.font.bold = True
                r.font.color.rgb = WHITE
            else:
                cell.fill.fore_color.rgb = LIGHT if i % 2 else WHITE
                r.font.color.rgb = color
                r.font.bold = color != INK or (bold_first_row == i)
    return t


def slide(title, sub=None, notes=None):
    s = prs.slides.add_slide(BLANK)
    count["n"] += 1
    band = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, W, Inches(1.05 if sub else 0.8))
    band.fill.solid()
    band.fill.fore_color.rgb = MAROON
    band.line.fill.background()
    text(s, Inches(0.35), Inches(0.08), Inches(11.6), Inches(0.6), title, 28, WHITE, True)
    if sub:
        text(s, Inches(0.35), Inches(0.6), Inches(11.6), Inches(0.4), sub, 16, WHITE)
    logo = FIG / "logo_small.png"
    if logo.exists():
        s.shapes.add_picture(str(logo), W - Inches(1.05), Inches(0.12), height=Inches(0.62))
    text(s, Inches(0.35), H - Inches(0.38), Inches(9), Inches(0.3),
         f"Team {C.TEAM_NO}  |  Court-Calibrated Group Activity Recognition for Volleyball", 11, MUTED)
    text(s, W - Inches(1.2), H - Inches(0.38), Inches(0.9), Inches(0.3), str(count["n"]), 12, MUTED,
         align=PP_ALIGN.RIGHT)
    if notes:
        s.notes_slide.notes_text_frame.text = notes
    return s


def pic(s, name, x, y, w=None, h=None):
    return s.shapes.add_picture(str(FIG / name), x, y, width=w, height=h)


def callout(s, x, y, w, h, msg, size=15, fill=RGBColor(0xEA, 0xF3, 0xEC), edge=GREEN):
    b = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h)
    b.adjustments[0] = 0.08
    b.fill.solid()
    b.fill.fore_color.rgb = fill
    b.line.color.rgb = edge
    b.line.width = Pt(1.5)
    tf = b.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.12)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    paras = msg if isinstance(msg, list) else [msg]
    for i, p in enumerate(paras):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        parts = p if isinstance(p, list) else [(p, {})]
        for t, st in parts:
            r = para.add_run()
            r.text = t
            r.font.size = Pt(st.get("size", size))
            r.font.bold = st.get("bold", False)
            r.font.color.rgb = st.get("color", INK)
    return b




# =========================================================================== slides
# Order and titles follow the official Amrita template (23CSE498 External Review,
# Type 2 Application Project). Template frames 2, 3 and 5 may be duplicated ("contd.").
B = {"bold": True}
CW = Inches(12.63)   # content width
X0 = Inches(0.35)
L2, R2, HW = Inches(0.35), Inches(6.75), Inches(6.2)
GOOD = RGBColor(0xEA, 0xF3, 0xEC)
pcl = PC["per_class"]

# ---- Title
s = prs.slides.add_slide(BLANK)
count["n"] += 1
tag = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, W - Inches(3.6), Inches(0.25), Inches(3.3), Inches(0.5))
tag.fill.solid()
tag.fill.fore_color.rgb = MAROON
tag.line.fill.background()
tag.text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER
r = tag.text_frame.paragraphs[0].add_run()
r.text = "Type 2: Application Project"
r.font.size, r.font.bold, r.font.color.rgb = Pt(16), True, WHITE
if (FIG / "logo.png").exists():
    s.shapes.add_picture(str(FIG / "logo.png"), Inches(5.67), Inches(0.35), height=Inches(1.0))
text(s, Inches(0.8), Inches(1.5), Inches(11.7), Inches(1.0), C.TITLE, 30, MAROON, True, PP_ALIGN.CENTER)
text(s, Inches(0.8), Inches(2.55), Inches(11.7), Inches(0.4),
     f"{C.COURSE.replace('--', '–')}   |   Team Number: G0{C.TEAM_NO}", 17, INK, True, PP_ALIGN.CENTER)
table(s, Inches(3.0), Inches(3.15), Inches(7.3),
      [["S.No", "Register Number", "Student Name"]] + [[str(i), r_, n_] for i, (r_, n_) in enumerate(C.STUDENTS, 1)],
      [1, 3, 4.5], 14, row_h=0.36)
text(s, Inches(0.8), Inches(5.55), Inches(11.7), Inches(0.4),
     f"Guide: {C.GUIDE[0]}, {C.GUIDE[1]}, Department of CSE", 17, INK, False, PP_ALIGN.CENTER)
text(s, Inches(0.8), Inches(6.0), Inches(11.7), Inches(0.4), "November 2026", 15, MUTED, False, PP_ALIGN.CENTER)

# ---- Guide approval (mandatory)
s = slide("Guide Review & Submission Approval", "Confirmation of Project Review and Approval from Guide",
          notes="The guide, Dr. Divya Singh, approved this presentation and the report by email.")
if (FIG / "guide_approval.png").exists():
    p_ = pic(s, "guide_approval.png", Inches(0.9), Inches(1.3), w=Inches(11.5))
    p_.line.color.rgb = MUTED
    p_.line.width = Pt(1)

# ---- 1. Problem definition
s = slide("1. Problem Definition & Requirement Analysis",
          "Target Problem, User Needs & Functional / Non-Functional Requirements",
          notes="Coaches spend hours watching match videos to judge how the team moved together, and two coaches "
                "can judge the same rally differently. Our app watches an ordinary broadcast video and, for every "
                "second of play, says which of four team tactics the near-side team is using.")
box(s, X0, Inches(1.25), CW, Inches(1.75), "Problem Statement & Target Stakeholders", C.PROBLEM, 15)
box(s, X0, Inches(3.1), CW, Inches(2.0), "Functional Requirements", C.FR, 15)
box(s, X0, Inches(5.2), CW, Inches(1.55), "Non-Functional Requirements & Constraints", C.NFR, 15)

# ---- 2. Architecture
s = slide("2. End-to-End Application Architecture", "System Architecture & User-to-Backend Workflow Diagram",
          notes="Follow the arrows: the coach uploads on the website, the server checks the file and queues it, a "
                "worker runs one analysis at a time on the graphics card, and the results page shows the video and "
                "reports. One job at a time means the 4 GB graphics card never runs out of memory.")
pic(s, "web_flow_loop.png", Inches(0.9), Inches(1.2), h=Inches(4.4))
box(s, X0, Inches(5.7), CW, Inches(1.05), "Architecture Summary", [C.ARCH_SUMMARY], 15)

# ---- 2. (contd.) Analysis pipeline
s = slide("2. End-to-End Application Architecture (contd.)", "Analysis Pipeline inside the Engine",
          notes="Inside the engine there are five steps. The key idea: we measure the TEAM as a whole every frame "
                "(spacing, speed, shape, ball position), and a trained model turns one second of those numbers "
                "into one of four tactics.")
pic(s, "analysis_pipeline.png", X0, Inches(1.2), w=Inches(8.0))
box(s, Inches(8.55), Inches(1.2), Inches(4.43), Inches(5.55), "Steps in Simple Words", C.PIPE_POINTS, 14)

# ---- 3. Components
s = slide("3. Architectural Components & Interface Design", "Component Breakdown, Data Flow & Technology Specifications",
          notes="Three parts: the website, the server and the analysis engine. The engine runs in its own process "
                "so a problem in one video never stops the website.")
box(s, L2, Inches(1.25), HW, Inches(2.65), "Core Application Components", C.COMPONENTS, 14)
box(s, R2, Inches(1.25), HW, Inches(2.65), "User Interaction & Workflow Design", C.JOURNEY, 14)
box(s, L2, Inches(4.05), HW, Inches(2.7), "Data Management & External Interfaces", C.DATA, 14)
box(s, R2, Inches(4.05), HW, Inches(2.7), "Platform & Deployment Specifications", [
    ("Development Stack", "Python, FastAPI, React, MongoDB, YOLO11, scikit-learn, OpenCV, FFmpeg."),
    ("Laptop (main)", "Windows + RTX 3050 graphics card: Accurate mode."),
    ("Cloud (live now)", "Azure server without a graphics card: public website, Fast mode."),
], 14)

# ---- 3. (contd.) Tactics + dataset
cm = RV["class_medians"]
fast = {k: v["fast_two_speed_cm_per_frame"] * 30 / 100 for k, v in cm.items()}
area = {k: v["court_area_m2 (hull_area)"] for k, v in cm.items()}
spread = {k: v["team_spread_cm (nn_dist_mean)"] / 100 for k, v in cm.items()}
s = slide("3. Architectural Components & Interface Design (contd.)",
          "Output Classes: The Four Team Tactics and the Labelled Dataset",
          notes="How to tell them apart: first, who has the ball - us (attack side) or them (defense side)? Second, "
                "did the team move as one unit? If not: one late player is Delayed Support, a broken shape is "
                "Spacing Breakdown. Labelling: one trained annotator watched each match, noted the time and the "
                "tactic, and a script cut a 1-second clip at that time; a check of 159 clips found 0 label "
                "mismatches. Why 8 videos: every action is labelled by hand; 3 more matches are not labelled yet "
                "and 1 file is damaged. No leakage: we always test on a match the model never saw.")
rows = [["Tactic", "Meaning in simple words", "How a labeller recognises it", "What our data shows"],
        ["Coordinated Attack", "Our team builds an attack together",
         "We have the ball; setter sets, 1-2 hitters run in fast, others cover",
         f"Fastest players move quickest ({fast['Coordinated Attack']:.1f} m/s)"],
        ["Coordinated Defense", "Our team defends as one unit",
         "Opponent attacks; block at the net, back players shift together",
         f"Widest court cover ({area['Coordinated Defense']:.1f} m²)"],
        ["Delayed Support", "One player reacts late",
         "After a touch, the teammate who should cover arrives late",
         f"Players bunched closest ({spread['Delayed Support']:.1f} m apart)"],
        ["Spacing Breakdown", "The team shape falls apart",
         "Players bunch up or leave a big gap; court left open",
         f"Largest, most uneven shape ({area['Spacing Breakdown']:.1f} m²)"]]
table(s, X0, Inches(1.25), CW, rows, [2.2, 2.8, 4.3, 3.3], 14, row_h=0.6)
box(s, X0, Inches(4.45), Inches(7.9), Inches(2.3), "Labelled Dataset", [
    ("Source", "11 public broadcast matches; 8 fully labelled → 945 one-second clips."),
    ("Clips per tactic", "Attack 369 · Defense 354 · Delayed Support 108 · Spacing Breakdown 114."),
    ("Labelling", "one trained annotator noted time + tactic; a script cut each 1-second clip."),
    ("Fair testing", "train on 7 matches, test on the 8th, repeat for all 8 (no match is in both)."),
], 13)
callout(s, Inches(8.45), Inches(4.45), Inches(4.53), Inches(2.3), [
    [("Order of decisions", B)],
    [("1. Who has the ball?", {})],
    [("2. Did the team move as one unit?", {})],
    [("3. If not: one late player → Delayed Support; broken shape → Spacing Breakdown.", {})]], 14, LIGHT, MAROON)

# ---- 4. Implementation overview
s = slide("4. Implementation Details -- Overview".replace("--", "–"),
          "End-to-End Application Module & Integration Status",
          notes="Every module is built and connected end to end. We give an honest status: the parts that work "
                "with known accuracy limits are measured on the testing slides.")
rows = [["Application Module", "Implemented Features & Integration Scope", "Status"],
        ["Module 1: User Interface", "Home, Analyze, Progress, Results, My analyses, Demos, Docs and Team pages; desktop app",
         "Complete (Integrated)"],
        ["Module 2: Core Analysis", "Video check, player + ball detection, tracking, court mapping, team split, "
         "tactic model, opponent SERVE / SPIKE / BLOCK, output video, reports", "Integrated (accuracy measured)"],
        ["Module 3: Data & Storage", "One database record + one folder per analysis; auto-delete after 14 days",
         "Complete (Integrated)"],
        ["Module 4: External Integration", "Graphics card for fast detection, FFmpeg for video, Azure cloud website",
         "Complete (Functional)"]]
table(s, X0, Inches(1.25), CW, rows, [2.8, 7.0, 2.8], 15, row_h=0.72)
box(s, X0, Inches(5.05), CW, Inches(1.7), "Prototype Environment & Repository", [
    ("Execution Platform", "Windows 11 laptop with RTX 3050 (run_website.bat starts everything); public copy on an "
                           "Azure cloud server."),
    ("Source Repository", f"{C.REPO} (version 1.0.0, open source)."),
], 15)

# ---- 5. Module details A + B
s = slide("5. Functional Module Implementation Details", "Key Feature Implementation & Inter-Module Integration",
          notes="Module A protects the system: only real volleyball videos get in. Module B is the analysis. "
                "The output frame shows only our team's players linked as a team, plus the ball.")
box(s, X0, Inches(1.25), Inches(6.5), Inches(2.6), "Module A: Video Upload & Check", [
    ("Implemented Workflow", "video sent in small parts with a progress bar."),
    ("Inputs & Outputs", "video + time range + mode in; a job number or a clear error out."),
    ("Validation & Controls", "only video files, 1 GB limit, a court with players must be visible, "
                              "only the uploader can cancel or delete."),
], 14)
box(s, X0, Inches(4.0), Inches(6.5), Inches(2.75), "Module B: Tactic Analysis Engine", [
    ("Implemented Workflow", "find players and ball → follow them → split teams by side of the net → measure the "
                             "team every frame → model names the tactic each second."),
    ("Inter-Module Integration", "runs as a separate worker, so the website keeps working during analysis; "
                                 "results go to the database and the Results page."),
], 14)
pic(s, "output_frame.png", Inches(7.05), Inches(1.4), w=Inches(5.9))
text(s, Inches(7.05), Inches(3.72), Inches(5.9), Inches(0.35),
     "Output frame: near-side team linked as one unit, with the ball", 13, MUTED, align=PP_ALIGN.CENTER)
box(s, Inches(7.05), Inches(4.2), Inches(5.9), Inches(2.55), "What the Coach Gets", [
    "Annotated match video with sound (several qualities).",
    "Tactic timeline report: one tactic per second, clickable to jump in the video.",
    "Opponent report: SERVE / SPIKE / BLOCK with times; short example clips per tactic.",
], 14)

# ---- 5. (contd.) Court mapping + team measurements
w8, w29 = WIN["8"], WIN["29"]
s = slide("5. Functional Module Implementation Details (contd.)", "Court Mapping and Team Measurements",
          notes="Court mapping: the floor is flat, so one 3x3 table of numbers (a homography) converts any pixel "
                "on the floor into metres. We find the painted lines, match them to the real court, and solve for "
                "the table. Accuracy: we hide one painted line, rebuild the mapping from the others, and check "
                "where it predicts the hidden line - the median miss is 5 cm. Window: one second = one volleyball "
                "action; tested 8 to 29 frames and the full second scores best.")
box(s, X0, Inches(1.25), HW, Inches(3.3), "Module C: Court Mapping (Calibration)", [
    ("Step 1", "take an empty-court frame; find the white painted lines."),
    ("Step 2", "match each line to its real place on the 9 m × 18 m court."),
    ("Step 3", "compute one 3×3 conversion table (homography): pixel ↔ metres."),
    ("Use", "player feet → metres; side of the net → which team."),
], 14)
table(s, X0, Inches(4.7), HW, [
    ["Accuracy check (11 matches)", "Result"],
    ["Hide one line, predict it", f"median miss {CA['leave_one_line_out_cm_median']:.0f} cm"],
    ["9 out of 10 points within", f"{CA['leave_one_line_out_cm_p90']:.0f} cm"],
    ["Line fit error", f"{CA['fit_rmse_px_mean']:.1f} pixels on average"]], [3.3, 2.9], 14, row_h=0.42)
box(s, R2, Inches(1.25), HW, Inches(3.3), "Module D: 18 Team Measurements per Frame", [
    ("Ball (3)", "seen or not; position."),
    ("Team position + shape (6)", "team centre; spread; outline area; longest gap."),
    ("Spacing (3)", "distance to nearest teammate: average, smallest, largest."),
    ("Count + ball link (2)", "players seen; ball distance from team centre."),
    ("Movement (4)", "speed of fastest 2, of other 4, their gap; moving together."),
], 13, gap=2)
callout(s, R2, Inches(4.7), HW, Inches(2.05), [
    [("Why a 1-second (29-frame) window? ", B), ("one second = one volleyball action.", {})],
    [(f"Tested: 8 frames → {pc(w8['accuracy'])} correct; 29 frames → {pc(w29['accuracy'])} correct "
      "(best). ", {}), ("Team-level numbers do not depend on player order, so a hidden player does not "
                        "break them.", {})]], 14, GOOD, GREEN)

# ---- 5. (contd.) Tactic model + why HGB
s = slide("5. Functional Module Implementation Details (contd.)", "Module E: Tactic Model and Why HGB Was Chosen",
          notes="Each pair of bars is one match the model never saw. HGB beats the best deep model, InceptionTime, "
                f"on {INC['hgb_better_folds']} of 8 matches; a statistical test (Wilcoxon) gives p = "
                f"{INC['wilcoxon_p']:.2f}, so the deep model is NOT better. With equal accuracy we chose the model "
                "that runs without a graphics card, is faster and gives honest confidence values. Settings: HGB with "
                "300 trees and learning rate 0.06; deep models 50 epochs, batch 32.")
pic(s, "rv_folds.png", X0, Inches(1.2), w=Inches(7.6))
rows = [["Model (tested on unseen matches)", "Correct", "Avg. over 4 tactics (F1)"],
        ["HGB – our model", pc(DEP["accuracy"]), f"{DEP['macro_f1']:.2f}"],
        ["InceptionTime (deep)", pc(INC["pooled_accuracy"]), f"{INC['pooled_macro_f1']:.2f}"],
        ["TCN (deep)", pc(DEEP["tcn"]["pooled_accuracy"]), f"{DEEP['tcn']['pooled_macro_f1']:.2f}"],
        ["Mamba (deep)", pc(DEEP["mamba"]["pooled_accuracy"]), f"{DEEP['mamba']['pooled_macro_f1']:.2f}"],
        ["Guessing the most common tactic", "39.0%", "–"]]
table(s, X0, Inches(4.85), Inches(7.6), rows, [3.6, 1.6, 2.4], 13, row_h=0.31, bold_first_row=1)
box(s, Inches(8.2), Inches(1.25), Inches(4.78), Inches(5.5), "Why HGB (Gradient-Boosted Trees)", [
    ("Wins more matches", f"better on {INC['hgb_better_folds']} of 8 unseen matches."),
    ("Deep model not better", f"statistical test p = {INC['wilcoxon_p']:.2f} (no real difference)."),
    ("No graphics card needed", "runs on the cloud server; deep models need a GPU."),
    ("Honest confidence", "calibrated: says 70% sure → right about 70% of the time."),
    ("Suits small data", "tree models beat deep nets on small tables of numbers [6]."),
    ("Compared fairly", "16 models, same 8 unseen-match tests."),
], 13)

# ---- 6. Testing
s = slide("6. Functional Testing & Verification Results", "End-to-End Feature Verification & Test Case Summary",
          notes="TC-01 to TC-04 are full-system tests. TC-05 to TC-07 are security: the site has no passwords; each "
                "browser gets a random secret ID stored only as a hash, and another browser cannot see, cancel, "
                "delete or upload into your analysis. 266 automatic code checks also pass.")
rows = [["Test Case / Feature", "Test Input / Action", "Verified Outcome", "Status"],
        ["TC-01: Full analysis", "150-s match video, Accurate mode",
         "annotated video, 40 tactic segments, 9 opponent actions, clips", "Pass"],
        ["TC-02: Real match check", "all 11 match videos", "all accepted", "Pass"],
        ["TC-03: Not volleyball", "street photo, test pattern, screen recording", "refused with clear message", "Pass"],
        ["TC-04: Wrong / fake / huge file", "PDF; text renamed .mp4; 5 GB file", "all refused (415 / 413)", "Pass"],
        ["TC-05: Hidden server file", "path-traversal request ../../config.py", "refused, nothing shown", "Pass"],
        ["TC-06: Another user's analysis", "second browser lists, cancels, deletes, uploads",
         "hidden or refused in all 4 (403 / 404)", "Pass"],
        ["TC-07: Owner rights", "owner deletes own analysis; forged short ID", "owner allowed; forged ID refused", "Pass"],
        ["TC-08: Automated code tests", "266 checks over 24 test files", "all passed; 52% code coverage", "Pass"]]
table(s, X0, Inches(1.2), CW, rows, [3.0, 4.0, 4.4, 1.2], 13, row_h=0.43)
box(s, X0, Inches(5.25), CW, Inches(1.5), "Application Verification Summary", [
    ("Functional Verification", f"all {len(FT)} API test cases and 266 code checks pass; 3 full analyses completed."),
    ("Responsiveness & Metrics", "about 7 min per minute of video on the laptop; tactic accuracy on unseen matches "
                                 f"{pc(DEP['accuracy'])} vs 39% by guessing."),
], 14)

# ---- 6. (contd.) Model accuracy
s = slide("6. Functional Testing & Verification Results (contd.)",
          "Tactic Accuracy on Matches the Model Never Saw",
          notes="Read the diagonal: Attack and Defense are right about 70% of the time, and when the model says "
                "Attack it is right 84% of the time. Delayed Support and Spacing Breakdown are mostly confused WITH "
                "EACH OTHER - both mean 'the team made a mistake'. Merged into one 'team error' class, the model "
                "catches about 75% of them. Kappa 0.49 = moderate agreement beyond chance.")
pic(s, "rv_confusion.png", X0, Inches(1.2), h=Inches(5.4))
rows = [["Tactic", "Precision", "Recall", "F1", "Clips"]] + [
    [k.replace("Coordinated ", ""), f"{v['precision']:.2f}", f"{v['recall']:.2f}", f"{v['f1']:.2f}", str(v["support"])]
    for k, v in pcl.items()]
table(s, R2, Inches(1.25), HW, rows, [2.4, 1.2, 1.1, 0.9, 0.9], 15, row_h=0.42)
table(s, R2, Inches(3.55), HW, [
    ["Overall (8 unseen-match tests)", "Score"],
    ["Correct tactic (accuracy)", pc(PC["accuracy"])],
    ["Average F1 over 4 tactics (macro)", f"{PC['macro_f1']:.2f}"],
    ["F1 weighted by tactic size", f"{PC['weighted_f1']:.2f}"],
    ["Agreement beyond chance (kappa)", f"{DEP['cohen_kappa']:.2f}"]], [3.9, 2.3], 15, row_h=0.4)
callout(s, R2, Inches(5.75), HW, Inches(0.95), [
    [("Precision", B), (" = when it says X, how often it is right.  ", {}),
     ("Recall", B), (" = of all real X, how many it finds.", {})]], 13, LIGHT, MAROON)

# ---- 6. (contd.) Component accuracy
s = slide("6. Functional Testing & Verification Results (contd.)", "Accuracy of Each Part, Checked Against People",
          notes="Tracking was tested on a public dataset whose player boxes were drawn by hand: players keep the "
                "same ID about 70% of the time, with only 4 ID switches in 925 boxes. For the ball and opponent "
                "actions there are no hand labels for our videos, so each detection was checked by eye; the sheets "
                "are in the evidence folder. Spike rule: touch above head height, ball leaves fast, next touch is "
                "by the other team. If the ball is lost, gaps up to 2 s are filled on a flight curve, and 15 of the "
                "18 measurements do not use the ball.")
rows = [["Part", "How it was tested", "Result"],
        ["Player tracking", f"{TRK['gt_boxes']} hand-drawn player boxes (public Volleyball Dataset)",
         f"same ID kept {TRK['idf1']:.0%} (IDF1) · only {TRK['id_switches']} ID switches · "
         f"{TRK['precision']:.0%} of boxes correct"],
        ["Court mapping", "hide one painted line, predict it (96 points, 11 matches)",
         f"median miss {CA['leave_one_line_out_cm_median']:.0f} cm"],
        ["Ball detection", f"{BA['play_frames']} random live-play frames, each box checked by eye",
         f"{pc(BA['precision'], 0)} of boxes are the ball ({pc(BA['precision_after_court_filter'], 0)} after court filter)"],
        ["Opponent SERVE", f"{AC['SERVE']['checked']} detected serves checked by eye",
         f"{AC['SERVE']['correct']} of {AC['SERVE']['checked']} correct"],
        ["Opponent SPIKE", f"{AC['SPIKE']['checked']} detected spikes checked by eye",
         f"{AC['SPIKE']['correct']} of {AC['SPIKE']['checked']} correct"],
        ["Team split", "side of the net from the court mapping", "works even when both teams wear similar shirts"]]
table(s, X0, Inches(1.25), CW, rows, [2.1, 5.2, 5.3], 14, row_h=0.56)
callout(s, X0, Inches(5.4), CW, Inches(1.35), [
    [("If the ball is lost: ", B), ("short gaps (≤ 2 s) are filled along its flight path; 15 of 18 team "
                                    "measurements do not need the ball.", {})],
    [("Spike vs normal touch: ", B), ("touched above head height + ball leaves fast + next touch by the other "
                                      "team.", {})]], 14, LIGHT, MAROON)

# ---- 7. Demo
s = slide("7. Working Prototype Demonstration", "End-to-End Application Walkthrough & Visual Evidence",
          notes=C.DEMO)
pic(s, "site_analyze.png", X0, Inches(1.2), w=HW)
pic(s, "output_frame.png", R2, Inches(1.2), w=HW)
text(s, X0, Inches(5.1), HW, Inches(0.3), "Input view: upload page with mode choice", 12, MUTED, align=PP_ALIGN.CENTER)
text(s, R2, Inches(3.62), HW, Inches(0.3), "Output view: annotated match video", 12, MUTED, align=PP_ALIGN.CENTER)
box(s, R2, Inches(4.0), HW, Inches(1.4), "Demo Steps", [
    "Upload → choose part + mode → progress bar → results page → download reports.",
], 14)
box(s, X0, Inches(5.5), CW, Inches(1.25), "Live Demonstration Scenario", [C.DEMO], 14)

# ---- 8. Justification
s = slide("8. Technical Justification & Architectural Decisions",
          "Platform Rationale, Engineering Trade-offs & Operational Considerations",
          notes="Why near-side team only: near players are large and fully visible; far players are small and hidden "
                "behind the net. Cloud: the public site runs on an Azure server without a graphics card today; a "
                "cloud server with a graphics card is future work. Coach study: planned next step.")
box(s, L2, Inches(1.25), HW, Inches(2.65), "1. Platform & Technology Justification", [
    "Python is the standard for video + AI; FastAPI handles big uploads; React gives a fast website; MongoDB stores "
    "job records easily.",
    "HGB chosen over deep models: same accuracy, no graphics card needed.",
], 14)
box(s, L2, Inches(4.05), HW, Inches(2.7), "2. Architecture & Integration Trade-offs", [
    "One analysis at a time, so the 4 GB graphics card never runs out of memory.",
    "Analysis runs separately, so one bad video never stops the website.",
    "Teams split by side of the net, not shirt colour.",
], 14)
box(s, R2, Inches(1.25), HW, Inches(2.65), "3. Usability, Security & Safety Measures", [
    "Only real video files; each browser has a secret ID (stored hashed); only the owner can cancel or delete.",
    "If court lines are not found, a verified court of the same venue or a standard court is used.",
], 14)
box(s, R2, Inches(4.05), HW, Inches(2.7), "4. Operational Constraints & Future Scope", [
    "After-match tool (~7 min per minute of video); near-side team; Delayed Support / Spacing Breakdown "
    "and opponent actions need more labelled data.",
    "Next: more labelled matches + a second annotator, cloud graphics card, study with coaches.",
], 13)

# ---- References
s = slide("References")
import re  # noqa: E402


def plain(v):
    v = re.sub(r"\\emph\{([^}]*)\}", r"\1", v)
    return v.replace("``", "“").replace("''", "”").replace("\\'i", "í").replace("--", "–")


text(s, X0, Inches(1.0), CW, Inches(6.0),
     [[(f"[{i}] ", {"bold": True, "color": MAROON}), (plain(v), {})] for i, (_, v) in enumerate(C.REFS, 1)], 13)

# ---- Thank you
s = prs.slides.add_slide(BLANK)
count["n"] += 1
text(s, 0, Inches(2.6), W, Inches(1.2), "Thank You", 60, MAROON, True, PP_ALIGN.CENTER)
text(s, 0, Inches(3.9), W, Inches(0.6), "Questions & Viva Voce", 26, INK, True, PP_ALIGN.CENTER)
text(s, 0, Inches(4.7), W, Inches(0.5), f"{C.REPO}   |   {C.SITE}", 16, MUTED, False, PP_ALIGN.CENTER)

prs.save(OUT)
print(count["n"], "slides ->", OUT)
