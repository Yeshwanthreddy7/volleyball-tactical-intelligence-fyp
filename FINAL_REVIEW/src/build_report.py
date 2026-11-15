"""Build the final-review report from content.py in two forms:
  report/final_report.tex   (official Amrita report template, compile on Overleaf)
  report/final_report.html  (same report, printed to an A4 PDF with Chrome)
Run: python src/build_report.py
"""
import html
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import content as C  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report"
OUT.mkdir(exist_ok=True)

ABSTRACT = (
    "Volleyball coaches usually study match videos by hand to see how their team moves together. This takes a long "
    "time, and different people can judge the same rally differently. This project is a web application that does "
    "this automatically from an ordinary match video recorded with one camera. The user uploads a video and chooses "
    "the part to analyse. The system first checks that the video shows a volleyball court with players. It then "
    "finds and follows every player and the ball, separates the two teams, and studies how the near-side team "
    "moves and positions itself around the ball. A trained machine-learning model then names the team's tactic for "
    "every second of play: Coordinated Attack, Coordinated Defense, Delayed Support or Spacing Breakdown. The user "
    "receives an annotated video, a timed tactic report, a list of the opponent's serves, spikes and blocks, and "
    "short example clips. The application is complete and works end to end on a laptop with a graphics card, and "
    "all functional tests passed. On match videos the model had never seen, it named the correct tactic 63.7% of "
    "the time, compared with 39.0% for simple guessing."
)
PROBLEM_ITEMS = [
    ("Problem Definition", "Coaches and analysts watch long match recordings to judge how a team attacks, "
     "defends and covers the court. Doing this by hand takes hours for each match, and two people can judge the "
     "same rally differently."),
    ("Target Users & Stakeholders", "Volleyball coaches, performance analysts and players at clubs and colleges, "
     "who usually have only one camera and no special tracking equipment."),
    ("Existing Limitations", "Existing scouting tools need a person to watch the match and type in every event. "
     "Research methods for recognising group activities [1, 2] need player positions marked by hand, so they cannot "
     "be used directly on a normal match video."),
]
OBJECTIVES = [
    "Take an ordinary match video and produce results without any manual marking.",
    "Name the near-side team's tactic for every second of play, and check the accuracy on match videos the model "
    "has never seen.",
    "Provide everything through an easy web application with input checks, a live progress bar and clear messages.",
]
REQ_ROWS = [(i, "Functional", t) for i, t in C.FR] + [
    ("NFR-1", "Non-Functional", "Refuse wrong or non-volleyball files early with a clear message; show live "
     "progress while the video is being analysed."),
    ("NFR-2", "Non-Functional", "Run on a normal laptop with a graphics card (NVIDIA RTX 3050); needs only one "
     "camera, no sensors or wearables."),
]
ARCH_TEXT = (
    "Figure 1 shows how the application works, from upload to result. "
    "(1) The coach opens the website and uploads a match video. "
    "(2) The web page sends the video to the server in small parts and shows an upload progress bar. "
    "(3) The server checks that the file is a real video and that a volleyball court with players is visible, "
    "then places the job in a waiting line (queue). "
    "(4) The video is saved, and a record of the job is stored in the database. "
    "(5) A worker picks up one job at a time and starts the analysis. "
    "(6) The analysis engine finds the players and the ball, follows them through the video, and the trained "
    "model names the tactic for each second. "
    "(7) The finished video, reports and clips are saved and shown on the Results page. "
    "Figure 2 shows the steps inside the analysis engine."
)
COMPONENT_ITEMS = [
    ("Main Components", "Website (React): the pages the user sees. Server (FastAPI): receives uploads, checks "
     "them and manages the job queue. Analysis Engine (Python): does the video analysis. Database (MongoDB): "
     "stores the record of every analysis."),
    ("User Flow", "Home, then Analyze (upload the video, choose the time range and the mode), then Progress, then "
     "Results (watch the video, see the tactic timeline, download the reports). A My analyses page lists earlier work."),
    ("Data Storage & External Interfaces", "Each analysis has one database record (status and summary) and one "
     "folder with the output video, reports and clips. The graphics card (GPU) speeds up player and ball "
     "detection, and FFmpeg creates the output videos."),
    ("Technology Stack", "Python, FastAPI, React, MongoDB, YOLO11 (player and ball detection), scikit-learn "
     "(tactic model), OpenCV and FFmpeg (video processing)."),
]
ENV_ITEMS = [
    ("Platform & Runtime Environment", "Windows 11 laptop with an NVIDIA RTX 3050 graphics card. One script "
     "(run_website.bat) starts the database and the server, and the website opens in the browser. A desktop app "
     "and a Windows installer are also available."),
    ("Libraries & Tools", "FastAPI, React, MongoDB, Ultralytics YOLO11, PyTorch, scikit-learn, OpenCV, FFmpeg "
     "and pytest (for automated testing)."),
    ("Source Code Repository", C.REPO + " (version 1.0.0, with documentation and automated tests)."),
]
MODULES = [
    ("Module 1: User Interface", "Home, Analyze, Progress, Results, My analyses, Demos, Methodology, "
     "Documentation and Team pages; desktop app.", "100% (Integrated)"),
    ("Module 2: Core Analysis", "Video check, player and ball detection, player tracking, team separation, "
     "tactic model, opponent action detection, annotated video, reports and clips.", "100% (Integrated)"),
    ("Module 3: Data & Storage", "A database record and a folder for every analysis; old analyses are deleted "
     "automatically after 14 days.", "100% (Integrated)"),
    ("Module 4: External Integration", "Graphics card (GPU) for fast detection, FFmpeg for video, public project "
     "website.", "100% (Functional)"),
]
DETAIL_ITEMS = [
    ("User Interface", "The Analyze page lets the user pick a video, the part to analyse and the mode (Accurate or "
     "Fast), and shows the upload progress. The Results page plays the annotated video and lists every tactic "
     "with its time."),
    ("Upload & Video Check", "Only video files (MP4, MOV, AVI, MKV, WebM) are accepted. Before the analysis, the "
     "server looks at a few frames to confirm that a volleyball court with players is visible; if not, it shows a "
     "clear message."),
    ("Analysis Engine", "A YOLO11 detector finds the players, and a second detector trained for volleyball finds "
     "the ball. A tracker keeps the same ID on each player across frames. Shirt colour and court side separate the "
     "two teams. For the near-side team, the engine measures how spread out the players are, how fast they move "
     "and where the ball is, one second at a time, and the trained model (HGB) names the tactic. Moments without "
     "live play, such as replays, are skipped. The opponent's serves, spikes and blocks are found with simple rules "
     "based on ball touches and body pose."),
    ("Results & Reports", "The engine makes the annotated video with the original sound in several qualities, "
     "writes the tactic and opponent reports (TXT, CSV and JSON), and cuts short example clips of each tactic."),
    ("End-to-End Integration", "The website, server, worker, analysis engine and database are fully connected, "
     "and each part can be restarted on its own. The same code runs on the laptop and on a cloud server."),
]
TESTS = [
    ["TC-01: Full analysis", "150-second match video, Accurate mode",
     "annotated video, 40 timed tactic segments, 9 opponent actions, example clips for all 4 tactics", "Pass"],
    ["TC-02: Video check (real match)", "all 11 match videos in our dataset", "all accepted", "Pass"],
    ["TC-03: Video check (not volleyball)", "street photo, test pattern, screen recording", "refused with a clear message", "Pass"],
    ["TC-04: Wrong file type", "a PDF document", "refused; the allowed file types are shown", "Pass"],
    ["TC-05: Fake video file", "a text file renamed to .mp4", "detected and refused", "Pass"],
    ["TC-06: File too large", "a 5 GB file", "refused before the upload starts", "Pass"],
    ["TC-07: Security check", "request for a missing analysis and for a hidden server file", "refused; nothing shown", "Pass"],
    ["TC-08: Automated tests", "266 automatic code checks", "all 266 passed", "Pass"],
]
EVAL_TEXT = (
    "All 8 test cases passed. Three full analyses of 1 to 2.5 minutes of match video finished on the laptop and "
    "produced every output. On the laptop's graphics card, one minute of 1080p video takes about 7 minutes to "
    "analyse, so the tool is meant for review after a match, not for live use. To measure accuracy fairly, the "
    "model was trained on 7 match videos and tested on the 8th, and this was repeated for all 8 videos (945 "
    "labelled one-second clips). Table 4 compares our model with other models tested in the same way."
)
MODEL_TABLE = (["Model", "Type", "Correct Tactic"], [
    ["HGB, our model (used in the app)", "Machine learning", "63.7%"],
    ["InceptionTime", "Deep learning", "61.9%"],
    ["Mamba", "Deep learning", "46.1%"],
    ["Always guessing the most common tactic", "Baseline", "39.0%"],
])
MODEL_CAP = "Accuracy on match videos the model never saw during training (945 clips from 8 videos)"
JUST_ITEMS = [
    ("Technology Choice", "Python is the standard language for video analysis and machine learning. FastAPI "
     "handles large video uploads well, React gives a fast and simple website, and MongoDB stores job records "
     "easily. For the tactic model we compared 16 models. HGB was as accurate as the best of them, suits the "
     "amount of data we have, and runs fast even without a graphics card, so we chose it over deep-learning models."),
    ("Design Trade-offs", "The server runs one analysis at a time, so the laptop's graphics card does not run out "
     "of memory; other jobs wait in the queue. The analysis runs separately from the website, so a problem in one "
     "analysis does not stop the website. Fast mode gives quicker results; Accurate mode finds the small ball more often."),
    ("Security & Error Handling", "Only real video files are accepted. A user's analyses can be cancelled or "
     "deleted only by that user. Requests for anything other than a real result file are refused. If the user "
     "closes the page, the unfinished analysis is stopped after 5 minutes. If the court lines cannot be found, a "
     "standard court layout is used."),
]
CONCLUSION = [
    ("Conclusion", "The application meets all its requirements: it accepts an ordinary match video, finds and "
     "follows the players and the ball, names the near-side team's tactic every second, and returns an annotated "
     "video and reports through a working website."),
    ("Current Limitations", "The analysis is done after the match, not live (about 7 minutes per minute of video). "
     "Tactics are given for the team on the near side of the court; because teams change sides between sets, the "
     "user analyses one set at a time. The two less common tactics, Delayed Support and Spacing Breakdown, are "
     "recognised less reliably than attack and defense. The model was trained on 945 clips from 8 match videos, "
     "so more data would improve it."),
    ("Future Enhancements", "Run the website on a cloud server with a graphics card for faster results. Add more "
     "match videos and labels to improve accuracy. Improve ball detection (a new ball detector is being trained). "
     "Recognise team names and follow the same team across sets."),
]
DELIVERABLES = [
    ("Source Code Repository", C.REPO + "; documentation, 266 automated tests, version 1.0.0", "Completed"),
    ("Working Application", "Website running on the laptop, desktop app and Windows installer", "Completed"),
    ("Individual Online Certifications", "; ".join(f"{n}: {c} ({p}, {hrs} h)" for n, c, p, hrs in C.CERTS), "Completed"),
    ("Group TEA Deliverable 1", "Project-specific dataset release: Volleyball Backcourt View Dataset on IEEE DataPort, "
     "https://ieee-dataport.org/documents/volleyball-backcourt-view-dataset", "Completed"),
    ("Group TEA Deliverable 2", "Project website (Hugging Face Space): "
     "https://huggingface.co/spaces/yeshwanthkumarreddy/volleyball-tactical-analysis", "Completed"),
    ("Group TEA Deliverable 3", "Open-source repository: "
     "https://github.com/Yeshwanthreddy7/volleyball-tactical-intelligence-fyp.git", "Completed"),
]
REFS = [r for r in C.REFS if r[0] not in ("grinsztajn2022",)] + [
    ("inceptiontime", "H. Ismail Fawaz \\emph{et al.}, ``InceptionTime: Finding AlexNet for time series "
     "classification,'' \\emph{Data Mining and Knowledge Discovery}, vol. 34, pp. 1936--1962, 2020."),
]


URL_RE = __import__("re").compile(r"https?://[^\s;,()]+")


def te(s):
    urls = URL_RE.findall(s)
    for i, u in enumerate(urls):
        s = s.replace(u, f"@@URL{i}@@", 1)
    s = (s.replace("\\", "\\textbackslash{}").replace("&", "\\&").replace("%", "\\%").replace("#", "\\#")
         .replace("_", "\\_").replace("$", "\\$").replace("→", "$\\rightarrow$"))
    for i, u in enumerate(urls):
        s = s.replace(f"@@URL{i}@@", f"\\url{{{u}}}")
    return s


def t_items(items):
    return "\\begin{itemize}[noitemsep, topsep=2pt]\n" + "\n".join(
        f"    \\item \\textbf{{{te(a)}:}} {te(b)}" for a, b in items) + "\n\\end{itemize}"


def t_rows(rows):
    return "\n".join("        " + " & ".join(te(x) for x in r) + " \\\\\n        \\hline" for r in rows)


stud_rows = []
for i, (r, n) in enumerate(C.STUDENTS):
    if i == 0:
        stud_rows.append(f"        \\small {r} & \\small {n} &\n        \\multirow{{{len(C.STUDENTS)}}}{{=}}{{\\raggedright \\small \\textbf{{{C.GUIDE[0]}}} \\newline \\textit{{{C.GUIDE[1]}, \\newline Dept. of CSE}}}} &\n        \\multirow{{{len(C.STUDENTS)}}}{{=}}{{\\raggedright \\small \\textbf{{Type 2}} \\newline Application-Based Project}} \\\\ \\cline{{1-2}}")
    else:
        end = "\\\\ \\cline{1-2}" if i < len(C.STUDENTS) - 1 else "\\\\\n        \\hline"
        stud_rows.append(f"        \\small {r} & \\small {n} & & {end}")

tex = r"""\documentclass[11pt, a4paper]{article}

% --- Packages ---
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{mathptmx}
\usepackage{graphicx}
\usepackage[margin=0.8in]{geometry}
\usepackage{xcolor}
\usepackage{titlesec}
\usepackage{enumitem}
\usepackage{float}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{array}
\usepackage{url}

\graphicspath{{../figures/}}

% --- Amrita Colors ---
\definecolor{amritaMaroon}{HTML}{A4123F}

% --- Title & Section Formatting ---
\titlespacing*{\section}{0pt}{12pt}{4pt}
\titlespacing*{\subsection}{0pt}{8pt}{3pt}
\titleformat{\section}{\large\bfseries\color{amritaMaroon}}{\thesection}{0.8em}{}
\titleformat{\subsection}{\normalsize\bfseries\color{amritaMaroon}}{\thesubsection}{0.6em}{}

\pagestyle{plain}

\begin{document}

\noindent
\begin{minipage}{\textwidth}
    \centering
    {\Large \textbf{\textcolor{amritaMaroon}{
    AMRITA VISHWA VIDYAPEETHAM \\
    SCHOOL OF COMPUTING \\
    DEPARTMENT OF COMPUTER SCIENCE AND ENGINEERING \\
    }}}
    \vspace{0.2cm}
    {\large \textbf{""" + te(C.TITLE) + r"""}} \\[0.1cm]
    \textbf{""" + C.COURSE + r"""} \quad | \quad \textbf{Team Number: """ + C.TEAM_NO + r"""} \\
    \today
\end{minipage}

\vspace{0.3cm}
\hrule
\vspace{0.3cm}

\noindent
\begin{center}
    \renewcommand{\arraystretch}{1.45}
    \setlength{\tabcolsep}{6pt}
    \begin{tabular}{|l|p{4.8cm}|p{4.6cm}|p{3.2cm}|}
        \hline
        \multicolumn{2}{|c|}{\textbf{Team Members}} & \textbf{Project Guide} & \textbf{Project Type} \\
        \hline
""" + "\n".join(stud_rows) + r"""
    \end{tabular}
\end{center}
\vspace{0.2cm}

\section{Abstract}
\noindent
""" + te(ABSTRACT) + r"""

\section{Problem Definition \& Requirement Analysis}

\subsection{Problem Statement, Context \& Target Users}
\begin{itemize}[noitemsep, topsep=2pt]
""" + "\n".join(f"    \\item \\textbf{{{te(a)}:}} {te(b).replace('[1, 2]', chr(92) + 'cite{zhang2024bicausal,ibrahim2016}')}" for a, b in PROBLEM_ITEMS) + r"""
    \item \textbf{Application Objectives:}
    \begin{enumerate}[noitemsep, topsep=2pt]
""" + "\n".join(f"        \\item {te(o)}" for o in OBJECTIVES) + r"""
    \end{enumerate}
\end{itemize}

\subsection{Functional \& Non-Functional Requirements}
\begin{table}[H]
    \centering
    \footnotesize
    \renewcommand{\arraystretch}{1.25}
    \begin{tabular}{|c|p{3.0cm}|p{9.2cm}|}
        \hline
        \textbf{Req. ID} & \textbf{Category} & \textbf{Requirement Specification} \\
        \hline
""" + t_rows(REQ_ROWS) + r"""
    \end{tabular}
    \caption{Summary of Application Functional and Non-Functional Requirements}
\end{table}

\section{Application Architecture \& Solution Design}

\subsection{End-to-End Application Architecture}
\noindent
""" + te(ARCH_TEXT) + r"""

\begin{figure}[H]
    \centering
    \includegraphics[width=\textwidth]{web_flow_loop.png}
    \caption{End-to-End Application Architecture and Workflow}
\end{figure}

\begin{figure}[H]
    \centering
    \includegraphics[width=\textwidth]{analysis_pipeline.png}
    \caption{Analysis Pipeline inside the Engine}
\end{figure}

\subsection{Component Design, Data Flow \& Technology Stack}
""" + t_items(COMPONENT_ITEMS) + r"""

\section{Implementation Details}

\subsection{Development \& Deployment Environment}
""" + t_items(ENV_ITEMS) + r"""

\subsection{Module-Wise Implementation \& Integration Summary}
\begin{table}[H]
    \centering
    \footnotesize
    \renewcommand{\arraystretch}{1.25}
    \begin{tabular}{|p{3.5cm}|p{7.8cm}|c|}
        \hline
        \textbf{Application Module} & \textbf{Implemented Features \& Integration Details} & \textbf{Status} \\
        \hline
""" + t_rows(MODULES) + r"""
    \end{tabular}
    \caption{Application Module Implementation and Integration Status}
\end{table}

\subsection{Detailed Module Descriptions}
""" + t_items(DETAIL_ITEMS) + r"""

\section{Functional Testing \& Prototype Demonstration}

\subsection{Functional Test Cases \& Verification}
\begin{table}[H]
    \centering
    \footnotesize
    \renewcommand{\arraystretch}{1.25}
    \begin{tabular}{|p{3.2cm}|p{3.6cm}|p{5.6cm}|c|}
        \hline
        \textbf{Test Case / Feature} & \textbf{Input / User Action} & \textbf{Observed Output} & \textbf{Result} \\
        \hline
""" + t_rows(TESTS) + r"""
    \end{tabular}
    \caption{Functional Testing Summary Across Core Application Scenarios}
\end{table}

\subsection{Working Prototype Demonstration}
\begin{figure}[H]
    \centering
    \includegraphics[width=0.49\textwidth]{site_analyze.png}\hfill
    \includegraphics[width=0.49\textwidth]{output_frame.png}
    \caption{Working Prototype: upload page (left) and annotated output video (right)}
\end{figure}

\noindent
\textbf{Evaluation \& Usability Summary:} """ + te(EVAL_TEXT) + r"""

\begin{table}[H]
    \centering
    \footnotesize
    \renewcommand{\arraystretch}{1.2}
    \begin{tabular}{|l|l|r|}
        \hline
        \textbf{Model} & \textbf{Type} & \textbf{Correct Tactic} \\
        \hline
""" + t_rows(MODEL_TABLE[1]) + r"""
    \end{tabular}
    \caption{""" + te(MODEL_CAP) + r"""}
\end{table}

\section{Technical Justification \& Architectural Decisions}
""" + t_items(JUST_ITEMS) + r"""

\section{Conclusion, Limitations \& Future Scope}
""" + t_items(CONCLUSION) + r"""

\section{Project Deliverables \& Technical Enrichment Summary}
\begin{table}[H]
    \centering
    \footnotesize
    \renewcommand{\arraystretch}{1.2}
    \begin{tabular}{|p{4.2cm}|p{8.6cm}|c|}
        \hline
        \textbf{Deliverable Category} & \textbf{Details / Title / Repository Link} & \textbf{Status} \\
        \hline
""" + t_rows(DELIVERABLES) + r"""
    \end{tabular}
    \caption{Summary of Completed Project and Technical Enrichment Deliverables}
\end{table}

\vspace{0.2cm}
\nocite{*}
\renewcommand{\refname}{\large\bfseries\color{amritaMaroon}References}
\bibliographystyle{ieeetr}
\bibliography{ref}

\end{document}
"""
(OUT / "final_report.tex").write_text(tex, encoding="utf-8")
(OUT / "ref.bib").write_text("\n\n".join(f"@misc{{{k},\n  note = {{{v}}}\n}}" for k, v in REFS) + "\n", encoding="utf-8")

# ------------------------------------------------------------------ HTML A4 version
h = html.escape


def hi(items):
    return "<ul>" + "".join(f"<li><b>{h(a)}:</b> {h(b)}</li>" for a, b in items) + "</ul>"


def linkify(x):
    return URL_RE.sub(lambda m: f'<a href="{m.group(0)}">{m.group(0)}</a>', h(x))


def htab(hd, rows, cap, n, cls=""):
    return (f'<table class="{cls}"><tr>' + "".join(f"<th>{h(x)}</th>" for x in hd) + "</tr>" +
            "".join("<tr>" + "".join(f"<td>{linkify(x)}</td>" for x in r) + "</tr>" for r in rows) +
            f'</table><div class="cap">Table {n}: {h(cap)}</div>')


def ref_html(v):
    return (h(v).replace("\\emph{", "<i>").replace("}", "</i>").replace("``", "“").replace("&#x27;&#x27;", "”")
            .replace("\\&#x27;i", "í").replace("--", "–"))


studs = "".join(
    f"<tr><td>{r}</td><td>{h(n)}</td>" + (f'<td rowspan="{len(C.STUDENTS)}"><b>{C.GUIDE[0]}</b><br><i>{C.GUIDE[1]},<br>Dept. of CSE</i></td>'
                                         f'<td rowspan="{len(C.STUDENTS)}"><b>Type 2</b><br>Application-Based Project</td>' if i == 0 else "") + "</tr>"
    for i, (r, n) in enumerate(C.STUDENTS))
body = f"""
<div class="head"><div class="uni">AMRITA VISHWA VIDYAPEETHAM<br>SCHOOL OF COMPUTING<br>DEPARTMENT OF COMPUTER SCIENCE AND ENGINEERING</div>
<div class="ptitle">{h(C.TITLE)}</div>
<div class="course"><b>{h(C.COURSE.replace('--', '–'))}</b> &nbsp;|&nbsp; <b>Team Number: {C.TEAM_NO}</b></div><div class="date">November 2026</div></div>
<hr>
<table class="team"><tr><th colspan="2">Team Members</th><th>Project Guide</th><th>Project Type</th></tr>{studs}</table>
<h2>1&nbsp;&nbsp; Abstract</h2><p>{h(ABSTRACT)}</p>
<h2>2&nbsp;&nbsp; Problem Definition &amp; Requirement Analysis</h2>
<h3>2.1&nbsp; Problem Statement, Context &amp; Target Users</h3>
<ul>{"".join(f"<li><b>{h(a)}:</b> {h(b)}</li>" for a, b in PROBLEM_ITEMS)}<li><b>Application Objectives:</b><ol>{"".join(f"<li>{h(o)}</li>" for o in OBJECTIVES)}</ol></li></ul>
<h3>2.2&nbsp; Functional &amp; Non-Functional Requirements</h3>
{htab(["Req. ID", "Category", "Requirement Specification"], REQ_ROWS, "Summary of Application Functional and Non-Functional Requirements", 1, "req")}
<h2>3&nbsp;&nbsp; Application Architecture &amp; Solution Design</h2>
<h3>3.1&nbsp; End-to-End Application Architecture</h3><p>{h(ARCH_TEXT)}</p>
<div class="fig"><img src="../figures/web_flow_loop.png"><div class="cap">Figure 1: End-to-End Application Architecture and Workflow</div></div>
<div class="fig"><img src="../figures/analysis_pipeline.png"><div class="cap">Figure 2: Analysis Pipeline inside the Engine</div></div>
<h3>3.2&nbsp; Component Design, Data Flow &amp; Technology Stack</h3>{hi(COMPONENT_ITEMS)}
<h2>4&nbsp;&nbsp; Implementation Details</h2>
<h3>4.1&nbsp; Development &amp; Deployment Environment</h3>{hi(ENV_ITEMS)}
<h3>4.2&nbsp; Module-Wise Implementation &amp; Integration Summary</h3>
{htab(["Application Module", "Implemented Features & Integration Details", "Status"], MODULES, "Application Module Implementation and Integration Status", 2, "mod")}
<h3>4.3&nbsp; Detailed Module Descriptions</h3>{hi(DETAIL_ITEMS)}
<h2>5&nbsp;&nbsp; Functional Testing &amp; Prototype Demonstration</h2>
<h3>5.1&nbsp; Functional Test Cases &amp; Verification</h3>
{htab(["Test Case / Feature", "Input / User Action", "Observed Output", "Result"], TESTS, "Functional Testing Summary Across Core Application Scenarios", 3, "tc")}
<h3>5.2&nbsp; Working Prototype Demonstration</h3>
<div class="fig two"><img src="../figures/site_analyze.png"><img src="../figures/output_frame.png"><div class="cap">Figure 3: Working Prototype: upload page (left) and annotated output video (right)</div></div>
<p><b>Evaluation &amp; Usability Summary:</b> {h(EVAL_TEXT)}</p>
{htab(MODEL_TABLE[0], MODEL_TABLE[1], MODEL_CAP, 4, "mt")}
<h2>6&nbsp;&nbsp; Technical Justification &amp; Architectural Decisions</h2>{hi(JUST_ITEMS)}
<h2>7&nbsp;&nbsp; Conclusion, Limitations &amp; Future Scope</h2>{hi(CONCLUSION)}
<h2>8&nbsp;&nbsp; Project Deliverables &amp; Technical Enrichment Summary</h2>
{htab(["Deliverable Category", "Details / Title / Repository Link", "Status"], DELIVERABLES, "Summary of Completed Project and Technical Enrichment Deliverables", 5, "del")}
<h2>References</h2>{"".join(f'<p class="ref">[{i}] {ref_html(v)}</p>' for i, (_, v) in enumerate(REFS, 1))}
"""
CSS = """
@page{size:A4;margin:0.8in 0.8in 0.7in}
body{font-family:'Times New Roman',Times,serif;font-size:11pt;color:#000;line-height:1.3;-webkit-print-color-adjust:exact;print-color-adjust:exact}
.head{text-align:center}
.uni{color:#A4123F;font-weight:bold;font-size:15.5pt;line-height:1.25}
.ptitle{font-weight:bold;font-size:13pt;margin-top:8px}
.course{margin-top:4px}.date{margin-top:2px}
hr{border:0;border-top:1px solid #000;margin:10px 0}
table{border-collapse:collapse;margin:8px auto 2px;font-size:9.5pt;width:100%}
th,td{border:1px solid #000;padding:4px 6px;vertical-align:top;text-align:left}
table.team{font-size:10.5pt}
table.req td:first-child,table.tc td:last-child,table.mod td:last-child,table.del td:last-child{text-align:center;white-space:nowrap}
a{color:#1a4fa0;word-break:break-all} table.mt{width:70%} table.mt td:not(:first-child){text-align:right}
.cap{text-align:center;font-size:10.5pt;margin:4px 0 10px}
h2{color:#A4123F;font-size:13pt;margin:14px 0 4px;page-break-after:avoid}
h3{color:#A4123F;font-size:11pt;margin:9px 0 3px;page-break-after:avoid}
p{margin:3px 0 6px;text-align:justify}
ul,ol{margin:2px 0 6px;padding-left:22px}
li{margin:1px 0;text-align:justify}
.fig{text-align:center;page-break-inside:avoid;margin:6px 0}
.fig img{width:100%}
.fig.two img{width:49%;height:180px;object-fit:cover;object-position:top left}
table,tr{page-break-inside:avoid}
.ref{font-size:10pt;margin:2px 0 4px;padding-left:24px;text-indent:-24px;text-align:left}
"""
(OUT / "final_report.html").write_text(f'<!doctype html><html><head><meta charset="utf-8"><title>Final Review Report</title><style>{CSS}</style></head><body>{body}</body></html>', encoding="utf-8")
print("report written")
