"""Build the final-review presentation from content.py in two forms:
  presentation/final_presentation.tex   (official Amrita Beamer template, compile on Overleaf)
  presentation/final_presentation.html  (same slides, printed to PDF with Chrome)
Run: python src/build_presentation.py
"""
import html
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import content as C  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "presentation"
OUT.mkdir(exist_ok=True)


def tex_esc(s):
    return (s.replace("\\", "\\textbackslash{}").replace("&", "\\&").replace("%", "\\%").replace("#", "\\#")
            .replace("_", "\\_").replace("$", "\\$").replace("~", "\\textasciitilde{}")
            .replace("--", "--").replace("→", "$\\rightarrow$").replace("≥", "$\\geq$"))


h = html.escape

# ------------------------------------------------------------------ slide model
# each slide: dict(title, sub, kind, ...)
slides = []
slides.append({"kind": "title"})
slides.append({"kind": "approval"})
slides.append({"kind": "blocks", "title": "1. Problem Definition & Requirement Analysis",
               "sub": "Target Problem, User Needs & Functional / Non-Functional Requirements",
               "blocks": [("Problem Statement & Target Stakeholders", C.PROBLEM),
                          ("Functional Requirements", C.FR),
                          ("Non-Functional Requirements & Constraints", C.NFR)]})
slides.append({"kind": "figure", "title": "2. End-to-End Application Architecture",
               "sub": "System Architecture & User-to-Backend Workflow Diagram",
               "img": "web_flow_loop.png", "summary": C.ARCH_SUMMARY})
slides.append({"kind": "figside", "title": "2. End-to-End Application Architecture (contd.)",
               "sub": "Analysis Pipeline inside the Engine",
               "img": "analysis_pipeline.png", "points": C.PIPE_POINTS})
slides.append({"kind": "grid4", "title": "3. Architectural Components & Interface Design",
               "sub": "Component Breakdown, Data Flow & Technology Specifications",
               "left": [("Core Application Components", C.COMPONENTS), ("User Interaction & Workflow Design", C.JOURNEY)],
               "right": [("Data Management & External Interfaces", C.DATA), ("Platform & Deployment Specifications", C.PLATFORM)]})
slides.append({"kind": "modtable", "title": "4. Implementation Details -- Overview",
               "sub": "End-to-End Application Module & Integration Status"})
slides.append({"kind": "module", "title": "5. Functional Module Implementation Details",
               "sub": "Key Feature Implementation & Inter-Module Integration",
               "mods": [C.MOD_A, C.MOD_B], "img": "output_frame.png",
               "cap": "Output frame: player links, ball and tactic for the near-side team"})
slides.append({"kind": "module_table", "title": "5. Functional Module Implementation Details (contd.)",
               "sub": "Tactic Model and Outputs", "mods": [C.MOD_C, C.MOD_D]})
slides.append({"kind": "tests", "title": "6. Functional Testing & Verification Results",
               "sub": "End-to-End Feature Verification & Test Case Summary"})
slides.append({"kind": "demo", "title": "7. Working Prototype Demonstration",
               "sub": "End-to-End Application Walkthrough & Visual Evidence"})
slides.append({"kind": "just", "title": "8. Technical Justification & Architectural Decisions",
               "sub": "Platform Rationale, Engineering Trade-offs & Operational Considerations"})
slides.append({"kind": "limits", "title": "9. Limitations & Future Scope",
               "sub": "What the Prototype Cannot Do Yet, and What Comes Next"})
slides.append({"kind": "refs"})
slides.append({"kind": "thanks"})
N = len(slides)


# ------------------------------------------------------------------ LaTeX
def t_items(items, size="\\footnotesize"):
    out = [size, "\\begin{itemize}"]
    for it in items:
        if isinstance(it, tuple):
            out.append(f"  \\item \\textbf{{{tex_esc(it[0])}:}} {tex_esc(it[1])}")
        else:
            out.append(f"  \\item {tex_esc(it)}")
    out.append("\\end{itemize}")
    return "\n".join(out)


def t_block(title, items, size="\\footnotesize"):
    return f"\\begin{{block}}{{{tex_esc(title)}}}\n{t_items(items, size)}\n\\end{{block}}"


def tex_frame(s):
    k = s["kind"]
    if k == "title":
        rows = "\n".join(f"            {i} & {r} & {n} \\\\\n            \\hline" for i, (r, n) in enumerate(C.STUDENTS, 1))
        return rf"""\begin{{frame}}
    \begin{{tikzpicture}}[remember picture,overlay]
        \node[fill=amritaMaroon, text=white, font=\bfseries\footnotesize, anchor=north east, rounded corners=2pt, inner sep=5pt]
        at ($(current page.north east) + (-0.2cm,-0.2cm)$) {{Type 2: Application Project}};
    \end{{tikzpicture}}
    \centering
    \includegraphics[width=2.8cm]{{logo.png}}
    \vspace{{0.1cm}}

    {{\large \textbf{{\textcolor{{amritaMaroon}}{{{tex_esc(C.TITLE)}}}}}}} \\
    \vspace{{0.05cm}}
    {{\footnotesize \textbf{{{C.COURSE}}} \quad | \quad \textbf{{Team Number: {C.TEAM_NO}}}}} \\
    \vspace{{0.1cm}}
    \begin{{table}}
        \centering
        \scriptsize
        \renewcommand{{\arraystretch}}{{1.0}}
        \begin{{tabular}}{{|c|c|l|}}
            \hline
            \textbf{{S.No}} & \textbf{{Register Number}} & \textbf{{Student Name}} \\
            \hline
{rows}
        \end{{tabular}}
    \end{{table}}
    \vspace{{0.05cm}}
    {{\footnotesize Guide: \textbf{{{C.GUIDE[0]}}}, {C.GUIDE[1]}, Department of CSE}} \\
    \vspace{{0.05cm}}
    {{\scriptsize \today}}
\end{{frame}}"""
    if k == "approval":
        return r"""\begin{frame}{Guide Review \& Submission Approval}
    \framesubtitle{Confirmation of Project Review and Approval from Guide}
    \centering
    \IfFileExists{guide_approval.png}{\includegraphics[height=5.6cm]{guide_approval.png}}{%
    \fbox{\begin{minipage}[c][5.4cm]{0.94\textwidth}
        \centering
        \textbf{[Insert Screenshot of Guide Approval Email Here]} \\[0.3cm]
        {\footnotesize Save the screenshot as guide\_approval.png next to this file and recompile.}
    \end{minipage}}}
\end{frame}"""
    head = (f"\\begin{{frame}}{{{tex_esc(s['title'])}}}\n    \\framesubtitle{{{tex_esc(s['sub'])}}}\n"
            if "title" in s else "")
    if k == "blocks":
        body = "\n".join(t_block(t, it, "\\scriptsize") for t, it in s["blocks"])
    elif k == "figure":
        body = (f"\\centering\n\\includegraphics[width=0.97\\textwidth]{{{s['img']}}}\n\\vspace{{0.2cm}}\n"
                f"\\begin{{block}}{{Summary}}\n\\footnotesize {tex_esc(s['summary'])}\n\\end{{block}}")
    elif k == "figside":
        body = (f"\\begin{{columns}}[T]\n\\column{{0.62\\textwidth}}\n\\centering\n"
                f"\\includegraphics[width=\\textwidth]{{{s['img']}}}\n\\column{{0.36\\textwidth}}\n"
                f"{t_block('Steps in Simple Words', s['points'], chr(92) + 'scriptsize')}\n\\end{{columns}}")
    elif k == "grid4":
        L = "\n".join(t_block(t, it, "\\scriptsize") for t, it in s["left"])
        R = "\n".join(t_block(t, it, "\\scriptsize") for t, it in s["right"])
        body = f"\\begin{{columns}}[T]\n\\column{{0.50\\textwidth}}\n{L}\n\\column{{0.48\\textwidth}}\n{R}\n\\end{{columns}}"
    elif k == "modtable":
        rows = "\n".join(f"{tex_esc(a)} & {tex_esc(b)} & {tex_esc(c)} \\\\" for a, b, c in C.MODULES)
        body = (r"""\begin{table}
    \centering
    \scriptsize
    \renewcommand{\arraystretch}{1.2}
    \begin{tabular}{p{3.2cm} p{7.4cm} c}
        \toprule
        \textbf{Application Module} & \textbf{Implemented Features \& Integration Scope} & \textbf{Status} \\
        \midrule
""" + rows + r"""
        \bottomrule
    \end{tabular}
\end{table}
""" + t_block("Prototype Environment & Repository", C.ENV, "\\scriptsize"))
    elif k == "module":
        L = "\n".join(t_block(t, it, "\\scriptsize") for t, it in s["mods"])
        body = (f"\\begin{{columns}}[T]\n\\column{{0.52\\textwidth}}\n{L}\n\\column{{0.46\\textwidth}}\n\\centering\n"
                f"\\includegraphics[width=\\textwidth]{{{s['img']}}}\\\\[0.1cm]\n{{\\scriptsize {tex_esc(s['cap'])}}}\n\\end{{columns}}")
    elif k == "module_table":
        L = "\n".join(t_block(t, it, "\\scriptsize") for t, it in s["mods"])
        hd, rows = C.MODEL_TABLE
        rr = "\n".join(" & ".join(tex_esc(x) for x in r) + " \\\\" for r in rows)
        body = (f"\\begin{{columns}}[T]\n\\column{{0.52\\textwidth}}\n{L}\n\\column{{0.46\\textwidth}}\n\\centering\n"
                "{\\scriptsize \\textbf{Accuracy on videos never seen in training}}\\\\[0.1cm]\n"
                "\\scriptsize\n\\begin{tabular}{llr}\n\\toprule\n" + " & ".join(f"\\textbf{{{x}}}" for x in hd) +
                " \\\\\n\\midrule\n" + rr + "\n\\bottomrule\n\\end{tabular}\\\\[0.1cm]\n"
                "{\\tiny 945 clips from 8 videos; each video tested once while the model trained on the other 7.}\n\\end{columns}")
    elif k == "tests":
        hd, rows = C.TESTS
        rr = "\n".join(" & ".join(tex_esc(x) for x in r) + " \\\\" for r in rows)
        body = (r"""\begin{table}
    \centering
    \tiny
    \renewcommand{\arraystretch}{1.15}
    \begin{tabular}{p{3.0cm} p{3.4cm} p{4.4cm} c}
        \toprule
        \textbf{Test Case / Feature} & \textbf{Test Input / Action} & \textbf{Verified Outcome} & \textbf{Status} \\
        \midrule
""" + rr + r"""
        \bottomrule
    \end{tabular}
\end{table}
""" + t_block("Application Verification Summary", C.VERIFY, "\\scriptsize"))
    elif k == "demo":
        body = (r"""\begin{columns}[T]
\column{0.48\textwidth}
\centering
\includegraphics[width=\textwidth,height=4.3cm,keepaspectratio]{site_analyze.png}\\
{\scriptsize Input view: upload page with mode choice}
\column{0.48\textwidth}
\centering
\includegraphics[width=\textwidth,height=4.3cm,keepaspectratio]{output_frame.png}\\
{\scriptsize Output view: annotated match video}
\end{columns}
\vspace{0.2cm}
\begin{block}{Live Demonstration Scenario}
\footnotesize """ + tex_esc(C.DEMO) + "\n\\end{block}")
    elif k == "just":
        L = "\n".join(t_block(t, it, "\\scriptsize") for t, it in C.JUST[:2])
        R = "\n".join(t_block(t, it, "\\scriptsize") for t, it in C.JUST[2:])
        body = f"\\begin{{columns}}[T]\n\\column{{0.48\\textwidth}}\n{L}\n\\column{{0.48\\textwidth}}\n{R}\n\\end{{columns}}"
    elif k == "limits":
        body = (f"\\begin{{columns}}[T]\n\\column{{0.48\\textwidth}}\n{t_block('Current Limitations', C.LIMITS, chr(92) + 'footnotesize')}\n"
                f"\\column{{0.48\\textwidth}}\n{t_block('Future Scope', C.FUTURE, chr(92) + 'footnotesize')}\n\\end{{columns}}")
    elif k == "refs":
        return r"""\section{References}
\nocite{*}
\begin{frame}[allowframebreaks]{References}
    \footnotesize
    \bibliographystyle{ieeetr}
    \bibliography{ref}
\end{frame}"""
    elif k == "thanks":
        return r"""\begin{frame}
    \centering
    {\Huge \textbf{\textcolor{amritaMaroon}{Thank You}}} \\[0.4cm]
    {\large \textbf{Questions \& Viva Voce}}
\end{frame}"""
    return head + body + "\n\\end{frame}"


preamble = r"""\documentclass[aspectratio=169, 11pt]{beamer}

% --- Packages & Setup ---
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{helvet}
\renewcommand{\familydefault}{\sfdefault}
\usepackage{xcolor}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{tikz}
\usetikzlibrary{calc}

\graphicspath{{../figures/}{./}}

% --- Branding Colors ---
\definecolor{amritaMaroon}{HTML}{A4123F}

% --- Theme Setup ---
\usetheme{Madrid}
\usecolortheme[named=amritaMaroon]{structure}
\setbeamertemplate{navigation symbols}{}
\setbeamertemplate{footline}[frame number]

\begin{document}
"""
tex = preamble + "\n\n".join(tex_frame(s) for s in slides) + "\n\n\\end{document}\n"
(OUT / "final_presentation.tex").write_text(tex, encoding="utf-8")
bib = "\n\n".join(f"@misc{{{k},\n  note = {{{v}}}\n}}" for k, v in C.REFS)
(OUT / "ref.bib").write_text(bib + "\n", encoding="utf-8")


# ------------------------------------------------------------------ HTML (beamer Madrid look)
def hx(s):
    return h(s).replace("--", "–")


def h_items(items):
    li = []
    for it in items:
        if isinstance(it, tuple):
            li.append(f"<li><b>{hx(it[0])}:</b> {hx(it[1])}</li>")
        else:
            li.append(f"<li>{hx(it)}</li>")
    return "<ul>" + "".join(li) + "</ul>"


def h_block(title, items, cls=""):
    return f'<div class="blk {cls}"><div class="bh">{hx(title)}</div><div class="bb">{h_items(items)}</div></div>'


def h_text_block(title, text, cls=""):
    return f'<div class="blk {cls}"><div class="bh">{hx(title)}</div><div class="bb"><p>{hx(text)}</p></div></div>'


def h_table(hd, rows, cls=""):
    th = "".join(f"<th>{hx(x)}</th>" for x in hd)
    tr = "".join("<tr>" + "".join(f"<td>{hx(x)}</td>" for x in r) + "</tr>" for r in rows)
    return f'<table class="bt {cls}"><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table>'


def html_slide(i, s):
    k = s["kind"]
    pg = f'<div class="pg">{i} / {N}</div>'
    if k == "title":
        rows = "".join(f"<tr><td>{n}</td><td>{r}</td><td class='l'>{h(nm)}</td></tr>" for n, (r, nm) in enumerate(C.STUDENTS, 1))
        return f"""<section class="slide"><div class="tag">Type 2: Application Project</div>
<div class="tc"><img src="../figures/logo.png" class="logo">
<div class="tt">{h(C.TITLE)}</div>
<div class="tsub">{hx(C.COURSE)} &nbsp;|&nbsp; Team Number: {C.TEAM_NO}</div>
<table class="st"><tr><th>S.No</th><th>Register Number</th><th class='l'>Student Name</th></tr>{rows}</table>
<div class="tg">Guide: <b>{C.GUIDE[0]}</b>, {C.GUIDE[1]}, Department of CSE</div>
<div class="td">November 2026</div></div>{pg}</section>"""
    if k == "approval":
        return f"""<section class="slide"><div class="hdr"><h1>Guide Review &amp; Submission Approval</h1><h2>Confirmation of Project Review and Approval from Guide</h2></div>
<div class="appr"><img src="../figures/guide_approval.png" onerror="this.style.display='none';this.nextElementSibling.style.display='flex'">
<div class="ph" style="display:none"><b>[Insert Screenshot of Guide Approval Email Here]</b><span>Save it as figures/guide_approval.png and rebuild.</span></div></div>{pg}</section>"""
    if k == "refs":
        refs = "".join(f'<p class="ref">[{n}] {h(v).replace(chr(92)+"emph{", "<i>").replace("}", "</i>").replace("``", "“").replace("&#x27;&#x27;", "”").replace(chr(92)+"&#x27;i", "í").replace("--", "–")}</p>'
                       for n, (_, v) in enumerate(C.REFS, 1))
        return f'<section class="slide"><div class="hdr short"><h1>References</h1></div><div class="body refs">{refs}</div>{pg}</section>'
    if k == "thanks":
        return f'<section class="slide"><div class="thx"><div>Thank You</div><span>Questions &amp; Viva Voce</span></div>{pg}</section>'
    hdr = f'<div class="hdr"><h1>{hx(s["title"])}</h1><h2>{hx(s["sub"])}</h2></div>'
    if k == "blocks":
        body = "".join(h_block(t, it) for t, it in s["blocks"])
    elif k == "figure":
        body = f'<div class="figbox"><img src="../figures/{s["img"]}"></div>' + h_text_block("Summary", s["summary"])
    elif k == "figside":
        body = (f'<div class="figside"><div class="fsimg"><img src="../figures/{s["img"]}"></div>'
                f'<div>{h_block("Steps in Simple Words", s["points"])}</div></div>')
    elif k == "grid4":
        body = ('<div class="cols"><div>' + "".join(h_block(t, it) for t, it in s["left"]) + "</div><div>" +
                "".join(h_block(t, it) for t, it in s["right"]) + "</div></div>")
    elif k == "modtable":
        body = h_table(["Application Module", "Implemented Features & Integration Scope", "Status"], C.MODULES, "mod") + h_block("Prototype Environment & Repository", C.ENV)
    elif k == "module":
        body = ('<div class="cols"><div>' + "".join(h_block(t, it) for t, it in s["mods"]) +
                f'</div><div class="side"><img src="../figures/{s["img"]}"><div class="cap">{hx(s["cap"])}</div></div></div>')
    elif k == "module_table":
        hd, rows = C.MODEL_TABLE
        body = ('<div class="cols"><div>' + "".join(h_block(t, it) for t, it in s["mods"]) +
                '</div><div class="side"><div class="tcap">Accuracy on videos never seen in training</div>' + h_table(hd, rows, "mt") +
                '<div class="cap">945 clips from 8 videos; each video tested once while the model trained on the other 7.</div></div></div>')
    elif k == "tests":
        hd, rows = C.TESTS
        body = h_table(hd, rows, "tests") + h_block("Application Verification Summary", C.VERIFY)
    elif k == "demo":
        body = ('<div class="cols demo"><div><img src="../figures/site_analyze.png"><div class="cap">Input view: upload page with mode choice</div></div>'
                '<div><img src="../figures/output_frame.png"><div class="cap">Output view: annotated match video</div></div></div>'
                + h_text_block("Live Demonstration Scenario", C.DEMO))
    elif k == "just":
        body = ('<div class="cols"><div>' + "".join(h_block(t, it) for t, it in C.JUST[:2]) + "</div><div>" +
                "".join(h_block(t, it) for t, it in C.JUST[2:]) + "</div></div>")
    elif k == "limits":
        body = '<div class="cols"><div>' + h_block("Current Limitations", C.LIMITS) + "</div><div>" + h_block("Future Scope", C.FUTURE, "g") + "</div></div>"
    return f'<section class="slide">{hdr}<div class="body">{body}</div>{pg}</section>'


CSS = """
@page { size: 1456px 819px; margin: 0; }
*{box-sizing:border-box}
html,body{margin:0;padding:0;background:#fff}
body{font-family:Helvetica,Arial,sans-serif;color:#111;-webkit-print-color-adjust:exact;print-color-adjust:exact}
.slide{width:1456px;height:819px;position:relative;overflow:hidden;page-break-after:always;break-after:page;background:#fff}
.slide:last-child{page-break-after:auto}
.hdr{background:#A4123F;color:#fff;padding:14px 28px 10px;height:120px}
.hdr.short{height:80px}
.hdr h1{margin:0;font-weight:normal;font-size:40px}
.hdr h2{margin:6px 0 0;font-weight:normal;font-size:24px}
.body{position:absolute;top:136px;left:22px;right:22px;bottom:36px}
.pg{position:absolute;right:20px;bottom:10px;font-size:17px}
.blk{border-radius:9px;overflow:hidden;box-shadow:3px 4px 6px rgba(0,0,0,.35);background:#f1e6eb;margin-bottom:14px}
.bh{background:#7d0f31;color:#fff;font-size:28px;padding:5px 14px}
.blk.g .bh{background:#1d7a35}.blk.g{background:#eaf3ec}
.bb{padding:8px 14px 10px;font-size:23px;line-height:1.32}
.bb p{margin:0}
.bb ul{margin:0;padding-left:34px;list-style:none}
.bb li{position:relative;margin:4px 0}
.bb li::before{content:"";position:absolute;left:-26px;top:7px;width:13px;height:13px;border-radius:50%;background:#7d0f31}
.cols{display:grid;grid-template-columns:1.05fr 1fr;gap:18px}
.figbox{border:1.5px solid #333;padding:10px;margin-bottom:14px;text-align:center}
.figbox{padding:2px}.figbox img{max-width:100%;max-height:520px}
.figside{display:grid;grid-template-columns:900px 1fr;gap:18px;align-items:start}
.fsimg{border:1.5px solid #333;padding:6px;text-align:center}.fsimg img{width:100%;max-height:620px;object-fit:contain}
.side{text-align:center}
.side img{width:100%;border:1px solid #333}
.cap{font-size:16px;color:#333;margin-top:6px;text-align:center}
.tcap{font-size:20px;font-weight:bold;margin:4px 0 8px}
table.bt{border-collapse:collapse;width:100%;font-size:21px;margin-bottom:16px}
table.bt th{text-align:left;padding:7px 10px;border-bottom:2px solid #000;border-top:3px solid #000}
table.bt td{padding:5px 10px;vertical-align:top}
table.bt tbody tr:last-child td{border-bottom:3px solid #000}
table.bt.mod td:last-child,table.bt.mod th:last-child{white-space:nowrap;text-align:center}
table.bt.tests{font-size:19px}
table.bt.tests td:last-child,table.bt.tests th:last-child{text-align:center}
table.bt.mt{font-size:21px} table.bt.mt td:first-child{text-align:left} table.bt.mt td:not(:first-child),table.bt.mt th:not(:first-child){text-align:right}
table.bt.mt td:nth-child(2),table.bt.mt th:nth-child(2){text-align:left} table.bt.mt tbody tr:first-child td{background:#eaf3ec;font-weight:bold}
.demo img{width:100%;height:390px;object-fit:cover;object-position:top left;border:1px solid #333}
.tag{position:absolute;right:20px;top:18px;background:#A4123F;color:#fff;font-weight:bold;font-size:28px;padding:9px 16px;border-radius:6px}
.tc{position:absolute;top:110px;left:0;right:0;text-align:center}
.logo{height:72px}
.tt{color:#A4123F;font-weight:bold;font-size:38px;margin-top:16px}
.tsub{font-weight:bold;font-size:24px;margin-top:12px}
table.st{border-collapse:collapse;margin:18px auto 0;font-size:22px}
table.st td,table.st th{border:1.5px solid #000;padding:3px 18px}
table.st .l{text-align:left}
.tg{font-size:24px;margin-top:18px}.td{font-size:20px;margin-top:10px}
.appr{position:absolute;top:150px;left:66px;right:66px;bottom:70px;display:flex;align-items:center;justify-content:center}
.appr img{max-width:100%;max-height:100%;border:1px solid #333}
.ph{width:100%;height:100%;border:1.5px solid #000;flex-direction:column;align-items:center;justify-content:center;text-align:center}
.ph b{font-size:32px}.ph span{font-size:24px;margin-top:16px}
.refs{top:100px}
.ref{font-size:19px;color:#A4123F;margin:0 0 12px;padding-left:10px}
.thx{position:absolute;top:250px;left:0;right:0;text-align:center}
.thx div{color:#A4123F;font-weight:bold;font-size:86px}.thx span{font-weight:bold;font-size:34px;display:block;margin-top:20px}
"""
page = (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Final Review Presentation</title><style>{CSS}</style></head><body>'
        + "".join(html_slide(i, s) for i, s in enumerate(slides, 1))
        + '<script>var m=location.hash.match(/s=(\\d+)/);if(m){document.querySelectorAll(".slide").forEach(function(s,i){if(i+1!=+m[1])s.style.display="none"})}</script></body></html>')
(OUT / "final_presentation.html").write_text(page, encoding="utf-8")
print(N, "slides written")
