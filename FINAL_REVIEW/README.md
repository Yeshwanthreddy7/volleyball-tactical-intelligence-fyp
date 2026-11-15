# FINAL_REVIEW – Team 56

Files for the Project Phase II external review.

| Folder | What is inside |
|---|---|
| `presentation/` | `final_presentation.pdf` (present this), `final_presentation.pptx` (same slides for PowerPoint), `final_presentation.tex` + `ref.bib` (official Amrita Beamer template, compile on Overleaf) |
| `report/` | `final_report.pdf` (submit this), `final_report.tex` + `ref.bib` (official Amrita report template, compile on Overleaf) |
| `figures/` | logo, architecture and pipeline diagrams, app screenshots. Add `guide_approval.png` here. |
| `evidence/` | scripts and outputs behind every number: functional API tests, pytest result, model evaluation |
| `src/` | `content.py` holds all text; `build_presentation.py` and `build_report.py` rebuild everything |

## Rebuild after editing `src/content.py`

```
python src/build_presentation.py
python src/build_report.py
```
Then print the two `.html` files to PDF in Chrome, or upload the `.tex`, `ref.bib` and `figures/` to Overleaf.

## On Overleaf
Upload the `presentation/` (or `report/`) folder together with `figures/` so that the path `../figures/` works,
or move the figures next to the `.tex` file. Compile with pdfLaTeX + BibTeX.
