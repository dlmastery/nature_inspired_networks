# Build status — `paper/icml2027/`

_Generated: 2026-06-01 by the paper-PDF-builder agent._

## Toolchain probe (Windows 11, this host)

| Tool | Installed? | Path |
|---|---|---|
| `pdflatex` | **NO** | `where pdflatex` returned non-zero |
| `xelatex` | **NO** | `where xelatex` returned non-zero |
| `pandoc` | **NO** | `where pandoc` returned non-zero |
| `weasyprint` (Py) | NO | `ModuleNotFoundError` |
| `pdfkit` (Py) | NO | `ModuleNotFoundError` |
| `reportlab` (Py) | NO | `ModuleNotFoundError` |
| `xhtml2pdf` (Py) | NO | `ModuleNotFoundError` |
| `playwright` / `pyppeteer` | NO | `ModuleNotFoundError` |
| **`PyMuPDF` (`fitz`)** | **YES** (1.27.1) | `Story` HTML/CSS→PDF API available |
| `Pillow` | YES (12.2.0) | used to down-scale figures |
| `markdown` | YES (3.4.1) | not used in current build path |

No LaTeX, pandoc, or HTML-to-PDF system tool is available on this host.
The preview falls back to a custom LaTeX→HTML→PyMuPDF-Story→PDF chain.

## Preview output

| File | Description | Size |
|---|---|---:|
| `main_preview.pdf` | Reviewer-facing PDF preview (10 pages) | ~1.4 MB |
| `main_preview.html` | Intermediate HTML (also human-viewable) | ~40 KB |
| `build_preview.py` | Build script (run anytime to regenerate) | — |
| `_figure_thumbs/` | Down-scaled JPEG copies of `paper/figures/*.png` (renamed `.png` to keep refs simple); kept so the build is incremental | — |

Build with: `python paper/icml2027/build_preview.py`

The preview contains:

- Title, anonymous-author block, abstract
- All 8 main sections + 4 appendices
- All tables (Track-A verdict distribution, Phase-8 certification,
  calibration comparison, per-seed CIFAR-100, fixer commits,
  reviewer-revision status, verdict tier counts)
- All 1 listing (the Fixer-PhiScaling mechanism-pinning test, Listing 1)
- Both LaTeX `\framebox` figure placeholders **replaced with the real
  PNG figures** from `paper/figures/` (`fig1_pareto.png`,
  `fig2_ablation_groups.png`) — the LaTeX source only references two
  `\begin{figure}` envs; the remaining 4 PNGs in `paper/figures/` are
  intended for later sections and are not yet referenced in `main.tex`.

## What this preview is NOT

- **NOT** the official ICML 2027 camera-ready format. ICML's
  two-column layout, font choices, header/footer banner, page-number
  format, and `\citep` expansion via BibTeX are NOT applied. A clear
  "PREVIEW BUILD" banner at the top of page 1 says so.
- **NOT** a `bibtex`-expanded references list. The bibliography
  appears as a one-paragraph stub explaining that `references.bib`
  must be processed by `bibtex` in the official build.

## How to build the OFFICIAL PDF

Once a machine has TeX Live (or MikTeX on Windows) installed, the
official `icml2027.sty` from <https://icml.cc> can be dropped into
this directory, and the standard four-step build runs:

```bash
cd paper/icml2027
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Until then, `main_preview.pdf` serves as the reviewer-facing artefact.

## Files written by this build

- `paper/icml2027/main_preview.pdf` (committed)
- `paper/icml2027/main_preview.html` (committed)
- `paper/icml2027/build_preview.py` (committed; self-contained, no
  network deps beyond stdlib + PyMuPDF + PIL)
- `paper/icml2027/_figure_thumbs/fig*.png` (committed; ~280 KB total,
  6 down-scaled JPEGs renamed to `.png` so the `<img src>` in the
  generated HTML matches)
- `paper/icml2027/BUILD_STATUS.md` (this file)

## Files that were NOT modified

Per the task constraints, NONE of `main.tex`, `abstract.tex`,
`references.bib`, `icml2027.sty`, or `README.md` were modified.
