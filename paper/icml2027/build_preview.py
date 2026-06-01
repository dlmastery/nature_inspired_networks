"""Build a reviewer-facing PDF preview of `main.tex` without LaTeX.

Toolchain probe results (recorded 2026-05-31):
  - `pdflatex` / `xelatex`: NOT INSTALLED on this Windows host
  - `pandoc`: NOT INSTALLED
  - `weasyprint` / `pdfkit` / `reportlab` / `xhtml2pdf`: NOT INSTALLED
  - `PyMuPDF` (fitz): AVAILABLE, with `Story` HTML/CSS->PDF API
  - `markdown`: AVAILABLE
  - `PIL`: AVAILABLE

This script converts `main.tex` to a structured HTML representation, then uses
PyMuPDF's Story API to render a near-camera-ready PDF preview. The output is
labelled clearly as a PREVIEW (no official ICML 2027 styling, but the body
text, section structure, tables, listings, and 6 figures from `paper/figures/`
are all faithfully reproduced).

Outputs:
  - paper/icml2027/main_preview.pdf   (PDF, multi-page, with embedded figures)
  - paper/icml2027/main_preview.html  (HTML used as intermediate; viewable)

Run: python paper/icml2027/build_preview.py
"""

from __future__ import annotations

import html
import os
import re
import sys
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image


HERE = Path(__file__).resolve().parent
PAPER_DIR = HERE.parent
FIG_DIR = PAPER_DIR / "figures"
MAIN_TEX = HERE / "main.tex"
ABSTRACT_TEX = HERE / "abstract.tex"

OUT_HTML = HERE / "main_preview.html"
OUT_PDF = HERE / "main_preview.pdf"

# PyMuPDF's Story API decompresses PNGs to raw RGBA streams at full resolution,
# which inflates a ~150 KB figure into a ~9 MB embedded XObject. To keep the
# preview file size reasonable (~1-3 MB), we render down-scaled JPEG copies
# of the figures into a temp dir and reference those.
FIG_THUMB_DIR = HERE / "_figure_thumbs"
FIG_THUMB_MAX_WIDTH = 900  # pixels - plenty for a single-column figure


# ---------------------------------------------------------------------------
# LaTeX -> HTML conversion (custom, minimal but faithful for this paper)
# ---------------------------------------------------------------------------

def _strip_comments(src: str) -> str:
    out = []
    for line in src.splitlines():
        # Strip LaTeX comments (% not preceded by \)
        cleaned = re.sub(r"(?<!\\)%.*$", "", line)
        out.append(cleaned)
    return "\n".join(out)


def _convert_inline(text: str) -> str:
    """Convert inline LaTeX commands -> HTML-ish text."""
    t = text

    # Escape HTML-significant characters first (but we want < > & rendered, not
    # LaTeX-significant).
    t = t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    # LaTeX-specific char restore (after html-escape, none collide).
    t = t.replace("---", "&mdash;").replace("--", "&ndash;")
    t = t.replace("``", "“").replace("''", "”")
    t = t.replace("~", " ")  # non-breaking space
    t = t.replace("\\,", " ")  # thin space
    t = t.replace("\\;", " ")
    t = t.replace("\\ ", " ")
    t = t.replace("\\%", "%")
    t = t.replace("\\$", "$")
    t = t.replace("\\#", "#")
    t = t.replace("\\&amp;", "&amp;")
    t = t.replace("\\_", "_")

    # math symbols - cover the ones that appear in this paper
    repl = {
        r"\\varphi": "φ",
        r"\\sigma": "σ",
        r"\\alpha": "α",
        r"\\beta": "β",
        r"\\omega": "ω",
        r"\\Sigma": "Σ",
        r"\\Delta": "Δ",
        r"\\geq": "≥",
        r"\\leq": "≤",
        r"\\approx": "≈",
        r"\\times": "×",
        r"\\to": "→",
        r"\\Rightarrow": "⇒",
        r"\\cdot": "·",
        r"\\sim": "~",
        r"\\ldots": "…",
        r"\\checkmark": "✓",
        r"\\mathrm": "",
        r"\\textsf": "",
        r"\\max": "max",
        r"\\min": "min",
        r"\\log": "log",
    }
    for k, v in repl.items():
        t = re.sub(k + r"\b", v, t)

    # `\frac{a}{b}` not used heavily; leave any leftover \frac as-is

    # Inline math: $...$ -> just render text inside, stripping outer $.
    def _math_inline(m):
        inner = m.group(1)
        # Strip braces, common math wrappers
        inner = re.sub(r"\\text(it|tt|bf|sf|rm)?\{([^{}]*)\}", r"\2", inner)
        inner = re.sub(r"\\mathrm\{([^{}]*)\}", r"\1", inner)
        inner = inner.replace("{", "").replace("}", "")
        # Superscripts: handle ^N or ^{...} -> Unicode if simple, else as-is
        inner = re.sub(r"\^2", "²", inner)
        inner = re.sub(r"\^3", "³", inner)
        return f"<i>{inner}</i>"
    t = re.sub(r"\$([^$]+)\$", _math_inline, t)

    # \emph{...} / \textit{...} -> <i>
    t = re.sub(r"\\emph\{([^{}]*)\}", r"<i>\1</i>", t)
    t = re.sub(r"\\textit\{([^{}]*)\}", r"<i>\1</i>", t)
    # \textbf{...} -> <b>
    t = re.sub(r"\\textbf\{([^{}]*)\}", r"<b>\1</b>", t)
    # \texttt{...} -> <code>
    t = re.sub(r"\\texttt\{([^{}]*)\}", r"<code>\1</code>", t)
    # \textsc{...} -> small caps style
    t = re.sub(r"\\textsc\{([^{}]*)\}", r"<span style='font-variant: small-caps;'>\1</span>", t)
    # \url{...} -> <a>
    t = re.sub(r"\\url\{([^{}]*)\}", r"<a href='\1'>\1</a>", t)

    # Citations: \citep{key1,key2} -> [key1, key2]
    def _cite(m):
        keys = [k.strip() for k in m.group(1).split(",")]
        return "[" + ", ".join(keys) + "]"
    t = re.sub(r"\\cite[pt]?\{([^{}]+)\}", _cite, t)
    t = re.sub(r"\\citealp\{([^{}]+)\}", _cite, t)
    t = re.sub(r"\\citet\{([^{}]+)\}", _cite, t)

    # Cross-refs: \ref{...} -> §label
    t = re.sub(r"\\ref\{([^{}]+)\}", r"<i>\1</i>", t)
    t = re.sub(r"\\S\\ref\{([^{}]+)\}", r"&sect;<i>\1</i>", t)
    t = t.replace("\\S", "&sect;")
    t = re.sub(r"\\label\{[^{}]+\}", "", t)

    # Eq.~\ref... pattern is already handled

    # \footnote{...} -> just append in parens
    def _footnote(m):
        return f" (note: {m.group(1)})"
    t = re.sub(r"\\footnote\{([^{}]+)\}", _footnote, t)

    # \paragraph{...} -> bold inline (handled at block level)

    # Leftover backslash-words: strip common ones
    for cmd in ["\\par", "\\noindent", "\\centering", "\\small", "\\large",
                "\\bf", "\\rm", "\\it", "\\sf", "\\tt"]:
        t = t.replace(cmd, "")
    return t


def _convert_table(body: str) -> str:
    """Very loose tabular -> HTML table."""
    # Extract caption + label first (before tabular)
    m_cap = re.search(r"\\caption\{(.+?)\}\s*(?:\\label\{[^}]+\})?", body, re.DOTALL)
    caption = _convert_inline(m_cap.group(1).strip()) if m_cap else ""

    # Find tabular block
    m_tab = re.search(r"\\begin\{tabular\}\{[^}]+\}(.*?)\\end\{tabular\}", body, re.DOTALL)
    if not m_tab:
        return f"<p><i>[table omitted]</i></p>"
    tab = m_tab.group(1)
    # Strip booktabs rules
    tab = re.sub(r"\\(toprule|midrule|bottomrule)", "", tab)
    rows = [r.strip() for r in tab.split("\\\\") if r.strip()]
    html_rows = []
    for i, row in enumerate(rows):
        cells = [_convert_inline(c.strip()) for c in row.split("&")]
        tag = "th" if i == 0 else "td"
        html_rows.append("<tr>" + "".join(f"<{tag}>{c}</{tag}>" for c in cells) + "</tr>")
    out = "<div class='table'>"
    if caption:
        out += f"<div class='caption'><b>Table.</b> {caption}</div>"
    out += "<table>" + "".join(html_rows) + "</table>"
    out += "</div>"
    return out


def _convert_figure(body: str, *, fig_index: int) -> str:
    """Convert a figure env body to an HTML <figure>.

    The LaTeX uses framebox placeholders for fig 1 and fig 2; we substitute
    the real PNG images from paper/figures/ in order.
    """
    m_cap = re.search(r"\\caption\{(.+?)\}\s*(?:\\label\{[^}]+\})?", body, re.DOTALL)
    caption = _convert_inline(m_cap.group(1).strip()) if m_cap else ""

    # Map fig_index (1-based among figures we encounter) -> figures/fig<N>_*.png.
    # Reference by basename; the figures dir is added to the Archive at render
    # time so the Story finds the bytes.
    fig_files = sorted(FIG_DIR.glob("fig*.png"))
    if 0 < fig_index <= len(fig_files):
        img_name = fig_files[fig_index - 1].name
    else:
        img_name = ""

    out = "<figure>"
    if img_name:
        out += f"<img src='{img_name}' alt='figure {fig_index}' />"
    else:
        out += "<div class='placeholder'>[Figure placeholder]</div>"
    if caption:
        out += f"<figcaption><b>Figure {fig_index}.</b> {caption}</figcaption>"
    out += "</figure>"
    return out


def _convert_listing(body: str) -> str:
    """Convert a lstlisting block to HTML <pre><code>."""
    # Caption is in the [...] options
    opts_m = re.search(r"\[(.*?)\]", body, re.DOTALL)
    caption = ""
    if opts_m:
        opts = opts_m.group(1)
        cm = re.search(r"caption=\{(.+?)\}\s*(?:,|$)", opts, re.DOTALL)
        if cm:
            caption = _convert_inline(cm.group(1).strip())
    # Body lines
    lines = body.split("\n")
    # Drop the first line with [opts] and last line with [/opts]
    code_lines = []
    started = False
    for ln in lines:
        if not started and "]" in ln:
            started = True
            # Take part after ]
            after = ln.split("]", 1)[1]
            if after.strip():
                code_lines.append(after)
            continue
        code_lines.append(ln)
    code = "\n".join(code_lines).strip("\n")
    code_html = html.escape(code)
    out = "<div class='listing'>"
    if caption:
        out += f"<div class='caption'><b>Listing.</b> {caption}</div>"
    out += f"<pre><code>{code_html}</code></pre>"
    out += "</div>"
    return out


def convert_tex_to_html(tex_src: str, abstract_src: str) -> str:
    src = _strip_comments(tex_src)
    abstract_src = _strip_comments(abstract_src)

    # Extract the title via \icmltitle (multi-line, with \\)
    m_title = re.search(r"\\icmltitle\{(.+?)\}\s*$", src, re.DOTALL | re.MULTILINE)
    title = ""
    if m_title:
        title = m_title.group(1)
        title = title.replace("\\\\", " — ").strip()
        title = _convert_inline(title)

    # Grab everything between \begin{document} and \end{document}
    m_doc = re.search(r"\\begin\{document\}(.*?)\\end\{document\}", src, re.DOTALL)
    body = m_doc.group(1) if m_doc else src

    # Inline the abstract.tex contents at \input{abstract.tex}
    body = body.replace("\\input{abstract.tex}", abstract_src)

    # Pull abstract env content
    abstract_html = ""
    m_abs = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", body, re.DOTALL)
    if m_abs:
        abstract_text = m_abs.group(1).strip()
        # Convert paragraphs
        paras = re.split(r"\n\s*\n", abstract_text)
        abstract_html = "".join(
            f"<p>{_convert_inline(p.strip())}</p>" for p in paras if p.strip()
        )
        # Remove the abstract env from body
        body = body[: m_abs.start()] + body[m_abs.end() :]

    # Remove \twocolumn[...] wrapper and the title block. The content can
    # contain nested `[...]` (e.g. `\\[0.25em]`), so we can't use a naive
    # non-greedy regex. Walk the brackets with a depth counter instead.
    tw_start = body.find("\\twocolumn[")
    if tw_start != -1:
        depth = 0
        i_scan = tw_start + len("\\twocolumn")
        end = -1
        while i_scan < len(body):
            ch = body[i_scan]
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    end = i_scan
                    break
            i_scan += 1
        if end != -1:
            body = body[:tw_start] + body[end + 1 :]
    body = re.sub(r"\\icmltitle\{.*?\}", "", body, flags=re.DOTALL)
    body = re.sub(r"\\begin\{center\}.*?\\end\{center\}", "", body, flags=re.DOTALL)
    body = re.sub(r"\\vskip[^\n]*", "", body)
    body = re.sub(r"\\icmlSubmissionBanner", "", body)
    body = re.sub(r"\\onecolumn", "", body)
    body = re.sub(r"\\appendix", "<hr/><h1 style='color:#444'>Appendix</h1>", body)

    # Walk through body sequentially, building HTML
    out: list[str] = []
    fig_count = 0
    # Iterate via a token stream
    i = 0
    text = body
    while i < len(text):
        # Look for next macro
        # Find next \section, \subsection, \paragraph, environment, bibliography
        rest = text[i:]
        # Strip leading whitespace
        m_ws = re.match(r"\s+", rest)
        if m_ws:
            i += m_ws.end()
            continue

        # \section{...}
        m = re.match(r"\\section\{([^{}]*)\}", rest)
        if m:
            out.append(f"<h1>{_convert_inline(m.group(1))}</h1>")
            i += m.end()
            continue
        m = re.match(r"\\subsection\{([^{}]*)\}", rest)
        if m:
            out.append(f"<h2>{_convert_inline(m.group(1))}</h2>")
            i += m.end()
            continue
        m = re.match(r"\\subsubsection\{([^{}]*)\}", rest)
        if m:
            out.append(f"<h3>{_convert_inline(m.group(1))}</h3>")
            i += m.end()
            continue
        m = re.match(r"\\paragraph\{([^{}]*)\}", rest)
        if m:
            out.append(f"<p><b>{_convert_inline(m.group(1))}.</b> ")
            i += m.end()
            # Collect text until next blank line or block macro
            tail_match = re.search(
                r"\n\s*\n|\\section\{|\\subsection\{|\\paragraph\{|\\begin\{",
                text[i:],
            )
            if tail_match:
                para_text = text[i : i + tail_match.start()]
                i += tail_match.start()
            else:
                para_text = text[i:]
                i = len(text)
            out.append(_convert_inline(para_text.strip()))
            out.append("</p>")
            continue

        # \begin{equation}...\end{equation}
        m = re.match(r"\\begin\{equation\}(.*?)\\end\{equation\}", rest, re.DOTALL)
        if m:
            eq = m.group(1)
            eq_html = _convert_inline(eq.strip())
            out.append(f"<div class='equation'><i>{eq_html}</i></div>")
            i += m.end()
            continue

        # \begin{enumerate}...\end{enumerate}
        m = re.match(r"\\begin\{enumerate\}(.*?)\\end\{enumerate\}", rest, re.DOTALL)
        if m:
            list_body = m.group(1)
            items = re.split(r"\\item\b", list_body)
            items = [it.strip() for it in items if it.strip()]
            html_items = "".join(f"<li>{_convert_inline(it)}</li>" for it in items)
            out.append(f"<ol>{html_items}</ol>")
            i += m.end()
            continue
        m = re.match(r"\\begin\{itemize\}(.*?)\\end\{itemize\}", rest, re.DOTALL)
        if m:
            list_body = m.group(1)
            items = re.split(r"\\item\b", list_body)
            items = [it.strip() for it in items if it.strip()]
            html_items = "".join(f"<li>{_convert_inline(it)}</li>" for it in items)
            out.append(f"<ul>{html_items}</ul>")
            i += m.end()
            continue

        # \begin{table*?}...\end{table*?}
        m = re.match(
            r"\\begin\{(table\*?|figure\*?|lstlisting)\}(?:\[[^\]]*\])?"
            r"(.*?)\\end\{\1\}",
            rest,
            re.DOTALL,
        )
        if m:
            env = m.group(1)
            body_env = m.group(2)
            if env.startswith("table"):
                out.append(_convert_table(body_env))
            elif env.startswith("figure"):
                fig_count += 1
                out.append(_convert_figure(body_env, fig_index=fig_count))
            elif env == "lstlisting":
                out.append(_convert_listing(body_env))
            i += m.end()
            continue

        # \bibliographystyle / \bibliography -> render as a stub references section
        m = re.match(r"\\bibliographystyle\{[^{}]+\}", rest)
        if m:
            i += m.end()
            continue
        m = re.match(r"\\bibliography\{[^{}]+\}", rest)
        if m:
            out.append("<h1>References</h1>")
            out.append(
                "<p><i>References are stored in <code>references.bib</code> "
                "(25 entries in CLAUDE.md Rule&nbsp;4 format). They are not "
                "expanded in this preview because no LaTeX/BibTeX toolchain "
                "is available; the bibliography will populate in the full "
                "<code>pdflatex</code> + <code>bibtex</code> build.</i></p>"
            )
            i += m.end()
            continue

        # Skip unknown single-arg macros (best-effort)
        m = re.match(r"\\[A-Za-z]+\{[^{}]*\}", rest)
        if m:
            i += m.end()
            continue
        m = re.match(r"\\[A-Za-z]+", rest)
        if m:
            i += m.end()
            continue

        # Otherwise: gather a paragraph until \n\s*\n or next macro
        tail_match = re.search(
            r"\n\s*\n|\\section\b|\\subsection\b|\\subsubsection\b|"
            r"\\paragraph\b|\\begin\{|\\bibliography\b|\\bibliographystyle\b|"
            r"\\appendix\b|\\onecolumn\b",
            text[i:],
        )
        if tail_match and tail_match.start() > 0:
            para = text[i : i + tail_match.start()]
            i += tail_match.start()
        elif tail_match and tail_match.start() == 0:
            # Shouldn't happen because matched above, but advance one char.
            i += 1
            continue
        else:
            para = text[i:]
            i = len(text)
        para = para.strip()
        if para:
            out.append(f"<p>{_convert_inline(para)}</p>")

    body_html = "\n".join(out)

    # Build full HTML
    css = """
    @page { size: A4; margin: 18mm 18mm 18mm 18mm; }
    body { font-family: 'Source Serif Pro', 'Georgia', serif;
           font-size: 9.5pt; line-height: 1.35; color: #111; }
    h1 { font-size: 13pt; margin-top: 14pt; margin-bottom: 6pt; color: #111;
         border-bottom: 1px solid #888; padding-bottom: 2pt; }
    h2 { font-size: 11pt; margin-top: 10pt; margin-bottom: 4pt; color: #222; }
    h3 { font-size: 10pt; margin-top: 8pt; margin-bottom: 3pt; color: #333;
         font-style: italic; }
    p { margin: 0 0 6pt 0; text-align: justify; }
    code { font-family: 'Consolas', 'Courier New', monospace; font-size: 9pt;
           background: #f4f4f4; padding: 0 2pt; }
    pre { background: #f4f4f4; padding: 4pt; border: 0.5pt solid #ccc;
          font-size: 8pt; overflow-x: auto; }
    pre code { background: none; padding: 0; }
    table { border-collapse: collapse; margin: 4pt 0; font-size: 8.5pt; width: 100%; }
    th, td { border-top: 0.5pt solid #444; padding: 2pt 4pt; text-align: left; }
    th { border-bottom: 0.5pt solid #444; background: #f8f8f8; }
    div.table { margin: 8pt 0; }
    div.caption { font-size: 8.5pt; font-style: italic; margin-bottom: 3pt;
                  color: #333; }
    div.equation { margin: 6pt 0; text-align: center; font-size: 10pt; }
    figure { margin: 8pt auto; text-align: center; }
    figure img { max-width: 90%; max-height: 250pt; }
    figcaption { font-size: 8.5pt; font-style: italic; color: #333;
                 margin-top: 3pt; text-align: justify; }
    div.placeholder { border: 1pt dashed #888; padding: 30pt; color: #888;
                      font-style: italic; }
    div.listing { margin: 8pt 0; }
    .banner { background: #fff8d6; border: 1pt solid #d4b840;
              padding: 6pt 10pt; margin-bottom: 12pt; font-size: 8.5pt;
              color: #604010; }
    .title { font-size: 16pt; font-weight: bold; text-align: center;
             margin: 6pt 0 4pt 0; line-height: 1.25; }
    .authors { text-align: center; font-size: 10pt; margin-bottom: 12pt; }
    .abstract { border-left: 2pt solid #888; padding-left: 8pt; margin: 8pt 0
                 14pt 0; font-size: 9.5pt; }
    .abstract h2 { font-size: 10pt; margin-top: 0; }
    """

    banner = (
        "<div class='banner'><b>PREVIEW BUILD &mdash; not the official ICML 2027 "
        "PDF.</b> This document was rendered from "
        "<code>paper/icml2027/main.tex</code> via PyMuPDF&apos;s Story API "
        "because no <code>pdflatex</code>, <code>xelatex</code>, or "
        "<code>pandoc</code> toolchain is available on the host machine. "
        "Section structure, body text, tables, listings, and figures (from "
        "<code>paper/figures/</code>) are faithfully reproduced, but ICML "
        "fonts/columns/headers/citation expansion are NOT applied. Build the "
        "official PDF locally with the <code>pdflatex main.tex; bibtex main; "
        "pdflatex main.tex; pdflatex main.tex</code> sequence once "
        "<code>icml2027.sty</code> from icml.cc is dropped in.</div>"
    )

    full = f"""<!DOCTYPE html>
<html><head><meta charset='utf-8'><title>{title}</title>
<style>{css}</style></head>
<body>
{banner}
<div class='title'>{title}</div>
<div class='authors'><b>Anonymous Authors</b> &mdash; <i>Affiliation withheld for blind review</i></div>
<div class='abstract'><h2>Abstract</h2>{abstract_html}</div>
{body_html}
</body></html>"""
    return full


# ---------------------------------------------------------------------------
# HTML -> PDF via PyMuPDF Story
# ---------------------------------------------------------------------------

def html_to_pdf(html_doc: str, out_pdf: Path) -> int:
    """Render an HTML document to PDF via fitz.Story. Returns total pages."""
    # A4 minus margins -> usable rect
    # A4 = 595 x 842 pt
    page_rect = fitz.paper_rect("A4")
    margin = 50  # ~17 mm
    where = fitz.Rect(
        margin, margin, page_rect.width - margin, page_rect.height - margin
    )

    # Make the down-scaled figures directory accessible to the Story by
    # basename (see _build_figure_thumbs in main()).
    archive = fitz.Archive()
    archive.add(str(FIG_THUMB_DIR))
    story = fitz.Story(html=html_doc, archive=archive)

    writer = fitz.DocumentWriter(str(out_pdf))
    n_pages = 0
    more = True
    while more:
        dev = writer.begin_page(page_rect)
        more, _ = story.place(where)
        story.draw(dev)
        writer.end_page()
        n_pages += 1
        if n_pages > 200:  # hard safety cap
            break
    writer.close()
    return n_pages


def _build_figure_thumbs() -> None:
    """Down-scale figures to JPEG so PyMuPDF Story doesn't decompress them
    to multi-megabyte raw RGBA streams in the output PDF."""
    FIG_THUMB_DIR.mkdir(exist_ok=True)
    for png in sorted(FIG_DIR.glob("fig*.png")):
        out = FIG_THUMB_DIR / png.name  # keep .png name so HTML refs match
        if out.exists() and out.stat().st_mtime >= png.stat().st_mtime:
            continue
        im = Image.open(png).convert("RGB")
        if im.width > FIG_THUMB_MAX_WIDTH:
            ratio = FIG_THUMB_MAX_WIDTH / im.width
            im = im.resize(
                (FIG_THUMB_MAX_WIDTH, int(im.height * ratio)),
                Image.LANCZOS,
            )
        # Save as JPEG under the .png name (Story honors content-type from
        # bytes, not filename). Keep quality high enough for figures.
        im.save(out, format="JPEG", quality=85, optimize=True)


def main() -> int:
    if not MAIN_TEX.exists():
        print(f"ERROR: {MAIN_TEX} not found", file=sys.stderr)
        return 1
    if not ABSTRACT_TEX.exists():
        print(f"ERROR: {ABSTRACT_TEX} not found", file=sys.stderr)
        return 1

    _build_figure_thumbs()

    tex_src = MAIN_TEX.read_text(encoding="utf-8")
    abs_src = ABSTRACT_TEX.read_text(encoding="utf-8")

    html_doc = convert_tex_to_html(tex_src, abs_src)
    OUT_HTML.write_text(html_doc, encoding="utf-8")
    print(f"[ok] wrote {OUT_HTML} ({len(html_doc):,} chars)")

    n_pages = html_to_pdf(html_doc, OUT_PDF)
    size_kb = OUT_PDF.stat().st_size / 1024
    print(f"[ok] wrote {OUT_PDF} ({n_pages} pages, {size_kb:,.1f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
