# ICML 2027 LaTeX submission — `paper/icml2027/`

Conversion of [`PAPER.md`](../../PAPER.md) (318 lines, protocol-as-contribution
framing) into an ICML 2027 submission template.

## Files

| File | Purpose |
|---|---|
| `main.tex` | Paper body, 8 pages main + 4 appendices. Includes `\input{abstract.tex}`. |
| `abstract.tex` | Stand-alone abstract, 199 words, lifted from `paper/paper_abstract.md`. |
| `references.bib` | BibTeX, 25 citations in Rule-4 format (`Author YEAR VENUE 'Title' arXiv:XXXX.XXXXX`). |
| `icml2027.sty` | **PLACEHOLDER** style file. Replace with the official ICML 2027 stylesheet from icml.cc when available. |

## Page budget (8 pages main + supplementary appendix)

| Section | Pages |
|---|---:|
| §1 Introduction | 1.0 |
| §2 Dual-track audit + Fixer protocol | 1.5 |
| §3 Methods (composite, screening sweep, stats floor) | 1.0 |
| §4 Empirical calibration on 84-hypothesis substrate | 2.0 |
| §5 Audit calibration on 62 third-party hypotheses | 1.0 |
| §6 Limitations + threats to validity | 0.5 |
| §7 Related work | 0.5 |
| §8 Conclusion | 0.5 |
| References | 1.0 (counts as page 9; ICML allows refs to overflow) |
| Appendix A Statistical-tests-details | supplementary |
| Appendix B Per-hypothesis verdict table | supplementary |
| Appendix C Reviewer-revision status | supplementary |
| Appendix D Reproducibility pointers | supplementary |

## Build

Once the official `icml2027.sty` is dropped in, build with:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The placeholder `icml2027.sty` here approximates the ICML 2024 author kit
(geometry, fonts, `\icmltitle` / `\icmlauthor` / `\icmlcorrespondingauthor`
macros) so `main.tex` compiles to a near-camera-ready PDF for internal review.
Replace before submission.

## Compilation notes

- Two figures are placeholder `\framebox` blocks pointing to existing
  artifacts at `experiments/plot_pareto.png` and `experiments/plot_ablation.png`.
  Swap the `\framebox` lines for `\includegraphics{...}` after the camera-ready
  Pareto + 8-group ablation regenerate.
- `listings` is used for the H09 phi_budget mechanism-pinning test code block
  (Listing 1 in §2.3).
- All numerical claims trace to existing repository artifacts. The Phase-8
  $n{=}7$ certification numbers come from `paper/STATISTICAL_TESTS.md` §0–§3;
  the audit-calibration $p$-values come from
  `audits/AUDIT_CALIBRATION_THIRD_PARTY.md` §4.3.1 + `paper/STATISTICAL_TESTS.md` §8.

## Blind-review convention

The author block is `Anonymous Authors / Affiliation withheld for blind review`.
Replace with the actual author block in the camera-ready (post-acceptance) pass.
The repository URL in §Repository pointers is left as the live GitHub URL but
should be redacted in the blind-submission build (replace with
`\url{https://anonymous.4open.science/r/...}` or equivalent).

## Provenance

| Source | Used for |
|---|---|
| `PAPER.md` (318 lines) | §1 introduction, §2 protocol, §4 empirical, §5 calibration, §6 limitations, §7 related work, §8 conclusion, Appendix A–D |
| `paper/paper_abstract.md` | `abstract.tex` |
| `paper/STATISTICAL_TESTS.md` §0–§10 | §3 methods, §4 Phase-8 table, Appendix A |
| `audits/AUDIT_CALIBRATION_THIRD_PARTY.md` §4 | §5 audit calibration |
| `audits/ICML_REVIEWS_2026-05-30/REBUTTAL.md` | Appendix C reviewer-revision status |
| `paper/NATURE_INSPIRED_NETWORKS.md` References | `references.bib` |
| `CLAUDE.md` Rules 1–28 | §2 (Rules 20–28 cited) + §3 (Rule 2, Rule 26) |

## Honest framing constraints

- No SOTA claim. The baseline sits 6.5\,pp below 164-ep convergence-budget SOTA.
- Three Phase-8 candidates are *illustrative protocol output*, not standalone
  empirical headlines (§4 Phase-8, §6 limitations).
- Hill-climbed CI for `sg_only_phi_budget` *includes 0*; reported honestly.
- Iso-tuned-cell extension at $n{=}3$ confirms direction but cannot re-certify
  at NeurIPS $\alpha$ — Phase-9f filed as future work.
- Audit model-family-independence is NOT achieved (all agents are Claude
  Opus 4.7); cross-family re-audit is partial.
