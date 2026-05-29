# Abstract — A Skeptical Protocol for Nature-Inspired Neural-Network Priors

We introduce **`nature_inspired_networks`**, an open autoresearch repository housing 84 nature-inspired neural-network hypotheses (φ-scaling, hexagonal lattices, Platonic / icosahedral equivariance, fractal recursion, toroidal closure, Chladni cymatic init, golden-angle modulation, and 15 cross-paradigm hybrids) across 8 thematic groups. Each hypothesis ships as a standalone pure-PyTorch primitive (~80 modules) with a committee-grade design doc, unit tests, and a SHA-256-fingerprinted composite metric.

**The methodological contribution** is the dual-track adversarial audit layered on top: a parallel 8-agent implementation-critic team verifies code-vs-doc correspondence (output: `audits/G<X>_audit.md`, four-tier verdicts PASS / MINOR / MAJOR / BROKEN), and a parallel 8-agent research-scientist-critic team independently challenges the scientific merit of each hypothesis (output: an addendum appended into every design doc, six-tier verdicts NOVEL+TESTABLE / DERIVATIVE+TESTABLE / NUMEROLOGY / FALSIFIED / UNFALSIFIABLE / INFRASTRUCTURE). An 8-agent Fixer campaign then patches every MAJOR / BROKEN finding with mechanism-verifying tests; every affected sweep row re-runs on the corrected code before any external claim is re-stated.

**Key empirical findings.** Across the 83 audited implementations, **51 % land non-PASS** (24 MINOR / 15 MAJOR / 3 BROKEN). Across 81 hypotheses scientifically critiqued, **exactly ONE rates NOVEL+TESTABLE** (H71 IcosaRoPE3D, never smoked yet); roughly half are decorative NUMEROLOGY where φ could be replaced by any constant in [1.3, 2.0] with the same outcome. The pre-audit "headline" — H09 phi_budget cross-dataset positive at 85.54 % CIFAR-10 / 58.05 % CIFAR-100 3-seed median — turned out to be produced by a network whose realized stage-parameter ratio was 1:1.41:2.45, not the claimed 1:φ:φ². The Fixer campaign corrected this; the post-fix re-run of the headline claim is in flight.

We make **no SOTA accuracy claim**. We claim (1) the protocol successfully catches headline claims produced by broken code BEFORE publication; (2) the audit + fixer + re-run cycle is portable to any autoresearch campaign as seven content-agnostic skills in `skills/`; (3) the strongest defensible category across 81 hypotheses is DERIVATIVE+TESTABLE — rediscoveries of established techniques under φ-flavoured rebranding — and the protocol successfully distinguishes them from numerology.

**The contribution is the protocol, not the priors.** Repo: [`dlmastery/nature_inspired_networks`](https://github.com/dlmastery/nature_inspired_networks). Live dashboard: [`https://dlmastery.github.io/nature_inspired_networks/`](https://dlmastery.github.io/nature_inspired_networks/).

---

*Composite metric:*
$$\textsf{composite} = \textsf{top-1} - 0.05\,\log_{10}(\textsf{params}_M) - 0.05\,\log_{10}(\textsf{latency}_{\text{ms}}).$$

*The composite formula is SHA-256-fingerprinted as `d65565e9c7b12d14cbce30a801ecc6753aea3eb148074256bfcc051fa61d0893`; editing it forces a `CompositeFingerprintError` (Rule 2).*
