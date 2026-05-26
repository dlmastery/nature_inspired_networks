# skills/

> Modular, **content-agnostic** auto-research skills extracted from
> `dlmastery/nature_inspired_networks`. None of these mention nature
> priors / sacred geometry / Platonic solids — they describe the
> *protocol* and *infrastructure* so any future project (tabular ML,
> medical OOD, FX prediction, anything) can pick them up.

## How to use

Each subdirectory is a self-contained skill. The `SKILL.md` inside is
the entry point — read it cover-to-cover before applying.

Skills are progressive: start with `autoresearch-experiment` if you
have one hypothesis to test; bring in the others as the project grows.

| skill | when to use |
|---|---|
| **autoresearch-experiment** | Running a single principled experiment with the 7-step ritual |
| **autoresearch-ablation-sweep** | Running an ablation matrix across many configs |
| **autoresearch-dashboard** | Generating the sortable HTML dashboard with Pareto / curves / topology panels |
| **autoresearch-reasoning-entry** | Authoring a citation-gated reasoning entry (Citation Rigor + Blob Completeness) |
| **autoresearch-modular-block** | Designing toggleable, ablation-friendly composable neural building blocks |
| **autoresearch-dataset-loader** | Wiring up benchmark datasets (CIFAR, MedMNIST, etc.) with cert workarounds |
| **autoresearch-topology-metrics** | Computing persistent-homology Betti curves, CKA, equivariance error |
| **autoresearch-experiment-archive** | The taxonomy structure for archiving each experiment in its own folder |
| **autoresearch-idea-scaffold** | Scaffolding a new idea sub-project from `_TEMPLATE/` |

## Design rules for these skills

1. **No domain content.** A skill that says "Chladni eigenmodes" or
   "Platonic equivariance" is leaking; rewrite it to say "filter
   initialization basis" or "group-equivariant layer."
2. **Each skill is one workflow.** If two workflows share 80%+ steps,
   they are the same skill — merge them. If a skill has more than ~5
   exit branches, split it.
3. **YAML frontmatter is mandatory.** `name` + `description` (one
   sentence on when to use). No optional fields needed unless the
   skill spawns Agents.
4. **Reference the canonical infra paths abstractly.** Use phrases
   like "the project's runner module" not
   `src/nature_inspired_networks/runner.py` — so the skill ports.

## Two cross-cutting disciplines every skill assumes

Every workflow above is wrapped in two project-wide disciplines that
the SKILL.md files reference but don't redefine each time:

- **Periodic GitHub checkpoint.** Commit + push on every milestone
  (file edit, test green, run-folder created, dashboard refresh),
  before AND after every background task launch, every ~15 min of
  active editing. Many small commits beat one big commit. Power
  outage must never lose progress. See
  `skills/autoresearch-checkpoint/SKILL.md` for the trigger table.
- **Test discipline.** Every new module/class/function ships with a
  unit test exercising shape, every Boolean-flag combination, and
  the bug class it was written to fix. Tests must pass before any
  training-loop background task launches. See
  `skills/autoresearch-modular-block/SKILL.md` for the mandatory
  smoke-test contract.
