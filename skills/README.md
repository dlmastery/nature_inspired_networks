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
