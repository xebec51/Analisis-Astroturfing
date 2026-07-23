# Repository Structure

This repository is organized as a research pipeline with frozen outputs. Repository cleanup must preserve
scientific artifacts and avoid rerunning analysis.

## Top-Level Layout

| path | role | notes |
|---|---|---|
| `dataset.csv` | canonical input | Root path retained for reproducibility and historical references. |
| `video_metadata_clean.csv` | canonical input | Root path retained for reproducibility and historical references. |
| `config/` | canonical configuration | Contains `individual_actor_registry.csv`. |
| `notebooks/` | notebook entrypoints | Target location for canonical and legacy notebooks after restructure. |
| `scripts/` | script entrypoints and utilities | Scripts stay flat unless moving them would not break imports. |
| `docs/` | repository documentation | Includes structure, entrypoints, artifact registry, reproducibility, and audit outputs. |
| `artifacts/` | project-level artifacts | Network project files such as Gephi and Visone projects. CSV files remain canonical data. |
| `archive/` | historical material | Not a trash folder; used only for documented legacy files. |
| `output/` | research outputs | Contains frozen, legacy, exploratory, and repository-integrity outputs. |
| `tests/` | tests | Read-only tests are preferred for repository cleanup. |

## Notebook Layout

Target notebook locations:

| path | status | notes |
|---|---|---|
| `notebooks/rm1/tiktok_coordination_analysis.ipynb` | canonical RM1 | LCN, Louvain, FSA_V, HCC, and RM1 outputs. |
| `notebooks/rm2/03_rm2_actor_type_typology.ipynb` | canonical RM2 actor type | Actor type and Gephi actor-type visualization. |
| `notebooks/legacy/02_rm2_sentiment_goals.ipynb` | legacy V1 sentiment/goals | Superseded by Sentiment V2 final scripts and outputs. |

Notebook files are not executed during repository cleanup. Static path changes are limited to project-root
resolution, input paths, output paths, and documentation of output contracts.

## Output Contract

Canonical output roots are intentionally stable:

| module | output roots |
|---|---|
| RM1 main | `output/tables/`, `output/gephi/`, `output/visualisasi/` |
| RM1 temporal | `output/rm1_temporal/tables/`, `output/rm1_temporal/visualisasi/` |
| RM2 actor type | `output/rm2_actor_type/tables/`, `output/rm2_actor_type/gephi/`, `output/rm2_actor_type/visualisasi/`, `output/rm2_actor_type/audit/` |
| RM2 Community-Mass | `output/rm2_actor_type/account_interaction/`, `output/rm2_actor_type/direct_interaction/` |
| RM2 comment similarity | `output/rm2_comment_similarity/`, `output/rm2_comment_similarity/presentation/` |
| RM2 sentiment legacy | `output/rm2_sentiment/tables/`, `output/rm2_sentiment/gephi/`, `output/rm2_sentiment/visualisasi/`, `output/rm2_sentiment/visualisasi_exploratory/`, `output/rm2_sentiment/human_validation/` |
| RM2 sentiment V2 | `output/rm2_sentiment/human_validation_v2/`, `output/rm2_sentiment/model_v2/`, `output/rm2_sentiment/final_v2/` |
| Repository audit | `docs/repository_audit/`, `output/repository_integrity/` |

New analysis outputs should not be written directly to the repository root. Generic folders such as
`results/`, `figures/`, `exports/`, and `tmp/` are not part of the contract unless explicitly documented.

## Immutable Artifacts

Frozen outputs, locked-test files, model artifacts, final predictions, final manifests, LCN/HCC outputs,
Actor Type outputs, Community-Mass outputs, and comment-similarity outputs are treated as immutable during
repository organization. Moving a project file must preserve SHA-256 content hashes.

## Interpretation Boundary

Repository structure documentation does not change scientific interpretation. Terms should remain conservative:
observed coordination evidence, observed association, observed message orientation, video brand context, and
textual similarity. These are not claims of payment, affiliation, control, causal influence, or intentional
coordination.
