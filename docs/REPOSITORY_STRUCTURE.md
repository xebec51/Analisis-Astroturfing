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
| `output/` | research outputs | Contains frozen, legacy, and exploratory outputs. |
| `tests/` | tests | Read-only tests are preferred for repository cleanup. |

## Notebook Layout

Target notebook locations:

| path | status | notes |
|---|---|---|
| `notebooks/rm1/tiktok_coordination_analysis.ipynb` | canonical RM1 | LCN, Louvain, FSA_V, HCC, and RM1 outputs. |
| `notebooks/rm2/02_rm2_sentiment_analysis.ipynb` | canonical RM2 sentiment reporting | Reads frozen final sentiment outputs and regenerates final PNG visuals. |
| `notebooks/rm2/03_rm2_actor_type_typology.ipynb` | canonical RM2 actor type | Actor type and Gephi actor-type visualization. |
| `notebooks/legacy/02_rm2_sentiment_goals.ipynb` | legacy V1 sentiment/goals | Superseded by Sentiment V2 final scripts and outputs. |

Network/model notebooks are not executed during repository cleanup. The final sentiment reporting notebook is
safe to rerun because it reads frozen final CSV/JSON artifacts and refreshes PNG figures only.

## Output Contract

Canonical output roots are intentionally stable:

| module | output roots |
|---|---|
| RM1 main | `output/tables/`, `output/gephi/`, `output/visualisasi/` |
| RM1 temporal | `output/rm1_temporal/tables/`, `output/rm1_temporal/visualisasi/` |
| RM2 actor type | `output/rm2_actor_type/tables/`, `output/rm2_actor_type/gephi/`, `output/rm2_actor_type/visualisasi/`, `output/rm2_actor_type/audit/` |
| RM2 Community-Mass | `output/rm2_actor_type/account_interaction/`, `output/rm2_actor_type/direct_interaction/` |
| RM2 comment similarity | `output/rm2_comment_similarity/`, `output/rm2_comment_similarity/presentation/` |
| RM2 sentiment final | `output/rm2_sentiment/final/`, `output/rm2_sentiment/final/tables/`, `output/rm2_sentiment/final/visualisasi/`, `output/rm2_sentiment/final/gephi/` |
| RM2 sentiment validation/model | `output/rm2_sentiment/validation/human_v1/`, `output/rm2_sentiment/validation/human_v2/`, `output/rm2_sentiment/model/frozen/` |
| RM2 sentiment legacy references | `output/rm2_sentiment/legacy/v1/visualisasi/`, `output/rm2_sentiment/legacy/exploratory/visualisasi/`, selected non-CSV reference files in `output/rm2_sentiment/legacy/v1/tables/` |
| Repository audit | `docs/repository_audit/` |

New analysis outputs should not be written directly to the repository root. Generic folders such as
`results/`, `figures/`, `exports/`, and `tmp/` are not part of the contract unless explicitly documented.

Bulky RM2 Sentiment V1 CSV outputs are intentionally trimmed from the tracked repository. The legacy V1
notebook is archived and must not be used to regenerate those CSV files; final sentiment reporting should use
`notebooks/rm2/02_rm2_sentiment_analysis.ipynb`.

## Immutable Artifacts

Frozen outputs, locked-test files, model artifacts, final predictions, final manifests, LCN/HCC outputs,
Actor Type outputs, Community-Mass outputs, and comment-similarity outputs are treated as immutable during
repository organization. Moving a project file must preserve SHA-256 content hashes.

## Interpretation Boundary

Repository structure documentation does not change scientific interpretation. Terms should remain conservative:
observed coordination evidence, observed association, observed message orientation, video brand context, and
textual similarity. These are not claims of payment, affiliation, control, causal influence, or intentional
coordination.
