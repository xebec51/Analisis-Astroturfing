# Repository Inventory Before Restructure

This inventory was generated read-only from Git tracked files before repository restructuring. Reference counts are static text references from reasonably sized text/code files; binary and large generated artifacts are not parsed for references.

- Tracked files before restructure: `424`
- Git LFS tracked paths: `1`
- Inventory rows including local audit/tooling observations: `429`

## Classification Counts (Tracked Files)

| classification | count |
|---|---:|
| CANONICAL_INPUT | 3 |
| CANONICAL_SOURCE | 13 |
| DEVELOPMENT_PROVENANCE | 73 |
| EXPLORATORY_OUTPUT | 3 |
| FROZEN_RESEARCH_ARTIFACT | 265 |
| GENERATED_CACHE | 1 |
| LEGACY_REQUIRED_FOR_REPRODUCIBILITY | 66 |

## High-Risk Immutable Groups

- Root input CSV files and `config/` are canonical inputs and must remain at their current paths.
- RM1, RM2 actor type, Community-Mass, comment similarity, and Sentiment V2 final outputs are frozen research artifacts.
- Sentiment V2 locked-test evaluation and model artifacts are immutable and must not be regenerated during repository cleanup.
- `output/rm2_comment_similarity/comment_similarity_pairs_all.csv` remains at its current LFS path.

## Files Requiring Special Handling

- `HCC.gephi`: Move to artifacts/network_projects with git mv (FROZEN_RESEARCH_ARTIFACT).
- `HCC_Visone.graphmlz`: Move to artifacts/network_projects with git mv (FROZEN_RESEARCH_ARTIFACT).
- `Type Actor.gephi`: Move to artifacts/network_projects with git mv (FROZEN_RESEARCH_ARTIFACT).
- `__pycache__/temporal_pipeline_code.cpython-312.pyc`: Delete tracked cache (GENERATED_CACHE).

## Unknown Files

No tracked files were classified as `UNKNOWN_REQUIRES_REVIEW` by the initial path audit.
