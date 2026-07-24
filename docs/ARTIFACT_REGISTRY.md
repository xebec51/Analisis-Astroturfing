# Artifact Registry

Per-file inventory values from the repository restructure audit are stored in
`docs/repository_audit/repository_inventory_before.csv`. This document lists the canonical artifact groups
used by the research pipeline.

| artifact | path | module | producer | consumer | status | immutable | hash | Git LFS | notes |
|---|---|---|---|---|---|---|---|---|---|
| Comment dataset | `dataset.csv` | Input | Source collection | RM1 and RM2 | CANONICAL_INPUT | Yes | `7e0ea9cbc82243445cccea1a14ccb847c61235e932f754f5e1bc42e390df32da` | No | Retained at root. |
| Video metadata | `video_metadata_clean.csv` | Input | Source collection | RM1 and RM2 | CANONICAL_INPUT | Yes | `92d4fdf7e75dacbd658871a8503829af6a151d83005c2a6716298019bb9ad094` | No | Retained at root. |
| Individual actor registry | `config/individual_actor_registry.csv` | Config | Manual registry | RM2 actor type | CANONICAL_INPUT | Yes | `debb299c4a4255793d7a31165e9aef319b79d613559d350c9fb24928f5bbd4d7` | No | Retained in `config/`. |
| LCN nodes | `output/gephi/gephi_lcn_nodes.csv` | RM1 | RM1 notebook | RM2 downstream, Gephi | FROZEN_RESEARCH_ARTIFACT | Yes | `4c18fa649771899e334ad53d6044031458f0822032cab54f229e1aa989643cd2` | No | Expected 724 nodes. |
| LCN edges | `output/gephi/gephi_lcn_edges.csv` | RM1 | RM1 notebook | RM2 downstream, Gephi | FROZEN_RESEARCH_ARTIFACT | Yes | `6b07f482a142d1151decd043cfd3cc465723491399642fc30882a567b7d2298b` | No | Expected 1357 edges. |
| HCC nodes | `output/gephi/gephi_hcc_nodes.csv` | RM1 | FSA_V/HCC extraction | RM2 downstream | FROZEN_RESEARCH_ARTIFACT | Yes | `a87aedc3eac4d1ad2ea89bcbb5d9b9156c70aa20e26df5ad20946ab18b840bb0` | No | Expected 218 HCC members across 42 HCC. |
| RM1 tables | `output/tables/` | RM1 | RM1 notebook and evidence builder | RM1/RM2 downstream | FROZEN_RESEARCH_ARTIFACT | Yes | See inventory CSV | No | Includes evidence, HCC, hashtag, brand, and temporal helper tables. |
| RM1 temporal | `output/rm1_temporal/` | RM1 temporal | RM1 temporal analysis | Temporal reporting | FROZEN_RESEARCH_ARTIFACT | Yes | See inventory CSV | No | Final temporal outputs use WIB. |
| Actor type universe | `output/rm2_actor_type/tables/actor_type_universe_summary.csv` | RM2 actor type | Actor type pipeline | RM2 typology | FROZEN_RESEARCH_ARTIFACT | Yes | `4bfd884ec9b93caed6d2a51296cbd4c2562fbea770f95d68da949604628f6f94` | No | Expected total actors 26427. |
| Community-Mass account pairs | `output/rm2_actor_type/account_interaction/community_mass_account_pairs.csv` | RM2 Community-Mass | Three-evidence account network script | Actor type downstream | FROZEN_RESEARCH_ARTIFACT | Yes | `5b64af89ded13e6a8cc1917b16ac9e7efe17d992ac959aeaf5d02dc2d7a75bdd` | No | Expected 434823 pairs. |
| Comment similarity pairs | `output/rm2_comment_similarity/comment_similarity_pairs_all.csv` | RM2 comment similarity | Similarity script | Presentation evidence package | FROZEN_RESEARCH_ARTIFACT | Yes | `94a3f32bafd1468abbde38c4d0495ba558f11a4b2022b666fd6d45e6323e318c` | Yes | Must stay at this path. |
| Sentiment V2 model | `output/rm2_sentiment/model/frozen/selected_model_development_frozen.joblib` | RM2 sentiment | Frozen development training | Locked-test evaluation and final inference | FROZEN_RESEARCH_ARTIFACT | Yes | `477bfe11ebc3463eeff5a5fb8359f86022d719c8bea49dfdad0460fa0c5e2ccc` | No | Do not retrain during cleanup. |
| Final locked test | `output/rm2_sentiment/validation/human_v2/locked_test_v2_observational_final.csv` | RM2 sentiment | Locked-test freeze | One-time evaluation | DEVELOPMENT_PROVENANCE | Yes | `663a69964d57ccd9dc4e05ec091a30fb6b1f3e12d0ecb271ba56ba70603bbfc8` | No | 300 observational rows, 0 synthetic/injected. |
| Evaluation lock | `output/rm2_sentiment/model/frozen/final_locked_test_evaluation_lock.json` | RM2 sentiment | One-time evaluator | Sentiment reporting | FROZEN_RESEARCH_ARTIFACT | Yes | `ea08186fc1bcb05507abea57991a06a5c918fffd66c3074a1849e6e9a96b81c2` | No | Records `FINAL_LOCKED_TEST_EVALUATED_ONCE`. |
| Final Sentiment V2 predictions | `output/rm2_sentiment/final/comment_sentiment_v2_observational.csv` | RM2 sentiment | Final inference script | RM2 reports and Gephi attributes | FROZEN_RESEARCH_ARTIFACT | Yes | `64b4fdacb9c623798035b5ed024e050d7e73de2a2260a7a8fcede5a4032b20d6` | No | 33063 observational rows. |
| Final Sentiment V2 manifest | `output/rm2_sentiment/final/FINAL_SENTIMENT_ANALYSIS_MANIFEST.json` | RM2 sentiment | Final inference script | Sentiment reporting | FROZEN_RESEARCH_ARTIFACT | Yes | `beed4dc304ca71ad8fb6069ec34b71103844f7b5806c4814cf14d592b7bfe040` | No | Final status `FINAL_MODEL_VALIDATED`. |
| Final sentiment visualizations | `output/rm2_sentiment/final/visualisasi/` | RM2 sentiment | `notebooks/rm2/02_rm2_sentiment_analysis.ipynb` | Presentation and report figures | CANONICAL_FINAL_OUTPUT | No | Generated from final CSV tables | No | Safe to regenerate; does not train, evaluate locked test, or infer. |
| Network project files | `artifacts/network_projects/` | Visualization projects | Manual Gephi/Visone work | Human visualization | FROZEN_RESEARCH_ARTIFACT | Yes | See migration map after move | No | Project files are not canonical data sources. |

## Legacy and Exploratory Artifact Groups

| group | path | status | notes |
|---|---|---|---|
| Sentiment V1 reference files | `output/rm2_sentiment/legacy/v1/tables/sentiment_heuristic_reference_guideline.md`, `output/rm2_sentiment/legacy/v1/tables/sentiment_model_selection.json` | LEGACY_REFERENCE_ONLY | Bulky legacy V1 CSV tables were trimmed; do not regenerate them from notebooks. |
| Sentiment V1 visual references | `output/rm2_sentiment/legacy/v1/visualisasi/` | LEGACY_VISUAL_REFERENCE | Retained PNG references only; final sentiment reporting uses `output/rm2_sentiment/final/`. |
| Exploratory wordcloud | `output/rm2_sentiment/legacy/exploratory/visualisasi/` | EXPLORATORY_OUTPUT | Presentation context only, not validation evidence. |
| Reply-only direct interaction | `output/rm2_actor_type/direct_interaction/` | LEGACY_DIRECT_INTERACTION | Optional diagnostic; three-evidence `account_interaction/` is canonical. |
