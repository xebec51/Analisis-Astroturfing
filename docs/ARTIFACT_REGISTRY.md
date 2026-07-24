# Artifact Registry

Per-file inventory values from the repository restructure audit are stored in
`docs/repository_audit/repository_inventory_before.csv`. This document lists the canonical artifact groups
used by the research pipeline.

| artifact | path | module | producer | consumer | status | immutable | hash | Git LFS | notes |
|---|---|---|---|---|---|---|---|---|---|
| Comment dataset | `dataset.csv` | Input | Source collection | RM1 and RM2 | CANONICAL_INPUT | Yes | `db11a4738a43aceafa01fe9b888e7c61a5168d9004f90ca4e254592b083978b2` | No | Retained at root; 2026-07-24 rerun reads 35334 rows. |
| Video metadata | `video_metadata_clean.csv` | Input | Source collection | RM1 and RM2 | CANONICAL_INPUT | Yes | `0cb228682705840fef691c2d9c6bcb38d6a4953985960d656394c067e7b666e9` | No | Retained at root; 55 rows. |
| Individual actor registry | `config/individual_actor_registry.csv` | Config | Manual registry | RM2 actor type | CANONICAL_INPUT | Yes | `debb299c4a4255793d7a31165e9aef319b79d613559d350c9fb24928f5bbd4d7` | No | Retained in `config/`. |
| LCN nodes | `output/gephi/gephi_lcn_nodes.csv` | RM1 | RM1 notebook | RM2 downstream, Gephi | FROZEN_RESEARCH_ARTIFACT | Yes | `8700613db132f0cc43c518295b5e7b8ec73f48925e9747c2808f6efce6c21edf` | No | 2026-07-24 rerun: 724 nodes. |
| LCN edges | `output/gephi/gephi_lcn_edges.csv` | RM1 | RM1 notebook | RM2 downstream, Gephi | FROZEN_RESEARCH_ARTIFACT | Yes | `f2fbce5966d753f5e536b61f99564405c40614c4555a12af89f84be94c4c4edb` | No | 2026-07-24 rerun: 1359 edges. |
| HCC nodes | `output/gephi/gephi_hcc_nodes.csv` | RM1 | FSA_V/HCC extraction | RM2 downstream | FROZEN_RESEARCH_ARTIFACT | Yes | `20f4cca797381167e801ecebeb799b557579b97097d59dd09485fe75c5f3fd9a` | No | 2026-07-24 rerun: 218 HCC members across 42 HCC. |
| RM1 tables | `output/tables/` | RM1 | RM1 notebook and evidence builder | RM1/RM2 downstream | FROZEN_RESEARCH_ARTIFACT | Yes | See inventory CSV | No | Includes evidence, HCC, hashtag, brand, and temporal helper tables. |
| RM1 temporal | `output/rm1_temporal/` | RM1 temporal | RM1 temporal analysis | Temporal reporting | FROZEN_RESEARCH_ARTIFACT | Yes | See inventory CSV | No | Final temporal outputs use WIB. |
| Actor type universe | `output/rm2_actor_type/tables/actor_type_universe_summary.csv` | RM2 actor type | Actor type pipeline | RM2 typology | FROZEN_RESEARCH_ARTIFACT | Yes | `4c74af1074a3ef2f8abc942eb95bfb302bf425865c1506b48d4e3eb142f30141` | No | 2026-07-24 rerun: 26427 total actors and 33063 V2-final sentiment comments. |
| Community-Mass account pairs | `output/rm2_actor_type/account_interaction/community_mass_account_pairs.csv` | RM2 Community-Mass | Three-evidence account network script | Actor type downstream | FROZEN_RESEARCH_ARTIFACT | Yes | `5cd039be8befc97ee48b8c135b97c36373c0e6157c75b3184cfcd0d8dcca9ce3` | Yes | 2026-07-24 rerun: 457628 pairs; large CSV stored through Git LFS. |
| Comment similarity pair sample | `output/rm2_comment_similarity/comment_similarity_pairs_all.csv` | RM2 comment similarity | Similarity script | Pair audit sample | FROZEN_RESEARCH_ARTIFACT | Yes | `ebd1469c7c98c4edb4d2c8ac04db08151e2dca78cf724e8d99b621e4b50f99f0` | No | Compatibility path; capped pair evidence sample only, not full pairwise enumeration. |
| Comment similarity groups | `output/rm2_comment_similarity/comment_similarity_groups.csv` | RM2 comment similarity | Similarity script | Similarity analysis and screenshot selection | FROZEN_RESEARCH_ARTIFACT | Yes | `39a9210f13ac5ba449c1dd9610046bd18350fbf96243cbd802f08691c69e8d46` | No | Canonical group-level output; 1452 groups. |
| Comment similarity group members | `output/rm2_comment_similarity/comment_similarity_group_members.csv` | RM2 comment similarity | Similarity script | Similarity analysis and screenshot lookup | FROZEN_RESEARCH_ARTIFACT | Yes | `257c6298b3c652f1c23db0f24cfdf64478d0b2bdcf6f3444b07358d84ac0cd59` | No | 10770 group-member rows with TikTok lookup fields. |
| Comment similarity screenshot queue | `output/rm2_comment_similarity/comment_similarity_screenshot_queue.csv` | RM2 comment similarity | Similarity script | Manual platform screenshot workflow | CANONICAL_REVIEW_OUTPUT | No | `68be5a9d30f52fc8a02d68986aa73f9b82c291a8aed3bf5af5bd0c1d44976213` | No | 933 prioritized substantive group-member rows for TikTok lookup. |
| Sentiment V2 model | `output/rm2_sentiment/model/frozen/selected_model_development_frozen.joblib` | RM2 sentiment | Frozen development training | Locked-test evaluation and final inference | FROZEN_RESEARCH_ARTIFACT | Yes | `477bfe11ebc3463eeff5a5fb8359f86022d719c8bea49dfdad0460fa0c5e2ccc` | No | Do not retrain during cleanup. |
| Final locked test | `output/rm2_sentiment/validation/human_v2/locked_test_v2_observational_final.csv` | RM2 sentiment | Locked-test freeze | One-time evaluation | DEVELOPMENT_PROVENANCE | Yes | `d94d322a877523e1317c8036d37752c89ec524c09b36f673910cc5219d270828` | No | 300 observational source rows; V3 same-test registry uses 274 evaluable three-class rows. |
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
