# Pipeline Entrypoints

This registry separates canonical, frozen, legacy, and diagnostic entrypoints. Final reporting notebooks may
be rerun to refresh tables/figures, but model training, locked-test evaluation, final inference, and network
formation scripts must not be rerun unless explicitly reproducing the frozen results.

| module | research question | canonical entrypoint | primary input | primary output | current status | safe to rerun | frozen | one-time | notes |
|---|---|---|---|---|---|---|---|---|---|
| RM1 LCN/HCC | Coordination network and HCC detection | `notebooks/rm1/tiktok_coordination_analysis.ipynb` | `dataset.csv`, `video_metadata_clean.csv` | `output/tables/`, `output/gephi/`, `output/visualisasi/` | CANONICAL_RM1 | No during cleanup | Yes outputs | No | Notebook moved and statically path-refactored only. |
| RM1 temporal | Temporal activity profile | RM1 notebook temporal section | RM1 outputs | `output/rm1_temporal/` | CANONICAL_RM1_TEMPORAL | No during cleanup | Yes outputs | No | Main timezone remains WIB for final temporal outputs. |
| RM2 Sentiment V1 | Legacy sentiment/goals | `notebooks/legacy/02_rm2_sentiment_goals.ipynb` and `scripts/rm2_sentiment_goals_pipeline.py` | RM1 outputs | Selected legacy reference files and PNGs only | LEGACY_V1_ARCHIVED | No | Legacy CSV outputs trimmed | No | Notebook is disabled as an execution entrypoint so trimmed V1 CSVs are not regenerated. |
| Sentiment V2 annotation package | Human validation package | `scripts/build_rm2_sentiment_v2_annotation_package.py` | Dataset and legacy diagnostics | `output/rm2_sentiment/validation/human_v2/` | CANONICAL_RM2_SENTIMENT | No during cleanup | Human labels frozen | No | Do not overwrite human labels. |
| Sentiment V2 validation/readiness | Annotation validation | `scripts/validate_rm2_sentiment_v2_annotations_and_readiness.py` | V2 human labels | V2 readiness and integrity CSVs | CANONICAL_RM2_SENTIMENT | Read-only validation only | Yes outputs | No | Does not train or evaluate model. |
| Sentiment V2 development training | Development model selection | `scripts/train_rm2_sentiment_v2_development_model.py` | Human development pool | `output/rm2_sentiment/model/frozen/` | FROZEN_MODEL_TRAINING_DO_NOT_RERUN | No | Yes | No | Model already frozen. |
| Sentiment V2 locked-test freeze | Final locked-test freeze | `scripts/freeze_rm2_sentiment_v2_locked_test.py` | Human validation V2 | Locked-test final CSV and manifest | ONE_TIME_EVALUATION_DO_NOT_RERUN | No | Yes | Yes | Locked test already frozen. |
| Sentiment V2 locked-test evaluation | One-time evaluation | `scripts/evaluate_rm2_sentiment_v2_locked_test_once.py` | Frozen model and locked test | `final_locked_test_*` outputs | ONE_TIME_EVALUATION_DO_NOT_RERUN | No | Yes | Yes | Status `FINAL_LOCKED_TEST_EVALUATED_ONCE`. |
| Sentiment V2 final inference | Final full inference | `scripts/apply_rm2_sentiment_v2_final_inference.py` | Frozen model and dataset | `output/rm2_sentiment/final/` | FINAL_INFERENCE_ALREADY_GENERATED | No during cleanup | Yes | No | Final V2 outputs already generated. |
| Sentiment final reporting | Tables, interpretation, and PNG visuals | `notebooks/rm2/02_rm2_sentiment_analysis.ipynb` | `output/rm2_sentiment/final/` | `output/rm2_sentiment/final/visualisasi/` | CANONICAL_RM2_SENTIMENT_FINAL_REPORTING | Yes | No | No | Reruns final visuals only; no training, locked-test evaluation, or full inference. |
| RM2 actor type | Actor type typology | `notebooks/rm2/03_rm2_actor_type_typology.ipynb` | RM1 and sentiment outputs | `output/rm2_actor_type/` | CANONICAL_RM2_ACTOR_TYPE | No during cleanup | Yes outputs | No | Actor type definitions are unchanged. |
| Community-Mass account evidence | Three-evidence account network | `scripts/build_rm2_community_mass_account_network.py` | RM1 evidence and actor type | `output/rm2_actor_type/account_interaction/` | CANONICAL_RM2_COMMUNITY_MASS | No during cleanup | Yes outputs | No | Three-evidence account-level layer is canonical. |
| Community-Mass direct reply | Optional reply diagnostic | `scripts/build_rm2_community_mass_direct_interactions.py` | Reply evidence | `output/rm2_actor_type/direct_interaction/` | LEGACY_DIRECT_INTERACTION | No during cleanup | Legacy outputs frozen | No | Optional diagnostic, not the main Community-Mass layer. |
| Comment similarity | Exact and near-similar comments | `scripts/build_rm2_comment_similarity_examples.py` | `dataset.csv`, actor attributes | `output/rm2_comment_similarity/` | CANONICAL_RM2_COMMENT_SIMILARITY | No during cleanup | Yes outputs | No | LFS pair file stays at its current path. |

## One-Time Warnings

- Do not rerun locked-test evaluation. It is recorded by `output/rm2_sentiment/model/frozen/final_locked_test_evaluation_lock.json`.
- Do not retrain Sentiment V2 during repository cleanup.
- Do not run full inference during repository cleanup.
- The final sentiment reporting notebook may be rerun because it reads frozen final outputs and refreshes presentation PNG files only.
