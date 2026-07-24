# Scripts Registry

Scripts stay flat to avoid breaking imports. Status values indicate whether a script is canonical, legacy,
frozen, or read-only.

| script | research module | status | primary input | primary output | safe to rerun | one-time | frozen | canonical entrypoint | notes |
|---|---|---|---|---|---|---|---|---|---|
| `rm1_evidence_builder.py` | RM1 | CANONICAL_RM1 | `dataset.csv` | RM1 evidence tables | No during cleanup | No | Outputs frozen | Yes | Shared evidence construction support. |
| `build_rm2_community_mass_account_network.py` | RM2 Community-Mass | CANONICAL_RM2_COMMUNITY_MASS | RM1 evidence, actor type | `output/rm2_actor_type/account_interaction/` | No during cleanup | No | Outputs frozen | Yes | Three-evidence account-level layer. |
| `build_rm2_community_mass_direct_interactions.py` | RM2 diagnostic | LEGACY_DIRECT_INTERACTION | Reply evidence | `output/rm2_actor_type/direct_interaction/` | No during cleanup | No | Legacy outputs frozen | No | Optional reply-only diagnostic. |
| `build_rm2_comment_similarity_examples.py` | RM2 comment similarity | CANONICAL_RM2_COMMENT_SIMILARITY | `dataset.csv`, actor attributes | `output/rm2_comment_similarity/` | No during cleanup | No | Outputs frozen | Yes | Does not alter LCN/HCC. |
| `build_rm2_sentiment_v2_annotation_package.py` | RM2 sentiment | CANONICAL_RM2_SENTIMENT | Dataset and diagnostics | `output/rm2_sentiment/human_validation_v2/` | No during cleanup | No | Human labels frozen | Yes | Annotation package builder. |
| `validate_rm2_sentiment_human_annotations_v2.py` | RM2 sentiment | CANONICAL_RM2_SENTIMENT | V2 annotation files | V2 validation report | Read-only validation only | No | Outputs frozen | Yes | Does not train model. |
| `validate_rm2_sentiment_v2_annotations_and_readiness.py` | RM2 sentiment | CANONICAL_RM2_SENTIMENT | V2 annotation files | Readiness/audit CSVs | Read-only validation only | No | Outputs frozen | Yes | Checks synthetic leakage/readiness. |
| `train_rm2_sentiment_v2_development_model.py` | RM2 sentiment | FROZEN_MODEL_TRAINING_DO_NOT_RERUN | Human development pool | `output/rm2_sentiment/model_v2/` | No | No | Yes | Historical canonical | Model already frozen. |
| `freeze_rm2_sentiment_v2_locked_test.py` | RM2 sentiment | ONE_TIME_EVALUATION_DO_NOT_RERUN | V2 labels | locked-test final files | No | Yes | Yes | Historical canonical | Do not rerun. |
| `evaluate_rm2_sentiment_v2_locked_test_once.py` | RM2 sentiment | ONE_TIME_EVALUATION_DO_NOT_RERUN | frozen model, locked test | `final_locked_test_*` | No | Yes | Yes | Historical canonical | Lock already records evaluation. |
| `apply_rm2_sentiment_v2_final_inference.py` | RM2 sentiment | FINAL_INFERENCE_ALREADY_GENERATED | frozen model, dataset | `output/rm2_sentiment/final_v2/` | No during cleanup | No | Yes | Historical canonical | Final output already generated. |
| `rm2_sentiment_goals_pipeline.py` | RM2 sentiment/goals | LEGACY_V1 | RM1 outputs | `output/rm2_sentiment/tables/` | No during cleanup | No | Legacy outputs frozen | No | Kept for V1 provenance. |
| `validate_rm2_sentiment_human_annotations.py` | RM2 sentiment | LEGACY_V1 | V1 annotation files | V1 validation metrics | Read-only only | No | Legacy outputs frozen | No | V1 provenance. |
| `project_paths.py` | Repository | CANONICAL_SOURCE | repository markers | path constants | Yes | No | No | Yes | Side-effect-free path resolver. |
| `io_utils.py` | Repository | CANONICAL_SOURCE | path constants | safe write helpers | Yes | No | No | Yes | For future reruns, not retroactive frozen scripts. |
| `validate_notebook_paths.py` | Repository audit | CANONICAL_SOURCE | notebooks JSON | `docs/repository_audit/` | Yes | No | No | Yes | Static only; does not execute notebooks. |
| `print_pipeline_output_plan.py` | Repository audit | CANONICAL_SOURCE | path constants | stdout only | Yes | No | No | Yes | Dry-run output plan. |
| `verify_repository_integrity.py` | Repository audit | CANONICAL_SOURCE | frozen outputs | `output/repository_integrity/` | Yes | No | No | Yes | Read-only verification; no analysis rerun. |
