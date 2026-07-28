# IndoBERT V4 Final Application Report

Generated at UTC: 2026-07-28T14:31:06.202983+00:00

## Git

- Branch: `research/indobert-v4-final-application`
- Audit commit: `545b565b5afe24153b91769e89dccbb0dc74c8b6`

## Model Artifact

- Artifact path: `artifacts/rm2_sentiment/indobert_v4_final/base_reference`
- Model ID: `indobenchmark/indobert-base-p2`
- Model revision: `94b4e0a82081fa57f227fcc2024d1ea89b57ac1f`
- Model SHA-256: `2b13c75df848ca0abe02cabf8a1ace0b26814730762a4a7d506095bb518c3c52`
- Artifact technical status: `PASS`
- Label order: `Negative`, `Neutral`, `Positive`
- Prediction rule: `argmax_no_threshold_tuning`

## Data

- Development evaluable: `1824`; counts `{"Negative": 470, "Neutral": 788, "Positive": 566}`
- Locked-test evaluable: `672`; counts `{"Negative": 160, "Neutral": 294, "Positive": 218}`
- Hard leakage counts: `{"comment_id": 0, "text_cluster_id": 0, "exact_duplicate_group_id": 0, "normalized_text": 0}`
- Near-duplicate cluster overlap, report-only: `12`
- Video overlap, report-only: `54`

## Locked-Test Metrics

| metric | value |
| --- | --- |
| accuracy | 0.7723 |
| macro_f1 | 0.7656 |
| weighted_f1 | 0.7746 |
| balanced_accuracy | 0.7754 |
| mcc | 0.6565 |
| min_class_recall | 0.7156 |
| negative_precision | 0.6377 |
| negative_recall | 0.8250 |
| negative_f1 | 0.7193 |
| neutral_precision | 0.8339 |
| neutral_recall | 0.7857 |
| neutral_f1 | 0.8091 |
| positive_precision | 0.8298 |
| positive_recall | 0.7156 |
| positive_f1 | 0.7685 |
| ece | 0.0629 |
| brier_score | 0.3289 |

## IndoBERT V4 Versus V2 Baseline

| metric | V2 baseline | IndoBERT V4 | delta |
| --- | --- | --- | --- |
| accuracy | 0.8359 | 0.7723 | -0.0636 |
| macro_f1 | 0.7309 | 0.7656 | 0.0347 |
| balanced_accuracy | 0.7188 | 0.7754 | 0.0566 |
| mcc | 0.6369 | 0.6565 | 0.0196 |
| positive_recall | 0.4773 | 0.7156 | 0.2383 |
| positive_f1 | 0.5753 | 0.7685 | 0.1932 |

## Acceptance Decision

Status: `INDOBERT_V4_NOT_ACCEPTED_KEEP_V2`

| criterion | passed |
| --- | --- |
| artifact_checksum_and_load_pass | True |
| no_hard_data_leakage | True |
| locked_test_evaluated_once | True |
| locked_test_excluded_from_training_or_selection | True |
| macro_f1_gte_0p7309 | True |
| positive_recall_gte_0p70 | True |
| accuracy_gte_0p8159 | False |
| minimum_class_recall_gte_0p60 | True |
| mcc_gte_0p60 | True |
| no_class_collapse | True |
| label_mapping_verified | True |
| model_reproducible_from_saved_artifact | True |

IndoBERT V4 is not promoted under the current strict gate because accuracy is `0.7723`, below the required `0.8159`. Positive recall improves over V2, but this does not override the accuracy gate.

## Downstream Action

Full inference was not run. `output/rm2_sentiment/final/CANONICAL_MODEL.json` was not created for IndoBERT V4, and `artifacts/rm2_sentiment/indobert_v4_final/final_model/` was not created. RM2 Goals outputs therefore remain on the already accepted V2 baseline until a separately accepted model replaces it.

## Reproduction Commands

```powershell
python scripts/audit_indobert_v4_final_application.py
python scripts/evaluate_rm2_sentiment_indobert_v4_locked_test_once.py  # only if LOCKED_TEST_EVALUATION_MANIFEST.json is absent or evaluated_once is not true
python scripts/predict_rm2_sentiment_cpu.py --model-dir artifacts/rm2_sentiment/indobert_v4_final/base_reference --text "produk ini bagus"
python -m unittest discover
python -m pytest
```

Do not retrain, retune, rerun locked-test evaluation, promote a model, or run full inference after seeing this failed strict acceptance decision.
