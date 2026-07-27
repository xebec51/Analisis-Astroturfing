# Sentiment Master Annotation V4 Protocol

Status: `MASTER_ANNOTATION_V4_READY_FOR_HUMAN_LABELING`

This package consolidates historical human sentiment annotations and creates one locked manual annotation package for the next V2 positive-recall development cycle. It does not train a model, tune thresholds, run locked-test evaluation, promote a model, or run full inference.

## Scope

- Historical human labels are copied as provenance and are not modified.
- Pending development candidates remain separate from pending locked-test candidates.
- Legacy opened tests remain `LEGACY_TEST_PROVENANCE` and are not eligible for development or the new locked test.
- Model predictions, probabilities, thresholds, HCC status, actor type, and goal orientation are not exposed in annotator workbooks.
- Pending final labels are blank until human annotation or adjudication is imported.

## Annotation Order

1. Fill `sentiment_v4_development_annotator_1.xlsx` and `sentiment_v4_locked_test_annotator_1.xlsx`.
2. Fill `sentiment_v4_development_annotator_2.xlsx` and `sentiment_v4_locked_test_annotator_2.xlsx` independently.
3. Run `python scripts/validate_sentiment_master_annotation_v4.py`.
4. Fill the generated adjudication workbooks for disagreement rows only.
5. Run `python scripts/import_completed_sentiment_master_annotation_v4.py`.

## Human Label Vocabulary

Allowed labels are:

- `Negative`
- `Neutral`
- `Positive`
- `Uncertain`
- `No Text`

Only `Negative`, `Neutral`, and `Positive` are evaluable for model training/evaluation after validation. `Uncertain` and `No Text` remain valid human annotation outcomes but are excluded from three-class evaluable counts.

## Locked Roles

- `HISTORICAL_DEVELOPMENT_FINAL`: final historical human labels eligible for development.
- `DEVELOPMENT_NEW_PENDING`: new development candidates awaiting human labels.
- `DEVELOPMENT_NEW_FINAL`: new development labels after completed import.
- `LEGACY_TEST_PROVENANCE`: opened old test/provenance rows, never training data.
- `LOCKED_TEST_NEW_PENDING`: new locked-test candidates awaiting human labels.
- `LOCKED_TEST_NEW_FINAL`: frozen new locked test after completed import.
- `EXCLUDED`: rows not eligible for development or locked-test evaluation.

## Integrity Rules

- No pending INJ rows.
- No hard `comment_id` overlap between development and locked test.
- No hard normalized-text cluster overlap between development and locked test.
- Video overlap is reported as a soft warning only.
- Duplicate or near-duplicate conflicts are audited before downstream use.
- No automatic sentiment labels are assigned to pending rows.

## Generated Package

The canonical package is in:

`output/rm2_sentiment/validation/human_master_v4/`

The codebook for annotators is:

`output/rm2_sentiment/validation/human_master_v4/SENTIMENT_ANNOTATION_CODEBOOK_V4.md`
