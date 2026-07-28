# IndoBERT V4 Fair Same-Test Comparison

Generated at UTC: 2026-07-28T14:57:05.882736+00:00

Status: `INDOBERT_V4_SAME_TEST_FAIR_COMPARISON`

This report compares the frozen legacy V2 sentiment model and frozen IndoBERT V4 base-reference model on the identical V4 locked-test denominator of 672 human-labeled comments. It does not change the existing strict decision `INDOBERT_V4_NOT_ACCEPTED_KEEP_V2`, does not tune V4 or V2, and does not update the canonical RM2 model.

## Same-Test Denominator

- Rows: 672
- Negative: 160
- Neutral: 294
- Positive: 218
- Dataset hash: `692aed0817a9d4f21f53049452e2501ad52a421cb027071a24f75c02307b26f6`

## Metrics

| mode | coverage | full_accuracy | covered_accuracy | macro_f1 | balanced_accuracy | mcc | positive_recall | abstained |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| V2_NATIVE_POLICY | 0.7188 | 0.4821 | 0.6708 | 0.5458 | 0.4702 | 0.3267 | 0.3670 | 189 |
| V2_FORCED_THREE_CLASS | 1.0000 | 0.5893 | 0.5893 | 0.5739 | 0.5781 | 0.3656 | 0.4404 | 0 |
| V4_ARGMAX_THREE_CLASS | 1.0000 | 0.7723 | 0.7723 | 0.7656 | 0.7754 | 0.6565 | 0.7156 | 0 |

## Paired Tests

- V2 forced three-class vs V4 argmax McNemar: `{"a_correct_b_wrong": 59, "a_wrong_b_correct": 182, "discordant_pairs": 241, "chi_square_continuity_corrected": 61.75933609958506, "p_value_chi_square_df1": 3.8810700873698555e-15, "p_value_exact_binomial_two_sided": 8.878901605594256e-16}`
- V2 native full-set vs V4 argmax McNemar: `{"a_correct_b_wrong": 42, "a_wrong_b_correct": 237, "discordant_pairs": 279, "chi_square_continuity_corrected": 134.89605734767025, "p_value_chi_square_df1": 3.4791398163349863e-31, "p_value_exact_binomial_two_sided": 3.5727954755038813e-34}`
- Bootstrap CI JSON: `output/rm2_sentiment/experiments/fair_same_test_comparison/same_test_bootstrap_ci.json`

## Interpretation

This comparison is descriptive and same-test only. V4 improves Positive recall relative to the historical V2 test summary, but this same-test report must not be used to retune V4, V5, thresholds, preprocessing, losses, class weights, or sampling rules. Locked-test errors are archived only for descriptive comparison.
