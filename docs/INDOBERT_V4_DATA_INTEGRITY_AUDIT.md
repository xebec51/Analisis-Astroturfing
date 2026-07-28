# IndoBERT V4 Data Integrity Audit

Generated at UTC: 2026-07-28T14:31:06.201982+00:00

The audit reads the completed Sentiment V4 human registry and the frozen locked-test registry. It does not edit labels or move rows between splits.

| item | value |
| --- | --- |
| Development registry | output/rm2_sentiment/validation/human_master_v4/sentiment_v4_development_final_registry.csv |
| Development evaluable rows | 1824 |
| Development class counts | {"Negative": 470, "Neutral": 788, "Positive": 566} |
| Locked-test registry | output/rm2_sentiment/validation/human_master_v4/sentiment_v4_locked_test_final_frozen.csv |
| Locked-test evaluable rows | 672 |
| Locked-test class counts | {"Negative": 160, "Neutral": 294, "Positive": 218} |
| Comment ID overlap | 0 |
| Text cluster overlap | 0 |
| Exact duplicate group overlap | 0 |
| Normalized text overlap | 0 |
| Near duplicate cluster overlap | 12 |
| Video ID overlap | 54 |
| Hard leakage status | PASS |

## Interpretation

Hard leakage is defined here as overlap in `comment_id`, `text_cluster_id`, `exact_duplicate_group_id`, or normalized text between development and locked test. Those checks pass. Near-duplicate cluster and video overlap are reported as diagnostics because the locked test is frozen and cannot be reshaped after evaluation.
