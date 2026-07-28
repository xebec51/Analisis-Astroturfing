# RM2 Sentiment Sensitivity Analysis

Status: `EXPLORATORY_SENSITIVITY_ANALYSIS_NOT_CANONICAL`

Generated at UTC: 2026-07-28T15:33:48.340052+00:00

This analysis compares canonical V2 RM2 sentiment outputs with non-canonical frozen IndoBERT V4 predictions. It does not modify `output/rm2_sentiment/final/` and does not change the canonical status of V2 or the previous V4 decision.

## Overall Distribution

| group | label | count | percentage_of_total | denominator |
| --- | --- | --- | --- | --- |
| Canonical V2 | Positive | 2718 | 8.220669630705018 | 33063 |
| Canonical V2 | Neutral | 23977 | 72.51913014547983 | 33063 |
| Canonical V2 | Negative | 4771 | 14.43002752321326 | 33063 |
| Canonical V2 | No Text | 4 | 0.012098115718476847 | 33063 |
| IndoBERT V4 sensitivity | Positive | 7594 | 22.968272691528295 | 33063 |
| IndoBERT V4 sensitivity | Neutral | 19538 | 59.09324622690016 | 33063 |
| IndoBERT V4 sensitivity | Negative | 5927 | 17.92638296585307 | 33063 |
| IndoBERT V4 sensitivity | No Text | 4 | 0.012098115718476847 | 33063 |

## HCC Versus Non-HCC, V4 Sensitivity

| group | total_comments | evaluable_comments | coverage | negative_count | negative_ratio_evaluable | neutral_count | neutral_ratio_evaluable | positive_count | positive_ratio_evaluable |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HCC | 945 | 945 | 1.0 | 80 | 0.08465608465608465 | 452 | 0.4783068783068783 | 413 | 0.43703703703703706 |
| Non-HCC | 32118 | 32114 | 0.9998754592440376 | 5847 | 0.18207012517904964 | 19086 | 0.5943202341657844 | 7181 | 0.22360964065516598 |

## HCC Goal Changes

- Changed HCC goal orientations: `3` of `42`
- Sensitivity conclusion: `model_dependent`

| hcc_id | v2_goal_orientation | v4_goal_orientation | goal_changed |
| --- | --- | --- | --- |
| 1 | Neutral Engagement | Neutral Engagement | False |
| 103 | Neutral Engagement | Neutral Engagement | False |
| 113 | Promotional / Supportive | Promotional / Supportive | False |
| 114 | Promotional / Supportive | Promotional / Supportive | False |
| 117 | Neutral Engagement | Neutral Engagement | False |
| 132 | Neutral Engagement | Neutral Engagement | False |
| 134 | Neutral Engagement | Neutral Engagement | False |
| 14 | Neutral Engagement | Mixed Goals | True |
| 15 | Neutral Engagement | Neutral Engagement | False |
| 150 | Neutral Engagement | Neutral Engagement | False |
| 16 | Neutral Engagement | Neutral Engagement | False |
| 168 | Promotional / Supportive | Promotional / Supportive | False |
| 17 | Neutral Engagement | Neutral Engagement | False |
| 18 | Neutral Engagement | Neutral Engagement | False |
| 180 | Promotional / Supportive | Promotional / Supportive | False |
| 181 | Promotional / Supportive | Promotional / Supportive | False |
| 182 | Neutral Engagement | Neutral Engagement | False |
| 183 | Neutral Engagement | Neutral Engagement | False |
| 191 | Promotional / Supportive | Promotional / Supportive | False |
| 2 | Neutral Engagement | Neutral Engagement | False |
