# Final RM2 Sentiment IndoBERT V5 Report

Status: `INDOBERT_V5_ACCEPTED_AS_FINAL_RM2_SENTIMENT_MODEL`.
Observational denominator: `33063` comments. INJ rows were excluded before inference.
Frozen model: `artifacts/rm2_sentiment/indobert_v5_final`.

## Sentiment Distribution

| label | count | percentage_of_total |
| --- | --- | --- |
| Positive | 10263 | 31.040740404681973 |
| Neutral | 17875 | 54.06345461694341 |
| Negative | 4925 | 14.895804978374619 |

## HCC Goal Orientation

| goal_orientation | n_hcc |
| --- | --- |
| Mixed Goals | 21 |
| Neutral Engagement | 13 |
| Promotional / Supportive | 7 |
| Polarized / Contested | 1 |

## Actor-Type Pooled Goals

| actor_type_primary | n_accounts | n_accounts_with_comments | n_valid_comments | pooled_positive_count | pooled_neutral_count | pooled_negative_count | pooled_positive_ratio | pooled_neutral_ratio | pooled_negative_ratio | pooled_dominant_sentiment | pooled_goal_orientation | goal_validation_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Community Actor | 207 | 207 | 945 | 489 | 406 | 50 | 0.5174603174603175 | 0.42962962962962964 | 0.05291005291005291 | Positive | Mixed Goals | INDOBERT_V5_ACCEPTED_AS_FINAL_RM2_SENTIMENT_MODEL |
| Individual Actor | 40 | 40 | 1384 | 599 | 731 | 54 | 0.4328034682080925 | 0.528179190751445 | 0.03901734104046243 | Neutral | Mixed Goals | INDOBERT_V5_ACCEPTED_AS_FINAL_RM2_SENTIMENT_MODEL |
| Mass Actor | 25943 | 25943 | 30734 | 9175 | 16738 | 4821 | 0.2985293160668966 | 0.5446085768204594 | 0.15686210711264398 | Neutral | Neutral Engagement | INDOBERT_V5_ACCEPTED_AS_FINAL_RM2_SENTIMENT_MODEL |

Goal orientation is a descriptive message-orientation aggregate from sentiment, not evidence of intent, payment, coordination, or causal influence.
