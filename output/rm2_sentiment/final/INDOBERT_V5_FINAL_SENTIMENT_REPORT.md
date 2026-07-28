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
| Community Actor | 218 | 207 | 945 | 289 | 541 | 87 | 0.31515812431842966 | 0.5899672846237731 | 0.09487459105779716 | Neutral | Neutral Engagement | INDOBERT_V5_ACCEPTED_AS_FINAL_RM2_SENTIMENT_MODEL |
| Individual Actor | 43 | 40 | 1384 | 164 | 1055 | 69 | 0.12732919254658384 | 0.8190993788819876 | 0.05357142857142857 | Neutral | Neutral Engagement | INDOBERT_V5_ACCEPTED_AS_FINAL_RM2_SENTIMENT_MODEL |
| Mass Actor | 26166 | 0 | 0 | 0 | 0 | 0 | 0.0 | 0.0 | 0.0 | No evaluable sentiment | Insufficient Text | INDOBERT_V5_ACCEPTED_AS_FINAL_RM2_SENTIMENT_MODEL |

Goal orientation is a descriptive message-orientation aggregate from sentiment, not evidence of intent, payment, coordination, or causal influence.
