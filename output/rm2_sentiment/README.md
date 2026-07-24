# RM2 Sentiment Output

This folder contains legacy Sentiment V1 outputs and final Sentiment V2 outputs.

| path | status | notes |
|---|---|---|
| `tables/` | LEGACY_REQUIRED_FOR_REPRODUCIBILITY | V1/provisional sentiment and goals outputs, retained for comparison. |
| `gephi/` | LEGACY_REQUIRED_FOR_REPRODUCIBILITY | V1 Gephi sentiment exports, not overwritten by V2. |
| `visualisasi/` | LEGACY_REQUIRED_FOR_REPRODUCIBILITY | V1 final PNG visualizations. |
| `visualisasi_exploratory/` | EXPLORATORY_OUTPUT | Wordcloud and exploratory visuals, not validation evidence. |
| `human_validation/` | DEVELOPMENT_PROVENANCE | V1 human validation package. |
| `human_validation_v2/` | DEVELOPMENT_PROVENANCE | V2 annotation, replacement, locked-test, and provenance files. |
| `model_v2/` | FROZEN_RESEARCH_ARTIFACT | Frozen development model, threshold, final locked-test evaluation outputs, and evaluation lock. |
| `final_v2/` | FROZEN_RESEARCH_ARTIFACT | Final Sentiment V2 full inference, reports, tables, presentation summary, and Gephi attributes. |

Important status:

- Final model status: `FINAL_MODEL_VALIDATED`.
- Final sentiment status: `FINAL_MODEL_VALIDATED_SENTIMENT_V2`.
- Threshold: `0.42`.
- Locked-test evaluation: `FINAL_LOCKED_TEST_EVALUATED_ONCE`.

Sentiment is an indicator of observed message orientation, not evidence of intent, payment, affiliation, control,
or causal influence.
