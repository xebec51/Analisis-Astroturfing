# RM2 Sentiment Human Validation V2

This directory contains the second human annotation package for RM2 comment-level sentiment analysis.

Phase status: package creation only. No model retraining, threshold selection, ensemble selection, inference rerun, HCC goal update, actor-type update, or Gephi topology update was performed.

Current V1 exclusion set: 579 unique comment_id values from `output/rm2_sentiment/human_validation/sentiment_human_annotation_validated.csv`. The V1 file is preserved and not overwritten by this package.

Files for annotators:

- `sentiment_v2_annotator_1_blind.csv`
- `sentiment_v2_annotator_2_blind.csv`
- `sentiment_v2_adjudication_template.csv`

Do not add model predictions, heuristic labels, HCC IDs, goal outputs, probabilities, or error flags to annotator blind files.

Use `sentiment_human_annotation_v2_guideline.md` and `sentiment_human_annotation_v2_codebook.csv` when labeling.

After annotation is complete, run:

```powershell
python scripts/validate_rm2_sentiment_human_annotations_v2.py
```

The locked-test V2 IDs are fixed in `locked_test_v2_manifest.csv`; they must not be used for model selection, threshold selection, ensemble weighting, preprocessing selection, or error-driven tuning.
