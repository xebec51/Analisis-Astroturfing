# IndoBERT V5 Development Plan

Status: `INDOBERT_V5_DEVELOPMENT_FOLDS_READY`

This plan prepares the next sentiment development cycle. V5 must not use the opened V4 locked-test errors or labels for model development. The V4 decision `INDOBERT_V4_NOT_ACCEPTED_KEEP_V2` remains unchanged.

## Data

Human labels for V5 development are collected in `output/rm2_sentiment/validation/human_v5/`.

Final adjudicated development registry:

- `output/rm2_sentiment/validation/human_v5/sentiment_v5_development_final_registry.csv`
- Evaluable three-class rows: `977`
- Distribution: Negative `178`, Neutral `569`, Positive `230`
- Excluded from three-class training denominator: Uncertain `13`, No Text `10`

The primary label is `sentiment_toward_target`. Rows with `Uncertain` or `No Text` are excluded from the three-class model denominator but retained in annotation audit files.

V5 development excludes:

- Existing V4 development rows.
- Existing opened V4 locked-test rows.
- Candidate V5 locked-test rows.
- Exact normalized-text duplicates across development and locked-test candidates.
- INJ or synthetic IDs.

Video overlap is audited. If hard grouping by `video_id` would make folds infeasible, video overlap is reported as a soft diagnostic while exact text and duplicate-cluster grouping remain hard.

## Model Candidates

Primary model family:

- `indobenchmark/indobert-base-p2`

Search grid:

- Loss: `cross_entropy`, `weighted_cross_entropy`, `focal_loss_gamma_1.0`, `focal_loss_gamma_1.5`, `focal_loss_gamma_2.0`
- Label smoothing: `0.0`, `0.025`, `0.05`
- Learning rate: `1e-5`, `2e-5`, `3e-5`
- Classifier dropout: `0.1`, `0.2`, `0.3`
- Max length: `128`, `256`
- Input mode: `comment_only`, `context_sep_comment`, `target_context_parent_comment`
- Seeds: `42`, `52`, `62`

## Selection

Selection uses development OOF only. Locked test V5 remains sealed until all candidate selection, threshold policy, and model freeze steps are complete.

Selection score prioritizes:

- Macro-F1.
- Balanced accuracy.
- MCC.
- Minimum class recall.
- Positive recall.
- Stability across seeds.

Penalties apply to:

- Excessive Negative prediction share.
- Class collapse.
- High seed instability.

## Ensemble And Abstention

Compare on development OOF only:

- Single best seed.
- Probability-average ensemble.
- Majority-vote ensemble.

Development-only abstention policies:

- Coverage target `93.43%` to match legacy V2 native coverage reference.
- Coverage target `95%`.
- Full coverage argmax `100%`.

Risk-coverage curves must be computed from development OOF only.

## Locked Test V5

The current V5 locked-test package has completed two-annotator labeling and human adjudication. It is frozen for future preregistered final evaluation, not for development-time tuning.

Frozen locked-test registry:

- `output/rm2_sentiment/validation/human_v5_locked_test/sentiment_v5_locked_test_final_frozen.csv`
- Evaluable three-class rows: `687`
- Distribution: Negative `134`, Neutral `380`, Positive `173`
- Excluded from three-class evaluation denominator: Uncertain `9`, No Text `4`

The Negative count is below the earlier target of 150. Do not change labels or move development data into the locked test to repair that target; report the actual distribution when V5 final evaluation is eventually run.

Do not evaluate V2 or V5 on locked-test V5 until the preregistered acceptance config has already been committed and all V5 model decisions are frozen.
