# IndoBERT V3 Full Rerun Report

Branch: `research/indobert-v3-full-rerun`

Execution date: 2026-07-24

Final status: `INDOBERT_V3_NOT_ACCEPTED_KEEP_V2`

## Scope

This run reran the RM1 coordination pipeline, built a human-supervised Sentiment V3 registry, developed IndoBERT-based V3 candidates, compared the frozen V3 candidate against frozen V2 on the same human test set, and regenerated downstream RM2 outputs. Human annotation and adjudication remain the only sentiment ground truth.

No automatic post-hoc Positive shifting was applied. No rule converted low-confidence predictions, Neutral predictions, promotional comments, or HCC membership into Positive. The final locked test was used only once for final comparison, not for tuning, threshold selection, preprocessing selection, epoch selection, class weighting, loss selection, or model selection.

## RM1 Method

The canonical RM1 notebook was rerun from the first cell to the last cell:

`notebooks/rm1/tiktok_coordination_analysis.ipynb`

LCN construction used the existing evidence method:

| Evidence | Role |
|---|---|
| Co-conv | LCN-forming evidence |
| Co-reply | LCN-forming evidence |
| Co-temporal | LCN-forming evidence |
| Co-hashtag | Brand/video context only, not LCN-forming |
| Co-Similarity | Post-HCC narrative similarity context only, not LCN-forming |

Louvain, FSA_V, and HCC followed the established method. The temporal activity profile used Asia/Jakarta/WIB. Co-Similarity was applied only after HCC formation. RM1 differences from the older baseline were recorded and were not forced to match historical counts.

## RM1 Rerun Results

| Metric | Latest rerun | Prior baseline | Delta |
|---|---:|---:|---:|
| Dataset rows | 35,334 | 33,847 | +1,487 |
| Video metadata rows | 55 | 55 | 0 |
| LCN nodes | 724 | 724 | 0 |
| LCN edges | 1,359 | 1,357 | +2 |
| HCC members | 218 | 218 | 0 |
| HCC communities | 42 | 42 | 0 |
| HCC edges | 465 | 464 | +1 |
| FSA_V focal structures | 207 | not fixed in baseline doc | recorded |
| HCC Co-Similarity summary rows | 42 | 42 | 0 |

Temporal summary:

| Scope | Result |
|---|---|
| All dataset dominant weekday | Selasa |
| HCC dominant weekday | Kamis |
| HCC peak hour | 17 |

HCC outputs are interpreted as indikasi koordinasi, pola teramati, asosiasi konteks brand/video, kesamaan narasi, and orientasi pesan. HCC is not evidence of buzzer activity, bot behavior, payment, brand affiliation, manipulative intent, or astroturfing.

## Human Label Registry V3

Sources:

| Source | Role |
|---|---|
| `output/rm2_sentiment/validation/human_v1/` | historical and development human labels |
| `output/rm2_sentiment/validation/human_v2/` | development human labels and frozen locked test source |
| `output/rm2_sentiment/validation/human_v3/` | generated V3 registry, conflict audit, and split manifest |

Generated files:

| File | Purpose |
|---|---|
| `output/rm2_sentiment/validation/human_v3/human_label_registry_v3.csv` | canonical V3 registry |
| `output/rm2_sentiment/validation/human_v3/human_label_conflict_audit_v3.csv` | conflict and integrity audit |
| `output/rm2_sentiment/validation/human_v3/data_split_manifest_v3.csv` | split and fold manifest |
| `output/rm2_sentiment/validation/human_v3/human_label_registry_v3_manifest.json` | registry manifest |

Registry counts:

| Split or role | Count |
|---|---:|
| Registry rows | 1,179 |
| Development evaluable rows | 800 |
| Final test evaluable rows | 274 |
| Excluded rows | 105 |
| Text clusters | 1,142 |
| Development CV groups | 58 |

Development labels:

| Label | Count |
|---|---:|
| Negative | 216 |
| Neutral | 429 |
| Positive | 155 |

Final test labels:

| Label | Count |
|---|---:|
| Negative | 41 |
| Neutral | 186 |
| Positive | 47 |

Rules applied:

- Only observational comments were eligible for three-class sentiment modeling.
- `comment_id` values beginning with `INJ` were excluded from training and final test classification.
- One `comment_id` has only one final label.
- Disagreement used human adjudication.
- `No Text` and `Uncertain` were excluded from three-class training/evaluation.
- Classification labels were only Negative, Neutral, and Positive.
- Exact duplicate and near-identical text clusters were not split across development and final test, nor across development folds.
- Split construction was group-aware, prioritizing `video_id` and duplicate text cluster groups.
- V2 predictions, pseudo-labels, lexicons, model-old outputs, and LLM outputs were not used as ground truth.

## V3 Candidates

Candidates considered:

| Candidate | Revision | Status |
|---|---|---|
| `apriandito/indobert-sentiment-classifier` | `ba24f8cea1c00090cc4fce2b63c61ed307943a78` | Run |
| `indobenchmark/indobert-large-p2` | `4b280c3bfcc1ed2d6b4589be5c876076b7d73568` | Resource-blocked on CPU model load |
| `indolem/indobertweet-base-uncased` | `32e28c05b47e33b6675d2670a1745c50a65e987a` | Run |
| V2 TF-IDF LinearSVC | frozen local artifact | Baseline only |

All Hugging Face loads used `trust_remote_code=False`. The base Hugging Face model checkpoints were not committed.

Input format:

`context [SEP] comment_text`

Allowed context fields were metadata available before sentiment labeling, such as product category and brand/video context. Labels, predictions, HCC goal, `goal_orientation`, confusion matrices, and evaluation results were not used as model input.

Preprocessing:

| Normalized | Preserved |
|---|---|
| Unicode NFKC | negation |
| HTML entity unescape | emoji and emoticons |
| control character removal | elongated words |
| URL to `HTTPURL` | abbreviations |
| mention to `@USER` | brand and product names |
| whitespace normalization | numbers, question marks, exclamation marks, and relevant Indonesian-English mixing |

## Development Procedure

Development selection used out-of-fold predictions only. It used 5 group-aware folds and seeds 42, 52, and 62. Class weights were calculated from each training fold only. The search covered weighted cross-entropy, focal loss, class-balanced focal loss, learning rates 1e-5/2e-5/3e-5, max lengths 128/192/256, warmup ratios 0.06/0.10, weight decay 0.01/0.05, and focal gamma 1.0/1.5/2.0 in the limited CPU-feasible grid.

The local environment was CPU-only (`torch 2.13.0+cpu`, CUDA unavailable). V3 was therefore developed as a frozen IndoBERT encoder plus supervised linear head candidate. Full encoder fine-tuning is not claimed for this run.

Best development candidate:

| Metric | Value |
|---|---:|
| Model | `apriandito/indobert-sentiment-classifier` |
| Loss | weighted cross-entropy |
| Learning rate | 3e-5 |
| Max length | 256 |
| Warmup ratio | 0.06 |
| Weight decay | 0.05 |
| Mean macro-F1 | 0.4724 |
| Std macro-F1 | 0.2848 |
| Mean balanced accuracy | 0.4787 |
| Mean MCC | 0.2293 |
| Mean Positive recall | 0.4000 |
| Mean Positive precision | 0.4012 |
| Mean Positive F1 | 0.3980 |

Frozen V3 candidate artifacts:

| File | Purpose |
|---|---|
| `output/rm2_sentiment/model/indobert_v3_candidate/v3_candidate_config.json` | frozen candidate config |
| `output/rm2_sentiment/model/indobert_v3_candidate/linear_head.safetensors` | supervised classifier head |
| `output/rm2_sentiment/model/indobert_v3_candidate/embedding_scaler.joblib` | feature scaler |
| `output/rm2_sentiment/model/indobert_v3_candidate/tokenizer/` | pinned tokenizer files |

## Same-Test Evaluation

V2 and V3 were evaluated on the same human final test rows from the V3 registry. The final test was not used for development selection or tuning.

| Metric | V2 frozen | V3 candidate |
|---|---:|---:|
| Evaluable test rows | 274 | 274 |
| Covered rows | 256 | 274 |
| Coverage | 0.9343 | 1.0000 |
| Abstention rate | 0.0657 | 0.0000 |
| Accuracy | 0.8359 | 0.1022 |
| Macro-F1 | 0.7309 | 0.0895 |
| Weighted-F1 | 0.8273 | 0.1241 |
| Balanced accuracy | 0.7188 | 0.1042 |
| MCC | 0.6369 | -0.2722 |
| Positive recall | 0.4773 | 0.1915 |
| Positive precision | 0.7241 | 0.0667 |
| Positive F1 | 0.5753 | 0.0989 |
| ECE | 0.1521 | 0.4266 |
| Brier score | 0.3003 | 0.9833 |

V3 failed the acceptance gate. It did not improve accuracy, macro-F1, balanced accuracy, MCC, Positive recall, Positive precision, Positive F1, minimum class recall, seed stability, or paired-bootstrap non-decrease. Paired bootstrap showed a meaningful decrease for all primary and secondary metrics. McNemar exact test reported 200 V2-correct/V3-wrong discordant rows and 14 V2-wrong/V3-correct rows (`p = 2.56e-43`).

Decision:

`INDOBERT_V3_NOT_ACCEPTED_KEEP_V2`

Because the gate failed, V3 was kept as an experiment and V2 remained the final sentiment model. V3 full inference was not run as a final output.

## Final Sentiment Status

Final sentiment reporting uses the existing accepted V2 final predictions:

`output/rm2_sentiment/final/comment_sentiment_v2_observational.csv`

Final V2 distribution:

| Label | Count |
|---|---:|
| Positive | 2,718 |
| Neutral | 23,977 |
| Negative | 4,771 |
| Uncertain | 1,593 |
| No Text | 4 |

The final sentiment notebook reads final prediction artifacts and generates tables, interpretation, visualizations, and reports. It does not train, tune, or use the test set to change a model.

## Downstream RM2 Outputs

Actor type:

| Actor type | Accounts | Comments |
|---|---:|---:|
| Individual Actor | 43 | 1,384 |
| Community Actor | 218 | 945 |
| Mass Actor | 26,166 | 30,734 |
| Total | 26,427 | 33,063 |

Community-Mass account evidence:

| Metric | Value |
|---|---:|
| Unique Community-Mass pairs | 457,628 |
| LCN edge pairs | 306 |
| Pre-LCN multi-evidence pairs | 2,943 |
| Pre-LCN single-evidence pairs | 454,379 |
| Community accounts involved | 218 |
| Mass accounts involved | 27,304 |
| Visual Gephi edges | 3,604 |

Sentiment was used only as an attribute in Community-Mass outputs. It did not form edges. The optional direct-reply diagnostic was not used as the main pipeline.

Large Community-Mass CSV exports are stored through Git LFS where required by GitHub file-size limits.

Comment-level exact and near-similarity:

| Metric | Value |
|---|---:|
| Dataset input rows | 35,334 |
| Observational comments analyzed | 35,334 |
| Unique normalized texts | 29,458 |
| Exact duplicate groups | 972 |
| Similarity pairs computed, not fully materialized | 421,478 |
| Similarity groups | 1,452 |
| Multi-account similarity groups | 1,241 |
| Screenshot-eligible substantive groups | 734 |
| Group member rows | 10,770 |
| Pair evidence sample rows | 5,000 |
| Screenshot queue rows | 933 |
| Near-exact pairs | 29,754 |
| High-similarity pairs | 75,132 |
| Presentation candidate examples | 30 |

Similarity is comment-level evidence only. The canonical output is now multi-comment similarity groups rather than
a full pairwise CSV. The full pair set is computed for summary metrics, but only a capped pair-evidence sample is
materialized. It does not change LCN, HCC, actor type, or sentiment.

## Key Output Locations

| Stage | Output |
|---|---|
| RM1 rerun log | `output/research_rerun/indobert_v3/01_rm1_execution.log` |
| RM1 LCN/HCC Gephi | `output/gephi/` |
| RM1 tables | `output/tables/` |
| RM1 temporal | `output/rm1_temporal/` |
| V3 human registry | `output/rm2_sentiment/validation/human_v3/` |
| V3 development experiments | `output/rm2_sentiment/experiments/indobert_v3/` |
| V3 frozen candidate | `output/rm2_sentiment/model/indobert_v3_candidate/` |
| V2 vs V3 same-test evaluation | `output/rm2_sentiment/experiments/indobert_v3/final_test_evaluation/` |
| Final accepted sentiment | `output/rm2_sentiment/final/` |
| Actor type | `output/rm2_actor_type/` |
| Community-Mass evidence | `output/rm2_actor_type/account_interaction/` |
| Comment similarity groups | `output/rm2_comment_similarity/comment_similarity_groups.csv` |
| Comment similarity members | `output/rm2_comment_similarity/comment_similarity_group_members.csv` |
| TikTok screenshot queue | `output/rm2_comment_similarity/comment_similarity_screenshot_queue.csv` |

## Integrity Confirmations

- No source dataset was overwritten.
- Sentiment V2 training, freeze, and one-time locked-test scripts were not rerun.
- V2 artifacts were retained as baseline and provenance.
- The locked test was not used for tuning or candidate selection.
- Human annotation/adjudication remained the ground truth.
- No prediction was forced into Positive.
- No pseudo-label, lexicon, older model output, or LLM output was used as final-test ground truth.
- No test leakage was detected in the V3 registry split checks.
- HCC is reported only as indikasi koordinasi and pola teramati, with context through asosiasi konteks brand/video, kesamaan narasi, and orientasi pesan.
