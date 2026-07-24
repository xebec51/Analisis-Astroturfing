# Reproducibility Notes

This repository stores both code and frozen research artifacts. Reproducibility work should preserve existing
outputs unless a future analysis run is explicitly planned and documented.

## Environment

- Python version recorded by the final Sentiment V2 manifest: `3.12.2`.
- Package versions for final Sentiment V2 are stored in
  `output/rm2_sentiment/final/FINAL_SENTIMENT_ANALYSIS_MANIFEST.json`.
- Git LFS is required. The LFS-tracked artifact is
  `output/rm2_comment_similarity/comment_similarity_pairs_all.csv`.

After cloning, run:

```bash
git lfs install
git lfs pull
```

## Project Root Resolution

Code should resolve paths from the repository root, not from a machine-specific working directory. The central
path module is `scripts/project_paths.py`. It looks for repository markers such as `README.md`, `dataset.csv`,
and `scripts/`.

## Pipeline Order

The historical scientific order is:

1. RM1 LCN/Louvain/FSA_V/HCC.
2. RM1 temporal profile.
3. RM2 Sentiment V1 legacy diagnostics.
4. RM2 Sentiment V2 human validation and development model.
5. Sentiment V2 locked-test freeze and one-time evaluation.
6. Sentiment V2 final inference.
7. RM2 actor type and Gephi exports.
8. Community-Mass account evidence network.
9. Comment-level exact and near-similarity evidence package.

Repository organization does not rerun these pipelines.

## Safe and Unsafe Runs

| entrypoint | status | cleanup behavior |
|---|---|---|
| `notebooks/rm1/tiktok_coordination_analysis.ipynb` | canonical but output-producing | Do not execute during cleanup. |
| `notebooks/rm2/03_rm2_actor_type_typology.ipynb` | canonical but output-producing | Do not execute during cleanup. |
| `notebooks/legacy/02_rm2_sentiment_goals.ipynb` | legacy | Do not execute during cleanup. |
| `scripts/train_rm2_sentiment_v2_development_model.py` | frozen training | Do not rerun. |
| `scripts/freeze_rm2_sentiment_v2_locked_test.py` | frozen locked-test preparation | Do not rerun. |
| `scripts/evaluate_rm2_sentiment_v2_locked_test_once.py` | one-time evaluation | Do not rerun. |
| `scripts/apply_rm2_sentiment_v2_final_inference.py` | final inference already generated | Do not rerun during cleanup. |
| `scripts/validate_notebook_paths.py` | static notebook audit | Safe to run. |
| `scripts/print_pipeline_output_plan.py` | dry-run output plan | Safe to run. |

## Expected Scientific Counts

| metric | expected |
|---|---:|
| dataset rows | 33847 |
| unique `comment_id` | 33847 |
| observational comments | 33063 |
| INJ diagnostic comments | 784 |
| LCN nodes | 724 |
| LCN edges | 1357 |
| HCC count | 42 |
| HCC members | 218 |
| Individual Actor | 43 |
| Community Actor | 218 |
| Mass Actor | 26166 |
| total actors | 26427 |
| Community-Mass pairs | 434823 |
| LCN Community-Mass pairs | 305 |
| pre-LCN multi-evidence pairs | 2667 |
| pre-LCN single-evidence pairs | 431851 |
| locked-test rows | 300 |
| Sentiment V2 Positive | 2718 |
| Sentiment V2 Neutral | 23977 |
| Sentiment V2 Negative | 4771 |
| Sentiment V2 Uncertain | 1593 |
| Sentiment V2 No Text | 4 |
| HCC comments | 945 |
| Non-HCC comments | 32118 |
| HCC goal orientation total | 42 |

## Locked-Test Warning

The final locked test has already been evaluated once. The lock file is
`output/rm2_sentiment/model/frozen/final_locked_test_evaluation_lock.json`. Do not rerun locked-test evaluation,
do not tune threshold from locked-test labels, and do not replace the lock file.

## Output Overwrite Warning

Many notebooks and scripts can overwrite scientific outputs. During repository cleanup, only static path
refactors and documentation changes should be made unless a rerun is explicitly requested. Any future rerun
should be planned as a separate analysis change with explicit output-diff review.
