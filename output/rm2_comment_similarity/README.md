# RM2 Comment Similarity Output

This folder contains comment-level exact and near-similarity outputs. The canonical analysis unit is now a
multi-comment similarity group, so repeated text patterns across many accounts are represented once as a group
rather than expanded into all possible account/comment pairs.

| file or folder | status | notes |
|---|---|---|
| `comment_similarity_groups.csv` | canonical | One row per similarity group/cluster; use this for analysis summaries. |
| `comment_similarity_group_members.csv` | canonical | One row per comment inside a similarity group, with `video_url`, candidate comment URL, account context, and sentiment attribute. |
| `comment_similarity_group_account_summary.csv` | canonical | One row per group-account combination; useful for seeing how many accounts join the same narrative pattern. |
| `comment_similarity_screenshot_queue.csv` | canonical screenshot aid | Prioritized substantive groups for TikTok lookup and screenshots. |
| `comment_similarity_pairs_all.csv` | compatibility sample | Capped pair evidence sample only; the full pairwise table is no longer materialized. |
| `comment_similarity_pair_evidence_sample.csv` | audit sample | Same capped pair evidence sample, with explicit materialization-scope column. |
| `exact_duplicate_comment_groups.csv` | frozen canonical | Exact normalized-text groups. |
| `near_similar_comment_clusters.csv` | compatibility alias | Same group-level data as `comment_similarity_groups.csv` using the legacy filename. |
| `near_similar_comment_cluster_members.csv` | compatibility alias | Same member-level data as `comment_similarity_group_members.csv` using the legacy filename. |
| `presentation/` | frozen review package | Candidate examples for manual presentation review. |
| `comment_similarity_integrity_report.csv` | frozen audit | Similarity pipeline integrity gates. |

For platform screenshots, start from `comment_similarity_screenshot_queue.csv`, open `video_url` or
`tiktok_comment_url_candidate`, then verify the `platform_username` and `comment_text_presentation` manually.

Local browser automation aid:

```powershell
python scripts/open_tiktok_similarity_comments.py --dry-run --limit 5
python scripts/open_tiktok_similarity_comments.py --group-rank 1 --limit 8
python scripts/open_tiktok_similarity_comments.py --group-rank 1-3 --channel msedge
```

The helper opens each TikTok candidate in a persistent local profile, attempts to highlight the visible
matching username/comment text, and appends review status to
`output/rm2_comment_similarity/tiktok_comment_lookup_status.csv`. Press `c` while reviewing to capture a
viewport screenshot under `output/rm2_comment_similarity/screenshots/`. TikTok may still require manual login,
comment-panel scrolling, or direct visual confirmation because comment permalinks are not always stable.

Textual similarity is descriptive evidence of kesamaan narasi and pola teramati. It is not proof of intent,
payment, affiliation, or deliberate coordination. Presentation examples remain candidates until manually reviewed.
