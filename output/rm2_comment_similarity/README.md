# RM2 Comment Similarity Output

This folder contains comment-level exact and near-similarity outputs.

| file or folder | status | notes |
|---|---|---|
| `comment_similarity_pairs_all.csv` | frozen canonical | LFS-tracked full pair table; must remain at this path. |
| `exact_duplicate_comment_groups.csv` | frozen canonical | Exact normalized-text groups. |
| `near_similar_comment_clusters.csv` | frozen canonical | Similarity clusters using thresholded pair graph. |
| `presentation/` | frozen review package | Candidate examples for manual presentation review. |
| `comment_similarity_integrity_report.csv` | frozen audit | Similarity pipeline integrity gates. |

Textual similarity is not proof of intent, payment, affiliation, or deliberate coordination. Presentation
examples remain candidates until manually reviewed.
