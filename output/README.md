# Output Directory

This directory contains research outputs and repository audit outputs. Most scientific outputs are frozen and
should not be regenerated during repository organization.

| path | research module | status | canonical/legacy/exploratory | producer | immutable | primary files | consumer | notes |
|---|---|---|---|---|---|---|---|---|
| `output/tables/` | RM1 | frozen | canonical | RM1 notebook | Yes | evidence, HCC, brand, hashtag tables | RM1/RM2 | Root RM1 output retained. |
| `output/gephi/` | RM1 | frozen | canonical | RM1 notebook | Yes | LCN/HCC node and edge CSV | Gephi/RM2 | CSV is canonical network data. |
| `output/visualisasi/` | RM1 | frozen | canonical | RM1 notebook | Yes | PNG visualizations | report/presentation | PNG only for final visual outputs. |
| `output/rm1_temporal/` | RM1 temporal | frozen | canonical | RM1 temporal section | Yes | temporal tables and line charts | report | Uses WIB. |
| `output/rm2_actor_type/` | RM2 actor type | frozen | canonical + legacy diagnostic | actor type notebook/scripts | Yes | actor type tables, Gephi, Community-Mass | RM2 | Direct reply subfolder is legacy diagnostic. |
| `output/rm2_comment_similarity/` | RM2 comment similarity | frozen | canonical | similarity script | Yes | pair, group, cluster, presentation files | presentation | `comment_similarity_pairs_all.csv` is LFS tracked. |
| `output/rm2_sentiment/` | RM2 sentiment | mixed | legacy + V2 final | sentiment scripts/notebook | Yes for final artifacts | V1 legacy, V2 validation/model/final | RM2 | See nested README. |
