# Network Project Files

This directory stores manual network visualization project files. These files are preserved for audit and
visual inspection, but they are not canonical data sources.

| file | software | research module | source node CSV | source edge CSV | purpose |
|---|---|---|---|---|---|
| `HCC.gephi` | Gephi | RM1 HCC | `output/gephi/gephi_hcc_nodes.csv` | `output/gephi/gephi_hcc_edges.csv` | Manual HCC network visualization project. |
| `Type Actor.gephi` | Gephi | RM2 actor type | `output/rm2_actor_type/gephi/gephi_actor_type_nodes.csv` | `output/rm2_actor_type/gephi/gephi_actor_type_edges.csv` | Manual actor type visualization project. |
| `HCC_Visone.graphmlz` | Visone | RM1 HCC | `output/gephi/gephi_hcc_nodes.csv` | `output/gephi/gephi_hcc_edges.csv` | Visone project/export for HCC visualization. |

CSV files in `output/` remain the canonical source for network data. Project files may contain manual layout,
styling, and viewport choices, so they should be interpreted as visualization projects rather than raw data.

Moving these files into this directory did not change their contents; hashes are recorded in
`docs/repository_audit/legacy_migration_map.csv`.
