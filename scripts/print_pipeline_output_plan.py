"""Print a read-only output plan for repository pipelines."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from scripts.project_paths import (
    LEGACY_RM2_SENTIMENT_NOTEBOOK_PATH,
    RM1_GEPHI_DIR,
    RM1_NOTEBOOK_PATH,
    RM1_TABLES_DIR,
    RM1_TEMPORAL_DIR,
    RM1_VISUALIZATION_DIR,
    RM2_ACTOR_TYPE_ACCOUNT_INTERACTION_DIR,
    RM2_ACTOR_TYPE_DIR,
    RM2_ACTOR_TYPE_NOTEBOOK_PATH,
    RM2_COMMENT_SIMILARITY_DIR,
    RM2_SENTIMENT_FINAL_V2_DIR,
    RM2_SENTIMENT_HUMAN_VALIDATION_V2_DIR,
    RM2_SENTIMENT_LEGACY_TABLES_DIR,
    RM2_SENTIMENT_MODEL_V2_DIR,
    relative_to_root,
)


PIPELINES = [
    {
        "pipeline": "RM1 LCN/HCC",
        "entrypoint": RM1_NOTEBOOK_PATH,
        "inputs": ["dataset.csv", "video_metadata_clean.csv"],
        "outputs": [RM1_TABLES_DIR, RM1_GEPHI_DIR, RM1_VISUALIZATION_DIR],
        "artifacts": ["co_*_edges.csv", "focal_structures.csv", "gephi_lcn_*.csv", "gephi_hcc_*.csv"],
        "overwrite": "output-producing; do not rerun during cleanup",
        "status": "CANONICAL_RM1",
    },
    {
        "pipeline": "RM1 temporal",
        "entrypoint": RM1_NOTEBOOK_PATH,
        "inputs": ["output/gephi/", "output/rm2_sentiment/tables/comment_sentiment.csv"],
        "outputs": [RM1_TEMPORAL_DIR],
        "artifacts": ["temporal_*", "hcc_temporal_profile.csv"],
        "overwrite": "output-producing; do not rerun during cleanup",
        "status": "CANONICAL_RM1_TEMPORAL",
    },
    {
        "pipeline": "RM2 Sentiment V1 legacy",
        "entrypoint": LEGACY_RM2_SENTIMENT_NOTEBOOK_PATH,
        "inputs": ["RM1 outputs"],
        "outputs": [RM2_SENTIMENT_LEGACY_TABLES_DIR],
        "artifacts": ["comment_sentiment.csv", "hcc_sentiment_goals_summary.csv"],
        "overwrite": "legacy output-producing; not canonical final",
        "status": "LEGACY_V1",
    },
    {
        "pipeline": "RM2 Sentiment V2 model/final",
        "entrypoint": "scripts/train/freeze/evaluate/apply Sentiment V2 scripts",
        "inputs": [RM2_SENTIMENT_HUMAN_VALIDATION_V2_DIR, RM2_SENTIMENT_MODEL_V2_DIR],
        "outputs": [RM2_SENTIMENT_FINAL_V2_DIR],
        "artifacts": ["final_locked_test_*", "comment_sentiment_v2_observational.csv"],
        "overwrite": "frozen or already generated; do not rerun during cleanup",
        "status": "FINAL_MODEL_VALIDATED",
    },
    {
        "pipeline": "RM2 Actor Type",
        "entrypoint": RM2_ACTOR_TYPE_NOTEBOOK_PATH,
        "inputs": ["RM1 outputs", "Sentiment outputs", "config/individual_actor_registry.csv"],
        "outputs": [RM2_ACTOR_TYPE_DIR],
        "artifacts": ["actor_type_*.csv", "gephi_*actor_type*.csv"],
        "overwrite": "output-producing; do not rerun during cleanup",
        "status": "CANONICAL_RM2_ACTOR_TYPE",
    },
    {
        "pipeline": "Community-Mass account evidence",
        "entrypoint": "scripts/build_rm2_community_mass_account_network.py",
        "inputs": ["RM1 pre-filter evidence", "actor type tables"],
        "outputs": [RM2_ACTOR_TYPE_ACCOUNT_INTERACTION_DIR],
        "artifacts": ["community_mass_account_pairs.csv"],
        "overwrite": "output-producing; do not rerun during cleanup",
        "status": "CANONICAL_RM2_COMMUNITY_MASS",
    },
    {
        "pipeline": "Comment similarity",
        "entrypoint": "scripts/build_rm2_comment_similarity_examples.py",
        "inputs": ["dataset.csv", "actor type attributes"],
        "outputs": [RM2_COMMENT_SIMILARITY_DIR],
        "artifacts": ["comment_similarity_pairs_all.csv", "presentation/"],
        "overwrite": "output-producing; do not rerun during cleanup",
        "status": "CANONICAL_RM2_COMMENT_SIMILARITY",
    },
]


def render_path(value: object) -> str:
    if hasattr(value, "resolve"):
        return relative_to_root(value)  # type: ignore[arg-type]
    return str(value)


def main() -> int:
    print("PIPELINE OUTPUT PLAN (read-only)")
    print("=" * 80)
    for item in PIPELINES:
        print(f"\nPipeline: {item['pipeline']}")
        print(f"Status: {item['status']}")
        print(f"Entrypoint: {render_path(item['entrypoint'])}")
        print("Inputs: " + "; ".join(render_path(p) for p in item["inputs"]))
        print("Output directories: " + "; ".join(render_path(p) for p in item["outputs"]))
        print("Primary artifacts: " + "; ".join(item["artifacts"]))
        print(f"Overwrite policy: {item['overwrite']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
