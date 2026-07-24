"""Central repository path configuration.

This module is intentionally side-effect free: importing it does not create
directories, read data, change the working directory, or alter environment
variables.
"""

from __future__ import annotations

from pathlib import Path


def find_project_root(start: Path | None = None) -> Path:
    """Find the repository root from a starting path.

    The resolver searches upward for stable repository markers. It does not
    depend on a drive letter, username, notebook location, or current working
    directory alone.
    """

    current = Path(start or __file__).resolve()
    if current.is_file():
        current = current.parent

    for candidate in (current, *current.parents):
        has_readme = (candidate / "README.md").exists()
        has_dataset = (candidate / "dataset.csv").exists()
        has_scripts = (candidate / "scripts").is_dir()
        has_git = (candidate / ".git").exists()
        if has_readme and has_dataset and has_scripts and (has_git or candidate.name):
            return candidate

    raise RuntimeError(
        "Project root tidak ditemukan. Jalankan dari dalam repository "
        "Analisis-Astroturfing."
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = PROJECT_ROOT / "dataset.csv"
VIDEO_METADATA_PATH = PROJECT_ROOT / "video_metadata_clean.csv"
CONFIG_DIR = PROJECT_ROOT / "config"
INDIVIDUAL_ACTOR_REGISTRY_PATH = CONFIG_DIR / "individual_actor_registry.csv"
OUTPUT_ROOT = PROJECT_ROOT / "output"

RM1_TABLES_DIR = OUTPUT_ROOT / "tables"
RM1_GEPHI_DIR = OUTPUT_ROOT / "gephi"
RM1_VISUALIZATION_DIR = OUTPUT_ROOT / "visualisasi"
RM1_TEMPORAL_DIR = OUTPUT_ROOT / "rm1_temporal"
RM1_TEMPORAL_TABLES_DIR = RM1_TEMPORAL_DIR / "tables"
RM1_TEMPORAL_VISUALIZATION_DIR = RM1_TEMPORAL_DIR / "visualisasi"

RM2_ACTOR_TYPE_DIR = OUTPUT_ROOT / "rm2_actor_type"
RM2_ACTOR_TYPE_TABLES_DIR = RM2_ACTOR_TYPE_DIR / "tables"
RM2_ACTOR_TYPE_GEPHI_DIR = RM2_ACTOR_TYPE_DIR / "gephi"
RM2_ACTOR_TYPE_VISUALIZATION_DIR = RM2_ACTOR_TYPE_DIR / "visualisasi"
RM2_ACTOR_TYPE_AUDIT_DIR = RM2_ACTOR_TYPE_DIR / "audit"
RM2_ACTOR_TYPE_ACCOUNT_INTERACTION_DIR = RM2_ACTOR_TYPE_DIR / "account_interaction"
RM2_ACTOR_TYPE_DIRECT_INTERACTION_DIR = RM2_ACTOR_TYPE_DIR / "direct_interaction"

RM2_COMMENT_SIMILARITY_DIR = OUTPUT_ROOT / "rm2_comment_similarity"
RM2_COMMENT_SIMILARITY_PRESENTATION_DIR = RM2_COMMENT_SIMILARITY_DIR / "presentation"
RM2_COMMENT_SIMILARITY_PAIRS_PATH = RM2_COMMENT_SIMILARITY_DIR / "comment_similarity_pairs_all.csv"

RM2_SENTIMENT_ROOT = OUTPUT_ROOT / "rm2_sentiment"
RM2_SENTIMENT_FINAL_DIR = RM2_SENTIMENT_ROOT / "final"
RM2_SENTIMENT_FINAL_TABLES_DIR = RM2_SENTIMENT_FINAL_DIR / "tables"
RM2_SENTIMENT_FINAL_GEPHI_DIR = RM2_SENTIMENT_FINAL_DIR / "gephi"
RM2_SENTIMENT_FINAL_PRESENTATION_DIR = RM2_SENTIMENT_FINAL_DIR / "presentation"
RM2_SENTIMENT_FINAL_VISUALIZATION_DIR = RM2_SENTIMENT_FINAL_DIR / "visualisasi"
RM2_SENTIMENT_MODEL_DIR = RM2_SENTIMENT_ROOT / "model" / "frozen"
RM2_SENTIMENT_VALIDATION_DIR = RM2_SENTIMENT_ROOT / "validation"
RM2_SENTIMENT_HUMAN_VALIDATION_DIR = RM2_SENTIMENT_VALIDATION_DIR / "human_v1"
RM2_SENTIMENT_HUMAN_VALIDATION_V2_DIR = RM2_SENTIMENT_VALIDATION_DIR / "human_v2"
RM2_SENTIMENT_LEGACY_DIR = RM2_SENTIMENT_ROOT / "legacy"
RM2_SENTIMENT_LEGACY_V1_DIR = RM2_SENTIMENT_LEGACY_DIR / "v1"
RM2_SENTIMENT_LEGACY_TABLES_DIR = RM2_SENTIMENT_LEGACY_V1_DIR / "tables"
RM2_SENTIMENT_LEGACY_GEPHI_DIR = RM2_SENTIMENT_LEGACY_V1_DIR / "gephi"
RM2_SENTIMENT_LEGACY_VISUALIZATION_DIR = RM2_SENTIMENT_LEGACY_V1_DIR / "visualisasi"
RM2_SENTIMENT_EXPLORATORY_VISUALIZATION_DIR = RM2_SENTIMENT_LEGACY_DIR / "exploratory" / "visualisasi"

DOCS_DIR = PROJECT_ROOT / "docs"
REPOSITORY_AUDIT_DIR = DOCS_DIR / "repository_audit"
ARCHIVE_DIR = PROJECT_ROOT / "archive"
NETWORK_PROJECTS_DIR = PROJECT_ROOT / "artifacts" / "network_projects"

NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
RM1_NOTEBOOK_PATH = NOTEBOOKS_DIR / "rm1" / "tiktok_coordination_analysis.ipynb"
RM2_ACTOR_TYPE_NOTEBOOK_PATH = NOTEBOOKS_DIR / "rm2" / "03_rm2_actor_type_typology.ipynb"
RM2_SENTIMENT_NOTEBOOK_PATH = NOTEBOOKS_DIR / "rm2" / "02_rm2_sentiment_analysis.ipynb"
LEGACY_RM2_SENTIMENT_NOTEBOOK_PATH = NOTEBOOKS_DIR / "legacy" / "02_rm2_sentiment_goals.ipynb"

ALLOWED_OUTPUT_ROOTS = (
    RM1_TABLES_DIR,
    RM1_GEPHI_DIR,
    RM1_VISUALIZATION_DIR,
    RM1_TEMPORAL_TABLES_DIR,
    RM1_TEMPORAL_VISUALIZATION_DIR,
    RM2_ACTOR_TYPE_TABLES_DIR,
    RM2_ACTOR_TYPE_GEPHI_DIR,
    RM2_ACTOR_TYPE_VISUALIZATION_DIR,
    RM2_ACTOR_TYPE_AUDIT_DIR,
    RM2_ACTOR_TYPE_ACCOUNT_INTERACTION_DIR,
    RM2_ACTOR_TYPE_DIRECT_INTERACTION_DIR,
    RM2_COMMENT_SIMILARITY_DIR,
    RM2_COMMENT_SIMILARITY_PRESENTATION_DIR,
    RM2_SENTIMENT_LEGACY_VISUALIZATION_DIR,
    RM2_SENTIMENT_EXPLORATORY_VISUALIZATION_DIR,
    RM2_SENTIMENT_HUMAN_VALIDATION_DIR,
    RM2_SENTIMENT_HUMAN_VALIDATION_V2_DIR,
    RM2_SENTIMENT_MODEL_DIR,
    RM2_SENTIMENT_FINAL_DIR,
    RM2_SENTIMENT_FINAL_TABLES_DIR,
    RM2_SENTIMENT_FINAL_GEPHI_DIR,
    RM2_SENTIMENT_FINAL_PRESENTATION_DIR,
    RM2_SENTIMENT_FINAL_VISUALIZATION_DIR,
    REPOSITORY_AUDIT_DIR,
)

FROZEN_OUTPUT_ROOTS = (
    RM1_TABLES_DIR,
    RM1_GEPHI_DIR,
    RM1_VISUALIZATION_DIR,
    RM1_TEMPORAL_DIR,
    RM2_ACTOR_TYPE_DIR,
    RM2_COMMENT_SIMILARITY_DIR,
    RM2_SENTIMENT_HUMAN_VALIDATION_V2_DIR,
    RM2_SENTIMENT_MODEL_DIR,
    RM2_SENTIMENT_FINAL_DIR,
)


def relative_to_root(path: Path) -> str:
    """Return a POSIX-style path relative to the project root."""

    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
