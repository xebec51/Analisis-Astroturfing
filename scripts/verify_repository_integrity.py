"""Read-only repository integrity verifier.

The verifier does not train, evaluate, infer, execute notebooks, or regenerate
scientific outputs. It only reads files and writes repository-audit reports.
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from scripts.project_paths import (  # noqa: E402
    DATASET_PATH,
    INDIVIDUAL_ACTOR_REGISTRY_PATH,
    LEGACY_RM2_SENTIMENT_NOTEBOOK_PATH,
    PROJECT_ROOT,
    REPOSITORY_AUDIT_DIR,
    REPOSITORY_INTEGRITY_DIR,
    RM1_GEPHI_DIR,
    RM1_NOTEBOOK_PATH,
    RM2_ACTOR_TYPE_ACCOUNT_INTERACTION_DIR,
    RM2_ACTOR_TYPE_NOTEBOOK_PATH,
    RM2_ACTOR_TYPE_TABLES_DIR,
    RM2_COMMENT_SIMILARITY_PAIRS_PATH,
    RM2_SENTIMENT_FINAL_TABLES_DIR,
    RM2_SENTIMENT_FINAL_V2_DIR,
    RM2_SENTIMENT_HUMAN_VALIDATION_V2_DIR,
    RM2_SENTIMENT_MODEL_V2_DIR,
    VIDEO_METADATA_PATH,
)


EXPECTED = {
    "dataset_rows": 33847,
    "unique_comment_id": 33847,
    "observational": 33063,
    "inj": 784,
    "lcn_nodes": 724,
    "lcn_edges": 1357,
    "hcc_count": 42,
    "hcc_members": 218,
    "individual_actor": 43,
    "community_actor": 218,
    "mass_actor": 26166,
    "total_actors": 26427,
    "community_mass_pairs": 434823,
    "lcn_community_mass": 305,
    "pre_lcn_multi": 2667,
    "pre_lcn_single": 431851,
    "locked_test": 300,
    "positive": 2718,
    "neutral": 23977,
    "negative": 4771,
    "uncertain": 1593,
    "no_text": 4,
    "hcc_comments": 945,
    "non_hcc_comments": 32118,
    "goal_orientation_total": 42,
}
EXPECTED_MODEL_HASH = "477bfe11ebc3463eeff5a5fb8359f86022d719c8bea49dfdad0460fa0c5e2ccc"
EXPECTED_THRESHOLD = 0.42


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def add_check(rows: list[dict[str, Any]], metric: str, expected: Any, observed: Any, passed: bool, notes: str = "") -> None:
    rows.append(
        {
            "metric": metric,
            "expected": expected,
            "observed": observed,
            "passed": bool(passed),
            "notes": notes,
        }
    )


def read_metric_csv(path: Path) -> dict[str, str]:
    df = pd.read_csv(path, dtype=str)
    return dict(zip(df["metric"], df["value"]))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def git_lines(args: list[str]) -> list[str]:
    out = subprocess.check_output(["git", *args], cwd=PROJECT_ROOT, text=True, encoding="utf-8", errors="replace")
    return [line.strip() for line in out.splitlines() if line.strip()]


def bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def verify_hashes() -> tuple[pd.DataFrame, bool]:
    before_path = REPOSITORY_AUDIT_DIR / "critical_artifact_hashes_before.csv"
    migration_path = REPOSITORY_AUDIT_DIR / "legacy_migration_map.csv"
    before = pd.read_csv(before_path, dtype=str)
    migration = pd.read_csv(migration_path, dtype=str) if migration_path.exists() else pd.DataFrame()
    move_map = dict(zip(migration.get("old_path", []), migration.get("new_path", [])))

    rows: list[dict[str, Any]] = []
    for _, row in before.iterrows():
        old_path = str(row["path"])
        new_path = move_map.get(old_path, old_path)
        target = PROJECT_ROOT / new_path
        exists = target.exists()
        sha_after = sha256_file(target) if exists and target.is_file() else ""
        expected_unchanged = True
        passed = exists and sha_after == row["sha256_before"]
        rows.append(
            {
                "path_before": old_path,
                "path_after": new_path,
                "sha256_before": row["sha256_before"],
                "sha256_after": sha_after,
                "expected_unchanged": expected_unchanged,
                "content_changed": sha_after != row["sha256_before"],
                "reason": "moved with git mv" if old_path != new_path else "critical artifact expected unchanged",
                "passed": bool(passed),
            }
        )

    result = pd.DataFrame(rows)
    REPOSITORY_AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(REPOSITORY_AUDIT_DIR / "critical_artifact_hash_verification.csv", index=False)
    return result, bool(result["passed"].all())


def verify() -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    critical_files = [
        DATASET_PATH,
        VIDEO_METADATA_PATH,
        INDIVIDUAL_ACTOR_REGISTRY_PATH,
        RM1_NOTEBOOK_PATH,
        RM2_ACTOR_TYPE_NOTEBOOK_PATH,
        LEGACY_RM2_SENTIMENT_NOTEBOOK_PATH,
        RM2_COMMENT_SIMILARITY_PAIRS_PATH,
        RM2_SENTIMENT_MODEL_V2_DIR / "selected_model_development_frozen.joblib",
        RM2_SENTIMENT_MODEL_V2_DIR / "final_locked_test_evaluation_lock.json",
        RM2_SENTIMENT_FINAL_V2_DIR / "comment_sentiment_v2_observational.csv",
    ]
    for path in critical_files:
        add_check(rows, f"exists:{path.relative_to(PROJECT_ROOT).as_posix()}", "present", path.exists(), path.exists())

    dataset = pd.read_csv(DATASET_PATH, dtype=str, low_memory=False)
    add_check(rows, "dataset_rows", EXPECTED["dataset_rows"], len(dataset), len(dataset) == EXPECTED["dataset_rows"])
    add_check(rows, "unique_comment_id", EXPECTED["unique_comment_id"], dataset["comment_id"].nunique(), dataset["comment_id"].nunique() == EXPECTED["unique_comment_id"])

    obs = pd.read_csv(RM2_SENTIMENT_FINAL_V2_DIR / "comment_sentiment_v2_observational.csv", dtype=str, low_memory=False)
    inj = pd.read_csv(RM2_SENTIMENT_FINAL_V2_DIR / "comment_sentiment_v2_injected_diagnostic.csv", dtype=str, low_memory=False)
    add_check(rows, "observational_rows", EXPECTED["observational"], len(obs), len(obs) == EXPECTED["observational"])
    add_check(rows, "inj_rows", EXPECTED["inj"], len(inj), len(inj) == EXPECTED["inj"])
    add_check(rows, "observational_unique_comment_id", EXPECTED["observational"], obs["comment_id"].nunique(), obs["comment_id"].nunique() == EXPECTED["observational"])
    add_check(rows, "observational_synthetic_ids", 0, int(obs["comment_id"].astype(str).str.match(r"(?i)^INJ").sum()), int(obs["comment_id"].astype(str).str.match(r"(?i)^INJ").sum()) == 0)

    label_counts = obs["final_sentiment_label"].value_counts().to_dict()
    add_check(rows, "sentiment_positive", EXPECTED["positive"], int(label_counts.get("Positive", 0)), int(label_counts.get("Positive", 0)) == EXPECTED["positive"])
    add_check(rows, "sentiment_neutral", EXPECTED["neutral"], int(label_counts.get("Neutral", 0)), int(label_counts.get("Neutral", 0)) == EXPECTED["neutral"])
    add_check(rows, "sentiment_negative", EXPECTED["negative"], int(label_counts.get("Negative", 0)), int(label_counts.get("Negative", 0)) == EXPECTED["negative"])
    add_check(rows, "sentiment_uncertain", EXPECTED["uncertain"], int(label_counts.get("Uncertain", 0)), int(label_counts.get("Uncertain", 0)) == EXPECTED["uncertain"])
    add_check(rows, "sentiment_no_text", EXPECTED["no_text"], int(label_counts.get("No Text", 0)), int(label_counts.get("No Text", 0)) == EXPECTED["no_text"])

    lcn_nodes = pd.read_csv(RM1_GEPHI_DIR / "gephi_lcn_nodes.csv")
    lcn_edges = pd.read_csv(RM1_GEPHI_DIR / "gephi_lcn_edges.csv")
    hcc_nodes = pd.read_csv(RM1_GEPHI_DIR / "gephi_hcc_nodes.csv")
    add_check(rows, "lcn_nodes", EXPECTED["lcn_nodes"], len(lcn_nodes), len(lcn_nodes) == EXPECTED["lcn_nodes"])
    add_check(rows, "lcn_edges", EXPECTED["lcn_edges"], len(lcn_edges), len(lcn_edges) == EXPECTED["lcn_edges"])
    add_check(rows, "hcc_count", EXPECTED["hcc_count"], hcc_nodes["community"].nunique(), hcc_nodes["community"].nunique() == EXPECTED["hcc_count"])
    add_check(rows, "hcc_members", EXPECTED["hcc_members"], len(hcc_nodes), len(hcc_nodes) == EXPECTED["hcc_members"])

    actor = pd.read_csv(RM2_ACTOR_TYPE_TABLES_DIR / "actor_type_universe_summary.csv")
    actor_map = dict(zip(actor["actor_type_primary"], actor["n_accounts"]))
    add_check(rows, "individual_actor", EXPECTED["individual_actor"], int(actor_map.get("Individual Actor", -1)), int(actor_map.get("Individual Actor", -1)) == EXPECTED["individual_actor"])
    add_check(rows, "community_actor", EXPECTED["community_actor"], int(actor_map.get("Community Actor", -1)), int(actor_map.get("Community Actor", -1)) == EXPECTED["community_actor"])
    add_check(rows, "mass_actor", EXPECTED["mass_actor"], int(actor_map.get("Mass Actor", -1)), int(actor_map.get("Mass Actor", -1)) == EXPECTED["mass_actor"])
    add_check(rows, "total_actors", EXPECTED["total_actors"], int(actor_map.get("Total", -1)), int(actor_map.get("Total", -1)) == EXPECTED["total_actors"])

    cm_summary = read_metric_csv(RM2_ACTOR_TYPE_ACCOUNT_INTERACTION_DIR / "community_mass_account_summary.csv")
    scope = pd.read_csv(RM2_ACTOR_TYPE_ACCOUNT_INTERACTION_DIR / "community_mass_by_interaction_scope.csv")
    scope_map = dict(zip(scope["interaction_scope"], scope["n_pairs"]))
    add_check(rows, "community_mass_pairs", EXPECTED["community_mass_pairs"], int(cm_summary["total_unique_community_mass_account_pairs"]), int(cm_summary["total_unique_community_mass_account_pairs"]) == EXPECTED["community_mass_pairs"])
    add_check(rows, "lcn_community_mass", EXPECTED["lcn_community_mass"], int(scope_map.get("LCN_EDGE", -1)), int(scope_map.get("LCN_EDGE", -1)) == EXPECTED["lcn_community_mass"])
    add_check(rows, "pre_lcn_multi", EXPECTED["pre_lcn_multi"], int(scope_map.get("PRE_LCN_MULTI_EVIDENCE", -1)), int(scope_map.get("PRE_LCN_MULTI_EVIDENCE", -1)) == EXPECTED["pre_lcn_multi"])
    add_check(rows, "pre_lcn_single", EXPECTED["pre_lcn_single"], int(scope_map.get("PRE_LCN_SINGLE_EVIDENCE", -1)), int(scope_map.get("PRE_LCN_SINGLE_EVIDENCE", -1)) == EXPECTED["pre_lcn_single"])

    locked = pd.read_csv(RM2_SENTIMENT_HUMAN_VALIDATION_V2_DIR / "locked_test_v2_observational_final.csv", dtype=str)
    training = pd.read_csv(RM2_SENTIMENT_MODEL_V2_DIR / "development_training_pool_provenance.csv", dtype=str)
    training_ids = set(training.loc[bool_series(training["included_in_training"]), "comment_id"].astype(str))
    locked_ids = set(locked["comment_id"].astype(str))
    synthetic_locked = int(locked["comment_id"].astype(str).str.match(r"(?i)^INJ").sum())
    overlap_training = len(training_ids & locked_ids)
    add_check(rows, "locked_test_rows", EXPECTED["locked_test"], len(locked), len(locked) == EXPECTED["locked_test"])
    add_check(rows, "locked_test_synthetic_ids", 0, synthetic_locked, synthetic_locked == 0)
    add_check(rows, "locked_test_training_overlap", 0, overlap_training, overlap_training == 0)

    lock = read_json(RM2_SENTIMENT_MODEL_V2_DIR / "final_locked_test_evaluation_lock.json")
    add_check(rows, "locked_test_status", "FINAL_LOCKED_TEST_EVALUATED_ONCE", lock.get("status"), lock.get("status") == "FINAL_LOCKED_TEST_EVALUATED_ONCE")
    add_check(rows, "model_hash", EXPECTED_MODEL_HASH, lock.get("model_sha256"), lock.get("model_sha256") == EXPECTED_MODEL_HASH)
    add_check(rows, "threshold", EXPECTED_THRESHOLD, lock.get("threshold"), float(lock.get("threshold")) == EXPECTED_THRESHOLD)

    hcc_summary = pd.read_csv(RM2_SENTIMENT_FINAL_TABLES_DIR / "hcc_sentiment_goals_summary_v2.csv")
    add_check(rows, "goal_orientation_total", EXPECTED["goal_orientation_total"], len(hcc_summary), len(hcc_summary) == EXPECTED["goal_orientation_total"])
    goal_counts = hcc_summary["goal_orientation"].value_counts().to_dict()
    add_check(rows, "goal_neutral_engagement", 31, int(goal_counts.get("Neutral Engagement", 0)), int(goal_counts.get("Neutral Engagement", 0)) == 31)
    add_check(rows, "goal_promotional_supportive", 11, int(goal_counts.get("Promotional / Supportive", 0)), int(goal_counts.get("Promotional / Supportive", 0)) == 11)

    hcc_non = pd.read_csv(RM2_SENTIMENT_FINAL_TABLES_DIR / "hcc_vs_nonhcc_comment_sentiment_v2.csv")
    group_total = dict(zip(hcc_non["group"], hcc_non["total_comments"]))
    add_check(rows, "hcc_comments", EXPECTED["hcc_comments"], int(group_total.get("HCC", -1)), int(group_total.get("HCC", -1)) == EXPECTED["hcc_comments"])
    add_check(rows, "non_hcc_comments", EXPECTED["non_hcc_comments"], int(group_total.get("Non-HCC", -1)), int(group_total.get("Non-HCC", -1)) == EXPECTED["non_hcc_comments"])

    add_check(rows, "comment_similarity_lfs_path_exists", "present", RM2_COMMENT_SIMILARITY_PAIRS_PATH.exists(), RM2_COMMENT_SIMILARITY_PAIRS_PATH.exists())
    lfs_lines = git_lines(["lfs", "ls-files"])
    lfs_paths = {line.split()[-1].replace("\\", "/") for line in lfs_lines if line.split()}
    add_check(rows, "comment_similarity_lfs_tracked", True, RM2_COMMENT_SIMILARITY_PAIRS_PATH.as_posix().replace(PROJECT_ROOT.as_posix() + "/", "") in lfs_paths, RM2_COMMENT_SIMILARITY_PAIRS_PATH.relative_to(PROJECT_ROOT).as_posix() in lfs_paths)

    tracked = git_lines(["ls-files"])
    pycache_tracked = [p for p in tracked if "__pycache__" in p]
    pyc_tracked = [p for p in tracked if p.endswith((".pyc", ".pyo", ".pyd"))]
    private_tracked = [p for p in tracked if "/private/" in p.replace("\\", "/")]
    add_check(rows, "tracked_pycache", 0, len(pycache_tracked), len(pycache_tracked) == 0, "; ".join(pycache_tracked[:5]))
    add_check(rows, "tracked_pyc", 0, len(pyc_tracked), len(pyc_tracked) == 0, "; ".join(pyc_tracked[:5]))
    add_check(rows, "tracked_private_output", 0, len(private_tracked), len(private_tracked) == 0, "; ".join(private_tracked[:5]))

    notebooks = [RM1_NOTEBOOK_PATH, RM2_ACTOR_TYPE_NOTEBOOK_PATH, LEGACY_RM2_SENTIMENT_NOTEBOOK_PATH]
    notebook_json_valid = True
    for notebook in notebooks:
        try:
            read_json(notebook)
        except Exception:
            notebook_json_valid = False
    add_check(rows, "notebook_json_valid", True, notebook_json_valid, notebook_json_valid)

    path_audit = pd.read_csv(REPOSITORY_AUDIT_DIR / "notebook_path_audit.csv")
    add_check(rows, "notebook_path_audit_failures", 0, int(path_audit["status"].eq("PATH_CONTRACT_FAIL").sum()), int(path_audit["status"].eq("PATH_CONTRACT_FAIL").sum()) == 0)
    add_check(rows, "absolute_local_paths_in_notebooks", 0, int(path_audit["absolute_windows_path_count"].sum() + path_audit["absolute_posix_path_count"].sum()), int(path_audit["absolute_windows_path_count"].sum() + path_audit["absolute_posix_path_count"].sum()) == 0)
    add_check(rows, "unregistered_notebook_outputs", 0, int(path_audit["unregistered_output_path_count"].sum()), int(path_audit["unregistered_output_path_count"].sum()) == 0)

    hash_df, hash_passed = verify_hashes()
    add_check(rows, "critical_artifact_hashes_unchanged", True, hash_passed, hash_passed, f"{len(hash_df)} critical hash rows")

    report = pd.DataFrame(rows)
    summary = {
        "overall_status": "PASS" if bool(report["passed"].all()) else "FAIL",
        "n_checks": int(len(report)),
        "n_passed": int(report["passed"].sum()),
        "n_failed": int((~report["passed"]).sum()),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip(),
        "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=PROJECT_ROOT, text=True).strip(),
        "notes": "Read-only repository integrity verification; no analysis pipeline was executed.",
    }
    return report, summary


def main() -> int:
    REPOSITORY_INTEGRITY_DIR.mkdir(parents=True, exist_ok=True)
    report, summary = verify()
    report.to_csv(REPOSITORY_INTEGRITY_DIR / "repository_integrity_report.csv", index=False)
    (REPOSITORY_INTEGRITY_DIR / "repository_integrity_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Repository integrity status: {summary['overall_status']}")
    print(f"Checks: {summary['n_passed']}/{summary['n_checks']} passed")
    return 0 if summary["overall_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
