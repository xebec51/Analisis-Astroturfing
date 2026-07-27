from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output/rm2_sentiment/validation/human_master_v4"
MASTER = OUT_DIR / "human_annotation_master_v4.csv"
DEV_A1 = OUT_DIR / "sentiment_v4_development_annotator_1.xlsx"
DEV_A2 = OUT_DIR / "sentiment_v4_development_annotator_2.xlsx"
DEV_ADJ = OUT_DIR / "sentiment_v4_development_adjudication.xlsx"
LOCK_A1 = OUT_DIR / "sentiment_v4_locked_test_annotator_1.xlsx"
LOCK_A2 = OUT_DIR / "sentiment_v4_locked_test_annotator_2.xlsx"
LOCK_ADJ = OUT_DIR / "sentiment_v4_locked_test_adjudication.xlsx"

OUT_DEV_FINAL = OUT_DIR / "sentiment_v4_development_final_registry.csv"
OUT_LOCKED_FINAL = OUT_DIR / "sentiment_v4_locked_test_final_frozen.csv"
OUT_IMPORT_REPORT = OUT_DIR / "sentiment_v4_import_report.csv"
OUT_IMPORT_MANIFEST = OUT_DIR / "SENTIMENT_V4_COMPLETED_IMPORT_MANIFEST.json"

LABELS = ["Negative", "Neutral", "Positive", "Uncertain", "No Text"]
EVALUABLE = ["Negative", "Neutral", "Positive"]
ANNOTATOR_COLUMNS = [
    "annotation_id",
    "comment_id",
    "video_id",
    "product_category",
    "brand_or_video_context",
    "comment_text",
    "human_label",
    "confidence_annotation",
    "needs_context",
    "annotator_notes",
]


def read_master() -> pd.DataFrame:
    return pd.read_csv(MASTER, dtype=str, keep_default_na=False, low_memory=False)


def read_workbook(path: Path, required_columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    sheets = pd.read_excel(path, sheet_name=None, dtype=str, keep_default_na=False)
    required = set(required_columns)
    for frame in sheets.values():
        if required.issubset(set(frame.columns)):
            return frame
    raise AssertionError(f"{path.name} does not contain a sheet with columns: {required_columns}")


def labels_from_pair(a1: pd.DataFrame, a2: pd.DataFrame, adj: pd.DataFrame) -> pd.DataFrame:
    left = a1[["annotation_id", "human_label", "annotator_notes"]].rename(
        columns={"human_label": "annotator_1_label_imported", "annotator_notes": "annotator_1_notes"}
    )
    right = a2[["annotation_id", "human_label", "annotator_notes"]].rename(
        columns={"human_label": "annotator_2_label_imported", "annotator_notes": "annotator_2_notes"}
    )
    merged = left.merge(right, on="annotation_id", how="outer")
    adj_cols = [col for col in ["annotation_id", "adjudicated_label", "adjudication_notes"] if col in adj.columns]
    if adj_cols:
        adj_frame = adj[adj_cols].rename(
            columns={"adjudicated_label": "adjudicated_label_imported", "adjudication_notes": "adjudication_notes_imported"}
        )
        merged = merged.merge(adj_frame, on="annotation_id", how="left")
    else:
        merged["adjudicated_label_imported"] = ""
        merged["adjudication_notes_imported"] = ""
    merged["annotator_1_label_imported"] = merged["annotator_1_label_imported"].fillna("")
    merged["annotator_2_label_imported"] = merged["annotator_2_label_imported"].fillna("")
    merged["adjudicated_label_imported"] = merged["adjudicated_label_imported"].fillna("")
    same = merged["annotator_1_label_imported"].eq(merged["annotator_2_label_imported"]) & merged["annotator_1_label_imported"].isin(LABELS)
    adjudicated = merged["adjudicated_label_imported"].isin(LABELS)
    merged["final_human_label_imported"] = ""
    merged.loc[same, "final_human_label_imported"] = merged.loc[same, "annotator_1_label_imported"]
    merged.loc[~same & adjudicated, "final_human_label_imported"] = merged.loc[~same & adjudicated, "adjudicated_label_imported"]
    merged["disagreement_flag_imported"] = merged["annotator_1_label_imported"].ne(merged["annotator_2_label_imported"])
    merged["adjudication_required_imported"] = merged["disagreement_flag_imported"] & merged["final_human_label_imported"].eq("")
    return merged


def import_scope(master: pd.DataFrame, role: str, a1_path: Path, a2_path: Path, adj_path: Path) -> pd.DataFrame:
    imported = labels_from_pair(
        read_workbook(a1_path, ANNOTATOR_COLUMNS),
        read_workbook(a2_path, ANNOTATOR_COLUMNS),
        read_workbook(adj_path, ["annotation_id"]),
    )
    rows = master.loc[master["annotation_role"].eq(role)].copy()
    rows = rows.merge(imported, left_on="master_annotation_id", right_on="annotation_id", how="left")
    rows["annotator_1_label"] = rows["annotator_1_label_imported"] if "annotator_1_label_imported" in rows else rows["annotator_1_label"]
    rows["annotator_2_label"] = rows["annotator_2_label_imported"] if "annotator_2_label_imported" in rows else rows["annotator_2_label"]
    rows["final_human_label"] = rows["final_human_label_imported"].fillna("")
    rows["adjudicated_label"] = rows["adjudicated_label_imported"].fillna("")
    rows["disagreement_flag"] = rows["disagreement_flag_imported"].fillna(False)
    rows["adjudication_required"] = rows["adjudication_required_imported"].fillna(False)
    rows["evaluable_three_class"] = rows["final_human_label"].isin(EVALUABLE)
    rows["annotation_status"] = rows["final_human_label"].map(lambda x: "FINAL_HUMAN_IMPORTED" if x in LABELS else "PENDING_ADJUDICATION")
    return rows


def count_metrics(scope: str, frame: pd.DataFrame) -> list[dict[str, object]]:
    counts = frame["final_human_label"].value_counts()
    rows: list[dict[str, object]] = [
        {"metric": f"{scope}_rows", "value": len(frame)},
        {"metric": f"{scope}_evaluable_three_class", "value": int(frame["final_human_label"].isin(EVALUABLE).sum())},
        {"metric": f"{scope}_blank_final_label", "value": int(frame["final_human_label"].eq("").sum())},
        {"metric": f"{scope}_inj_count", "value": int(frame["comment_id"].map(lambda x: str(x).upper().startswith("INJ")).sum())},
    ]
    for label in LABELS:
        rows.append({"metric": f"{scope}_label_{label.lower().replace(' ', '_')}", "value": int(counts.get(label, 0))})
    return rows


def main() -> None:
    master = read_master()
    historical_before = master.loc[
        master["annotation_role"].eq("HISTORICAL_DEVELOPMENT_FINAL"),
        ["master_annotation_id", "final_human_label"],
    ].reset_index(drop=True)
    dev = import_scope(master, "DEVELOPMENT_NEW_PENDING", DEV_A1, DEV_A2, DEV_ADJ)
    locked = import_scope(master, "LOCKED_TEST_NEW_PENDING", LOCK_A1, LOCK_A2, LOCK_ADJ)

    unresolved = int(dev["final_human_label"].eq("").sum() + locked["final_human_label"].eq("").sum())
    report_rows = [
        {"metric": "development_rows_imported", "value": len(dev)},
        {"metric": "locked_rows_imported", "value": len(locked)},
        {"metric": "unresolved_pending_adjudication", "value": unresolved},
        {"metric": "historical_labels_overwritten", "value": 0},
    ]
    report = pd.DataFrame(report_rows)
    report.to_csv(OUT_IMPORT_REPORT, index=False, encoding="utf-8-sig")
    if unresolved:
        raise AssertionError(f"Import blocked: {unresolved} rows still lack final human labels/adjudication.")

    dev_final = pd.concat(
        [
            master.loc[master["annotation_role"].eq("HISTORICAL_DEVELOPMENT_FINAL")],
            dev.assign(annotation_role="DEVELOPMENT_NEW_FINAL", split_lock="DEVELOPMENT_FINAL_LOCKED"),
        ],
        ignore_index=True,
        sort=False,
    )
    locked_final = locked.assign(annotation_role="LOCKED_TEST_NEW_FINAL", split_lock="LOCKED_TEST_FINAL_LOCKED")

    report_rows.extend(count_metrics("development_final", dev_final))
    report_rows.extend(count_metrics("locked_test_final", locked_final))
    dev_counts = dev_final["final_human_label"].value_counts()
    locked_counts = locked_final["final_human_label"].value_counts()
    dev_target_pass = (
        int(dev_final["final_human_label"].isin(EVALUABLE).sum()) >= 1450
        and int(dev_counts.get("Negative", 0)) >= 400
        and int(dev_counts.get("Neutral", 0)) >= 650
        and int(dev_counts.get("Positive", 0)) >= 400
    )
    locked_target_pass = (
        int(locked_final["final_human_label"].isin(EVALUABLE).sum()) >= 600
        and int(locked_counts.get("Negative", 0)) >= 150
        and int(locked_counts.get("Neutral", 0)) >= 300
        and int(locked_counts.get("Positive", 0)) >= 150
    )
    report_rows.extend(
        [
            {"metric": "development_final_class_target_pass", "value": dev_target_pass},
            {"metric": "locked_test_final_class_target_pass", "value": locked_target_pass},
            {
                "metric": "locked_test_neutral_target_shortfall",
                "value": max(0, 300 - int(locked_counts.get("Neutral", 0))),
            },
        ]
    )
    report = pd.DataFrame(report_rows)
    report.to_csv(OUT_IMPORT_REPORT, index=False, encoding="utf-8-sig")

    historical_after = dev_final.loc[
        dev_final["annotation_role"].eq("HISTORICAL_DEVELOPMENT_FINAL"),
        ["master_annotation_id", "final_human_label"],
    ].reset_index(drop=True)
    if not historical_before.equals(historical_after):
        raise AssertionError("Historical labels changed during import; refusing to write outputs.")

    dev_final.to_csv(OUT_DEV_FINAL, index=False, encoding="utf-8-sig")
    locked_final.to_csv(OUT_LOCKED_FINAL, index=False, encoding="utf-8-sig")
    OUT_IMPORT_MANIFEST.write_text(
        json.dumps(
            {
                "status": "SENTIMENT_MASTER_V4_COMPLETED_LABELS_IMPORTED",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "development_final_registry": str(OUT_DEV_FINAL.relative_to(ROOT)).replace("\\", "/"),
                "locked_test_final_frozen": str(OUT_LOCKED_FINAL.relative_to(ROOT)).replace("\\", "/"),
                "counts": {
                    "development_final_rows": int(len(dev_final)),
                    "development_final_evaluable_three_class": int(dev_final["final_human_label"].isin(EVALUABLE).sum()),
                    "locked_test_final_rows": int(len(locked_final)),
                    "locked_test_final_evaluable_three_class": int(locked_final["final_human_label"].isin(EVALUABLE).sum()),
                    "locked_test_neutral_target_shortfall": max(0, 300 - int(locked_counts.get("Neutral", 0))),
                },
                "historical_labels_overwritten": False,
                "model_used_as_adjudicator": False,
                "majority_vote_used": False,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print("SENTIMENT_MASTER_V4_COMPLETED_LABELS_IMPORTED")


if __name__ == "__main__":
    main()
