"""Static notebook path validator.

The validator parses notebook JSON and code-cell text only. It never executes
notebook cells or imports output-producing notebook code.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from scripts.project_paths import (
    ALLOWED_OUTPUT_ROOTS,
    LEGACY_RM2_SENTIMENT_NOTEBOOK_PATH,
    PROJECT_ROOT,
    REPOSITORY_AUDIT_DIR,
    RM1_NOTEBOOK_PATH,
    RM2_ACTOR_TYPE_NOTEBOOK_PATH,
    RM2_SENTIMENT_NOTEBOOK_PATH,
    relative_to_root,
)


CANONICAL_NOTEBOOKS = {
    RM1_NOTEBOOK_PATH: "RM1",
    RM2_SENTIMENT_NOTEBOOK_PATH: "RM2_SENTIMENT",
    RM2_ACTOR_TYPE_NOTEBOOK_PATH: "RM2_ACTOR_TYPE",
}
LEGACY_NOTEBOOKS = {
    LEGACY_RM2_SENTIMENT_NOTEBOOK_PATH: "LEGACY_SENTIMENT_V1",
}
OLD_NOTEBOOK_PATHS = {
    "tiktok_coordination_analysis.ipynb": "notebooks/rm1/tiktok_coordination_analysis.ipynb",
    "03_rm2_actor_type_typology.ipynb": "notebooks/rm2/03_rm2_actor_type_typology.ipynb",
    "02_rm2_sentiment_analysis.ipynb": "notebooks/rm2/02_rm2_sentiment_analysis.ipynb",
    "02_rm2_sentiment_goals.ipynb": "notebooks/legacy/02_rm2_sentiment_goals.ipynb",
}
ABS_WINDOWS_RE = re.compile(r"[A-Za-z]:[\\/][^\\n\\r\\t\"']+")
ABS_POSIX_RE = re.compile(r"(?<![\\w])/(?:home|Users|mnt|var|tmp)/[^\\n\\r\\t\"']+")
OUTPUT_LITERAL_RE = re.compile(r"['\"]((?:\\.\\./|\\.\\\\)?output[/\\\\][^'\"]+)['\"]")
WRITE_OP_RE = re.compile(r"to_csv|to_json|to_excel|ExcelWriter|joblib\\.dump|pickle\\.dump|savefig|write_graphml|write_gexf|write_")
MKDIR_RE = re.compile(r"mkdir|makedirs")


def load_notebook(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def code_cells(nb: dict) -> list[tuple[int, str]]:
    cells = []
    for idx, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") == "code":
            source = cell.get("source", [])
            cells.append((idx, "".join(source) if isinstance(source, list) else str(source)))
    return cells


def in_allowed_contract(path_text: str) -> bool:
    clean = path_text.replace("\\", "/").lstrip("./")
    if clean == "output":
        return False
    candidate = (PROJECT_ROOT / clean).resolve()
    allowed = [Path(p).resolve() for p in ALLOWED_OUTPUT_ROOTS]
    return any(candidate == root or root in candidate.parents for root in allowed)


def validate() -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    registry: list[dict] = []
    output_paths_by_notebook: dict[str, list[str]] = defaultdict(list)

    for nb_path, module in {**CANONICAL_NOTEBOOKS, **LEGACY_NOTEBOOKS}.items():
        rel = relative_to_root(nb_path)
        try:
            nb = load_notebook(nb_path)
            json_valid = True
            parse_error = ""
        except Exception as exc:  # pragma: no cover - diagnostic path
            nb = {"cells": []}
            json_valid = False
            parse_error = str(exc)

        cells = code_cells(nb)
        all_code = "\n".join(src for _, src in cells)
        is_legacy = nb_path in LEGACY_NOTEBOOKS
        has_bootstrap = "find_project_root" in all_code and "PROJECT_ROOT" in all_code
        uses_project_paths = "scripts.project_paths" in all_code
        legacy_status = "LEGACY_PATHS_DOCUMENTED" if is_legacy else ""

        abs_windows = ABS_WINDOWS_RE.findall(all_code)
        abs_posix = ABS_POSIX_RE.findall(all_code)
        old_refs = []
        for old, new in OLD_NOTEBOOK_PATHS.items():
            if old in all_code and new not in all_code:
                old_refs.append(old)

        output_literals = OUTPUT_LITERAL_RE.findall(all_code)
        unregistered = [p for p in output_literals if not in_allowed_contract(p)]
        root_outputs = [p for p in output_literals if p.replace("\\", "/").strip("./") == "output"]

        for cell_index, src in cells:
            for literal in OUTPUT_LITERAL_RE.findall(src):
                op = "reference"
                if WRITE_OP_RE.search(src):
                    op = "write_or_save_reference"
                elif MKDIR_RE.search(src):
                    op = "directory_creation_reference"
                output_paths_by_notebook[rel].append(literal.replace("\\", "/").lstrip("./"))
                registry.append(
                    {
                        "notebook": rel,
                        "cell_index": cell_index,
                        "producer_section": module,
                        "operation": op,
                        "output_path": literal.replace("\\", "/").lstrip("./"),
                        "artifact_type": Path(literal).suffix.lstrip(".") or "directory",
                        "canonical_module": module,
                        "overwrite_behavior": "would_overwrite_if_notebook_is_rerun",
                        "directory_contract_valid": in_allowed_contract(literal),
                        "duplicate_output_path": False,
                        "intended_duplicate": False,
                        "frozen_target": literal.replace("\\", "/").startswith("output/rm2_sentiment/final"),
                        "notes": "Static reference only; notebook was not executed.",
                    }
                )

        critical_failures = []
        if not json_valid:
            critical_failures.append("invalid_json")
        if not is_legacy and not has_bootstrap:
            critical_failures.append("missing_bootstrap")
        if not is_legacy and not uses_project_paths:
            critical_failures.append("missing_project_paths")
        if abs_windows:
            critical_failures.append("absolute_windows_path")
        if abs_posix:
            critical_failures.append("absolute_posix_path")
        if old_refs and not is_legacy:
            critical_failures.append("old_notebook_reference")
        if unregistered:
            critical_failures.append("unregistered_output_path")
        if root_outputs:
            critical_failures.append("root_output_path")

        status = legacy_status if is_legacy else "PATH_CONTRACT_PASS"
        if critical_failures:
            status = "PATH_CONTRACT_FAIL"

        rows.append(
            {
                "notebook": rel,
                "module": module,
                "json_valid": json_valid,
                "parse_error": parse_error,
                "code_cell_count": len(cells),
                "has_bootstrap": has_bootstrap,
                "uses_project_paths": uses_project_paths,
                "absolute_windows_path_count": len(abs_windows),
                "absolute_posix_path_count": len(abs_posix),
                "old_notebook_reference_count": len(old_refs),
                "unregistered_output_path_count": len(unregistered),
                "root_output_path_count": len(root_outputs),
                "hardcoded_output_reference_count": len(output_literals),
                "status": status,
                "notes": "; ".join(critical_failures) if critical_failures else "Static path audit passed or legacy paths documented.",
            }
        )

    duplicate_counts = {
        notebook: Counter(paths)
        for notebook, paths in output_paths_by_notebook.items()
    }
    for row in registry:
        count = duplicate_counts[row["notebook"]][row["output_path"]]
        row["duplicate_output_path"] = count > 1
        row["intended_duplicate"] = count > 1 and "visualisasi" in row["output_path"]

    return rows, registry


def main() -> int:
    REPOSITORY_AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    rows, registry = validate()

    audit_csv = REPOSITORY_AUDIT_DIR / "notebook_path_audit.csv"
    with audit_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    registry_csv = REPOSITORY_AUDIT_DIR / "notebook_output_registry.csv"
    if registry:
        with registry_csv.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(registry[0].keys()))
            writer.writeheader()
            writer.writerows(registry)
    else:
        registry_csv.write_text("", encoding="utf-8")

    md = [
        "# Notebook Path Audit",
        "",
        "| notebook | status | absolute paths | old refs | unregistered outputs | notes |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in rows:
        md.append(
            "| {notebook} | {status} | {abs_count} | {old_count} | {unreg_count} | {notes} |".format(
                notebook=row["notebook"],
                status=row["status"],
                abs_count=row["absolute_windows_path_count"] + row["absolute_posix_path_count"],
                old_count=row["old_notebook_reference_count"],
                unreg_count=row["unregistered_output_path_count"],
                notes=row["notes"],
            )
        )
    (REPOSITORY_AUDIT_DIR / "notebook_path_audit.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    failures = [row for row in rows if row["status"] == "PATH_CONTRACT_FAIL"]
    print(f"Notebook path audit rows: {len(rows)}")
    print(f"Notebook output registry rows: {len(registry)}")
    print(f"Failures: {len(failures)}")
    for row in rows:
        print(f"{row['status']}: {row['notebook']}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
