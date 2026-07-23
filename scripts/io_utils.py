"""Safe I/O helpers for future reruns.

These helpers are not applied retroactively to frozen one-time scripts. They
exist to keep future notebook/script writes inside documented output roots.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from scripts.project_paths import ALLOWED_OUTPUT_ROOTS, PROJECT_ROOT


def ensure_parent(path: Path | str) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def assert_within_project(path: Path | str) -> Path:
    target = Path(path).resolve()
    root = PROJECT_ROOT.resolve()
    if target != root and root not in target.parents:
        raise ValueError(f"Path di luar project root: {target}")
    return target


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def assert_allowed_output_path(path: Path | str) -> Path:
    target = assert_within_project(path)
    root = PROJECT_ROOT.resolve()
    if target == root:
        raise ValueError("Repository root tidak boleh menjadi output file.")
    forbidden_roots = [
        root / ".git",
        root / "config",
        root / "notebooks",
        root / "scripts",
        root / "tests",
        root / "archive",
        root / "artifacts" / "network_projects",
    ]
    if target in {root / "dataset.csv", root / "video_metadata_clean.csv"}:
        raise ValueError(f"Input kanonis tidak boleh ditulis ulang: {target}")
    if any(target == forbidden or _is_relative_to(target, forbidden) for forbidden in forbidden_roots):
        raise ValueError(f"Path output tidak diizinkan: {target}")
    allowed = tuple(Path(p).resolve() for p in ALLOWED_OUTPUT_ROOTS)
    if not any(target == base or _is_relative_to(target, base) for base in allowed):
        raise ValueError(f"Path tidak termasuk output contract: {target}")
    return target


def _atomic_replace_bytes(data: bytes, path: Path) -> Path:
    target = assert_allowed_output_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        tmp_path.replace(target)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return target


def write_csv_atomic(df: Any, path: Path | str, **kwargs: Any) -> Path:
    encoding = kwargs.pop("encoding", "utf-8")
    index = kwargs.pop("index", False)
    csv_text = df.to_csv(index=index, **kwargs)
    return _atomic_replace_bytes(csv_text.encode(encoding), Path(path))


def write_json_atomic(data: Any, path: Path | str, **kwargs: Any) -> Path:
    text = json.dumps(data, ensure_ascii=kwargs.pop("ensure_ascii", False), indent=kwargs.pop("indent", 2), **kwargs)
    return _atomic_replace_bytes((text + "\n").encode("utf-8"), Path(path))


def save_figure(fig: Any, path: Path | str, **kwargs: Any) -> Path:
    target = assert_allowed_output_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, **kwargs)
    return target
