"""Local browser helper for manually finding TikTok comments from similarity groups.

The script reads the prioritized screenshot queue, opens each candidate TikTok URL
in a persistent local browser profile, tries to highlight matching visible text,
and appends a manual review status. It is deliberately a review aid, not a
sentiment/modeling input.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
    from playwright.sync_api import sync_playwright
except Exception:  # pragma: no cover - import guard for friendly CLI error
    PlaywrightError = Exception  # type: ignore[assignment]
    PlaywrightTimeoutError = Exception  # type: ignore[assignment]
    sync_playwright = None  # type: ignore[assignment]


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUEUE = ROOT / "output/rm2_comment_similarity/comment_similarity_screenshot_queue.csv"
DEFAULT_STATUS = ROOT / "output/rm2_comment_similarity/tiktok_comment_lookup_status.csv"
DEFAULT_SCREENSHOT_DIR = ROOT / "output/rm2_comment_similarity/screenshots"
DEFAULT_PROFILE_DIR = Path.home() / ".tiktok_comment_finder_profile"

STATUS_FIELDS = [
    "checked_at_local",
    "screenshot_group_rank",
    "group_id",
    "member_rank",
    "comment_id",
    "video_id",
    "opened_url",
    "platform_username",
    "comment_text_presentation",
    "auto_match_status",
    "auto_match_term",
    "auto_match_text",
    "manual_status",
    "screenshot_file",
    "notes",
]

HIGHLIGHT_SCRIPT = r"""
({ terms }) => {
  const styleId = "codex-comment-finder-style";
  const oldStyle = document.getElementById(styleId);
  if (!oldStyle) {
    const style = document.createElement("style");
    style.id = styleId;
    style.textContent = `
      .codex-comment-finder-hit {
        outline: 4px solid #ffcc00 !important;
        background: rgba(255, 204, 0, 0.20) !important;
        box-shadow: 0 0 0 9999px rgba(0,0,0,0.08) !important;
        scroll-margin: 120px !important;
      }
    `;
    document.head.appendChild(style);
  }
  document.querySelectorAll(".codex-comment-finder-hit").forEach((el) => {
    el.classList.remove("codex-comment-finder-hit");
  });

  const normalizedTerms = terms
    .map((term) => ({
      value: String(term.value || "").toLowerCase().replace(/\s+/g, " ").trim(),
      kind: String(term.kind || ""),
      weight: Number(term.weight || 1),
    }))
    .filter((term) => term.value.length >= 3);

  function isVisible(el) {
    const rect = el.getBoundingClientRect();
    const style = window.getComputedStyle(el);
    return (
      rect.width > 0 &&
      rect.height > 0 &&
      style.visibility !== "hidden" &&
      style.display !== "none" &&
      Number(style.opacity || "1") > 0
    );
  }

  const selectors = "span,p,div,a,strong,em,h1,h2,h3,button";
  const elements = Array.from(document.querySelectorAll(selectors));
  let best = null;
  let bestScore = 0;
  let bestTerm = "";
  let bestText = "";

  for (const el of elements) {
    if (!isVisible(el)) continue;
    const rawText = (el.innerText || el.textContent || "").replace(/\s+/g, " ").trim();
    if (!rawText || rawText.length > 1500) continue;
    const text = rawText.toLowerCase();
    for (const term of normalizedTerms) {
      if (!text.includes(term.value)) continue;
      const score = term.weight + Math.min(25, term.value.length / 4) - Math.min(10, rawText.length / 400);
      if (score > bestScore) {
        best = el;
        bestScore = score;
        bestTerm = term.value;
        bestText = rawText.slice(0, 500);
      }
    }
  }

  if (best) {
    const target = best.closest("article, [data-e2e], [role='listitem'], li, div") || best;
    target.classList.add("codex-comment-finder-hit");
    target.scrollIntoView({ block: "center", inline: "nearest", behavior: "smooth" });
    return { found: true, term: bestTerm, text: bestText };
  }

  const pageText = (document.body.innerText || "").toLowerCase().replace(/\s+/g, " ");
  const pageTerm = normalizedTerms.find((term) => pageText.includes(term.value));
  if (pageTerm) {
    return { found: true, term: pageTerm.value, text: "Term appears in page text, but no compact visible element was isolated." };
  }
  return { found: false, term: "", text: "" };
}
"""

SCROLL_SCRIPT = r"""
() => {
  const candidates = Array.from(document.querySelectorAll("*"))
    .filter((el) => {
      const style = window.getComputedStyle(el);
      const overflowY = style.overflowY || "";
      return /(auto|scroll)/.test(overflowY) && el.scrollHeight > el.clientHeight + 120;
    })
    .sort((a, b) => (b.scrollHeight - b.clientHeight) - (a.scrollHeight - a.clientHeight));
  const target = candidates[0] || document.scrollingElement || document.body;
  const step = Math.max(500, Math.floor((target.clientHeight || window.innerHeight || 800) * 0.75));
  target.scrollBy({ top: step, behavior: "smooth" });
  return {
    tag: target.tagName,
    className: target.className || "",
    scrollTop: target.scrollTop || window.scrollY || 0,
    scrollHeight: target.scrollHeight || document.body.scrollHeight || 0,
  };
}
"""


def normalize_blank(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null", "<na>"}:
        return ""
    return text


def parse_rank_selector(value: str) -> set[int]:
    ranks: set[int] = set()
    for part in re.split(r"[,;\s]+", value.strip()):
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            ranks.update(range(int(start), int(end) + 1))
        else:
            ranks.add(int(part))
    return ranks


def safe_filename(value: object, fallback: str = "item") -> str:
    text = normalize_blank(value) or fallback
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("._")
    return text[:120] or fallback


def sort_queue_rows(df: pd.DataFrame) -> pd.DataFrame:
    sort_cols = [col for col in ["screenshot_group_rank", "member_rank", "group_id", "comment_id"] if col in df.columns]
    out = df.copy()
    for col in ["screenshot_group_rank", "member_rank"]:
        if col in out.columns:
            out[f"__{col}_num"] = pd.to_numeric(out[col], errors="coerce").fillna(10**9)
    actual_sort = []
    for col in sort_cols:
        actual_sort.append(f"__{col}_num" if f"__{col}_num" in out.columns else col)
    out = out.sort_values(actual_sort).drop(columns=[col for col in out.columns if col.startswith("__")])
    return out.reset_index(drop=True)


def load_queue(args: argparse.Namespace) -> pd.DataFrame:
    queue_path = Path(args.queue)
    if not queue_path.exists():
        raise FileNotFoundError(queue_path)
    df = pd.read_csv(queue_path, dtype=str, low_memory=False).fillna("")
    required = {"comment_id", "video_url", "platform_username", "comment_text_presentation"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Queue file is missing required columns: {', '.join(missing)}")

    if args.group_rank:
        ranks = parse_rank_selector(args.group_rank)
        rank_col = "screenshot_group_rank" if "screenshot_group_rank" in df.columns else "group_rank"
        if rank_col not in df.columns:
            raise ValueError("Queue file has no group-rank column.")
        df = df.loc[pd.to_numeric(df[rank_col], errors="coerce").isin(ranks)].copy()

    if args.group_id:
        allowed = {normalize_blank(value) for value in args.group_id}
        df = df.loc[df.get("group_id", "").map(normalize_blank).isin(allowed)].copy()

    if args.comment_id:
        allowed = {normalize_blank(value) for value in args.comment_id}
        df = df.loc[df["comment_id"].map(normalize_blank).isin(allowed)].copy()

    df = sort_queue_rows(df)
    if args.resume:
        done = completed_comment_ids(Path(args.status))
        df = df.loc[~df["comment_id"].isin(done)].copy()

    if args.start_row > 1:
        df = df.iloc[args.start_row - 1 :].copy()
    if args.limit:
        df = df.head(args.limit).copy()
    return df.reset_index(drop=True)


def completed_comment_ids(status_path: Path) -> set[str]:
    if not status_path.exists():
        return set()
    try:
        status = pd.read_csv(status_path, dtype=str).fillna("")
    except Exception:
        return set()
    if "comment_id" not in status.columns:
        return set()
    return set(status["comment_id"].map(normalize_blank))


def row_url(row: pd.Series) -> str:
    candidate = normalize_blank(row.get("tiktok_comment_url_candidate", ""))
    return candidate or normalize_blank(row.get("video_url", ""))


def build_search_terms(row: pd.Series) -> list[dict[str, object]]:
    username = normalize_blank(row.get("platform_username", "")).lstrip("@")
    text = normalize_blank(row.get("comment_text_presentation", ""))
    compact_text = re.sub(r"\s+", " ", text).strip()
    terms: list[dict[str, object]] = []
    if compact_text:
        terms.append({"value": compact_text[:140], "kind": "comment_text", "weight": 100})
        words = re.findall(r"(?u)\b\w+\b", compact_text.lower())
        for n_words, weight in [(10, 85), (8, 80), (6, 75), (4, 65)]:
            if len(words) >= n_words:
                terms.append({"value": " ".join(words[:n_words]), "kind": "comment_phrase", "weight": weight})
        if len(compact_text) > 45:
            terms.append({"value": compact_text[:70], "kind": "comment_prefix", "weight": 70})
    if username:
        terms.append({"value": username, "kind": "username", "weight": 45})

    seen: set[str] = set()
    unique: list[dict[str, object]] = []
    for term in terms:
        key = normalize_blank(term["value"]).lower()
        if key and key not in seen:
            seen.add(key)
            unique.append(term)
    return unique


def append_status(status_path: Path, record: dict[str, object]) -> None:
    status_path.parent.mkdir(parents=True, exist_ok=True)
    exists = status_path.exists()
    with status_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=STATUS_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({field: normalize_blank(record.get(field, "")) for field in STATUS_FIELDS})


def print_row_summary(index: int, total: int, row: pd.Series, opened_url: str) -> None:
    print("\n" + "=" * 88)
    print(f"Target {index}/{total}")
    print(f"group_id: {normalize_blank(row.get('group_id', ''))}")
    print(f"group_rank: {normalize_blank(row.get('screenshot_group_rank', ''))} | member_rank: {normalize_blank(row.get('member_rank', ''))}")
    print(f"comment_id: {normalize_blank(row.get('comment_id', ''))}")
    print(f"username: {normalize_blank(row.get('platform_username', ''))}")
    print(f"video_id: {normalize_blank(row.get('video_id', ''))}")
    print(f"opened_url: {opened_url}")
    print(f"text: {normalize_blank(row.get('comment_text_presentation', ''))}")


def try_find_comment(page: Any, row: pd.Series, scroll_steps: int, scroll_delay_ms: int) -> dict[str, object]:
    terms = build_search_terms(row)
    if not terms:
        return {"found": False, "term": "", "text": "No search terms built."}

    for step in range(scroll_steps + 1):
        try:
            result = page.evaluate(HIGHLIGHT_SCRIPT, {"terms": terms})
        except PlaywrightError as exc:
            return {"found": False, "term": "", "text": f"Playwright evaluate failed: {exc}"}
        if result.get("found"):
            result["scroll_step"] = step
            return result
        try:
            page.evaluate(SCROLL_SCRIPT)
        except PlaywrightError:
            page.mouse.wheel(0, 700)
        page.wait_for_timeout(scroll_delay_ms)
    return {"found": False, "term": "", "text": "No visible match after scrolling."}


def prompt_for_action(page: Any, row: pd.Series, args: argparse.Namespace, record: dict[str, object]) -> tuple[dict[str, object], bool]:
    if args.no_pause:
        record["manual_status"] = "AUTO_ONLY"
        return record, False

    print("\nActions: Enter=next | c=capture screenshot | f=mark found | n=mark not found | s=skip | q=quit")
    while True:
        choice = input("Action: ").strip().lower()
        if choice == "":
            record["manual_status"] = record.get("manual_status") or "REVIEWED_NEXT"
            return record, False
        if choice == "c":
            args.screenshot_dir.mkdir(parents=True, exist_ok=True)
            filename = (
                f"{safe_filename(row.get('group_id', 'group'))}_"
                f"{safe_filename(row.get('comment_id', 'comment'))}.png"
            )
            path = args.screenshot_dir / filename
            page.screenshot(path=str(path), full_page=False)
            record["screenshot_file"] = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
            record["manual_status"] = "SCREENSHOT_CAPTURED"
            print(f"Saved screenshot: {record['screenshot_file']}")
            continue
        if choice == "f":
            record["manual_status"] = "FOUND_MANUALLY"
            return record, False
        if choice == "n":
            record["manual_status"] = "NOT_FOUND_MANUALLY"
            return record, False
        if choice == "s":
            record["manual_status"] = "SKIPPED"
            return record, False
        if choice == "q":
            record["manual_status"] = "QUIT"
            return record, True
        print("Unknown action. Use Enter, c, f, n, s, or q.")


def open_rows(rows: pd.DataFrame, args: argparse.Namespace) -> None:
    if sync_playwright is None:
        raise RuntimeError("Playwright is not installed. Install with: pip install playwright")

    args.profile_dir.mkdir(parents=True, exist_ok=True)
    args.screenshot_dir.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        launch_kwargs: dict[str, object] = {
            "headless": args.headless,
            "viewport": {"width": args.viewport_width, "height": args.viewport_height},
        }
        if args.channel:
            launch_kwargs["channel"] = args.channel
        try:
            context = p.chromium.launch_persistent_context(str(args.profile_dir), **launch_kwargs)
        except PlaywrightError as exc:
            raise RuntimeError(
                "Could not launch browser. Try one of these:\n"
                "  python -m playwright install chromium\n"
                "  python scripts/open_tiktok_similarity_comments.py --channel msedge --limit 1\n"
                "  python scripts/open_tiktok_similarity_comments.py --channel \"\" --limit 1\n"
                f"Original error: {exc}"
            ) from exc

        page = context.pages[0] if context.pages else context.new_page()
        page.set_default_timeout(args.timeout_ms)

        for idx, row in enumerate(rows.to_dict("records"), start=1):
            row_series = pd.Series(row)
            url = row_url(row_series)
            print_row_summary(idx, len(rows), row_series, url)
            record: dict[str, object] = {
                "checked_at_local": datetime.now().astimezone().isoformat(timespec="seconds"),
                "screenshot_group_rank": row.get("screenshot_group_rank", ""),
                "group_id": row.get("group_id", ""),
                "member_rank": row.get("member_rank", ""),
                "comment_id": row.get("comment_id", ""),
                "video_id": row.get("video_id", ""),
                "opened_url": url,
                "platform_username": row.get("platform_username", ""),
                "comment_text_presentation": row.get("comment_text_presentation", ""),
                "auto_match_status": "",
                "auto_match_term": "",
                "auto_match_text": "",
                "manual_status": "",
                "screenshot_file": "",
                "notes": "",
            }
            if not url:
                record["auto_match_status"] = "NO_URL"
                append_status(args.status, record)
                continue
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=args.navigation_timeout_ms)
                page.wait_for_timeout(args.initial_wait_ms)
                result = try_find_comment(page, row_series, args.scroll_steps, args.scroll_delay_ms)
                record["auto_match_status"] = "AUTO_MATCH_FOUND" if result.get("found") else "AUTO_MATCH_NOT_FOUND"
                record["auto_match_term"] = result.get("term", "")
                record["auto_match_text"] = result.get("text", "")
                print(f"auto_match_status: {record['auto_match_status']}")
                if record["auto_match_term"]:
                    print(f"matched_term: {record['auto_match_term']}")
                if record["auto_match_text"]:
                    print(f"matched_text: {record['auto_match_text']}")
            except PlaywrightTimeoutError:
                record["auto_match_status"] = "NAVIGATION_TIMEOUT"
                print("Navigation timeout.")
            except PlaywrightError as exc:
                record["auto_match_status"] = "PLAYWRIGHT_ERROR"
                record["notes"] = str(exc)
                print(f"Playwright error: {exc}")

            record, should_quit = prompt_for_action(page, row_series, args, record)
            append_status(args.status, record)
            if should_quit:
                break

        if args.keep_open and not args.no_pause:
            input("\nDone. Press Enter to close browser...")
        context.close()


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open TikTok videos from the comment-similarity screenshot queue and help locate comments."
    )
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE, help="CSV queue to read.")
    parser.add_argument("--status", type=Path, default=DEFAULT_STATUS, help="CSV lookup status log to append.")
    parser.add_argument("--profile-dir", type=Path, default=DEFAULT_PROFILE_DIR, help="Persistent browser profile directory.")
    parser.add_argument("--screenshot-dir", type=Path, default=DEFAULT_SCREENSHOT_DIR, help="Directory for optional screenshots.")
    parser.add_argument("--group-rank", default="", help="Group ranks to open, e.g. 1,2,5-8.")
    parser.add_argument("--group-id", action="append", default=[], help="Specific group_id to open; can be repeated.")
    parser.add_argument("--comment-id", action="append", default=[], help="Specific comment_id to open; can be repeated.")
    parser.add_argument("--start-row", type=int, default=1, help="1-based start row after filters.")
    parser.add_argument("--limit", type=int, default=10, help="Maximum rows to open. Use 0 for all filtered rows.")
    parser.add_argument("--resume", action="store_true", help="Skip comment_ids already present in the status file.")
    parser.add_argument("--channel", default="chrome", help='Browser channel: chrome, msedge, or "" for Playwright Chromium.')
    parser.add_argument("--headless", action="store_true", help="Run browser headless.")
    parser.add_argument("--no-pause", action="store_true", help="Do not prompt between rows; append auto status only.")
    parser.add_argument("--keep-open", action="store_true", help="Wait before closing the browser after the last row.")
    parser.add_argument("--scroll-steps", type=int, default=18, help="Number of comment-panel scroll attempts.")
    parser.add_argument("--scroll-delay-ms", type=int, default=700, help="Delay after each scroll.")
    parser.add_argument("--initial-wait-ms", type=int, default=5500, help="Wait after page load for TikTok UI/comments.")
    parser.add_argument("--timeout-ms", type=int, default=8000, help="Default Playwright action timeout.")
    parser.add_argument("--navigation-timeout-ms", type=int, default=60000, help="Page navigation timeout.")
    parser.add_argument("--viewport-width", type=int, default=1440, help="Browser viewport width.")
    parser.add_argument("--viewport-height", type=int, default=1000, help="Browser viewport height.")
    parser.add_argument("--dry-run", action="store_true", help="Print selected rows and exit without opening a browser.")
    args = parser.parse_args(argv)
    args.limit = None if args.limit == 0 else args.limit
    if normalize_blank(args.channel) == "":
        args.channel = None
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    rows = load_queue(args)
    if rows.empty:
        print("No queue rows matched the selected filters.")
        return 0

    preview_cols = [
        col
        for col in [
            "screenshot_group_rank",
            "group_id",
            "member_rank",
            "comment_id",
            "platform_username",
            "video_url",
            "comment_text_presentation",
        ]
        if col in rows.columns
    ]
    print(f"Selected {len(rows)} queue row(s).")
    print(rows[preview_cols].head(min(len(rows), 10)).to_string(index=False))
    if args.dry_run:
        print("Dry run only. Browser was not opened.")
        return 0

    open_rows(rows, args)
    print(f"Status appended to: {args.status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
