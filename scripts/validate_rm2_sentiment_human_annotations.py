from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.rm2_sentiment_goals_pipeline import validate_human_annotation_frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate RM2 sentiment human annotation labels.")
    parser.add_argument(
        "annotation_csv",
        nargs="?",
        default="output/rm2_sentiment/validation/human_v1/sentiment_human_validation_template.csv",
        help="Path to completed human annotation CSV.",
    )
    parser.add_argument(
        "--output",
        default="output/rm2_sentiment/validation/human_v1/human_validation_metrics.csv",
        help="Path for validation metrics CSV.",
    )
    args = parser.parse_args()
    frame = pd.read_csv(args.annotation_csv, dtype=str, low_memory=False).fillna("")
    metrics = validate_human_annotation_frame(frame)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output, index=False)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
