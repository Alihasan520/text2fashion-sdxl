from __future__ import annotations

from pathlib import Path

import pandas as pd

SCORE_COLUMNS = [
    "prompt_alignment",
    "category_correctness",
    "color_correctness",
    "visual_quality",
    "artifact_control",
    "product_view_correctness",
]


def summarize_visual_scores(score_csv: str | Path) -> pd.DataFrame:
    """Summarize manual visual scores by checkpoint."""
    df = pd.read_csv(score_csv)
    for column in SCORE_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    summary = df.groupby("checkpoint")[SCORE_COLUMNS].mean().round(3)
    summary["final_score"] = summary.mean(axis=1).round(3)
    return summary.sort_values("final_score", ascending=False)


def save_score_summary(score_csv: str | Path, output_path: str | Path) -> Path:
    """Save checkpoint score summary as CSV."""
    summary = summarize_visual_scores(score_csv)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path)
    return output_path
