from __future__ import annotations

import csv
from pathlib import Path

METRICS = [
    "prompt_alignment",
    "category_correctness",
    "color_correctness",
    "visual_quality",
    "artifact_control",
    "product_view_correctness",
]


def create_visual_score_template(
    output_path: str | Path,
    checkpoint_names: list[str],
    prompt_records: list[dict[str, str]],
    seeds: list[int],
) -> Path:
    """Create a CSV template for manual visual evaluation."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["checkpoint", "category", "prompt_id", "seed", "prompt", *METRICS, "notes"]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for checkpoint_name in checkpoint_names:
            for record in prompt_records:
                for seed in seeds:
                    row = {
                        "checkpoint": checkpoint_name,
                        "category": record["category"],
                        "prompt_id": record["prompt_id"],
                        "seed": seed,
                        "prompt": record["prompt"],
                        "notes": "",
                    }
                    for metric in METRICS:
                        row[metric] = ""
                    writer.writerow(row)

    return output_path
