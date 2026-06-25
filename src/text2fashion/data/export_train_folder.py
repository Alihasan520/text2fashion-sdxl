from __future__ import annotations

import json
from pathlib import Path

from datasets import Dataset
from tqdm.auto import tqdm


def export_dataset_for_diffusers(
    dataset: Dataset,
    export_dir: str | Path,
    image_column: str = "image",
    caption_column: str = "text",
    image_format: str = "png",
    overwrite: bool = True,
) -> Path:
    """Export a dataset split to image folder + metadata.jsonl format."""
    export_dir = Path(export_dir)
    images_dir = export_dir / "images"
    metadata_path = export_dir / "metadata.jsonl"

    images_dir.mkdir(parents=True, exist_ok=True)

    if metadata_path.exists() and overwrite:
        metadata_path.unlink()

    mode = "w" if overwrite else "a"
    with metadata_path.open(mode, encoding="utf-8") as f:
        for idx, example in enumerate(tqdm(dataset, desc="Exporting images")):
            image = example[image_column].convert("RGB")
            caption = example[caption_column]

            image_filename = f"{idx:06d}.{image_format}"
            image_path = images_dir / image_filename
            image.save(image_path)

            record = {
                "file_name": f"images/{image_filename}",
                "text": caption,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return metadata_path
