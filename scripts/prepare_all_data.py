from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running the script before installing the package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from text2fashion.data.caption_cleaning import clean_dataset_captions
from text2fashion.data.export_train_folder import export_dataset_for_diffusers
from text2fashion.data.load_data import DatasetConfig, add_category_column, filter_categories, load_fashion_dataset
from text2fashion.data.split_dataset import create_train_validation_test_split
from text2fashion.data.validate_dataset import validate_splits
from text2fashion.settings import load_yaml, resolve_path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Prepare the full Text2Fashion dataset for SDXL LoRA training.")
    parser.add_argument("--config", default="configs/data_all.yaml", help="Path to the data YAML config.")
    return parser.parse_args()


def main() -> None:
    """Prepare all-category data for training."""
    args = parse_args()
    config = load_yaml(args.config)

    dataset_cfg = DatasetConfig(
        name=config["dataset"]["name"],
        split=config["dataset"].get("split", "train"),
        image_column=config["dataset"].get("image_column", "image"),
        text_column=config["dataset"].get("text_column", "text"),
    )

    target_categories = config["categories"]["target"]
    seed = int(config["processing"].get("seed", 42))
    validation_size = float(config["processing"].get("validation_size", 0.10))
    test_size = float(config["processing"].get("test_size", 0.10))

    processed_dataset_dir = resolve_path(config["paths"]["processed_dataset_dir"])
    train_export_dir = resolve_path(config["paths"]["train_export_dir"])
    validation_report_path = resolve_path(config["paths"]["validation_report_path"])

    print("Loading dataset...")
    dataset = load_fashion_dataset(dataset_cfg)

    print("Adding category column...")
    dataset = add_category_column(dataset, text_column=dataset_cfg.text_column)

    print(f"Filtering categories: {target_categories}")
    dataset = filter_categories(dataset, target_categories)

    print("Cleaning captions...")
    dataset = clean_dataset_captions(
        dataset,
        text_column=dataset_cfg.text_column,
        category_column="category",
        output_column="text",
    )

    print("Creating train/validation/test splits...")
    splits = create_train_validation_test_split(
        dataset,
        validation_size=validation_size,
        test_size=test_size,
        seed=seed,
    )

    print("Validating splits...")
    summary = validate_splits(splits, text_column="text", category_column="category")
    validation_report_path.parent.mkdir(parents=True, exist_ok=True)
    validation_report_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    if config["processing"].get("save_processed_splits", True):
        print(f"Saving processed splits to: {processed_dataset_dir}")
        processed_dataset_dir.parent.mkdir(parents=True, exist_ok=True)
        splits.save_to_disk(str(processed_dataset_dir))

    print(f"Exporting training split to: {train_export_dir}")
    metadata_path = export_dataset_for_diffusers(
        splits["train"],
        export_dir=train_export_dir,
        image_column=dataset_cfg.image_column,
        caption_column="text",
        overwrite=True,
    )

    print("Done.")
    print(f"Metadata file: {metadata_path}")
    print(f"Validation report: {validation_report_path}")
    print("Split sizes:")
    for split_name, split_data in splits.items():
        print(f"- {split_name}: {len(split_data)}")


if __name__ == "__main__":
    main()
