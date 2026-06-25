from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from datasets import Dataset, load_dataset


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    split: str = "train"
    image_column: str = "image"
    text_column: str = "text"


def load_fashion_dataset(config: DatasetConfig) -> Dataset:
    """Load the source fashion dataset from Hugging Face."""
    dataset = load_dataset(config.name, split=config.split)
    return dataset


def extract_prefix(text: str) -> str:
    """Extract the category-like prefix before the first comma."""
    if not text:
        return ""
    return text.split(",", 1)[0].strip().lower()


def add_category_column(dataset: Dataset, text_column: str = "text") -> Dataset:
    """Add a category column extracted from the caption prefix."""
    def _map(example: dict) -> dict:
        example["category"] = extract_prefix(example[text_column])
        return example

    return dataset.map(_map)


def filter_categories(dataset: Dataset, categories: Iterable[str]) -> Dataset:
    """Keep only rows whose category is in the target categories."""
    allowed = {c.lower().strip() for c in categories}
    return dataset.filter(lambda example: example.get("category", "").lower().strip() in allowed)
