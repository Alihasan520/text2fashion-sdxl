from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, asdict
from typing import Any

from datasets import Dataset, DatasetDict


@dataclass
class DatasetValidationSummary:
    num_rows: int
    empty_captions: int
    duplicate_captions: int
    category_counts: dict[str, int]
    min_caption_words: int
    max_caption_words: int
    avg_caption_words: float

    def to_dict(self) -> dict[str, Any]:
        """Convert the validation summary to a dictionary."""
        return asdict(self)


def validate_dataset(
    dataset: Dataset,
    text_column: str = "text",
    category_column: str = "category",
) -> DatasetValidationSummary:
    """Compute simple validation statistics for a dataset."""
    captions = [str(example.get(text_column, "")).strip() for example in dataset]
    categories = [str(example.get(category_column, "unknown")).strip() for example in dataset]

    empty_captions = sum(1 for caption in captions if not caption)
    caption_counts = Counter(captions)
    duplicate_captions = sum(1 for _, count in caption_counts.items() if count > 1)

    word_counts = [len(caption.split()) for caption in captions if caption]
    min_words = min(word_counts) if word_counts else 0
    max_words = max(word_counts) if word_counts else 0
    avg_words = sum(word_counts) / len(word_counts) if word_counts else 0.0

    return DatasetValidationSummary(
        num_rows=len(dataset),
        empty_captions=empty_captions,
        duplicate_captions=duplicate_captions,
        category_counts=dict(Counter(categories)),
        min_caption_words=min_words,
        max_caption_words=max_words,
        avg_caption_words=round(avg_words, 2),
    )


def validate_splits(splits: DatasetDict, text_column: str = "text", category_column: str = "category") -> dict[str, dict[str, Any]]:
    """Validate every split in a DatasetDict."""
    return {
        split_name: validate_dataset(split, text_column=text_column, category_column=category_column).to_dict()
        for split_name, split in splits.items()
    }
