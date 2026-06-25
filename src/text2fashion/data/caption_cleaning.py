from __future__ import annotations

import re
from typing import Iterable

from datasets import Dataset

STANDARD_SUFFIX = "studio product photo, minimal background"


def _normalize_spaces(text: str) -> str:
    """Normalize spaces and comma spacing."""
    text = re.sub(r"\s*,\s*", ", ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r",\s*,", ", ", text)
    return text.strip(" ,")


def _remove_brand_noise(text: str) -> str:
    """Remove a small set of common product-title noise patterns."""
    replacements = {
        " - asia": "",
        "white label": "",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _remove_unstable_gender_terms(text: str) -> str:
    """Remove unstable gender wording that does not help product generation."""
    patterns = [
        r"\bman's\b",
        r"\bmen's\b",
        r"\bmens\b",
        r"\bwoman's\b",
        r"\bwomen's\b",
        r"\bwomens\b",
    ]
    for pattern in patterns:
        text = re.sub(pattern, "", text)
    return text


def _normalize_fashion_terms(text: str) -> str:
    """Normalize common fashion caption wording."""
    text = re.sub(r"\bzip\b", "zipper", text)
    text = re.sub(r"\bzippered\b", "zipper", text)
    text = text.replace("half - zipper", "half-zipper")
    text = text.replace("hood and hoodie", "hood")
    text = text.replace("zippers", "zipper")
    return text


def strip_original_caption(text: str) -> tuple[str, str, str]:
    """Split the source caption into prefix, product name, and visual description."""
    parts = [part.strip() for part in text.split(",")]
    prefix = parts[0].lower() if len(parts) > 0 else ""
    product_name = parts[1] if len(parts) > 1 else ""
    visual_desc = ", ".join(parts[2:]).strip() if len(parts) > 2 else ""
    return prefix, product_name, visual_desc


def clean_caption(text: str, category: str | None = None) -> str:
    """Create a short and consistent training caption."""
    prefix, product_name, visual_desc = strip_original_caption(text)
    category = (category or prefix).lower().strip()

    # Prefer the visual BLIP-like description because it is less brand-heavy.
    base = visual_desc or product_name or text
    base = base.lower()
    base = base.replace("a photography of", "")
    base = base.replace("a photo of", "")
    base = base.replace("on white background", "")
    base = base.replace("on a white background", "")
    base = _remove_brand_noise(base)
    base = _remove_unstable_gender_terms(base)
    base = _normalize_fashion_terms(base)
    base = _normalize_spaces(base)

    # Keep category signal when the caption is too generic.
    if category and category not in {"top", "bottom", "outer"}:
        category_hint = category
    else:
        category_hint = ""

    if len(base.split()) <= 2 and category_hint:
        base = f"{base} {category_hint}"

    base = _normalize_spaces(base)
    base = re.sub(r",?\s*studio product photo, minimal background\s*$", "", base).strip(" ,")
    final_caption = f"{base}, {STANDARD_SUFFIX}" if base else STANDARD_SUFFIX
    return final_caption


def clean_dataset_captions(
    dataset: Dataset,
    text_column: str = "text",
    category_column: str = "category",
    output_column: str = "text",
) -> Dataset:
    """Clean captions for the full fashion dataset."""
    def _map(example: dict) -> dict:
        category = example.get(category_column, "")
        example[output_column] = clean_caption(example[text_column], category=category)
        return example

    return dataset.map(_map)


def sample_captions(dataset: Dataset, n: int = 10, text_column: str = "text") -> list[str]:
    """Return a small sample of captions for quick inspection."""
    n = min(n, len(dataset))
    return [dataset[i][text_column] for i in range(n)]
