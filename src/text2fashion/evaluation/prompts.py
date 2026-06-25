from __future__ import annotations

EVALUATION_PROMPTS: dict[str, list[str]] = {
    "top": [
        "black oversized hoodie with front pocket, studio product photo, minimal background",
        "white t-shirt with clean minimal design, studio product photo, minimal background",
        "gray sweatshirt with ribbed cuffs, studio product photo, minimal background",
        "blue denim shirt with front buttons, studio product photo, minimal background",
        "red knit sweater with round collar, studio product photo, minimal background",
        "green crop top with short sleeves, studio product photo, minimal background",
    ],
    "bottom": [
        "blue straight fit jeans with front pockets, studio product photo, minimal background",
        "black cargo pants with side pockets, studio product photo, minimal background",
        "beige wide leg trousers with pleats, studio product photo, minimal background",
        "gray sweatpants with drawstring waist, studio product photo, minimal background",
        "black skirt with clean silhouette, studio product photo, minimal background",
        "dark denim shorts with visible stitching, studio product photo, minimal background",
    ],
    "outer": [
        "black puffer jacket with hood, studio product photo, minimal background",
        "white down jacket with black zipper, studio product photo, minimal background",
        "denim jacket with front pockets, studio product photo, minimal background",
        "beige coat with belt and high collar, studio product photo, minimal background",
        "black leather jacket with zipper pockets, studio product photo, minimal background",
        "blue jacket with hood and zipper, studio product photo, minimal background",
    ],
    "composite": [
        "black hoodie with leather sleeves, studio product photo, minimal background",
        "blue denim jeans with cargo side pockets, studio product photo, minimal background",
        "beige coat with oversized collar and belt, studio product photo, minimal background",
        "white sweatshirt with black stripe details, studio product photo, minimal background",
        "green jacket with color block pattern and multiple pockets, studio product photo, minimal background",
        "black trousers with wide leg cut and clean tailoring, studio product photo, minimal background",
    ],
}


def get_prompts(categories: list[str] | None = None, prompts_per_category: int | None = None) -> list[dict[str, str]]:
    """Return evaluation prompts as structured records."""
    selected = categories or list(EVALUATION_PROMPTS.keys())
    records: list[dict[str, str]] = []

    for category in selected:
        prompts = EVALUATION_PROMPTS.get(category, [])
        if prompts_per_category is not None:
            prompts = prompts[:prompts_per_category]
        for idx, prompt in enumerate(prompts, start=1):
            records.append({"category": category, "prompt_id": f"{category}_{idx:02d}", "prompt": prompt})

    return records
