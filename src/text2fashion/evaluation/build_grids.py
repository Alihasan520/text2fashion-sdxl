from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw


def build_prompt_grids(
    eval_output_dir: str | Path,
    prompt_records: list[dict[str, str]],
    checkpoint_names: list[str],
    seeds: list[int],
    grid_dir: str | Path | None = None,
) -> Path:
    """Build comparison grids for each prompt and seed."""
    eval_output_dir = Path(eval_output_dir)
    grid_dir = Path(grid_dir) if grid_dir else eval_output_dir / "grids"
    grid_dir.mkdir(parents=True, exist_ok=True)

    for record in prompt_records:
        for seed in seeds:
            images = []
            labels = []

            for checkpoint_name in checkpoint_names:
                image_path = eval_output_dir / checkpoint_name / f"{record['prompt_id']}_seed{seed}.png"
                if image_path.exists():
                    images.append(Image.open(image_path).convert("RGB"))
                    labels.append(checkpoint_name)

            if not images:
                continue

            img_w, img_h = images[0].size
            label_h = 44
            grid = Image.new("RGB", (img_w * len(images), img_h + label_h), "white")
            draw = ImageDraw.Draw(grid)

            for col, (image, label) in enumerate(zip(images, labels)):
                x = col * img_w
                grid.paste(image, (x, label_h))
                draw.text((x + 12, 12), label, fill="black")

            output_name = f"{record['prompt_id']}_seed{seed}_grid.png"
            grid.save(grid_dir / output_name)

    return grid_dir


def build_showcase_grid(image_paths: list[str | Path], output_path: str | Path, columns: int = 3) -> Path:
    """Build a simple showcase grid from selected images."""
    images = [Image.open(path).convert("RGB") for path in image_paths]
    if not images:
        raise ValueError("No images were provided.")

    img_w, img_h = images[0].size
    rows = (len(images) + columns - 1) // columns
    grid = Image.new("RGB", (columns * img_w, rows * img_h), "white")

    for idx, image in enumerate(images):
        row = idx // columns
        col = idx % columns
        grid.paste(image, (col * img_w, row * img_h))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output_path)
    return output_path
