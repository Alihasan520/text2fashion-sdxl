from __future__ import annotations

from pathlib import Path

from .hub_config import HubConfig


def build_model_card(config: HubConfig) -> str:
    """Build a Hugging Face model card for the LoRA repository."""
    categories = "\n".join(f"- {category}" for category in config.trained_categories)

    return f"""---
base_model: {config.base_model}
library_name: diffusers
license: other
tags:
  - stable-diffusion-xl
  - lora
  - text-to-image
  - fashion
  - diffusers
---

# {config.project_name}

This repository contains LoRA weights for an SDXL-based text-to-fashion image generation model.

## Model Details

- Base model: `{config.base_model}`
- Fine-tuning method: {config.fine_tuning_method}
- Dataset: `{config.dataset_name}`
- Target categories:
{categories}

## Intended Use

The model is intended for generating fashion product-style images from text prompts. It is designed as a portfolio project demonstrating data preparation, LoRA fine-tuning, checkpoint evaluation, and Gradio deployment.

## Example Prompt

```text
black puffer jacket with hood, studio product photo, minimal background
```

## Evaluation

Checkpoint selection is based on a visual evaluation rubric covering:

- Prompt alignment
- Category correctness
- Color correctness
- Visual quality
- Artifact control
- Product view correctness

## Limitations

The model is optimized for product-style fashion images and may not perform well on scenes, people, complex backgrounds, or non-fashion prompts.
"""


def save_model_card(config: HubConfig, output_path: str | Path) -> Path:
    """Save the generated model card."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_model_card(config), encoding="utf-8")
    return output_path
