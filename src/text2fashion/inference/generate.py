from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from diffusers import StableDiffusionXLPipeline


def generate_image(
    pipe: StableDiffusionXLPipeline,
    prompt: str,
    negative_prompt: str | None = None,
    seed: int = 42,
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
    height: int = 768,
    width: int = 768,
) -> Image.Image:
    """Generate one image from a text prompt."""
    generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(int(seed))
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=int(num_inference_steps),
        guidance_scale=float(guidance_scale),
        height=int(height),
        width=int(width),
        generator=generator,
    )
    return result.images[0]


def save_image(image: Image.Image, output_path: str | Path) -> Path:
    """Save a generated image."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return output_path
