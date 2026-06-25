from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
from diffusers import StableDiffusionXLPipeline


def load_sdxl_pipeline(base_model: str, device: str = "cuda") -> StableDiffusionXLPipeline:
    """Load the base SDXL pipeline for evaluation."""
    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        use_safetensors=True,
    ).to(device)
    pipe.set_progress_bar_config(disable=False)
    return pipe


def collect_lora_sources(lora_root: str | Path, include_final: bool = True, names: list[str] | None = None) -> list[tuple[str, Path]]:
    """Collect available LoRA checkpoints and final weights."""
    lora_root = Path(lora_root)
    sources: list[tuple[str, Path]] = []

    if names:
        checkpoint_dirs = [lora_root / name for name in names]
    else:
        checkpoint_dirs = sorted(
            [p for p in lora_root.glob("checkpoint-*") if p.is_dir()],
            key=lambda p: int(p.name.split("-")[-1]),
        )

    for checkpoint_dir in checkpoint_dirs:
        if (checkpoint_dir / "pytorch_lora_weights.safetensors").exists():
            sources.append((checkpoint_dir.name, checkpoint_dir))

    if include_final and (lora_root / "pytorch_lora_weights.safetensors").exists():
        sources.append(("final", lora_root))

    return sources


def generate_for_lora_sources(
    pipe: StableDiffusionXLPipeline,
    lora_sources: Iterable[tuple[str, Path]],
    prompt_records: list[dict[str, str]],
    output_dir: str | Path,
    seeds: list[int],
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
    height: int = 768,
    width: int = 768,
) -> None:
    """Generate samples for every LoRA source, prompt, and seed."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for source_name, lora_path in lora_sources:
        source_dir = output_dir / source_name
        source_dir.mkdir(parents=True, exist_ok=True)

        try:
            pipe.unload_lora_weights()
        except Exception:
            pass

        pipe.load_lora_weights(str(lora_path))

        for record in prompt_records:
            for seed in seeds:
                generator = torch.Generator(device="cuda").manual_seed(int(seed))
                image = pipe(
                    prompt=record["prompt"],
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    height=height,
                    width=width,
                    generator=generator,
                ).images[0]

                filename = f"{record['prompt_id']}_seed{seed}.png"
                image.save(source_dir / filename)

        print(f"Finished generation for {source_name}")
