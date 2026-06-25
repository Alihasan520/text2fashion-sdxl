from __future__ import annotations

from pathlib import Path

import torch
from diffusers import StableDiffusionXLPipeline


def load_pipeline(
    base_model: str,
    lora_path: str | Path,
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.float16,
) -> StableDiffusionXLPipeline:
    """Load SDXL and attach LoRA weights."""
    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model,
        torch_dtype=torch_dtype,
        use_safetensors=True,
    ).to(device)
    pipe.load_lora_weights(str(lora_path))
    pipe.set_progress_bar_config(disable=True)
    return pipe


def unload_pipeline(pipe: StableDiffusionXLPipeline) -> None:
    """Release pipeline memory."""
    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
