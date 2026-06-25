from __future__ import annotations

import shutil
from pathlib import Path

from .resume_utils import list_checkpoints


def clean_incomplete_outputs(output_dir: str | Path) -> None:
    """Remove partial output files when no valid checkpoints exist."""
    output_dir = Path(output_dir)
    checkpoints = list_checkpoints(output_dir)

    if checkpoints:
        print("Valid checkpoints found. Output directory was not cleaned.")
        return

    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        print("Output directory created.")
        return

    for item in output_dir.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()
    print("Incomplete outputs removed.")


def keep_last_checkpoints(output_dir: str | Path, limit: int) -> None:
    """Keep only the most recent checkpoint directories."""
    if limit <= 0:
        return

    checkpoints = list_checkpoints(output_dir)
    excess = checkpoints[:-limit]
    for checkpoint in excess:
        shutil.rmtree(checkpoint)
        print(f"Removed old checkpoint: {checkpoint.name}")


def checkpoint_summary(output_dir: str | Path) -> dict[str, list[str]]:
    """Return checkpoint and final-weight availability."""
    output_dir = Path(output_dir)
    checkpoints = [p.name for p in list_checkpoints(output_dir)]
    final_weights = []
    if (output_dir / "pytorch_lora_weights.safetensors").exists():
        final_weights.append("pytorch_lora_weights.safetensors")
    return {"checkpoints": checkpoints, "final_weights": final_weights}
