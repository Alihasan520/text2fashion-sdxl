from __future__ import annotations

from pathlib import Path


def list_checkpoints(output_dir: str | Path) -> list[Path]:
    """Return checkpoint directories sorted by training step."""
    output_dir = Path(output_dir)
    checkpoints = [p for p in output_dir.glob("checkpoint-*") if p.is_dir()]
    return sorted(checkpoints, key=lambda p: int(p.name.split("-")[-1]))


def latest_checkpoint(output_dir: str | Path) -> Path | None:
    """Return the latest checkpoint directory, if one exists."""
    checkpoints = list_checkpoints(output_dir)
    return checkpoints[-1] if checkpoints else None


def resolve_resume_argument(output_dir: str | Path, requested: str | None = "latest") -> str | None:
    """Resolve the resume argument used by the Diffusers training script."""
    if not requested:
        return None
    if requested == "latest":
        return "latest" if latest_checkpoint(output_dir) else None
    return requested


def print_resume_status(output_dir: str | Path, requested: str | None = "latest") -> str | None:
    """Print the current resume state and return the resume argument."""
    checkpoints = list_checkpoints(output_dir)
    if checkpoints:
        print("Available checkpoints:")
        for checkpoint in checkpoints:
            print(f"- {checkpoint.name}")
    else:
        print("No checkpoint found. Training will start from scratch.")

    resume_arg = resolve_resume_argument(output_dir, requested=requested)
    if resume_arg:
        print(f"Resume mode: {resume_arg}")
    return resume_arg
