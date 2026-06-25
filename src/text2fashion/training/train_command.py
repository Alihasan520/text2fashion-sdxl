from __future__ import annotations

from pathlib import Path
from typing import Any


def build_train_command(
    config: dict[str, Any],
    script_path: str | Path = "train_text_to_image_lora_sdxl.py",
    resume_from_checkpoint: str | None = None,
    num_processes: int = 1,
) -> list[str]:
    """Build an accelerate command for SDXL LoRA training."""
    model_cfg = config["model"]
    path_cfg = config["paths"]
    train_cfg = config["training"]

    mixed_precision = str(train_cfg.get("mixed_precision", "fp16"))

    cmd = [
        "accelerate",
        "launch",
        f"--num_processes={num_processes}",
        f"--mixed_precision={mixed_precision}",
        str(script_path),
        "--pretrained_model_name_or_path",
        str(model_cfg["base_model"]),
        "--train_data_dir",
        str(path_cfg["train_data_dir"]),
        "--caption_column",
        "text",
        "--image_column",
        "image",
        "--resolution",
        str(train_cfg["resolution"]),
        "--train_batch_size",
        str(train_cfg["train_batch_size"]),
        "--gradient_accumulation_steps",
        str(train_cfg["gradient_accumulation_steps"]),
        "--max_train_steps",
        str(train_cfg["max_train_steps"]),
        "--learning_rate",
        str(train_cfg["learning_rate"]),
        "--lr_scheduler",
        str(train_cfg.get("lr_scheduler", "constant")),
        "--lr_warmup_steps",
        str(train_cfg.get("lr_warmup_steps", 0)),
        "--output_dir",
        str(path_cfg["output_dir"]),
        "--checkpointing_steps",
        str(train_cfg["checkpointing_steps"]),
        "--checkpoints_total_limit",
        str(train_cfg["checkpoints_total_limit"]),
        "--mixed_precision",
        mixed_precision,
        "--seed",
        str(train_cfg.get("seed", 42)),
    ]

    if train_cfg.get("random_flip", True):
        cmd.append("--random_flip")

    if train_cfg.get("gradient_checkpointing", True):
        cmd.append("--gradient_checkpointing")

    if resume_from_checkpoint:
        cmd.extend(["--resume_from_checkpoint", resume_from_checkpoint])

    return cmd


def command_to_string(command: list[str]) -> str:
    """Convert a command list to a readable shell command."""
    return " ".join(str(part) for part in command)
