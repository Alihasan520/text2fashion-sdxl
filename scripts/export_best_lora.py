from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the selected LoRA checkpoint for deployment.")
    parser.add_argument("--source", required=True, help="Checkpoint or final LoRA directory.")
    parser.add_argument("--target", default="outputs/exports/best_lora", help="Export directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.source)
    target = Path(args.target)

    weight_file = source / "pytorch_lora_weights.safetensors"
    if not weight_file.exists():
        raise FileNotFoundError(f"LoRA weights not found: {weight_file}")

    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(weight_file, target / "pytorch_lora_weights.safetensors")

    print(f"Exported LoRA weights to: {target}")


if __name__ == "__main__":
    main()
