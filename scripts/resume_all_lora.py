from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from urllib.request import urlretrieve

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from text2fashion.settings import load_yaml, resolve_path
from text2fashion.training.resume_utils import print_resume_status
from text2fashion.training.train_command import build_train_command, command_to_string

SCRIPT_URL = "https://raw.githubusercontent.com/huggingface/diffusers/v0.37.1/examples/text_to_image/train_text_to_image_lora_sdxl.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resume the all-category SDXL LoRA model.")
    parser.add_argument("--config", default="configs/resume_all_kaggle.yaml", help="Resume config path.")
    parser.add_argument("--script-path", default="train_text_to_image_lora_sdxl.py", help="Diffusers training script path.")
    parser.add_argument("--download-script", action="store_true", help="Download the Diffusers training script before training.")
    parser.add_argument("--num-processes", type=int, default=1, help="Number of accelerate processes.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    script_path = Path(args.script_path)

    if args.download_script or not script_path.exists():
        print("Downloading Diffusers SDXL LoRA training script...")
        urlretrieve(SCRIPT_URL, script_path)

    output_dir = resolve_path(config["paths"]["output_dir"])
    resume_cfg = config.get("resume", {})
    requested = resume_cfg.get("checkpoint", "latest")
    resume_arg = print_resume_status(output_dir, requested=requested)

    if not resume_arg:
        print("No checkpoint is available for resume.")
        raise SystemExit(1)

    command = build_train_command(
        config=config,
        script_path=script_path,
        resume_from_checkpoint=resume_arg,
        num_processes=args.num_processes,
    )

    print("Running command:")
    print(command_to_string(command))
    result = subprocess.run(command, text=True)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
