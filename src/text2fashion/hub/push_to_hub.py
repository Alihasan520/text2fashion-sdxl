from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi, create_repo, upload_folder

from text2fashion.hub.hub_config import load_hub_config
from text2fashion.hub.model_card import save_model_card


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Push the selected LoRA weights to Hugging Face Hub.")
    parser.add_argument("--config", default="configs/hub.yaml", help="Local Hugging Face config path.")
    parser.add_argument("--env", default=".env", help="Local environment file path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_hub_config(args.config, env_path=args.env)

    if not config.token:
        raise ValueError("HF_TOKEN is required in the local .env file.")

    local_lora_path = Path(config.local_lora_path)
    if not local_lora_path.exists():
        raise FileNotFoundError(f"LoRA directory not found: {local_lora_path}")

    private = config.model_visibility != "public"
    create_repo(repo_id=config.model_repo_id, repo_type="model", private=private, token=config.token, exist_ok=True)

    readme_path = local_lora_path / "README.md"
    save_model_card(config, readme_path)

    upload_folder(
        repo_id=config.model_repo_id,
        repo_type="model",
        folder_path=str(local_lora_path),
        token=config.token,
    )

    print(f"Uploaded model files to: {config.model_repo_id}")


if __name__ == "__main__":
    main()
