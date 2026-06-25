from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from text2fashion.settings import load_yaml, resolve_path


@dataclass(frozen=True)
class HubConfig:
    username: str
    model_repo_name: str
    space_repo_name: str
    model_visibility: str
    space_visibility: str
    local_lora_path: Path
    base_model: str
    project_name: str
    dataset_name: str
    fine_tuning_method: str
    trained_categories: list[str]
    token: str | None

    @property
    def model_repo_id(self) -> str:
        return f"{self.username}/{self.model_repo_name}"

    @property
    def space_repo_id(self) -> str:
        return f"{self.username}/{self.space_repo_name}"


def load_hub_config(config_path: str | Path = "configs/hub.yaml", env_path: str | Path = ".env") -> HubConfig:
    """Load Hugging Face deployment configuration."""
    load_dotenv(env_path)
    config = load_yaml(config_path)

    hf_cfg = config["huggingface"]
    model_cfg = config["model"]
    card_cfg = config["model_card"]

    username = os.getenv("HF_USERNAME") or hf_cfg["username"]
    token = os.getenv("HF_TOKEN")

    return HubConfig(
        username=username,
        model_repo_name=hf_cfg["model_repo_name"],
        space_repo_name=hf_cfg["space_repo_name"],
        model_visibility=hf_cfg.get("model_visibility", "public"),
        space_visibility=hf_cfg.get("space_visibility", "public"),
        local_lora_path=resolve_path(model_cfg["local_lora_path"]),
        base_model=model_cfg["base_model"],
        project_name=card_cfg["project_name"],
        dataset_name=card_cfg["dataset_name"],
        fine_tuning_method=card_cfg.get("fine_tuning_method", "LoRA"),
        trained_categories=list(card_cfg.get("trained_categories", [])),
        token=token,
    )
