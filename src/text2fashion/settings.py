from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    """Return the project root based on the current file location."""
    return Path(__file__).resolve().parents[2]


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def resolve_path(path: str | Path, base_dir: str | Path | None = None) -> Path:
    """Resolve a path relative to the project root or a provided base directory."""
    path = Path(path)
    if path.is_absolute():
        return path
    if base_dir is None:
        base_dir = project_root()
    return Path(base_dir) / path
