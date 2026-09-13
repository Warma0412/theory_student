from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from .paths import CONFIGS_ROOT


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = CONFIGS_ROOT / config_path
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping: {config_path}")
    return data


def apply_config_defaults(parser: argparse.ArgumentParser, config: dict[str, Any]) -> None:
    valid_keys = {
        action.dest
        for action in parser._actions
        if getattr(action, "dest", None) and action.dest != "help"
    }
    defaults = {key: value for key, value in config.items() if key in valid_keys}
    if defaults:
        parser.set_defaults(**defaults)
