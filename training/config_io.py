"""Helpers for loading config files and merging with CLI namespaces."""

from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


def load_config_file(path: str) -> Dict[str, Any]:
    """Load a JSON or YAML config file into a dictionary."""
    cfg_path = Path(path)
    text = cfg_path.read_text(encoding="utf-8")
    suffix = cfg_path.suffix.lower()

    if suffix in {".json"}:
        payload = json.loads(text)
    elif suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise ImportError(
                "YAML config requested but pyyaml is not installed."
            ) from exc
        payload = yaml.safe_load(text)
    else:
        raise ValueError(f"Unsupported config format: {cfg_path.suffix}")

    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError("Config file must contain a mapping/object at top-level.")
    return payload


def pick_config_section(
    config: Mapping[str, Any],
    section: Optional[str] = None,
) -> Dict[str, Any]:
    """Pick a subsection if present, otherwise return top-level mapping."""
    if section and section in config and isinstance(config[section], dict):
        return dict(config[section])
    return dict(config)


def pick_train_section(config: Mapping[str, Any], subcommand: str) -> Dict[str, Any]:
    """Pick `train.<subcommand>` section when present.

    Supports either:
    - top-level `<subcommand>` mapping
    - top-level `train.<subcommand>` mapping
    """
    if "train" in config and isinstance(config["train"], Mapping):
        train_cfg = config["train"]
        if subcommand in train_cfg and isinstance(train_cfg[subcommand], Mapping):
            return dict(train_cfg[subcommand])
        return dict(train_cfg)
    return pick_config_section(config, subcommand)


def merge_namespace_with_config(
    args: Namespace,
    defaults: Mapping[str, Any],
    config: Mapping[str, Any],
) -> Namespace:
    """Merge config values into args when CLI value remains at parser default.

    CLI values always take precedence over config values.
    """
    merged = Namespace(**vars(args))
    for key, default in defaults.items():
        if key not in config:
            continue
        current = getattr(merged, key, default)
        if current == default:
            setattr(merged, key, config[key])
    return merged
