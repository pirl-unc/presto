"""Training configuration for Presto.

Canonical production training uses one unified mixed-source loop with
time-varying task/regularizer weight schedules.
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal, Optional
import json


@dataclass
class LRSchedule:
    """Learning-rate schedule configuration."""

    warmup_steps: int = 1000
    warmup_ratio: float = 0.0
    schedule: Literal["cosine", "linear", "constant", "inverse_sqrt"] = "cosine"
    min_lr_ratio: float = 0.01


@dataclass
class OptimizerConfig:
    """Optimizer configuration."""

    name: Literal["adamw", "muon", "sgd"] = "adamw"

    # AdamW settings
    lr: float = 1e-4
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8

    # Muon settings
    muon_lr: float = 0.02
    muon_momentum: float = 0.95

    # Gradient clipping
    grad_clip: float = 1.0
    grad_clip_norm: Literal["l2", "inf"] = "l2"


@dataclass
class Config:
    """Main Presto training configuration."""

    # Model architecture
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 8
    dropout: float = 0.1

    # Optimization
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    lr_schedule: LRSchedule = field(default_factory=LRSchedule)

    # Training loop
    batch_size: int = 32
    accumulation_steps: int = 1
    epochs: int = 40
    mixed_precision: bool = True
    compile: bool = False

    # Data
    train_data: Optional[str] = None
    val_data: Optional[str] = None
    num_workers: int = 4

    # Hardware
    device: str = "auto"

    # Checkpointing
    save_dir: str = "checkpoints"
    save_every: int = 1
    keep_last: int = 3

    # Logging/eval
    log_every: int = 100
    eval_every: int = 1

    # Experiment tracking
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None
    seed: int = 42

    def to_dict(self) -> dict:
        """Convert to dict, handling nested dataclasses."""

        def convert(obj):
            if hasattr(obj, "__dataclass_fields__"):
                return {k: convert(v) for k, v in asdict(obj).items()}
            if isinstance(obj, (list, tuple)):
                return [convert(x) for x in obj]
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            return obj

        return convert(self)

    def to_yaml(self, path: str = None) -> str:
        """Serialize config to YAML."""
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("pip install pyyaml") from exc

        def str_representer(dumper, data):
            if "\n" in data:
                return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
            return dumper.represent_scalar("tag:yaml.org,2002:str", data)

        yaml.add_representer(str, str_representer)
        serialized = yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)
        if path:
            Path(path).write_text(serialized)
        return serialized

    def to_json(self, path: str = None) -> str:
        """Serialize config to JSON."""
        serialized = json.dumps(self.to_dict(), indent=2)
        if path:
            Path(path).write_text(serialized)
        return serialized

    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        """Construct config from plain dictionary."""
        payload = dict(data)
        if "optimizer" in payload and isinstance(payload["optimizer"], dict):
            payload["optimizer"] = OptimizerConfig(**payload["optimizer"])
        if "lr_schedule" in payload and isinstance(payload["lr_schedule"], dict):
            payload["lr_schedule"] = LRSchedule(**payload["lr_schedule"])

        allowed = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in payload.items() if k in allowed}
        return cls(**filtered)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load config from YAML file."""
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("pip install pyyaml") from exc
        return cls.from_dict(yaml.safe_load(Path(path).read_text()))

    @classmethod
    def from_json(cls, path: str) -> "Config":
        """Load config from JSON file."""
        return cls.from_dict(json.loads(Path(path).read_text()))


def default_config() -> Config:
    """Standard unified-training config."""
    return Config()


def muon_config() -> Config:
    """Unified-training config using Muon optimizer."""
    cfg = default_config()
    cfg.optimizer = OptimizerConfig(name="muon", muon_lr=0.02)
    return cfg


def fast_config() -> Config:
    """Short unified-training config for debugging."""
    cfg = default_config()
    cfg.epochs = 2
    return cfg


def binding_only_config() -> Config:
    """Compact preset for binding-focused experiments."""
    cfg = default_config()
    cfg.epochs = 20
    return cfg
