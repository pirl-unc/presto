"""Checkpoint serialization helpers for Presto."""

from __future__ import annotations

from datetime import datetime, timezone
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch

from ..models.presto import Presto
from ..models.affinity import (
    DEFAULT_MAX_AFFINITY_NM,
    DEFAULT_BINDING_MIDPOINT_NM,
    DEFAULT_BINDING_LOG10_SCALE,
)


CHECKPOINT_FORMAT = "presto.v2"
CHECKPOINT_FORMAT_VERSION = 2
MODEL_CLASS = "presto.models.presto.Presto"


def _count_transformer_layers(state_dict: Dict[str, torch.Tensor], prefix: str) -> Optional[int]:
    pattern = re.compile(rf"^{re.escape(prefix)}\.(\d+)\.")
    max_idx = -1
    for key in state_dict:
        match = pattern.match(key)
        if match:
            max_idx = max(max_idx, int(match.group(1)))
    return max_idx + 1 if max_idx >= 0 else None


def infer_model_config_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """Infer a best-effort model config from state dict keys/shapes."""
    d_model = None
    n_layers = None
    n_heads = None

    # v3+ single-stream checkpoints
    aa_emb = state_dict.get("aa_embedding.weight")
    if aa_emb is not None and aa_emb.ndim == 2:
        d_model = int(aa_emb.shape[1])
    else:
        # v2 fallback
        pep_emb = state_dict.get("pmhc_encoder.pep_embedding.weight")
        if pep_emb is not None and pep_emb.ndim == 2:
            d_model = int(pep_emb.shape[1])

    n_layers = _count_transformer_layers(state_dict, "stream_encoder.layers")
    if n_layers is None:
        # v2 fallback
        n_layers = _count_transformer_layers(
            state_dict,
            "pmhc_encoder.mhc_encoder.transformer.layers",
        )

    # `n_heads` is not encoded directly in state dict. Use a safe divisor fallback.
    if d_model is not None:
        for candidate in (16, 12, 8, 6, 4, 3, 2, 1):
            if d_model % candidate == 0:
                n_heads = candidate
                break

    config: Dict[str, Any] = {}
    if d_model is not None:
        config["d_model"] = d_model
    if n_layers is not None:
        config["n_layers"] = n_layers
    if n_heads is not None:
        config["n_heads"] = n_heads
    return config


def build_model_config(model: Presto) -> Dict[str, Any]:
    """Extract model constructor config from a Presto instance."""
    if hasattr(model, "stream_encoder"):
        n_layers = len(model.stream_encoder.layers)
        if n_layers == 0:
            n_layers = 1
        if model.stream_encoder.layers:
            n_heads = int(model.stream_encoder.layers[0].self_attn.num_heads)
        else:
            n_heads = 4
    else:
        # Fallback path for non-canonical model wrappers.
        n_layers = len(model.pmhc_encoder.mhc_encoder.transformer.layers)
        if n_layers == 0:
            n_layers = 1
        if model.pmhc_encoder.mhc_encoder.transformer.layers:
            n_heads = int(model.pmhc_encoder.mhc_encoder.transformer.layers[0].self_attn.num_heads)
        else:
            n_heads = 4
    return {
        "d_model": int(model.d_model),
        "n_layers": int(n_layers),
        "n_heads": int(n_heads),
        "latent_topology": str(model.latent_topology),
        "max_affinity_nM": float(getattr(model, "max_affinity_nM", DEFAULT_MAX_AFFINITY_NM)),
        "binding_midpoint_nM": float(
            getattr(model, "binding_midpoint_nM", DEFAULT_BINDING_MIDPOINT_NM)
        ),
        "binding_log10_scale": float(
            getattr(model, "binding_log10_scale", DEFAULT_BINDING_LOG10_SCALE)
        ),
    }


def _extract_state_and_config(
    checkpoint_payload: Union[Dict[str, Any], Dict[str, torch.Tensor]],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    if isinstance(checkpoint_payload, dict) and "model_state_dict" in checkpoint_payload:
        state_dict = checkpoint_payload["model_state_dict"]
        model_config = (
            checkpoint_payload.get("model_config") or checkpoint_payload.get("config") or {}
        )
    else:
        state_dict = checkpoint_payload
        model_config = {}
    if not model_config:
        model_config = infer_model_config_from_state_dict(state_dict)
    return state_dict, dict(model_config)


def save_model_checkpoint(
    path: Union[str, Path],
    *,
    model: Presto,
    optimizer_state_dict: Optional[Dict[str, Any]] = None,
    epoch: Optional[int] = None,
    step: Optional[int] = None,
    metrics: Optional[Dict[str, Any]] = None,
    train_config: Optional[Dict[str, Any]] = None,
    run_config: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Serialize a self-describing Presto checkpoint."""
    payload: Dict[str, Any] = {
        "checkpoint_format": CHECKPOINT_FORMAT,
        "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
        "model_class": MODEL_CLASS,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_config": build_model_config(model),
        "model_state_dict": model.state_dict(),
    }
    if optimizer_state_dict is not None:
        payload["optimizer_state_dict"] = optimizer_state_dict
    if epoch is not None:
        payload["epoch"] = int(epoch)
    if step is not None:
        payload["step"] = int(step)
    if metrics is not None:
        payload["metrics"] = metrics
    if train_config is not None:
        payload["train_config"] = train_config
    if run_config is not None:
        payload["run_config"] = run_config
    if extra is not None:
        payload["extra"] = extra

    torch.save(payload, str(path))
    return payload


def load_model_from_checkpoint(
    checkpoint_path: Union[str, Path],
    *,
    map_location: Union[str, torch.device, None] = "cpu",
    d_model: Optional[int] = None,
    n_layers: Optional[int] = None,
    n_heads: Optional[int] = None,
    latent_topology: Optional[str] = None,
    max_affinity_nM: Optional[float] = None,
    binding_midpoint_nM: Optional[float] = None,
    binding_log10_scale: Optional[float] = None,
    strict: bool = True,
) -> Tuple[Presto, Dict[str, Any]]:
    """Load Presto model + raw payload from checkpoint path.

    Explicit args override checkpoint config if provided.
    Affinity calibration parameters are loaded from checkpoint config unless
    explicitly overridden.
    """
    try:
        payload = torch.load(checkpoint_path, map_location=map_location)
    except Exception as exc:
        # PyTorch 2.6+ defaults to weights_only=True. Our checkpoints include
        # config/metadata objects, so fall back to full trusted deserialization.
        if "Weights only load failed" not in str(exc):
            raise
        payload = torch.load(
            checkpoint_path,
            map_location=map_location,
            weights_only=False,
        )
    state_dict, model_config = _extract_state_and_config(payload)

    resolved = {
        "d_model": d_model if d_model is not None else model_config.get("d_model"),
        "n_layers": n_layers if n_layers is not None else model_config.get("n_layers"),
        "n_heads": n_heads if n_heads is not None else model_config.get("n_heads"),
        "latent_topology": (
            latent_topology if latent_topology is not None else model_config.get("latent_topology")
        ),
        "max_affinity_nM": (
            max_affinity_nM if max_affinity_nM is not None else model_config.get("max_affinity_nM")
        ),
        "binding_midpoint_nM": (
            binding_midpoint_nM
            if binding_midpoint_nM is not None
            else model_config.get("binding_midpoint_nM")
        ),
        "binding_log10_scale": (
            binding_log10_scale
            if binding_log10_scale is not None
            else model_config.get("binding_log10_scale")
        ),
    }

    # Stable defaults for state-dict-only checkpoints.
    if resolved["d_model"] is None:
        resolved["d_model"] = 256
    if resolved["n_layers"] is None:
        resolved["n_layers"] = 4
    if resolved["n_heads"] is None:
        resolved["n_heads"] = 8
    if resolved["latent_topology"] is None:
        resolved["latent_topology"] = "expanded"
    if resolved["max_affinity_nM"] is None:
        resolved["max_affinity_nM"] = DEFAULT_MAX_AFFINITY_NM
    if resolved["binding_midpoint_nM"] is None:
        resolved["binding_midpoint_nM"] = DEFAULT_BINDING_MIDPOINT_NM
    if resolved["binding_log10_scale"] is None:
        resolved["binding_log10_scale"] = DEFAULT_BINDING_LOG10_SCALE

    model = Presto(
        d_model=int(resolved["d_model"]),
        n_layers=int(resolved["n_layers"]),
        n_heads=int(resolved["n_heads"]),
        latent_topology=str(resolved["latent_topology"]),
        max_affinity_nM=float(resolved["max_affinity_nM"]),
        binding_midpoint_nM=float(resolved["binding_midpoint_nM"]),
        binding_log10_scale=float(resolved["binding_log10_scale"]),
    )
    model.load_state_dict(state_dict, strict=strict)
    return model, payload
