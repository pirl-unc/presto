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
    n_categories = None

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

    # n_categories is deprecated (CategoryHead removed); skip inference.

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
        model_config = checkpoint_payload.get("model_config") or checkpoint_payload.get("config") or {}
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
    n_categories: Optional[int] = None,
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
        "n_categories": n_categories if n_categories is not None else model_config.get("n_categories"),
        "max_affinity_nM": (
            max_affinity_nM
            if max_affinity_nM is not None
            else model_config.get("max_affinity_nM")
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
    if resolved["max_affinity_nM"] is None:
        resolved["max_affinity_nM"] = DEFAULT_MAX_AFFINITY_NM
    if resolved["binding_midpoint_nM"] is None:
        resolved["binding_midpoint_nM"] = DEFAULT_BINDING_MIDPOINT_NM
    if resolved["binding_log10_scale"] is None:
        resolved["binding_log10_scale"] = DEFAULT_BINDING_LOG10_SCALE

    # Migrate 3-class chain type heads to 6-class fine types if needed
    state_dict = _migrate_chain_type_heads(state_dict)
    state_dict = _drop_legacy_dead_keys(state_dict)

    model = Presto(
        d_model=int(resolved["d_model"]),
        n_layers=int(resolved["n_layers"]),
        n_heads=int(resolved["n_heads"]),
        n_categories=resolved["n_categories"],
        max_affinity_nM=float(resolved["max_affinity_nM"]),
        binding_midpoint_nM=float(resolved["binding_midpoint_nM"]),
        binding_log10_scale=float(resolved["binding_log10_scale"]),
    )
    model.load_state_dict(state_dict, strict=strict)
    return model, payload


def _migrate_chain_type_heads(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Migrate MHC chain type heads to current 5-class fine types.

    Current mapping (5 classes):
      MHC_I=0, MHC_IIa=1, MHC_IIb=2, B2M=3, unknown=4

    Old 3-class mapping:
      a: class_I_alpha=0, class_II_alpha=1, unknown=2
      b: class_I_beta=0,  class_II_beta=1,  unknown=2

    Old 6-class mapping:
      MHC_Ia=0, MHC_Ib=1, MHC_IIa=2, MHC_IIb=3, B2M=4, unknown=5
    """
    for prefix in ("mhc_a_type_head", "mhc_b_type_head"):
        w_key = f"{prefix}.weight"
        b_key = f"{prefix}.bias"
        if w_key not in state_dict:
            continue
        w = state_dict[w_key]

        if w.shape[0] == 3:
            # Migrate 3 → 5
            d_in = w.shape[1]
            new_w = torch.zeros(5, d_in, dtype=w.dtype, device=w.device)
            new_b = torch.zeros(5, dtype=w.dtype, device=w.device)
            old_b = state_dict.get(b_key)

            if prefix == "mhc_a_type_head":
                # Old: class_I_alpha=0 → New: MHC_I=0
                # Old: class_II_alpha=1 → New: MHC_IIa=1
                # Old: unknown=2 → New: unknown=4
                new_w[0] = w[0]
                new_w[1] = w[1]
                new_w[4] = w[2]
                if old_b is not None:
                    new_b[0] = old_b[0]
                    new_b[1] = old_b[1]
                    new_b[4] = old_b[2]
            else:
                # Old: class_I_beta=0 → New: B2M=3
                # Old: class_II_beta=1 → New: MHC_IIb=2
                # Old: unknown=2 → New: unknown=4
                new_w[3] = w[0]
                new_w[2] = w[1]
                new_w[4] = w[2]
                if old_b is not None:
                    new_b[3] = old_b[0]
                    new_b[2] = old_b[1]
                    new_b[4] = old_b[2]

            state_dict[w_key] = new_w
            state_dict[b_key] = new_b

        elif w.shape[0] == 6:
            # Migrate 6 → 5 (merge MHC_Ia + MHC_Ib → MHC_I)
            d_in = w.shape[1]
            new_w = torch.zeros(5, d_in, dtype=w.dtype, device=w.device)
            new_b = torch.zeros(5, dtype=w.dtype, device=w.device)
            old_b = state_dict.get(b_key)

            # Old: MHC_Ia=0, MHC_Ib=1, MHC_IIa=2, MHC_IIb=3, B2M=4, unknown=5
            # New: MHC_I=0, MHC_IIa=1, MHC_IIb=2, B2M=3, unknown=4
            # Average Ia and Ib weights for MHC_I
            new_w[0] = (w[0] + w[1]) / 2.0  # MHC_I ← avg(MHC_Ia, MHC_Ib)
            new_w[1] = w[2]                   # MHC_IIa
            new_w[2] = w[3]                   # MHC_IIb
            new_w[3] = w[4]                   # B2M
            new_w[4] = w[5]                   # unknown
            if old_b is not None:
                new_b[0] = (old_b[0] + old_b[1]) / 2.0
                new_b[1] = old_b[2]
                new_b[2] = old_b[3]
                new_b[3] = old_b[4]
                new_b[4] = old_b[5]

            state_dict[w_key] = new_w
            state_dict[b_key] = new_b

    # Migrate chain_compat_head input dimension if needed
    compat_key = "chain_compat_head.0.weight"
    if compat_key in state_dict:
        cw = state_dict[compat_key]
        a_w_key = "mhc_a_type_head.weight"
        if a_w_key in state_dict:
            d_model = state_dict[a_w_key].shape[1]
            n_species_w = state_dict.get("mhc_a_species_head.weight")
            n_species = n_species_w.shape[0] if n_species_w is not None else 4
            expected_new = d_model * 2 + 5 + 5 + n_species * 2
            d_in = cw.shape[1]

            # From old 3+3 format
            expected_old_3 = d_model * 2 + 3 + 3 + n_species * 2
            # From old 6+6 format
            expected_old_6 = d_model * 2 + 6 + 6 + n_species * 2

            if d_in == expected_old_3:
                dm2 = d_model * 2
                new_cw = torch.zeros(cw.shape[0], expected_new, dtype=cw.dtype, device=cw.device)
                new_cw[:, :dm2] = cw[:, :dm2]
                # Old a: [I_alpha=0, II_alpha=1, unknown=2]
                # New a: [MHC_I=0, MHC_IIa=1, MHC_IIb=2, B2M=3, unk=4]
                new_cw[:, dm2 + 0] = cw[:, dm2 + 0]  # MHC_I ← I_alpha
                new_cw[:, dm2 + 1] = cw[:, dm2 + 1]  # MHC_IIa ← II_alpha
                new_cw[:, dm2 + 4] = cw[:, dm2 + 2]  # unk ← unknown
                # Old b: [I_beta=0, II_beta=1, unknown=2]
                new_cw[:, dm2 + 5 + 3] = cw[:, dm2 + 3 + 0]  # B2M ← I_beta
                new_cw[:, dm2 + 5 + 2] = cw[:, dm2 + 3 + 1]  # MHC_IIb ← II_beta
                new_cw[:, dm2 + 5 + 4] = cw[:, dm2 + 3 + 2]  # unk ← unknown
                # Species probs
                new_cw[:, dm2 + 10:] = cw[:, dm2 + 6:]
                state_dict[compat_key] = new_cw

            elif d_in == expected_old_6:
                dm2 = d_model * 2
                new_cw = torch.zeros(cw.shape[0], expected_new, dtype=cw.dtype, device=cw.device)
                new_cw[:, :dm2] = cw[:, :dm2]
                # Old a 6-class: [Ia=0, Ib=1, IIa=2, IIb=3, B2M=4, unk=5]
                # New a 5-class: [MHC_I=0, MHC_IIa=1, MHC_IIb=2, B2M=3, unk=4]
                new_cw[:, dm2 + 0] = (cw[:, dm2 + 0] + cw[:, dm2 + 1]) / 2.0
                new_cw[:, dm2 + 1] = cw[:, dm2 + 2]
                new_cw[:, dm2 + 2] = cw[:, dm2 + 3]
                new_cw[:, dm2 + 3] = cw[:, dm2 + 4]
                new_cw[:, dm2 + 4] = cw[:, dm2 + 5]
                # Old b 6-class
                new_cw[:, dm2 + 5 + 0] = (cw[:, dm2 + 6 + 0] + cw[:, dm2 + 6 + 1]) / 2.0
                new_cw[:, dm2 + 5 + 1] = cw[:, dm2 + 6 + 2]
                new_cw[:, dm2 + 5 + 2] = cw[:, dm2 + 6 + 3]
                new_cw[:, dm2 + 5 + 3] = cw[:, dm2 + 6 + 4]
                new_cw[:, dm2 + 5 + 4] = cw[:, dm2 + 6 + 5]
                # Species probs
                new_cw[:, dm2 + 10:] = cw[:, dm2 + 12:]
                state_dict[compat_key] = new_cw

    return state_dict


def _drop_legacy_dead_keys(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Drop parameters from modules that were removed from canonical Presto."""
    dead_prefixes = (
        "presentation.",
    )
    return {
        key: value
        for key, value in state_dict.items()
        if not any(key.startswith(prefix) for prefix in dead_prefixes)
    }
