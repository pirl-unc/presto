"""Fixed 4-condition matrix and model factory for the warm-start follow-up."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn as nn

from .assay_context import AssayContextEncoder
from .backbone import FixedBackbone
from .heads import AffinityHead, HEAD_REGISTRY


@dataclass(frozen=True)
class ConditionSpec:
    """One cell of the fixed 4-condition cold/warm matrix."""

    cond_id: int
    head_type: str
    assay_mode: str
    max_nM: float
    warm_start: bool = False
    n_bins: int = 128
    sigma_mult: float = 0.75
    embed_dim: int = 128
    label: str = ""

    def __post_init__(self) -> None:
        if self.label:
            return
        base = f"c{self.cond_id:02d}_{self.head_type}_{self.assay_mode}_max{int(self.max_nM / 1000)}k"
        base += "_warm" if self.warm_start else "_cold"
        if self.head_type in {"twohot", "hlgauss"}:
            base += f"_K{self.n_bins}"
        if self.head_type == "hlgauss":
            base += f"_s{self.sigma_mult}"
        if self.embed_dim != 128:
            base += f"_d{self.embed_dim}"
        object.__setattr__(self, "label", base)


def _make_conditions() -> List[ConditionSpec]:
    conds: List[ConditionSpec] = []
    cond_id = 0

    def _append(
        head_type: str,
        assay_mode: str,
        max_nM: float,
        *,
        warm_start: bool,
        n_bins: int = 128,
        sigma_mult: float = 0.75,
    ) -> None:
        nonlocal cond_id
        cond_id += 1
        conds.append(
            ConditionSpec(
                cond_id=cond_id,
                head_type=head_type,
                assay_mode=assay_mode,
                max_nM=max_nM,
                warm_start=warm_start,
                n_bins=n_bins,
                sigma_mult=sigma_mult,
            )
        )

    _append("mhcflurry", "additive", 200_000, warm_start=False)
    _append("mhcflurry", "additive", 200_000, warm_start=True)
    _append("log_mse", "additive", 200_000, warm_start=False)
    _append("log_mse", "additive", 200_000, warm_start=True)

    assert len(conds) == 4, f"Expected 4 conditions, got {len(conds)}"
    return conds


CONDITIONS: List[ConditionSpec] = _make_conditions()
CONDITIONS_BY_ID: Dict[int, ConditionSpec] = {spec.cond_id: spec for spec in CONDITIONS}


class DistributionalModel(nn.Module):
    """Encoder + assay context + head wrapper."""

    def __init__(
        self,
        encoder: FixedBackbone,
        assay_ctx: AssayContextEncoder,
        head: AffinityHead,
        spec: ConditionSpec,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.assay_ctx = assay_ctx
        self.head = head
        self.spec = spec

    def _assay_embedding(
        self,
        batch_size: int,
        device: torch.device | str,
        assay_type_idx: torch.Tensor | None = None,
        assay_prep_idx: torch.Tensor | None = None,
        assay_geometry_idx: torch.Tensor | None = None,
        assay_readout_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if assay_type_idx is None:
            assay_type_idx = torch.zeros(batch_size, dtype=torch.long, device=device)
        if assay_prep_idx is None:
            assay_prep_idx = torch.zeros(batch_size, dtype=torch.long, device=device)
        if assay_geometry_idx is None:
            assay_geometry_idx = torch.zeros(batch_size, dtype=torch.long, device=device)
        if assay_readout_idx is None:
            assay_readout_idx = torch.zeros(batch_size, dtype=torch.long, device=device)
        return self.assay_ctx(
            assay_type_idx,
            assay_prep_idx,
            assay_geometry_idx,
            assay_readout_idx,
        )

    def forward(
        self,
        pep_tok: torch.Tensor,
        mhc_a_tok: torch.Tensor,
        mhc_b_tok: torch.Tensor,
        assay_type_idx: torch.Tensor | None = None,
        assay_prep_idx: torch.Tensor | None = None,
        assay_geometry_idx: torch.Tensor | None = None,
        assay_readout_idx: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        h = self.encoder(pep_tok, mhc_a_tok, mhc_b_tok)
        assay_emb = self._assay_embedding(
            batch_size=h.shape[0],
            device=h.device,
            assay_type_idx=assay_type_idx,
            assay_prep_idx=assay_prep_idx,
            assay_geometry_idx=assay_geometry_idx,
            assay_readout_idx=assay_readout_idx,
        )
        return self.head(h, assay_emb)


def build_model(
    spec: ConditionSpec,
    *,
    embed_dim: int | None = None,
    n_heads: int = 4,
    n_layers: int = 2,
    ff_dim: int | None = None,
    ctx_dim: int = 32,
) -> DistributionalModel:
    """Build the fixed clean-benchmark model for one condition."""

    actual_embed_dim = int(embed_dim if embed_dim is not None else spec.embed_dim)
    actual_ff_dim = int(ff_dim if ff_dim is not None else actual_embed_dim)
    encoder = FixedBackbone(
        embed_dim=actual_embed_dim,
        n_heads=int(n_heads),
        n_layers=int(n_layers),
        ff_dim=actual_ff_dim,
    )
    assay_ctx = AssayContextEncoder(ctx_dim=int(ctx_dim))
    in_dim = encoder.out_dim

    head_cls = HEAD_REGISTRY[spec.head_type]
    head_kwargs = {
        "in_dim": in_dim,
        "ctx_dim": int(ctx_dim),
        "max_nM": float(spec.max_nM),
        "assay_mode": spec.assay_mode,
    }
    if spec.head_type in {"twohot", "hlgauss"}:
        head_kwargs["n_bins"] = int(spec.n_bins)
    if spec.head_type == "hlgauss":
        head_kwargs["sigma_mult"] = float(spec.sigma_mult)

    head = head_cls(**head_kwargs)
    return DistributionalModel(encoder=encoder, assay_ctx=assay_ctx, head=head, spec=spec)
