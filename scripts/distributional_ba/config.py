"""32-condition experiment matrix and model factory."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn as nn

from .assay_context import AssayContextEncoder
from .encoders import ENCODER_BACKBONES, build_encoder
from .heads import HEAD_REGISTRY, AffinityHead


@dataclass(frozen=True)
class ConditionSpec:
    """One cell of the 32-condition matrix."""

    cond_id: int
    head_type: str          # mhcflurry | log_mse | twohot | hlgauss
    assay_mode: str         # additive | affine | d1_affine | d2_logit
    max_nM: float           # 50_000 or 100_000
    n_bins: int = 128       # distributional only
    sigma_mult: float = 0.75  # hlgauss only
    embed_dim: int = 128    # encoder embedding dimension
    label: str = ""

    def __post_init__(self):
        if not self.label:
            object.__setattr__(
                self, "label",
                f"c{self.cond_id:02d}_{self.head_type}_{self.assay_mode}_max{int(self.max_nM/1000)}k"
                + (f"_K{self.n_bins}" if self.head_type in ("twohot", "hlgauss") else "")
                + (f"_s{self.sigma_mult}" if self.head_type == "hlgauss" else "")
                + (f"_d{self.embed_dim}" if self.embed_dim != 128 else ""),
            )


def _make_conditions() -> List[ConditionSpec]:
    """Build the full 32-condition matrix."""
    conds: List[ConditionSpec] = []
    cid = 0

    def _add(head_type, assay_mode, max_nM, n_bins=128, sigma_mult=0.75):
        nonlocal cid
        cid += 1
        conds.append(ConditionSpec(
            cond_id=cid,
            head_type=head_type,
            assay_mode=assay_mode,
            max_nM=max_nM,
            n_bins=n_bins,
            sigma_mult=sigma_mult,
        ))

    # Block 1 (8): Core method × MAX — K=128, D1-affine, sigma=0.75
    for max_nM in (50_000, 100_000):
        _add("mhcflurry", "affine", max_nM)
    for max_nM in (50_000, 100_000):
        _add("log_mse", "affine", max_nM)
    for max_nM in (50_000, 100_000):
        _add("twohot", "d1_affine", max_nM)
    for max_nM in (50_000, 100_000):
        _add("hlgauss", "d1_affine", max_nM, sigma_mult=0.75)

    # Block 2 (4): Bin count K=64
    for max_nM in (50_000, 100_000):
        _add("twohot", "d1_affine", max_nM, n_bins=64)
    for max_nM in (50_000, 100_000):
        _add("hlgauss", "d1_affine", max_nM, n_bins=64, sigma_mult=0.75)

    # Block 3 (8): D2-logit assay integration
    for n_bins in (128, 64):
        for max_nM in (50_000, 100_000):
            _add("twohot", "d2_logit", max_nM, n_bins=n_bins)
    for n_bins in (128, 64):
        for max_nM in (50_000, 100_000):
            _add("hlgauss", "d2_logit", max_nM, n_bins=n_bins, sigma_mult=0.75)

    # Block 4 (8): HL-Gauss sigma sweep (K=128)
    for max_nM in (50_000, 100_000):
        _add("hlgauss", "d1_affine", max_nM, sigma_mult=0.5)
    for max_nM in (50_000, 100_000):
        _add("hlgauss", "d1_affine", max_nM, sigma_mult=1.5)
    for max_nM in (50_000, 100_000):
        _add("hlgauss", "d2_logit", max_nM, sigma_mult=0.5)
    for max_nM in (50_000, 100_000):
        _add("hlgauss", "d2_logit", max_nM, sigma_mult=1.5)

    # Block 5 (4): Regression additive ablation
    for max_nM in (50_000, 100_000):
        _add("mhcflurry", "additive", max_nM)
    for max_nM in (50_000, 100_000):
        _add("log_mse", "additive", max_nM)

    assert len(conds) == 32, f"Expected 32 conditions, got {len(conds)}"
    return conds


CONDITIONS: List[ConditionSpec] = _make_conditions()
CONDITIONS_BY_ID: Dict[int, ConditionSpec] = {c.cond_id: c for c in CONDITIONS}


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------

class DistributionalModel(nn.Module):
    """Encoder + assay context + head wrapper.

    When the assay context encoder has ``repr_dim > 0``, the model
    feeds the head's pre-integration binding signal and the encoder's
    molecular representation (both detached) into the assay context
    encoder so that the assay bias is content-conditioned.
    """

    def __init__(
        self,
        encoder: nn.Module,
        assay_ctx: AssayContextEncoder,
        head: AffinityHead,
        spec: ConditionSpec,
        assay_input_mode: str = "factorized",
    ):
        super().__init__()
        self.encoder = encoder
        self.assay_ctx = assay_ctx
        self.head = head
        self.spec = spec
        assay_input_mode = str(assay_input_mode).strip().lower()
        if assay_input_mode not in {"factorized", "none"}:
            raise ValueError(
                f"Unsupported assay_input_mode {assay_input_mode!r}. Expected 'factorized' or 'none'."
            )
        self.assay_input_mode = assay_input_mode

    def encode_input(self, pep_tok, mhc_a_tok, mhc_b_tok):
        """Get (B, 3*embed_dim) representation from encoder."""
        if hasattr(self.encoder, 'encode'):
            return self.encoder.encode(pep_tok, mhc_a_tok, mhc_b_tok)
        return self.encoder(pep_tok, mhc_a_tok, mhc_b_tok)

    def _compute_assay_emb(
        self,
        h: torch.Tensor,
        assay_type_idx: torch.Tensor,
        assay_prep_idx: torch.Tensor,
        assay_geometry_idx: torch.Tensor,
        assay_readout_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Compute assay embedding, optionally conditioned on molecular context."""
        if self.assay_input_mode == "none":
            return torch.zeros(
                h.shape[0],
                self.assay_ctx.ctx_dim,
                device=h.device,
                dtype=h.dtype,
            )
        kwargs = {}
        if self.assay_ctx.repr_dim > 0:
            binding_signal = self.head.compute_binding_signal(h)
            kwargs["binding_logit"] = binding_signal.detach()
            kwargs["mol_repr"] = h.detach()
        return self.assay_ctx(
            assay_type_idx, assay_prep_idx, assay_geometry_idx, assay_readout_idx,
            **kwargs,
        )

    def forward(
        self,
        pep_tok, mhc_a_tok, mhc_b_tok,
        assay_type_idx=None, assay_prep_idx=None,
        assay_geometry_idx=None, assay_readout_idx=None,
    ):
        import torch
        h = self.encode_input(pep_tok, mhc_a_tok, mhc_b_tok)
        B = h.shape[0]
        device = h.device
        # Default to "unknown" (index 0) if not provided
        if assay_type_idx is None:
            assay_type_idx = torch.zeros(B, dtype=torch.long, device=device)
        if assay_prep_idx is None:
            assay_prep_idx = torch.zeros(B, dtype=torch.long, device=device)
        if assay_geometry_idx is None:
            assay_geometry_idx = torch.zeros(B, dtype=torch.long, device=device)
        if assay_readout_idx is None:
            assay_readout_idx = torch.zeros(B, dtype=torch.long, device=device)

        assay_emb = self._compute_assay_emb(
            h, assay_type_idx, assay_prep_idx, assay_geometry_idx, assay_readout_idx,
        )
        return self.head(h, assay_emb)


def build_model(
    spec: ConditionSpec,
    embed_dim: int | None = None,
    n_heads: int = 4,
    n_layers: int = 2,
    ctx_dim: int = 32,
    content_conditioned: bool = False,
    encoder_backbone: str = "historical_ablation",
    assay_input_mode: str = "factorized",
) -> DistributionalModel:
    """Build encoder + assay context + head for one condition.

    Args:
        content_conditioned: If True, the assay context encoder receives the
            binding logit and mean-pooled molecular representation (detached)
            so the assay bias depends on input content.
    """
    assay_input_mode = str(assay_input_mode).strip().lower()
    if assay_input_mode not in {"factorized", "none"}:
        raise ValueError(
            f"Unsupported assay_input_mode {assay_input_mode!r}. Expected 'factorized' or 'none'."
        )
    if assay_input_mode == "none" and content_conditioned:
        raise ValueError("content_conditioned=True is incompatible with assay_input_mode='none'")
    actual_embed_dim = embed_dim if embed_dim is not None else spec.embed_dim
    if encoder_backbone not in ENCODER_BACKBONES:
        raise ValueError(
            f"Unknown encoder_backbone {encoder_backbone!r}. Expected one of {ENCODER_BACKBONES}.",
        )
    encoder = build_encoder(
        encoder_backbone=encoder_backbone,
        embed_dim=actual_embed_dim,
        n_heads=n_heads,
        n_layers=n_layers,
    )
    repr_dim = encoder.out_dim if content_conditioned else 0
    assay_ctx = AssayContextEncoder(ctx_dim=ctx_dim, repr_dim=repr_dim)
    in_dim = encoder.out_dim  # 3 * embed_dim

    head_cls = HEAD_REGISTRY[spec.head_type]
    kwargs = dict(in_dim=in_dim, ctx_dim=ctx_dim, max_nM=spec.max_nM, assay_mode=spec.assay_mode)
    if spec.head_type in ("twohot", "hlgauss"):
        kwargs["n_bins"] = spec.n_bins
    if spec.head_type == "hlgauss":
        kwargs["sigma_mult"] = spec.sigma_mult

    head = head_cls(**kwargs)
    return DistributionalModel(
        encoder,
        assay_ctx,
        head,
        spec,
        assay_input_mode=assay_input_mode,
    )
