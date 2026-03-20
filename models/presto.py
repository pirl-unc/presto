"""Presto: Full unified model for pMHC presentation, recognition, and immunogenicity.

Canonical forward path:
- Single token stream encoder over peptide/flanks/groove-half-1/groove-half-2.
- Segmented latent-query attention DAG produces biologic latent vectors.
- Assay and task outputs are readouts of those shared latent vectors.
- Canonical assay prediction never consumes assay-selector metadata as model input.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .pmhc import (
    PROCESSING_SPECIES_BUCKETS,
    _class_probs_from_input,
    _species_probs_from_input,
)
from .presto_modules import (
    AffinityPredictor,
    ClassPresentationPredictor,
    ClassProcessingPredictor,
    PrestoTrunkState,
)
from .heads import TCellAssayHead, ElutionHead
from .affinity import (
    AFFINITY_TARGET_ENCODINGS,
    DEFAULT_MAX_AFFINITY_NM,
    DEFAULT_BINDING_MIDPOINT_NM,
    DEFAULT_BINDING_LOG10_SCALE,
    max_log10_nM,
)
from ..data.allele_resolver import (
    normalize_processing_species_label,
)
from ..data.vocab import (
    AA_VOCAB,
    AA_TO_IDX,
    N_ORGANISM_CATEGORIES,
    N_MHC_SPECIES,
    ORGANISM_TO_IDX,
    normalize_organism,
    TCELL_APC_TYPES,
    TCELL_ASSAY_METHODS,
    TCELL_ASSAY_READOUTS,
    TCELL_CULTURE_CONTEXTS,
    TCELL_PEPTIDE_FORMATS,
    TCELL_STIM_CONTEXTS,
)


def _species_idx_tensor(
    species_of_origin: Any,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Convert species-of-origin override to ORGANISM index tensor."""
    def _idx_from_label(label: Any) -> int:
        normalized = normalize_organism(str(label)) if label is not None else None
        if normalized is None:
            normalized = str(label).strip().lower() if label is not None else ""
        return ORGANISM_TO_IDX.get(normalized, 0)

    if isinstance(species_of_origin, str):
        idx = _idx_from_label(species_of_origin)
        return torch.full((batch_size,), idx, dtype=torch.long, device=device)

    if isinstance(species_of_origin, (list, tuple)):
        out = torch.zeros(batch_size, dtype=torch.long, device=device)
        for i in range(batch_size):
            value = species_of_origin[i] if i < len(species_of_origin) else None
            out[i] = _idx_from_label(value)
        return out

    if isinstance(species_of_origin, torch.Tensor):
        sp = species_of_origin.to(device=device)
        if sp.ndim == 1:
            if sp.shape[0] == 1:
                sp = sp.expand(batch_size)
            elif sp.shape[0] != batch_size:
                raise ValueError(
                    "species_of_origin id tensor batch mismatch: "
                    f"expected {batch_size}, got {sp.shape[0]}"
                )
            return sp.to(dtype=torch.long).clamp(min=0, max=N_ORGANISM_CATEGORIES - 1)
        if sp.ndim == 2:
            if sp.shape[0] == 1:
                sp = sp.expand(batch_size, -1)
            elif sp.shape[0] != batch_size:
                raise ValueError(
                    "species_of_origin probs tensor batch mismatch: "
                    f"expected {batch_size}, got {sp.shape[0]}"
                )
            return sp.argmax(dim=-1).to(dtype=torch.long).clamp(min=0, max=N_ORGANISM_CATEGORIES - 1)
        raise ValueError(
            f"unsupported species_of_origin tensor rank {sp.ndim}; expected 1 or 2"
        )

    return torch.zeros(batch_size, dtype=torch.long, device=device)


def _processing_species_idx_tensor(
    species: Any,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Convert processing species override to per-sample embedding ids."""
    species_map = {name: idx for idx, name in enumerate(PROCESSING_SPECIES_BUCKETS)}

    if isinstance(species, str):
        normalized = normalize_processing_species_label(species, default=None)
        idx = species_map.get(normalized, 0)
        return torch.full((batch_size,), idx, dtype=torch.long, device=device)

    if isinstance(species, (list, tuple)):
        ids = torch.zeros(batch_size, dtype=torch.long, device=device)
        for i in range(batch_size):
            label = normalize_processing_species_label(
                species[i] if i < len(species) else None,
                default=None,
            )
            ids[i] = species_map.get(label, 0)
        return ids

    if isinstance(species, torch.Tensor):
        sp = species.to(device=device)
        if sp.ndim == 1:
            if sp.shape[0] == 1:
                sp = sp.expand(batch_size)
            elif sp.shape[0] != batch_size:
                raise ValueError(
                    f"species id tensor batch mismatch: expected {batch_size}, got {sp.shape[0]}"
                )
            return sp.to(dtype=torch.long).clamp(min=0, max=len(PROCESSING_SPECIES_BUCKETS) - 1)
        if sp.ndim == 2:
            if sp.shape[0] == 1:
                sp = sp.expand(batch_size, -1)
            elif sp.shape[0] != batch_size:
                raise ValueError(
                    f"species probs tensor batch mismatch: expected {batch_size}, got {sp.shape[0]}"
                )
            return sp.argmax(dim=-1).to(dtype=torch.long).clamp(
                min=0,
                max=len(PROCESSING_SPECIES_BUCKETS) - 1,
            )
        raise ValueError(f"unsupported species tensor rank {sp.ndim}; expected 1 or 2")

    return torch.zeros(batch_size, dtype=torch.long, device=device)


class Presto(nn.Module):
    """Presto unified model with single token stream + latent DAG."""

    SEG_NFLANK = 0
    SEG_PEPTIDE = 1
    SEG_CFLANK = 2
    SEG_GROOVE_1 = 3
    SEG_GROOVE_2 = 4
    # Backward-compatible aliases for the public 2-segment MHC interface.
    SEG_MHC_A = SEG_GROOVE_1
    SEG_MHC_B = SEG_GROOVE_2

    LATENT_ORDER = [
        "processing",
        "ms_detectability",
        "species_of_origin",
        "pmhc_interaction",
        "recognition",
    ]

    # Per design S7.5: segment access table
    LATENT_SEGMENTS = {
        "processing": ["nflank", "peptide", "cflank"],
        "ms_detectability": ["peptide"],
        "species_of_origin": ["peptide"],  # peptide-only cross-attention
        "pmhc_interaction": ["peptide", "mhc_a", "mhc_b"],
        "recognition": ["peptide"],
    }

    # Per design S7.2: DAG dependencies
    LATENT_DEPS = {
        "processing": [],
        "ms_detectability": [],
        "species_of_origin": [],
        "pmhc_interaction": [],
        "recognition": ["foreignness"],
    }

    # Latents computed via cross-attention.
    CROSS_ATTN_LATENTS = list(LATENT_ORDER)
    N_LATENT_LAYERS = 2

    # Binding latent names that use enhanced query path when available
    BINDING_LATENT_NAMES = {"pmhc_interaction"}

    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        n_categories: Optional[int] = None,
        max_affinity_nM: float = DEFAULT_MAX_AFFINITY_NM,
        affinity_target_encoding: str = "log10",
        binding_midpoint_nM: float = DEFAULT_BINDING_MIDPOINT_NM,
        binding_log10_scale: float = DEFAULT_BINDING_LOG10_SCALE,
        # --- Binding latent architecture flags ---
        binding_n_latent_layers: int = 2,           # A: depth per binding latent
        binding_n_queries: int = 8,                  # B: multi-token queries
        binding_use_decoder_layers: bool = False,    # B: self-attn among queries
        binding_query_pool: str = "mean",            # B: pooling strategy
        use_pmhc_interaction_block: bool = False,    # C: pMHC interaction block
        pmhc_interaction_layers: int = 2,            # C: num layers
        use_groove_prior: bool = True,               # D: groove attention bias
        peptide_pos_mode: str = "triple",
        groove_pos_mode: str = "sequential",
        core_window_lengths: Optional[Sequence[int]] = None,
        core_refinement_mode: str = "shared",
        affinity_assay_residual_mode: str = "legacy",
        kd_grouping_mode: str = "merged_kd",
        binding_kinetic_input_mode: str = "affinity_vec",
        binding_direct_segment_mode: str = "off",
    ):
        """Initialize Presto."""
        super().__init__()
        self.d_model = d_model
        self.n_categories = None  # deprecated; kept for backward compat
        self.max_affinity_nM = float(max_affinity_nM)
        affinity_target_encoding = str(affinity_target_encoding).strip().lower()
        if affinity_target_encoding not in AFFINITY_TARGET_ENCODINGS:
            raise ValueError(
                f"Unsupported affinity_target_encoding: {affinity_target_encoding!r}"
            )
        self.affinity_target_encoding = affinity_target_encoding
        self.max_log10_nM = max_log10_nM(self.max_affinity_nM)
        self.binding_midpoint_nM = float(binding_midpoint_nM)
        self.binding_midpoint_log10_nM = math.log10(max(self.binding_midpoint_nM, 1e-12))
        self.binding_log10_scale = max(float(binding_log10_scale), 1e-6)
        self.missing_token_idx = int(AA_TO_IDX["<MISSING>"])
        self.x_token_idx = int(AA_TO_IDX["X"])
        # Binding latent architecture config
        self.binding_n_latent_layers = binding_n_latent_layers
        self.binding_n_queries = binding_n_queries
        self.binding_use_decoder_layers = binding_use_decoder_layers
        self.binding_query_pool = binding_query_pool
        self.use_pmhc_interaction_block = use_pmhc_interaction_block
        self.pmhc_interaction_layers = pmhc_interaction_layers
        self.pmhc_interaction_token_dim = min(64, d_model)
        self.pmhc_interaction_vec_dim = (
            max(1, self.binding_n_queries) * self.pmhc_interaction_token_dim
        )
        normalized_core_lengths = sorted(
            {
                max(1, int(length))
                for length in (core_window_lengths if core_window_lengths is not None else (9,))
            }
        )
        self.core_window_lengths = tuple(normalized_core_lengths or [9])
        self.core_window_size = max(self.core_window_lengths)
        self.max_pfr_length = 50
        self.pfr_length_dim = 32
        self.use_groove_prior = use_groove_prior
        peptide_pos_mode = str(peptide_pos_mode).strip().lower()
        if peptide_pos_mode not in {
            "triple",
            "triple_baseline",
            "abs_only",
            "triple_plus_abs",
            "start_only",
            "end_only",
            "start_plus_end",
            "concat_start_end",
            "concat_start_end_frac",
            "mlp_start_end",
            "mlp_start_end_frac",
        }:
            raise ValueError(f"Unsupported peptide_pos_mode: {peptide_pos_mode!r}")
        self.peptide_pos_mode = peptide_pos_mode
        groove_pos_mode = str(groove_pos_mode).strip().lower()
        if groove_pos_mode not in {
            "sequential",
            "triple",
            "triple_baseline",
            "abs_only",
            "triple_plus_abs",
            "start_only",
            "end_only",
            "start_plus_end",
            "concat_start_end",
            "concat_start_end_frac",
            "mlp_start_end",
            "mlp_start_end_frac",
        }:
            raise ValueError(f"Unsupported groove_pos_mode: {groove_pos_mode!r}")
        self.groove_pos_mode = groove_pos_mode
        core_refinement_mode = str(core_refinement_mode).strip().lower()
        if core_refinement_mode not in {"shared", "class_specific"}:
            raise ValueError(f"Unsupported core_refinement_mode: {core_refinement_mode!r}")
        self.core_refinement_mode = core_refinement_mode
        affinity_assay_residual_mode = str(affinity_assay_residual_mode).strip().lower()
        if affinity_assay_residual_mode not in {
            "legacy",
            "pooled_single_output",
            "shared_base_segment_residual",
            "shared_base_factorized_context_residual",
            "shared_base_factorized_context_plus_segment_residual",
            "dag_family",
            "dag_method_leaf",
            "dag_prep_readout_leaf",
        }:
            raise ValueError(
                f"Unsupported affinity_assay_residual_mode: {affinity_assay_residual_mode!r}"
            )
        self.affinity_assay_residual_mode = affinity_assay_residual_mode
        kd_grouping_mode = str(kd_grouping_mode).strip().lower()
        if kd_grouping_mode not in {"merged_kd", "split_kd_proxy"}:
            raise ValueError(f"Unsupported kd_grouping_mode: {kd_grouping_mode!r}")
        self.kd_grouping_mode = kd_grouping_mode
        binding_kinetic_input_mode = str(binding_kinetic_input_mode).strip().lower()
        if binding_kinetic_input_mode not in {"affinity_vec", "interaction_vec", "fused"}:
            raise ValueError(
                "Unsupported binding_kinetic_input_mode: "
                f"{binding_kinetic_input_mode!r}"
            )
        self.binding_kinetic_input_mode = binding_kinetic_input_mode
        binding_direct_segment_mode = str(binding_direct_segment_mode).strip().lower()
        if binding_direct_segment_mode not in {
            "off",
            "affinity_residual",
            "affinity_stability_residual",
            "gated_affinity",
        }:
            raise ValueError(
                "Unsupported binding_direct_segment_mode: "
                f"{binding_direct_segment_mode!r}"
            )
        self.binding_direct_segment_mode = binding_direct_segment_mode
        self._has_binding_enhancements = (
            binding_n_latent_layers != self.N_LATENT_LAYERS
            or binding_n_queries > 1
            or use_pmhc_interaction_block
            or use_groove_prior
        )

        # ------------------------------------------------------------------
        # Single token stream encoder
        # ------------------------------------------------------------------
        self.aa_embedding = nn.Embedding(len(AA_VOCAB), d_model, padding_idx=0)
        # Keep ambiguous 'X' neutral: fixed zero vector.
        with torch.no_grad():
            self.aa_embedding.weight[self.x_token_idx].zero_()
        self.aa_embedding.weight.register_hook(self._zero_x_embedding_grad)
        self.segment_embedding = nn.Embedding(5, d_model)

        # Segment-specific positional encoding (design S3.2.3)
        # Peptide: triple-frame encoding
        self.pep_nterm_pos = nn.Embedding(50, d_model)
        self.pep_cterm_pos = nn.Embedding(50, d_model)
        self.pep_abs_pos = nn.Embedding(50, d_model)
        self.pep_frac_mlp = nn.Sequential(
            nn.Linear(1, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.pep_pos_concat_proj = nn.Linear(2 * d_model, d_model)
        self.pep_pos_concat_frac_proj = nn.Linear(2 * d_model + 2, d_model)
        self.pep_pos_concat_mlp = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.pep_pos_concat_frac_mlp = nn.Sequential(
            nn.Linear(2 * d_model + 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        # Flanks: distance-from-cleavage
        self.nflank_dist_pos = nn.Embedding(25, d_model)
        self.cflank_dist_pos = nn.Embedding(25, d_model)
        # Groove halves: per-segment sequential
        self.groove_1_pos = nn.Embedding(120, d_model)
        self.groove_2_pos = nn.Embedding(120, d_model)
        self.groove_1_abs_pos = nn.Embedding(120, d_model)
        self.groove_2_abs_pos = nn.Embedding(120, d_model)
        self.groove_1_end_pos = nn.Embedding(120, d_model)
        self.groove_2_end_pos = nn.Embedding(120, d_model)
        self.groove_frac_mlp = nn.Sequential(
            nn.Linear(1, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.groove_pos_concat_proj = nn.Linear(2 * d_model, d_model)
        self.groove_pos_concat_frac_proj = nn.Linear(2 * d_model + 2, d_model)
        self.groove_pos_concat_mlp = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.groove_pos_concat_frac_mlp = nn.Sequential(
            nn.Linear(2 * d_model + 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        # Binding-core positions within the candidate window.
        self.core_position_embed = nn.Embedding(self.core_window_size, d_model)
        self.pfr_length_embed = nn.Embedding(self.max_pfr_length + 1, self.pfr_length_dim)

        # Global conditioning embedding (design S3.2.4)
        self.species_cond_embed = nn.Embedding(7, d_model)    # 7 species categories
        self.chain_completeness_embed = nn.Embedding(64, d_model)  # 6-bit bitfield

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.stream_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            enable_nested_tensor=False,
        )
        self.stream_norm = nn.LayerNorm(d_model)

        self.pmhc_interaction_token_proj = nn.Linear(d_model, self.pmhc_interaction_token_dim)
        self.pmhc_interaction_vec_norm = nn.LayerNorm(self.pmhc_interaction_vec_dim)
        self.binding_affinity_readout_proj = nn.Sequential(
            nn.Linear(self.pmhc_interaction_vec_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.binding_stability_readout_proj = nn.Sequential(
            nn.Linear(self.pmhc_interaction_vec_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.binding_direct_segment_affinity_proj = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.binding_direct_segment_stability_proj = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.binding_direct_segment_gate = nn.Sequential(
            nn.Linear(2 * d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )
        # Class-specific processing projections
        self.processing_class1_proj = nn.Linear(d_model, d_model)
        self.processing_class2_proj = nn.Linear(d_model, d_model)
        # Class-specific presentation MLPs
        self.presentation_class1_mlp = nn.Sequential(
            nn.Linear(d_model + self.pmhc_interaction_vec_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.presentation_class2_mlp = nn.Sequential(
            nn.Linear(d_model + self.pmhc_interaction_vec_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.presentation_class1_vec_norm = nn.LayerNorm(d_model)
        self.presentation_class2_vec_norm = nn.LayerNorm(d_model)
        # Lineage-specific immunogenicity MLPs
        self.immunogenicity_cd8_mlp = nn.Sequential(
            nn.Linear(self.pmhc_interaction_vec_dim + d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.immunogenicity_cd4_mlp = nn.Sequential(
            nn.Linear(self.pmhc_interaction_vec_dim + d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.immunogenicity_cd8_vec_norm = nn.LayerNorm(d_model)
        self.immunogenicity_cd4_vec_norm = nn.LayerNorm(d_model)
        self.core_window_fuse = nn.Sequential(
            nn.Linear(
                self.pmhc_interaction_vec_dim + 2 * d_model + 2 * self.pfr_length_dim,
                d_model,
            ),
            nn.GELU(),
            nn.Linear(d_model, self.pmhc_interaction_vec_dim),
        )
        self.core_window_vec_norm = nn.LayerNorm(self.pmhc_interaction_vec_dim)
        self.core_window_score = nn.Sequential(
            nn.Linear(self.pmhc_interaction_vec_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.core_window_score_class1 = nn.Sequential(
            nn.Linear(self.pmhc_interaction_vec_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.core_window_score_class2 = nn.Sequential(
            nn.Linear(self.pmhc_interaction_vec_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.core_window_prior = nn.Sequential(
            nn.Linear(5, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

        # ------------------------------------------------------------------
        # Per-chain MHC inference heads (design S5.1-S5.3)
        # ------------------------------------------------------------------
        n_species = N_MHC_SPECIES
        # Per-chain fine type (5 classes):
        # {MHC_I, MHC_IIa, MHC_IIb, B2M, unknown}
        from ..data.vocab import N_MHC_CHAIN_FINE_TYPES
        self.mhc_a_type_head = nn.Linear(d_model, N_MHC_CHAIN_FINE_TYPES)
        self.mhc_b_type_head = nn.Linear(d_model, N_MHC_CHAIN_FINE_TYPES)
        # Per-chain species
        self.mhc_a_species_head = nn.Linear(d_model, n_species)
        self.mhc_b_species_head = nn.Linear(d_model, n_species)
        # Chain compatibility (design S5.3)
        n_fine_types = N_MHC_CHAIN_FINE_TYPES
        self.chain_compat_head = nn.Sequential(
            nn.Linear(d_model * 2 + n_fine_types * 2 + n_species * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

        # ------------------------------------------------------------------
        # Latent query DAG (segmented attention per latent, N_LATENT_LAYERS deep)
        # ------------------------------------------------------------------
        # Binding latents may use multi-token queries (Variant B)
        if binding_n_queries > 1:
            self.latent_queries = nn.ParameterDict({
                name: nn.Parameter(
                    torch.randn(binding_n_queries, d_model) * 0.02
                    if name in self.BINDING_LATENT_NAMES
                    else torch.randn(d_model) * 0.02
                )
                for name in self.CROSS_ATTN_LATENTS
            })
        else:
            self.latent_queries = nn.ParameterDict({
                name: nn.Parameter(torch.randn(d_model) * 0.02)
                for name in self.CROSS_ATTN_LATENTS
            })

        def _n_layers_for(name: str) -> int:
            if name in self.BINDING_LATENT_NAMES:
                return binding_n_latent_layers
            return self.N_LATENT_LAYERS

        self.latent_layers = nn.ModuleDict({
            name: nn.ModuleList([
                nn.ModuleDict({
                    "attn": nn.MultiheadAttention(d_model, n_heads, dropout=0.1, batch_first=True),
                    "norm1": nn.LayerNorm(d_model),
                    "ffn": nn.Sequential(
                        nn.Linear(d_model, d_model * 4),
                        nn.GELU(),
                        nn.Linear(d_model * 4, d_model),
                        nn.Dropout(0.1),
                    ),
                    "norm2": nn.LayerNorm(d_model),
                })
                for _ in range(_n_layers_for(name))
            ])
            for name in self.CROSS_ATTN_LATENTS
        })

        # Variant B: self-attention layers among multi-token binding queries
        if binding_n_queries > 1 and binding_use_decoder_layers:
            self.binding_query_self_attn = nn.ModuleDict({
                name: nn.ModuleList([
                    nn.ModuleDict({
                        "self_attn": nn.MultiheadAttention(d_model, n_heads, dropout=0.1, batch_first=True),
                        "norm": nn.LayerNorm(d_model),
                    })
                    for _ in range(_n_layers_for(name))
                ])
                for name in self.BINDING_LATENT_NAMES
            })
        # Variant B: pooling projection for multi-token queries
        if binding_n_queries > 1:
            if binding_query_pool == "attention":
                self.binding_pool_attn = nn.Linear(d_model, 1)
            # mean pooling needs no extra params

        # Variant C: pMHC interaction block
        if use_pmhc_interaction_block:
            self.pmhc_interaction = nn.ModuleList([
                nn.ModuleDict({
                    "pep_to_mhc_attn": nn.MultiheadAttention(d_model, n_heads, dropout=0.1, batch_first=True),
                    "pep_norm1": nn.LayerNorm(d_model),
                    "pep_ffn": nn.Sequential(
                        nn.Linear(d_model, d_model * 4), nn.GELU(),
                        nn.Linear(d_model * 4, d_model), nn.Dropout(0.1),
                    ),
                    "pep_norm2": nn.LayerNorm(d_model),
                    "mhc_to_pep_attn": nn.MultiheadAttention(d_model, n_heads, dropout=0.1, batch_first=True),
                    "mhc_norm1": nn.LayerNorm(d_model),
                    "mhc_ffn": nn.Sequential(
                        nn.Linear(d_model, d_model * 4), nn.GELU(),
                        nn.Linear(d_model * 4, d_model), nn.Dropout(0.1),
                    ),
                    "mhc_norm2": nn.LayerNorm(d_model),
                })
                for _ in range(pmhc_interaction_layers)
            ])

        self.groove_attn = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=0.1,
            batch_first=True,
        )
        self.groove_query = nn.Parameter(torch.randn(1, d_model) * 0.02)

        # Context vector MLP (design S5.4): class_probs + a_species + b_species + compat
        self.context_token_proj = nn.Sequential(
            nn.Linear(2 + n_species * 2 + 1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # ------------------------------------------------------------------
        # Readout heads from latent vectors
        # ------------------------------------------------------------------
        self.class1_processing_predictor = ClassProcessingPredictor(d_model)
        self.class2_processing_predictor = ClassProcessingPredictor(d_model)
        self.affinity_predictor = AffinityPredictor(
            d_model=d_model,
            interaction_dim=self.pmhc_interaction_vec_dim,
            max_log10_nM=self.max_log10_nM,
            binding_midpoint_nM=self.binding_midpoint_nM,
            binding_log10_scale=self.binding_log10_scale,
            binding_kinetic_input_mode=self.binding_kinetic_input_mode,
            affinity_assay_residual_mode=self.affinity_assay_residual_mode,
            kd_grouping_mode=self.kd_grouping_mode,
            affinity_target_encoding=self.affinity_target_encoding,
            max_affinity_nM=self.max_affinity_nM,
        )
        self.class1_presentation_predictor = ClassPresentationPredictor(d_model)
        self.class2_presentation_predictor = ClassPresentationPredictor(d_model)
        self.recognition_cd8_head = nn.Linear(d_model, 1)
        self.recognition_cd4_head = nn.Linear(d_model, 1)
        self.immunogenicity_cd8_latent_head = nn.Linear(d_model, 1)
        self.immunogenicity_cd4_latent_head = nn.Linear(d_model, 1)

        # MS detectability readout from latent (design S7.4)
        self.ms_detectability_head = nn.Linear(d_model, 1)

        # Elution head (S9.3: pres_logit + ms_detect_logit, no pmhc_vec).
        self.elution_head = ElutionHead()
        self.tcr_evidence_head = nn.Sequential(
            nn.Linear(self.pmhc_interaction_vec_dim, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )
        self.tcr_evidence_method_head = nn.Linear(self.pmhc_interaction_vec_dim, 3)

        self.tcell_assay_head = TCellAssayHead(
            d_model=d_model,
            n_assay_methods=len(TCELL_ASSAY_METHODS),
            n_assay_readouts=len(TCELL_ASSAY_READOUTS),
            n_apc_types=len(TCELL_APC_TYPES),
            n_culture_contexts=len(TCELL_CULTURE_CONTEXTS),
            n_stim_contexts=len(TCELL_STIM_CONTEXTS),
            n_peptide_formats=len(TCELL_PEPTIDE_FORMATS),
        )

        # Species of origin and foreignness heads (replaces old CategoryHead)
        self.foreignness_proj = nn.Linear(d_model, d_model)
        self.species_of_origin_head = nn.Linear(d_model, N_ORGANISM_CATEGORIES)
        self.foreignness_head = nn.Linear(d_model, 1)
        self.species_override_embed = nn.Embedding(N_ORGANISM_CATEGORIES, d_model)
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize transformer-style components to a stable working scale."""
        query_std = 1.0 / math.sqrt(self.d_model)

        for param in self.latent_queries.values():
            nn.init.normal_(param.data, std=query_std)
        nn.init.normal_(self.groove_query.data, std=query_std)

        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=query_std)
                if module.padding_idx is not None:
                    with torch.no_grad():
                        module.weight[module.padding_idx].zero_()
            elif isinstance(module, nn.MultiheadAttention):
                if getattr(module, "in_proj_weight", None) is not None:
                    nn.init.xavier_uniform_(module.in_proj_weight)
                if getattr(module, "in_proj_bias", None) is not None:
                    nn.init.zeros_(module.in_proj_bias)
                if getattr(module, "q_proj_weight", None) is not None:
                    nn.init.xavier_uniform_(module.q_proj_weight)
                if getattr(module, "k_proj_weight", None) is not None:
                    nn.init.xavier_uniform_(module.k_proj_weight)
                if getattr(module, "v_proj_weight", None) is not None:
                    nn.init.xavier_uniform_(module.v_proj_weight)
                if getattr(module, "out_proj", None) is not None:
                    nn.init.xavier_uniform_(module.out_proj.weight)
                    if module.out_proj.bias is not None:
                        nn.init.zeros_(module.out_proj.bias)
        with torch.no_grad():
            self.aa_embedding.weight[self.x_token_idx].zero_()

    @property
    def apc_cell_type_context_proj(self) -> nn.Sequential:
        """Renamed APC/cell-type context projection module."""
        return self.context_token_proj

    @property
    def mhc_a_pos(self) -> nn.Embedding:
        """Backward-compatible alias for groove-half-1 positional embeddings."""
        return self.groove_1_pos

    @property
    def mhc_b_pos(self) -> nn.Embedding:
        """Backward-compatible alias for groove-half-2 positional embeddings."""
        return self.groove_2_pos

    @property
    def processing_class1_head(self) -> nn.Linear:
        return self.class1_processing_predictor.head

    @property
    def processing_class2_head(self) -> nn.Linear:
        return self.class2_processing_predictor.head

    @property
    def presentation_class1_latent_head(self) -> nn.Linear:
        return self.class1_presentation_predictor.head

    @property
    def presentation_class2_latent_head(self) -> nn.Linear:
        return self.class2_presentation_predictor.head

    @property
    def binding_affinity_probe(self) -> nn.Module:
        return self.affinity_predictor.binding_affinity_probe

    @property
    def binding(self) -> nn.Module:
        return self.affinity_predictor.binding

    @property
    def assay_heads(self) -> nn.Module:
        return self.affinity_predictor.assay_heads

    @property
    def kd_assay_bias(self) -> nn.Module:
        return self.affinity_predictor.kd_assay_bias

    @kd_assay_bias.setter
    def kd_assay_bias(self, module: nn.Module) -> None:
        self.affinity_predictor.kd_assay_bias = module

    @property
    def kd_assay_bias_scale(self) -> nn.Parameter:
        return self.affinity_predictor.kd_assay_bias_scale

    @kd_assay_bias_scale.setter
    def kd_assay_bias_scale(self, param: nn.Parameter) -> None:
        self.affinity_predictor.kd_assay_bias_scale = param

    @property
    def binding_probe_mix_logit(self) -> nn.Parameter:
        return self.affinity_predictor.binding_probe_mix_logit

    @binding_probe_mix_logit.setter
    def binding_probe_mix_logit(self, param: nn.Parameter) -> None:
        self.affinity_predictor.binding_probe_mix_logit = param

    @property
    def w_binding_class1_calibration(self) -> nn.Parameter:
        return self.affinity_predictor.w_binding_class1_calibration

    @w_binding_class1_calibration.setter
    def w_binding_class1_calibration(self, param: nn.Parameter) -> None:
        self.affinity_predictor.w_binding_class1_calibration = param

    @property
    def w_binding_class2_calibration(self) -> nn.Parameter:
        return self.affinity_predictor.w_binding_class2_calibration

    @w_binding_class2_calibration.setter
    def w_binding_class2_calibration(self, param: nn.Parameter) -> None:
        self.affinity_predictor.w_binding_class2_calibration = param

    def _load_from_state_dict(
        self,
        state_dict: Dict[str, torch.Tensor],
        prefix: str,
        local_metadata: Dict[str, Any],
        strict: bool,
        missing_keys: List[str],
        unexpected_keys: List[str],
        error_msgs: List[str],
    ) -> None:
        """Drop deprecated weights for backward compatibility."""
        legacy_key = f"{prefix}mhc_class_cond_embed.weight"
        if legacy_key in state_dict:
            state_dict.pop(legacy_key)
        head_key_map = {
            f"{prefix}processing_class1_head.": f"{prefix}class1_processing_predictor.head.",
            f"{prefix}processing_class2_head.": f"{prefix}class2_processing_predictor.head.",
            f"{prefix}presentation_class1_latent_head.": f"{prefix}class1_presentation_predictor.head.",
            f"{prefix}presentation_class2_latent_head.": f"{prefix}class2_presentation_predictor.head.",
            f"{prefix}binding_affinity_probe.": f"{prefix}affinity_predictor.binding_affinity_probe.",
            f"{prefix}binding.": f"{prefix}affinity_predictor.binding.",
            f"{prefix}assay_heads.": f"{prefix}affinity_predictor.assay_heads.",
            f"{prefix}kd_assay_bias.": f"{prefix}affinity_predictor.kd_assay_bias.",
        }
        exact_key_map = {
            f"{prefix}kd_assay_bias_scale": f"{prefix}affinity_predictor.kd_assay_bias_scale",
            f"{prefix}binding_probe_mix_logit": f"{prefix}affinity_predictor.binding_probe_mix_logit",
            f"{prefix}w_binding_class1_calibration": f"{prefix}affinity_predictor.w_binding_class1_calibration",
            f"{prefix}w_binding_class2_calibration": f"{prefix}affinity_predictor.w_binding_class2_calibration",
        }
        for old_prefix, new_prefix in head_key_map.items():
            for key in list(state_dict.keys()):
                if not key.startswith(old_prefix):
                    continue
                new_key = new_prefix + key[len(old_prefix) :]
                if new_key not in state_dict:
                    state_dict[new_key] = state_dict[key]
                state_dict.pop(key)
        for old_key, new_key in exact_key_map.items():
            if old_key in state_dict:
                if new_key not in state_dict:
                    state_dict[new_key] = state_dict[old_key]
                state_dict.pop(old_key)
        old_pos_to_new = {
            f"{prefix}mhc_a_pos.weight": f"{prefix}groove_1_pos.weight",
            f"{prefix}mhc_b_pos.weight": f"{prefix}groove_2_pos.weight",
        }
        for old_key, new_key in old_pos_to_new.items():
            if old_key in state_dict and new_key not in state_dict:
                old_weight = state_dict.pop(old_key)
                new_weight = getattr(self, new_key[len(prefix) : -len('.weight')]).weight
                if old_weight.shape == new_weight.shape:
                    state_dict[new_key] = old_weight
                else:
                    state_dict[new_key] = old_weight[: new_weight.shape[0], :]
            elif old_key in state_dict:
                state_dict.pop(old_key)
        # Drop old CategoryHead keys from pre-foreignness checkpoints
        # Drop old DAG-violating head keys (elution context_head, MSDetectionHead,
        # RepertoireHead, old TCellAssayHead keys)
        drop_prefixes = [
            f"{prefix}category_head.",
            f"{prefix}elution_head.context_head.",
            f"{prefix}ms_detection.",
            f"{prefix}repertoire.",
            f"{prefix}tcr_encoder.",
            f"{prefix}matcher.",
            f"{prefix}chain_classifier.",
            f"{prefix}chain_attribute_classifier.",
            f"{prefix}cell_classifier.",
            f"{prefix}tcell_assay_head.base_with_tcr.",
            f"{prefix}tcell_assay_head.base_without_tcr.",
            f"{prefix}tcell_assay_head.context_bias.",
            f"{prefix}tcell_assay_head.lineage_projection.",
            f"{prefix}tcell_assay_head.assay_method_classifier.",
            f"{prefix}tcell_assay_head.assay_readout_classifier.",
            f"{prefix}tcell_assay_head.apc_type_classifier.",
            f"{prefix}tcell_assay_head.culture_context_classifier.",
            f"{prefix}tcell_assay_head.stim_context_classifier.",
            # Removed in information-flow simplification
            f"{prefix}pmhc_vec_proj.",
            f"{prefix}presentation_mlp.",
            f"{prefix}presentation_vec_norm.",
            f"{prefix}immunogenicity_mlp.",
            f"{prefix}immunogenicity_vec_norm.",
        ]
        drop_exact = [
            f"{prefix}groove_bias_a",
            f"{prefix}groove_bias_b",
            f"{prefix}elution_head.w_context",
            f"{prefix}elution_head.w_processing",
            f"{prefix}tcell_assay_head.w_bio",
            f"{prefix}tcell_assay_head.w_base",
            f"{prefix}tcell_assay_head.w_ctx",
            f"{prefix}tcell_assay_head.w_lineage",
            f"{prefix}tcell_assay_head.bias",
            # Dead scalar parameters removed in information-flow simplification
            f"{prefix}w_presentation_class1_latent",
            f"{prefix}w_presentation_class2_latent",
            f"{prefix}w_class1_presentation_stability",
            f"{prefix}w_class2_presentation_stability",
            f"{prefix}w_class1_presentation_class",
            f"{prefix}w_class2_presentation_class",
        ]
        for key in list(state_dict.keys()):
            if any(key.startswith(dp) for dp in drop_prefixes):
                state_dict.pop(key)
            elif key in drop_exact:
                state_dict.pop(key)
        # Drop keys with shape mismatches from information-flow simplification:
        # binding_affinity_readout_proj (1536→512 input), binding heads (512→256),
        # tcr_evidence heads (256→512 input)
        for key in list(state_dict.keys()):
            if not key.startswith(prefix):
                continue
            local_key = key[len(prefix):]
            parts = local_key.split(".")
            try:
                mod = self
                for part in parts[:-1]:
                    if part.isdigit():
                        mod = mod[int(part)]
                    else:
                        mod = getattr(mod, part)
                param_name = parts[-1]
                if hasattr(mod, param_name):
                    own = getattr(mod, param_name)
                    if isinstance(own, torch.Tensor) and own.shape != state_dict[key].shape:
                        state_dict.pop(key)
            except (AttributeError, IndexError, KeyError, TypeError):
                pass
        super()._load_from_state_dict(
            state_dict=state_dict,
            prefix=prefix,
            local_metadata=local_metadata,
            strict=strict,
            missing_keys=missing_keys,
            unexpected_keys=unexpected_keys,
            error_msgs=error_msgs,
        )
        with torch.no_grad():
            self.aa_embedding.weight[self.x_token_idx].zero_()

    @staticmethod
    def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Mask-aware mean over sequence dimension."""
        mask_f = mask.unsqueeze(-1).float()
        return (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)

    def _zero_x_embedding_grad(self, grad: torch.Tensor) -> torch.Tensor:
        """Keep X token embedding fixed at zero by zeroing its gradient row."""
        if grad is None:
            return grad
        if grad.ndim != 2:
            return grad
        grad = grad.clone()
        grad[self.x_token_idx].zero_()
        return grad

    def _segment_tensor(self, length: int, seg_id: int, device: torch.device) -> torch.Tensor:
        return torch.full((length,), seg_id, device=device, dtype=torch.long)

    def _ensure_optional_segment(
        self,
        tok: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Ensure optional segment is present as at least one non-pad token."""
        if tok is None:
            return torch.full(
                (batch_size, 1),
                self.missing_token_idx,
                dtype=torch.long,
                device=device,
            )
        seg = tok.to(device=device, dtype=torch.long)
        if seg.ndim != 2:
            raise ValueError(f"Expected 2D token tensor, got shape={tuple(seg.shape)}")

        # For rows that are fully padded, force a sentinel non-pad token.
        mask = seg != 0
        empty = ~mask.any(dim=1)
        if empty.any():
            seg = seg.clone()
            seg[empty, 0] = self.missing_token_idx
        return seg

    def _build_single_stream(
        self,
        pep_tok: torch.Tensor,
        mhc_a_tok: torch.Tensor,
        mhc_b_tok: torch.Tensor,
        flank_n_tok: Optional[torch.Tensor] = None,
        flank_c_tok: Optional[torch.Tensor] = None,
        species_id: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Build and encode single token stream."""
        device = pep_tok.device
        batch_size = pep_tok.shape[0]

        pep = self._ensure_optional_segment(pep_tok, batch_size=batch_size, device=device)
        mhc_a = self._ensure_optional_segment(mhc_a_tok, batch_size=batch_size, device=device)
        mhc_b = self._ensure_optional_segment(mhc_b_tok, batch_size=batch_size, device=device)
        nflank = self._ensure_optional_segment(flank_n_tok, batch_size=batch_size, device=device)
        cflank = self._ensure_optional_segment(flank_c_tok, batch_size=batch_size, device=device)

        parts = [
            (nflank, "nflank", self.SEG_NFLANK),
            (pep, "peptide", self.SEG_PEPTIDE),
            (cflank, "cflank", self.SEG_CFLANK),
            (mhc_a, "mhc_a", self.SEG_MHC_A),
            (mhc_b, "mhc_b", self.SEG_MHC_B),
        ]

        tokens = torch.cat([p[0] for p in parts], dim=1)
        valid_mask = torch.cat([(p[0] != 0) for p in parts], dim=1)

        seg_ids = torch.cat(
            [self._segment_tensor(p[0].shape[1], p[2], device=device) for p in parts],
            dim=0,
        )

        offsets: Dict[str, slice] = {}
        start = 0
        for seg_tok, seg_name, _ in parts:
            end = start + seg_tok.shape[1]
            offsets[seg_name] = slice(start, end)
            start = end

        seq_len = tokens.shape[1]
        tok_ids = tokens.clamp(min=0, max=len(AA_VOCAB) - 1)

        # Peptide position encoding.
        pep_sl = offsets["peptide"]
        pep_len_per = (tokens[:, pep_sl] != 0).sum(dim=1).clamp(min=1)  # (B,)
        pep_idx = torch.arange(pep_sl.stop - pep_sl.start, device=device).unsqueeze(0).expand(batch_size, -1)
        nterm_idx = pep_idx.clamp(max=self.pep_nterm_pos.num_embeddings - 1)
        cterm_dist = (pep_len_per.unsqueeze(1) - 1 - pep_idx).clamp(min=0)
        cterm_idx = cterm_dist.clamp(max=self.pep_cterm_pos.num_embeddings - 1)
        nterm_frac = pep_idx.float() / (pep_len_per.unsqueeze(1) - 1).clamp(min=1).float()
        cterm_frac = cterm_dist.float() / (pep_len_per.unsqueeze(1) - 1).clamp(min=1).float()
        pep_pos_embed = self._compose_position_signal(
            mode=self.peptide_pos_mode,
            start_embed=self.pep_nterm_pos(nterm_idx),
            end_embed=self.pep_cterm_pos(cterm_idx),
            start_frac=nterm_frac,
            end_frac=cterm_frac,
            frac_mlp=self.pep_frac_mlp,
            abs_embed=self.pep_abs_pos(
                pep_idx.clamp(max=self.pep_abs_pos.num_embeddings - 1)
            ),
            concat_proj=self.pep_pos_concat_proj,
            concat_frac_proj=self.pep_pos_concat_frac_proj,
            concat_mlp=self.pep_pos_concat_mlp,
            concat_frac_mlp=self.pep_pos_concat_frac_mlp,
        )

        # N-flank: distance-from-cleavage (reversed: last position = closest to cleavage)
        nfl_sl = offsets["nflank"]
        nfl_len_per = (tokens[:, nfl_sl] != 0).sum(dim=1).clamp(min=1)
        nfl_idx = torch.arange(nfl_sl.stop - nfl_sl.start, device=device).unsqueeze(0).expand(batch_size, -1)
        nfl_dist = (nfl_len_per.unsqueeze(1) - 1 - nfl_idx).clamp(min=0)
        nfl_pos_embed = self.nflank_dist_pos(
            nfl_dist.clamp(max=self.nflank_dist_pos.num_embeddings - 1)
        )

        # C-flank: distance-from-cleavage (first position = closest to cleavage)
        cfl_sl = offsets["cflank"]
        cfl_idx = torch.arange(cfl_sl.stop - cfl_sl.start, device=device).unsqueeze(0).expand(batch_size, -1)
        cfl_pos_embed = self.cflank_dist_pos(
            cfl_idx.clamp(max=self.cflank_dist_pos.num_embeddings - 1)
        )

        # Groove half 1 position encoding.
        mhc_a_sl = offsets["mhc_a"]
        mhc_a_idx = torch.arange(mhc_a_sl.stop - mhc_a_sl.start, device=device).unsqueeze(0).expand(batch_size, -1)
        if self.groove_pos_mode == "sequential":
            mhc_a_pos_embed = self.groove_1_pos(
                mhc_a_idx.clamp(max=self.groove_1_pos.num_embeddings - 1)
            )
        else:
            mhc_a_len_per = (tokens[:, mhc_a_sl] != 0).sum(dim=1).clamp(min=1)
            mhc_a_start_idx = mhc_a_idx.clamp(max=self.groove_1_pos.num_embeddings - 1)
            mhc_a_end_dist = (mhc_a_len_per.unsqueeze(1) - 1 - mhc_a_idx).clamp(min=0)
            mhc_a_end_idx = mhc_a_end_dist.clamp(max=self.groove_1_end_pos.num_embeddings - 1)
            mhc_a_start_frac = mhc_a_idx.float() / (mhc_a_len_per.unsqueeze(1) - 1).clamp(min=1).float()
            mhc_a_end_frac = mhc_a_end_dist.float() / (mhc_a_len_per.unsqueeze(1) - 1).clamp(min=1).float()
            mhc_a_pos_embed = self._compose_position_signal(
                mode=self.groove_pos_mode,
                start_embed=self.groove_1_pos(mhc_a_start_idx),
                end_embed=self.groove_1_end_pos(mhc_a_end_idx),
                start_frac=mhc_a_start_frac,
                end_frac=mhc_a_end_frac,
                frac_mlp=self.groove_frac_mlp,
                abs_embed=self.groove_1_abs_pos(
                    mhc_a_idx.clamp(max=self.groove_1_abs_pos.num_embeddings - 1)
                ),
                concat_proj=self.groove_pos_concat_proj,
                concat_frac_proj=self.groove_pos_concat_frac_proj,
                concat_mlp=self.groove_pos_concat_mlp,
                concat_frac_mlp=self.groove_pos_concat_frac_mlp,
            )

        # Groove half 2 position encoding.
        mhc_b_sl = offsets["mhc_b"]
        mhc_b_idx = torch.arange(mhc_b_sl.stop - mhc_b_sl.start, device=device).unsqueeze(0).expand(batch_size, -1)
        if self.groove_pos_mode == "sequential":
            mhc_b_pos_embed = self.groove_2_pos(
                mhc_b_idx.clamp(max=self.groove_2_pos.num_embeddings - 1)
            )
        else:
            mhc_b_len_per = (tokens[:, mhc_b_sl] != 0).sum(dim=1).clamp(min=1)
            mhc_b_start_idx = mhc_b_idx.clamp(max=self.groove_2_pos.num_embeddings - 1)
            mhc_b_end_dist = (mhc_b_len_per.unsqueeze(1) - 1 - mhc_b_idx).clamp(min=0)
            mhc_b_end_idx = mhc_b_end_dist.clamp(max=self.groove_2_end_pos.num_embeddings - 1)
            mhc_b_start_frac = mhc_b_idx.float() / (mhc_b_len_per.unsqueeze(1) - 1).clamp(min=1).float()
            mhc_b_end_frac = mhc_b_end_dist.float() / (mhc_b_len_per.unsqueeze(1) - 1).clamp(min=1).float()
            mhc_b_pos_embed = self._compose_position_signal(
                mode=self.groove_pos_mode,
                start_embed=self.groove_2_pos(mhc_b_start_idx),
                end_embed=self.groove_2_end_pos(mhc_b_end_idx),
                start_frac=mhc_b_start_frac,
                end_frac=mhc_b_end_frac,
                frac_mlp=self.groove_frac_mlp,
                abs_embed=self.groove_2_abs_pos(
                    mhc_b_idx.clamp(max=self.groove_2_abs_pos.num_embeddings - 1)
                ),
                concat_proj=self.groove_pos_concat_proj,
                concat_frac_proj=self.groove_pos_concat_frac_proj,
                concat_mlp=self.groove_pos_concat_mlp,
                concat_frac_mlp=self.groove_pos_concat_frac_mlp,
            )
        pos_embed = torch.cat(
            [
                nfl_pos_embed,
                pep_pos_embed,
                cfl_pos_embed,
                mhc_a_pos_embed,
                mhc_b_pos_embed,
            ],
            dim=1,
        )

        # Global conditioning embedding (design S3.2.4)
        if species_id is None:
            species_id = torch.zeros(batch_size, dtype=torch.long, device=device)

        # Chain completeness bitfield (computed per sample; avoids cross-sample leakage)
        has_nflank = (
            (flank_n_tok != 0).any(dim=1)
            if flank_n_tok is not None
            else torch.zeros(batch_size, dtype=torch.bool, device=device)
        )
        has_cflank = (
            (flank_c_tok != 0).any(dim=1)
            if flank_c_tok is not None
            else torch.zeros(batch_size, dtype=torch.bool, device=device)
        )
        has_mhc_a = (
            (mhc_a_tok != 0).any(dim=1)
            if mhc_a_tok is not None
            else torch.zeros(batch_size, dtype=torch.bool, device=device)
        )
        has_mhc_b = (
            (mhc_b_tok != 0).any(dim=1)
            if mhc_b_tok is not None
            else torch.zeros(batch_size, dtype=torch.bool, device=device)
        )
        completeness_id = (
            (has_nflank.long() << 0)
            | (has_cflank.long() << 1)
            | (has_mhc_a.long() << 2)
            | (has_mhc_b.long() << 3)
        )

        species_cond = self.species_cond_embed(species_id)

        completeness_cond = self.chain_completeness_embed(completeness_id)
        global_cond = species_cond + completeness_cond  # (batch_size, d_model)

        # Keep peptide token states independent of non-peptide modality presence
        # and immune-system side info. This preserves strict peptide-only flow for
        # species_of_origin -> foreignness -> recognition.
        token_cond = global_cond.unsqueeze(1).expand(-1, tok_ids.shape[1], -1).clone()
        pep_positions = (seg_ids == self.SEG_PEPTIDE).unsqueeze(0).expand(batch_size, -1)
        token_cond[pep_positions] = 0.0

        x = (
            self.aa_embedding(tok_ids)
            + self.segment_embedding(seg_ids).unsqueeze(0)
            + pos_embed
            + token_cond
        )

        # Segment-blocked base attention: tokens only self-attend within their segment.
        seg_block_mask = seg_ids.unsqueeze(0) != seg_ids.unsqueeze(1)
        key_padding_mask = ~valid_mask
        all_masked = key_padding_mask.all(dim=1)
        if all_masked.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_masked, 0] = False

        # Manual layer loop to capture early-layer states for aux heads.
        h = x
        early_states = None
        for i, layer in enumerate(self.stream_encoder.layers):
            h = layer(h, src_mask=seg_block_mask, src_key_padding_mask=key_padding_mask)
            if i == 0:
                early_states = h
        h = self.stream_norm(h)

        segment_masks = {
            "nflank": (seg_ids == self.SEG_NFLANK).unsqueeze(0).expand(batch_size, -1) & valid_mask,
            "peptide": (seg_ids == self.SEG_PEPTIDE).unsqueeze(0).expand(batch_size, -1) & valid_mask,
            "cflank": (seg_ids == self.SEG_CFLANK).unsqueeze(0).expand(batch_size, -1) & valid_mask,
            "mhc_a": (seg_ids == self.SEG_MHC_A).unsqueeze(0).expand(batch_size, -1) & valid_mask,
            "mhc_b": (seg_ids == self.SEG_MHC_B).unsqueeze(0).expand(batch_size, -1) & valid_mask,
        }

        return {
            "tokens": tokens,
            "states": h,
            "early_states": early_states,
            "valid_mask": valid_mask,
            "segment_ids": seg_ids,
            "segment_masks": segment_masks,
            "offsets": offsets,
        }

    def _latent_query(
        self,
        name: str,
        h: torch.Tensor,
        allowed_mask: torch.Tensor,
        latent_store: Dict[str, torch.Tensor],
        dep_names: List[str],
        extra_tokens: Optional[List[torch.Tensor]] = None,
        collect_attn: bool = False,
    ) -> tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """Compute one latent via N_LATENT_LAYERS of segmented cross-attention."""
        batch_size = h.shape[0]
        kv, kv_valid = self._prepare_latent_kv(
            h=h,
            allowed_mask=allowed_mask,
            latent_store=latent_store,
            dep_names=dep_names,
            extra_tokens=extra_tokens,
        )

        q = self.latent_queries[name].view(1, 1, -1).expand(batch_size, 1, -1)
        collected_attn: List[torch.Tensor] = []
        for layer in self.latent_layers[name]:
            attn_out, attn_weights = layer["attn"](
                layer["norm1"](q),
                kv,
                kv,
                key_padding_mask=~kv_valid,
                need_weights=collect_attn,
                average_attn_weights=False,
            )
            if collect_attn and isinstance(attn_weights, torch.Tensor):
                collected_attn.append(attn_weights)
            q = q + attn_out
            q = q + layer["ffn"](layer["norm2"](q))
        return q.squeeze(1), (collected_attn if collect_attn else None)

    @staticmethod
    def _ensure_nonempty_kv_mask(kv_valid: torch.Tensor) -> torch.Tensor:
        empty = ~kv_valid.any(dim=1)
        if empty.any():
            kv_valid = kv_valid.clone()
            kv_valid[empty, 0] = True
        return kv_valid

    @staticmethod
    def _gather_prefix_states(
        prefix: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        batch_idx = torch.arange(prefix.shape[0], device=prefix.device).unsqueeze(1)
        batch_idx = batch_idx.expand_as(indices)
        clipped = indices.clamp(min=0, max=prefix.shape[1] - 1)
        return prefix[batch_idx, clipped]

    @staticmethod
    def _gather_sequence_positions(
        seq: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        batch_idx = torch.arange(seq.shape[0], device=seq.device).view(-1, 1, 1)
        batch_idx = batch_idx.expand_as(positions)
        clipped = positions.clamp(min=0, max=seq.shape[1] - 1)
        return seq[batch_idx, clipped]

    def _compose_position_signal(
        self,
        *,
        mode: str,
        start_embed: torch.Tensor,
        end_embed: torch.Tensor,
        start_frac: torch.Tensor,
        end_frac: torch.Tensor,
        frac_mlp: nn.Module,
        abs_embed: Optional[torch.Tensor],
        concat_proj: nn.Module,
        concat_frac_proj: nn.Module,
        concat_mlp: nn.Module,
        concat_frac_mlp: nn.Module,
    ) -> torch.Tensor:
        if mode in {"triple", "triple_baseline"}:
            return start_embed + end_embed + frac_mlp(start_frac.unsqueeze(-1))
        if mode == "abs_only":
            if abs_embed is None:
                raise ValueError("abs_only requires abs_embed")
            return abs_embed
        if mode == "triple_plus_abs":
            if abs_embed is None:
                raise ValueError("triple_plus_abs requires abs_embed")
            return start_embed + end_embed + frac_mlp(start_frac.unsqueeze(-1)) + abs_embed
        if mode == "start_only":
            return start_embed
        if mode == "end_only":
            return end_embed
        if mode == "start_plus_end":
            return start_embed + end_embed
        if mode == "concat_start_end":
            return concat_proj(torch.cat([start_embed, end_embed], dim=-1))

        frac_features = torch.cat(
            [start_frac.unsqueeze(-1), end_frac.unsqueeze(-1)],
            dim=-1,
        )
        if mode == "concat_start_end_frac":
            return concat_frac_proj(torch.cat([start_embed, end_embed, frac_features], dim=-1))
        if mode == "mlp_start_end":
            return concat_mlp(torch.cat([start_embed, end_embed], dim=-1))
        if mode == "mlp_start_end_frac":
            return concat_frac_mlp(torch.cat([start_embed, end_embed, frac_features], dim=-1))
        raise ValueError(f"Unsupported positional composition mode: {mode!r}")

    @staticmethod
    def _repeat_candidates(x: torch.Tensor, n_candidates: int) -> torch.Tensor:
        return x.unsqueeze(1).expand(-1, n_candidates, *x.shape[1:]).reshape(
            x.shape[0] * n_candidates,
            *x.shape[1:],
        )

    def _prepare_latent_kv(
        self,
        h: torch.Tensor,
        allowed_mask: torch.Tensor,
        latent_store: Dict[str, torch.Tensor],
        dep_names: List[str],
        extra_tokens: Optional[List[torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = h.shape[0]
        kv = h
        kv_valid = allowed_mask

        dep_tokens: List[torch.Tensor] = []
        for dep in dep_names:
            dep_tensor = latent_store[dep]
            if dep_tensor.ndim == 2:
                dep_tokens.append(dep_tensor.unsqueeze(1))
            elif dep_tensor.ndim == 3:
                dep_tokens.append(dep_tensor)
            else:
                raise ValueError(
                    f"latent dependency {dep} has unsupported rank {dep_tensor.ndim}"
                )
        if extra_tokens:
            dep_tokens.extend(extra_tokens)

        if dep_tokens:
            dep_cat = torch.cat(dep_tokens, dim=1)
            dep_valid = torch.ones(
                (batch_size, dep_cat.shape[1]),
                dtype=torch.bool,
                device=h.device,
            )
            kv = torch.cat([kv, dep_cat], dim=1)
            kv_valid = torch.cat([kv_valid, dep_valid], dim=1)

        return kv, self._ensure_nonempty_kv_mask(kv_valid)

    def _run_binding_query(
        self,
        name: str,
        kv: torch.Tensor,
        kv_valid: torch.Tensor,
        collect_attn: bool = False,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[List[torch.Tensor]]]:
        batch_size = kv.shape[0]
        kv_valid = self._ensure_nonempty_kv_mask(kv_valid)

        query_param = self.latent_queries[name]
        if query_param.ndim == 1:
            q = query_param.view(1, 1, -1).expand(batch_size, 1, -1)
            n_q = 1
        else:
            q = query_param.unsqueeze(0).expand(batch_size, -1, -1)
            n_q = query_param.shape[0]

        if attn_mask is not None and attn_mask.shape[0] != n_q:
            attn_mask = attn_mask[:1, :].expand(n_q, -1)

        collected_attn: List[torch.Tensor] = []
        layers = self.latent_layers[name]
        has_self_attn = (
            self.binding_n_queries > 1
            and self.binding_use_decoder_layers
            and hasattr(self, "binding_query_self_attn")
        )

        for layer_idx, layer in enumerate(layers):
            if has_self_attn:
                sa_layer = self.binding_query_self_attn[name][layer_idx]
                sa_out, _ = sa_layer["self_attn"](
                    sa_layer["norm"](q),
                    sa_layer["norm"](q),
                    sa_layer["norm"](q),
                    need_weights=False,
                )
                q = q + sa_out

            key_padding_mask = ~kv_valid
            if attn_mask is not None:
                key_padding_mask = key_padding_mask.float().masked_fill(
                    key_padding_mask,
                    float("-inf"),
                )
            attn_out, attn_weights = layer["attn"](
                layer["norm1"](q),
                kv,
                kv,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
                need_weights=collect_attn,
                average_attn_weights=False,
            )
            if collect_attn and isinstance(attn_weights, torch.Tensor):
                collected_attn.append(attn_weights)
            q = q + attn_out
            q = q + layer["ffn"](layer["norm2"](q))

        interaction_tokens = q
        interaction_vec = self.pmhc_interaction_token_proj(interaction_tokens).reshape(
            batch_size,
            -1,
        )
        return interaction_vec, interaction_tokens, (
            collected_attn if collect_attn else None
        )

    def _run_pmhc_interaction(
        self,
        h: torch.Tensor,
        seg_masks: Dict[str, torch.Tensor],
        offsets: Dict[str, slice],
    ) -> torch.Tensor:
        """Run bidirectional pMHC interaction block (Variant C).

        Peptide tokens cross-attend to MHC tokens and vice versa.
        Returns a modified hidden state tensor for binding latents only.
        Non-peptide/MHC positions are copied from h unchanged.
        """
        pep_sl = offsets["peptide"]
        mhc_a_sl = offsets["mhc_a"]
        mhc_b_sl = offsets["mhc_b"]

        # Extract and detach from h's graph to build new representations
        pep_h = h[:, pep_sl, :].clone()
        mhc_a_h = h[:, mhc_a_sl, :].clone()
        mhc_b_h = h[:, mhc_b_sl, :].clone()
        mhc_h = torch.cat([mhc_a_h, mhc_b_h], dim=1)

        pep_mask = seg_masks["peptide"][:, pep_sl]
        mhc_a_mask = seg_masks["mhc_a"][:, mhc_a_sl]
        mhc_b_mask = seg_masks["mhc_b"][:, mhc_b_sl]
        mhc_mask = torch.cat([mhc_a_mask, mhc_b_mask], dim=1)

        # Ensure at least one valid token for each mask to avoid NaN
        pep_pad = ~pep_mask
        mhc_pad = ~mhc_mask
        pep_all_masked = pep_pad.all(dim=1)
        mhc_all_masked = mhc_pad.all(dim=1)
        if pep_all_masked.any():
            pep_pad = pep_pad.clone()
            pep_pad[pep_all_masked, 0] = False
        if mhc_all_masked.any():
            mhc_pad = mhc_pad.clone()
            mhc_pad[mhc_all_masked, 0] = False

        for layer in self.pmhc_interaction:
            pep_out, _ = layer["pep_to_mhc_attn"](
                layer["pep_norm1"](pep_h), mhc_h, mhc_h,
                key_padding_mask=mhc_pad,
                need_weights=False,
            )
            pep_h = pep_h + pep_out
            pep_h = pep_h + layer["pep_ffn"](layer["pep_norm2"](pep_h))

            # MHC cross-attends to peptide
            mhc_out, _ = layer["mhc_to_pep_attn"](
                layer["mhc_norm1"](mhc_h), pep_h, pep_h,
                key_padding_mask=pep_pad,
                need_weights=False,
            )
            mhc_h = mhc_h + mhc_out
            mhc_h = mhc_h + layer["mhc_ffn"](layer["mhc_norm2"](mhc_h))

        # Construct new tensor with enriched peptide/MHC, original elsewhere
        mhc_a_len = mhc_a_sl.stop - mhc_a_sl.start
        replacements = sorted(
            [
                (pep_sl, pep_h),
                (mhc_a_sl, mhc_h[:, :mhc_a_len, :]),
                (mhc_b_sl, mhc_h[:, mhc_a_len:, :]),
            ],
            key=lambda pair: pair[0].start,
        )
        parts = []
        prev_end = 0
        for sl, replacement in replacements:
            if sl.start > prev_end:
                parts.append(h[:, prev_end:sl.start, :])
            parts.append(replacement)
            prev_end = sl.stop
        if prev_end < h.shape[1]:
            parts.append(h[:, prev_end:, :])
        return torch.cat(parts, dim=1)

    def _compute_groove_vec(
        self,
        h: torch.Tensor,
        seg_masks: Dict[str, torch.Tensor],
        offsets: Dict[str, slice],
        class_probs: torch.Tensor,
    ) -> torch.Tensor:
        """Summarize the allele-specific peptide-binding groove.

        The runtime MHC input is already reduced to two structurally aligned
        groove halves:
        - `mhc_a`: alpha1 (class I/II)
        - `mhc_b`: alpha2 (class I) or beta1 (class II)

        So groove summarization is class-agnostic and no longer needs the old
        full-chain masking heuristics.
        """
        mhc_a_slice = offsets["mhc_a"]
        mhc_b_slice = offsets["mhc_b"]
        mhc_a_h = h[:, mhc_a_slice, :]
        mhc_b_h = h[:, mhc_b_slice, :]
        mhc_a_mask = seg_masks["mhc_a"][:, mhc_a_slice]
        mhc_b_mask = seg_masks["mhc_b"][:, mhc_b_slice]
        mhc_h = torch.cat([mhc_a_h, mhc_b_h], dim=1)
        groove_mask = torch.cat([mhc_a_mask, mhc_b_mask], dim=1)

        batch_size = h.shape[0]
        query = self.groove_query.unsqueeze(0).expand(batch_size, -1, -1)
        groove_mask = self._ensure_nonempty_kv_mask(groove_mask)
        groove_out, _ = self.groove_attn(
            query,
            mhc_h,
            mhc_h,
            key_padding_mask=~groove_mask,
            need_weights=False,
        )
        return groove_out.squeeze(1)

    def _binding_latent_query(
        self,
        name: str,
        h: torch.Tensor,
        seg_masks: Dict[str, torch.Tensor],
        offsets: Dict[str, slice],
        latent_store: Dict[str, torch.Tensor],
        dep_names: List[str],
        class_probs: torch.Tensor,
        extra_tokens: Optional[List[torch.Tensor]] = None,
        collect_attn: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Enumerate peptide core windows, score them, and marginalize binding."""
        batch_size = h.shape[0]
        pep_slice = offsets["peptide"]
        pep_h = h[:, pep_slice, :]
        pep_valid = seg_masks["peptide"][:, pep_slice]
        pep_len = pep_valid.sum(dim=1).clamp(min=1)

        configured_core_lengths = {int(length) for length in self.core_window_lengths}
        min_configured = min(configured_core_lengths)
        fallback_lengths = {
            max(1, int(observed_len))
            for observed_len in pep_len.detach().cpu().tolist()
            if int(observed_len) < min_configured
        }
        candidate_lengths = sorted(configured_core_lengths | fallback_lengths)

        pep_max_len = int(pep_len.max().item())
        starts_list: List[int] = []
        lens_list: List[int] = []
        for core_len_i in candidate_lengths:
            if pep_max_len < core_len_i:
                continue
            for start_i in range(pep_max_len - core_len_i + 1):
                starts_list.append(start_i)
                lens_list.append(core_len_i)
        if not starts_list:
            starts_list = [0]
            lens_list = [max(1, pep_max_len)]

        starts_1d = torch.tensor(starts_list, device=h.device, dtype=torch.long)
        core_lens_1d = torch.tensor(lens_list, device=h.device, dtype=torch.long)
        max_candidates = int(starts_1d.shape[0])

        starts = starts_1d.view(1, max_candidates).expand(batch_size, -1)
        core_lens = core_lens_1d.view(1, max_candidates).expand(batch_size, -1)
        ends = starts + core_lens
        pep_len_expanded = pep_len.unsqueeze(1).expand_as(starts)
        configured_length_mask = torch.zeros_like(core_lens, dtype=torch.bool)
        for configured_len in configured_core_lengths:
            configured_length_mask |= core_lens == int(configured_len)
        short_peptide_fallback = (
            (pep_len_expanded < int(min_configured))
            & (core_lens == pep_len_expanded.clamp(min=1))
        )
        length_allowed = configured_length_mask | short_peptide_fallback
        candidate_mask = length_allowed & (starts < pep_len_expanded) & (ends <= pep_len_expanded)

        core_offsets = torch.arange(self.core_window_size, device=h.device).view(1, 1, -1)
        core_positions = starts.unsqueeze(-1) + core_offsets
        core_token_mask = (
            candidate_mask.unsqueeze(-1)
            & (core_offsets < core_lens.unsqueeze(-1))
            & (core_positions < pep_len.view(batch_size, 1, 1))
        )
        core_tokens = self._gather_sequence_positions(pep_h, core_positions)
        core_pos_embed = self.core_position_embed(
            torch.arange(self.core_window_size, device=h.device)
        ).view(1, 1, self.core_window_size, -1)
        core_tokens = (
            core_tokens + core_pos_embed * core_token_mask.unsqueeze(-1).float()
        ) * core_token_mask.unsqueeze(-1).float()

        pep_prefix = torch.cat(
            [
                pep_h.new_zeros((batch_size, 1, self.d_model)),
                (pep_h * pep_valid.unsqueeze(-1).float()).cumsum(dim=1),
            ],
            dim=1,
        )
        pep_total = self._gather_prefix_states(
            pep_prefix,
            pep_len.unsqueeze(1).expand(-1, max_candidates),
        )
        npfr_sum = self._gather_prefix_states(pep_prefix, starts)
        cpfr_sum = pep_total - self._gather_prefix_states(pep_prefix, ends)
        npfr_len = starts
        cpfr_len = (pep_len_expanded - ends).clamp(min=0)
        npfr_repr = npfr_sum / npfr_len.float().unsqueeze(-1).clamp(min=1.0)
        cpfr_repr = cpfr_sum / cpfr_len.float().unsqueeze(-1).clamp(min=1.0)
        npfr_repr = npfr_repr * candidate_mask.unsqueeze(-1).float()
        cpfr_repr = cpfr_repr * candidate_mask.unsqueeze(-1).float()

        npfr_len_embed = self.pfr_length_embed(npfr_len.clamp(max=self.max_pfr_length))
        cpfr_len_embed = self.pfr_length_embed(cpfr_len.clamp(max=self.max_pfr_length))

        mhc_a_slice = offsets["mhc_a"]
        mhc_b_slice = offsets["mhc_b"]
        mhc_a_h = h[:, mhc_a_slice, :]
        mhc_b_h = h[:, mhc_b_slice, :]
        mhc_h = torch.cat([mhc_a_h, mhc_b_h], dim=1)
        mhc_mask = torch.cat(
            [
                seg_masks["mhc_a"][:, mhc_a_slice],
                seg_masks["mhc_b"][:, mhc_b_slice],
            ],
            dim=1,
        )

        core_flat = core_tokens.reshape(batch_size * max_candidates, self.core_window_size, self.d_model)
        core_mask_flat = core_token_mask.reshape(batch_size * max_candidates, self.core_window_size)
        mhc_flat = self._repeat_candidates(mhc_h, max_candidates)
        mhc_mask_flat = self._repeat_candidates(mhc_mask, max_candidates)

        kv_parts = [core_flat, mhc_flat]
        kv_valid_parts = [core_mask_flat, mhc_mask_flat]
        dep_token_list: List[torch.Tensor] = []
        for dep in dep_names:
            dep_tensor = latent_store[dep]
            if dep_tensor.ndim == 2:
                dep_tensor = dep_tensor.unsqueeze(1)
            elif dep_tensor.ndim != 3:
                raise ValueError(
                    f"latent dependency {dep} has unsupported rank {dep_tensor.ndim}"
                )
            dep_token_list.append(dep_tensor)
        if extra_tokens:
            dep_token_list.extend(extra_tokens)
        for token_tensor in dep_token_list:
            kv_parts.append(self._repeat_candidates(token_tensor, max_candidates))
            kv_valid_parts.append(
                torch.ones(
                    (batch_size * max_candidates, token_tensor.shape[1]),
                    dtype=torch.bool,
                    device=h.device,
                )
            )

        kv = torch.cat(kv_parts, dim=1)
        kv_valid = torch.cat(kv_valid_parts, dim=1)

        candidate_interaction_flat, candidate_tokens_flat, attn_layers = self._run_binding_query(
            name=name,
            kv=kv,
            kv_valid=kv_valid,
            collect_attn=collect_attn,
            attn_mask=None,
        )

        candidate_vec = self.core_window_vec_norm(
            self.core_window_fuse(
                torch.cat(
                    [
                        candidate_interaction_flat,
                        npfr_repr.reshape(batch_size * max_candidates, -1),
                        npfr_len_embed.reshape(batch_size * max_candidates, -1),
                        cpfr_repr.reshape(batch_size * max_candidates, -1),
                        cpfr_len_embed.reshape(batch_size * max_candidates, -1),
                    ],
                    dim=-1,
                )
            )
        ).reshape(batch_size, max_candidates, -1)

        pep_len_f = pep_len.float().unsqueeze(1).clamp(min=1.0)
        prior_features = torch.cat(
            [
                core_lens.float().unsqueeze(-1) / pep_len_f.unsqueeze(-1),
                npfr_len.float().unsqueeze(-1) / pep_len_f.unsqueeze(-1),
                cpfr_len.float().unsqueeze(-1) / pep_len_f.unsqueeze(-1),
                class_probs.unsqueeze(1).expand(-1, max_candidates, -1),
            ],
            dim=-1,
        )
        core_window_prior_logit = self.core_window_prior(prior_features).squeeze(-1)
        if self.core_refinement_mode == "class_specific":
            class1_score = self.core_window_score_class1(candidate_vec).squeeze(-1)
            class2_score = self.core_window_score_class2(candidate_vec).squeeze(-1)
            class1_weight = class_probs[:, :1].expand(-1, max_candidates)
            class2_weight = class_probs[:, 1:2].expand(-1, max_candidates)
            core_window_score_logit = class1_weight * class1_score + class2_weight * class2_score
        else:
            core_window_score_logit = self.core_window_score(candidate_vec).squeeze(-1)
        core_window_logit = core_window_score_logit + core_window_prior_logit
        core_window_prior_logit = core_window_prior_logit.masked_fill(~candidate_mask, -1e4)
        core_window_score_logit = core_window_score_logit.masked_fill(~candidate_mask, -1e4)
        core_window_logit = core_window_logit.masked_fill(~candidate_mask, -1e4)

        core_window_posterior = F.softmax(core_window_logit, dim=1)
        core_window_posterior = core_window_posterior * candidate_mask.to(
            dtype=core_window_posterior.dtype
        )
        core_window_posterior = core_window_posterior / core_window_posterior.sum(
            dim=1,
            keepdim=True,
        ).clamp(min=1e-8)
        core_window_posterior = core_window_posterior.to(dtype=candidate_vec.dtype)

        interaction_vec = torch.sum(
            core_window_posterior.unsqueeze(-1) * candidate_vec,
            dim=1,
        )
        candidate_tokens = candidate_tokens_flat.reshape(
            batch_size,
            max_candidates,
            candidate_tokens_flat.shape[1],
            candidate_tokens_flat.shape[2],
        )
        interaction_tokens = torch.sum(
            core_window_posterior.unsqueeze(-1).unsqueeze(-1) * candidate_tokens,
            dim=1,
        )

        core_start_logit = pep_h.new_full((batch_size, pep_h.shape[1]), -1e4)
        core_start_prob = pep_h.new_zeros((batch_size, pep_h.shape[1]))
        batch_idx = torch.arange(batch_size, device=h.device).unsqueeze(1).expand_as(starts)
        valid_batch_idx = batch_idx[candidate_mask]
        valid_starts = starts[candidate_mask]
        core_start_logit[valid_batch_idx, valid_starts] = core_window_logit[candidate_mask]
        core_start_prob[valid_batch_idx, valid_starts] = core_window_posterior[candidate_mask]
        core_start_logit = core_start_logit.masked_fill(~pep_valid, -1e4)

        core_membership = pep_h.new_zeros((batch_size, pep_h.shape[1]))
        for offset in range(self.core_window_size):
            contrib = core_window_posterior * core_token_mask[:, :, offset].to(
                dtype=core_window_posterior.dtype
            )
            pos = core_positions[:, :, offset].clamp(max=pep_h.shape[1] - 1)
            core_membership.scatter_add_(1, pos, contrib)
        core_membership = core_membership * pep_valid.to(dtype=core_membership.dtype)
        core_membership = core_membership / core_membership.sum(dim=1, keepdim=True).clamp(
            min=1e-8,
        )

        map_idx = core_window_logit.argmax(dim=1, keepdim=True)
        map_start = starts.gather(1, map_idx).squeeze(1)
        pos = torch.arange(pep_h.shape[1], device=h.device).view(1, -1).expand(batch_size, -1)
        core_relative_position_index = pos - map_start.unsqueeze(1)

        weighted_attn_layers: List[torch.Tensor] = []
        if collect_attn and attn_layers:
            mhc_start = self.core_window_size
            mhc_stop = mhc_start + mhc_h.shape[1]
            attn_weight = core_window_posterior.view(batch_size, max_candidates, 1, 1, 1)
            for layer_attn in attn_layers:
                layer_attn = layer_attn.reshape(
                    batch_size,
                    max_candidates,
                    layer_attn.shape[1],
                    layer_attn.shape[2],
                    layer_attn.shape[3],
                )
                weighted_attn_layers.append(
                    (attn_weight * layer_attn[..., mhc_start:mhc_stop]).sum(dim=1)
                )

        expected_start = torch.sum(
            core_window_posterior * starts.float(),
            dim=1,
            keepdim=True,
        )
        expected_core_len = torch.sum(
            core_window_posterior * core_lens.float(),
            dim=1,
            keepdim=True,
        )
        expected_cpfr = torch.sum(
            core_window_posterior * cpfr_len.float(),
            dim=1,
            keepdim=True,
        )
        map_len = core_lens.gather(1, map_idx).squeeze(1)

        diagnostics: Dict[str, torch.Tensor] = {
            "core_window_mask": candidate_mask,
            "core_window_start": starts,
            "core_window_length": core_lens,
            "core_window_prior_logit": core_window_prior_logit,
            "core_window_score_logit": core_window_score_logit,
            "core_window_logit": core_window_logit,
            "core_window_posterior_prob": core_window_posterior,
            "core_start_logit": core_start_logit,
            "core_start_prob": core_start_prob,
            "core_start_probs": core_start_prob,
            "core_membership_prob": core_membership,
            "core_relative_position_index": core_relative_position_index,
            "core_length": map_len.unsqueeze(1).expand_as(pos),
            "npfr_length": pos,
            "cpfr_length": (pep_len.view(-1, 1) - pos - map_len.view(-1, 1)).clamp(min=0),
            "core_length_norm": expected_core_len / pep_len.float().unsqueeze(-1),
            "npfr_length_norm": expected_start / pep_len.float().unsqueeze(-1),
            "cpfr_length_norm": expected_cpfr / pep_len.float().unsqueeze(-1),
            "attn_layers": weighted_attn_layers,
        }
        return interaction_vec, interaction_tokens, diagnostics

    @staticmethod
    def _binding_attention_stats(
        binding_attn: Mapping[str, Sequence[torch.Tensor]],
        mhc_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Summarize binding-latent attention concentration over MHC residues."""
        base_k = int(mhc_mask.shape[-1])
        attn_per_layer: List[torch.Tensor] = []
        for layers in binding_attn.values():
            for layer_attn in layers:
                if not isinstance(layer_attn, torch.Tensor):
                    continue
                # (B, heads, Q=1, K) -> (B, K) averaged across heads and query slot.
                if layer_attn.ndim != 4:
                    continue
                # Latent dependencies/context tokens can extend K beyond the
                # base stream tokens. Keep only base-stream attention slots so
                # MHC masking aligns with token positions.
                k = layer_attn.shape[-1]
                if k > base_k:
                    layer_attn = layer_attn[..., :base_k]
                attn_per_layer.append(layer_attn.mean(dim=(1, 2)))
        if not attn_per_layer:
            return {}

        mean_attn = torch.stack(attn_per_layer, dim=0).mean(dim=0)  # (B, K)
        mhc_mask_f = mhc_mask.float()
        mhc_token_count = mhc_mask_f.sum(dim=-1)
        mhc_weight = mean_attn * mhc_mask_f
        mhc_mass = mhc_weight.sum(dim=-1)
        valid = (mhc_token_count > 0.0) & (mhc_mass > 0.0)
        mhc_prob = mhc_weight / mhc_mass.unsqueeze(-1).clamp(min=1e-8)
        effective = 1.0 / mhc_prob.pow(2).sum(dim=-1).clamp(min=1e-8)
        effective = torch.where(valid, effective, torch.zeros_like(effective))

        return {
            "binding_mhc_attention_effective_residues": effective,
            "binding_mhc_attention_mass": mhc_mass,
            "binding_mhc_attention_valid_mask": valid.float(),
            "binding_mhc_attention_token_count": mhc_token_count,
        }

    def _trunk_state_from_outputs(self, outputs: Mapping[str, Any]) -> PrestoTrunkState:
        latent_vecs = outputs["latent_vecs"]
        return PrestoTrunkState(
            processing_vec=latent_vecs["processing"],
            interaction_vec=latent_vecs["pmhc_interaction"],
            pep_vec=outputs["pep_vec"],
            mhc_a_vec=outputs["mhc_a_vec"],
            mhc_b_vec=outputs["mhc_b_vec"],
            binding_affinity_vec=latent_vecs["binding_affinity"],
            binding_stability_vec=latent_vecs["binding_stability"],
            recognition_vec=latent_vecs["recognition"],
            processing_class1_vec=latent_vecs["processing_class1"],
            processing_class2_vec=latent_vecs["processing_class2"],
            presentation_class1_vec=latent_vecs["presentation_class1"],
            presentation_class2_vec=latent_vecs["presentation_class2"],
            immunogenicity_cd8_vec=latent_vecs["immunogenicity_cd8"],
            immunogenicity_cd4_vec=latent_vecs["immunogenicity_cd4"],
            class_probs=outputs["mhc_class_probs"],
        )

    def predict_processing_from_trunk(
        self,
        trunk_state: PrestoTrunkState,
    ) -> Dict[str, torch.Tensor]:
        class1 = self.class1_processing_predictor(trunk_state.processing_class1_vec)
        class2 = self.class2_processing_predictor(trunk_state.processing_class2_vec)
        mixed_logit = (
            trunk_state.class_probs[:, :1] * class1["logit"]
            + trunk_state.class_probs[:, 1:2] * class2["logit"]
        )
        return {
            "processing_class1_logit": class1["logit"],
            "processing_class2_logit": class2["logit"],
            "processing_logit": mixed_logit,
            "processing_mixed_logit": mixed_logit,
            "processing_class1_prob": class1["prob"],
            "processing_class2_prob": class2["prob"],
            "processing_prob": torch.sigmoid(mixed_logit),
            "processing_mixed_prob": torch.sigmoid(mixed_logit),
        }

    def predict_affinity_from_trunk(
        self,
        trunk_state: PrestoTrunkState,
        *,
        mhc_class: Optional[Any] = None,
        binding_context: Optional[Dict[str, torch.Tensor]] = None,
        species_probs: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        return self.affinity_predictor(
            interaction_vec=trunk_state.interaction_vec,
            binding_affinity_vec=trunk_state.binding_affinity_vec,
            binding_stability_vec=trunk_state.binding_stability_vec,
            pep_vec=trunk_state.pep_vec,
            mhc_a_vec=trunk_state.mhc_a_vec,
            mhc_b_vec=trunk_state.mhc_b_vec,
            class_probs=trunk_state.class_probs,
            species_probs=species_probs,
            mhc_class=mhc_class,
            binding_context=binding_context,
        )

    def predict_presentation_from_trunk(
        self,
        trunk_state: PrestoTrunkState,
    ) -> Dict[str, torch.Tensor]:
        class1 = self.class1_presentation_predictor(trunk_state.presentation_class1_vec)
        class2 = self.class2_presentation_predictor(trunk_state.presentation_class2_vec)
        mixed_logit = (
            trunk_state.class_probs[:, :1] * class1["logit"]
            + trunk_state.class_probs[:, 1:2] * class2["logit"]
        )
        return {
            "presentation_class1_logit": class1["logit"],
            "presentation_class2_logit": class2["logit"],
            "presentation_logit": mixed_logit,
            "presentation_mixed_logit": mixed_logit,
            "presentation_class1_prob": class1["prob"],
            "presentation_class2_prob": class2["prob"],
            "presentation_prob": torch.sigmoid(mixed_logit),
            "presentation_mixed_prob": torch.sigmoid(mixed_logit),
        }

    def forward_mhc_only(
        self,
        mhc_a_tok: torch.Tensor,
        mhc_b_tok: torch.Tensor,
    ) -> Dict[str, Any]:
        """Run only the shared stream + MHC auxiliary heads.

        Used for MHC-only warm-start pretraining. This intentionally avoids the
        pMHC latent DAG and downstream task heads.
        """
        if mhc_a_tok.ndim != 2 or mhc_b_tok.ndim != 2:
            raise ValueError(
                f"forward_mhc_only expects 2D token tensors; got "
                f"{tuple(mhc_a_tok.shape)} and {tuple(mhc_b_tok.shape)}"
            )
        device = mhc_a_tok.device
        batch_size = int(mhc_a_tok.shape[0])
        if int(mhc_b_tok.shape[0]) != batch_size:
            raise ValueError(
                f"mhc_a/mhc_b batch mismatch: {batch_size} vs {int(mhc_b_tok.shape[0])}"
            )

        dummy_pep = torch.full(
            (batch_size, 1),
            self.missing_token_idx,
            dtype=torch.long,
            device=device,
        )
        stream = self._build_single_stream(
            pep_tok=dummy_pep,
            mhc_a_tok=mhc_a_tok,
            mhc_b_tok=mhc_b_tok,
            flank_n_tok=None,
            flank_c_tok=None,
            species_id=torch.zeros(batch_size, dtype=torch.long, device=device),
        )
        early = stream["early_states"]
        seg_masks = stream["segment_masks"]

        mhc_a_vec = self._masked_mean(early, seg_masks["mhc_a"])
        mhc_b_vec = self._masked_mean(early, seg_masks["mhc_b"])
        mhc_a_type_logits = self.mhc_a_type_head(mhc_a_vec)
        mhc_b_type_logits = self.mhc_b_type_head(mhc_b_vec)
        mhc_a_species_logits = self.mhc_a_species_head(mhc_a_vec)
        mhc_b_species_logits = self.mhc_b_species_head(mhc_b_vec)

        return {
            "mhc_a_vec": mhc_a_vec,
            "mhc_b_vec": mhc_b_vec,
            "mhc_a_type_logits": mhc_a_type_logits,
            "mhc_b_type_logits": mhc_b_type_logits,
            "mhc_a_species_logits": mhc_a_species_logits,
            "mhc_b_species_logits": mhc_b_species_logits,
        }

    def forward(
        self,
        pep_tok: torch.Tensor,
        mhc_a_tok: torch.Tensor,
        mhc_b_tok: torch.Tensor,
        mhc_class: Optional[Any] = None,
        species: Optional[Any] = None,
        mhc_species: Optional[Any] = None,
        immune_species: Optional[Any] = None,
        species_of_origin: Optional[Any] = None,
        flank_n_tok: Optional[torch.Tensor] = None,
        flank_c_tok: Optional[torch.Tensor] = None,
        tcell_context: Optional[Dict[str, torch.Tensor]] = None,
        return_binding_attention: bool = False,
        peptide_species: Optional[Any] = None,  # deprecated alias for species_of_origin
        binding_context: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, Any]:
        """Forward pass through full model under the canonical outputs-only assay contract."""
        outputs: Dict[str, Any] = {}

        immune_species_input = immune_species if immune_species is not None else species
        mhc_species_input = mhc_species if mhc_species is not None else species
        species_of_origin_override = (
            species_of_origin if species_of_origin is not None else peptide_species
        )

        # ------------------------------------------------------------------
        # 1) Build single token stream and pooled representations
        # ------------------------------------------------------------------
        # Resolve species_id for global conditioning (per-sample)
        species_id = _processing_species_idx_tensor(
            species=immune_species_input,
            batch_size=pep_tok.shape[0],
            device=pep_tok.device,
        )

        stream = self._build_single_stream(
            pep_tok=pep_tok,
            mhc_a_tok=mhc_a_tok,
            mhc_b_tok=mhc_b_tok,
            flank_n_tok=flank_n_tok,
            flank_c_tok=flank_c_tok,
            species_id=species_id,
        )
        h = stream["states"]
        early = stream["early_states"]
        valid_mask = stream["valid_mask"]
        seg_masks = stream["segment_masks"]
        offsets = stream["offsets"]

        # Aux vectors pooled from early-layer states (type, species, chain compat heads only)
        pep_vec = self._masked_mean(early, seg_masks["peptide"])
        mhc_a_vec = self._masked_mean(early, seg_masks["mhc_a"])
        mhc_b_vec = self._masked_mean(early, seg_masks["mhc_b"])

        outputs["pep_vec"] = pep_vec
        outputs["mhc_a_vec"] = mhc_a_vec
        outputs["mhc_b_vec"] = mhc_b_vec

        # ------------------------------------------------------------------
        # 2) Per-chain MHC inference (design S5.1-S5.4)
        # ------------------------------------------------------------------
        batch_size = pep_tok.shape[0]

        # Per-chain type classification
        mhc_a_type_logits = self.mhc_a_type_head(mhc_a_vec)
        mhc_b_type_logits = self.mhc_b_type_head(mhc_b_vec)
        mhc_a_type_probs = F.softmax(mhc_a_type_logits, dim=-1)
        mhc_b_type_probs = F.softmax(mhc_b_type_logits, dim=-1)
        outputs["mhc_a_type_logits"] = mhc_a_type_logits
        outputs["mhc_b_type_logits"] = mhc_b_type_logits

        # Per-chain species classification
        mhc_a_species_logits = self.mhc_a_species_head(mhc_a_vec)
        mhc_b_species_logits = self.mhc_b_species_head(mhc_b_vec)
        mhc_a_species_probs = F.softmax(mhc_a_species_logits, dim=-1)
        mhc_b_species_probs = F.softmax(mhc_b_species_logits, dim=-1)
        outputs["mhc_a_species_logits"] = mhc_a_species_logits
        outputs["mhc_b_species_logits"] = mhc_b_species_logits

        # Compositional class probabilities from groove-half identities.
        # Fine type indices remain:
        # MHC_I=0, MHC_IIa=1, MHC_IIb=2, B2M=3, unknown=4
        # But the runtime inputs are now:
        # - class I:  (alpha1, alpha2)  -> (MHC_I, MHC_I)
        # - class II: (alpha1, beta1)   -> (MHC_IIa, MHC_IIb)
        class1_prob_raw = mhc_a_type_probs[:, 0] * mhc_b_type_probs[:, 0]
        class2_prob_raw = mhc_a_type_probs[:, 1] * mhc_b_type_probs[:, 2]
        class_sum = (class1_prob_raw + class2_prob_raw).clamp(min=1e-8)
        inferred_class_probs = torch.stack([
            class1_prob_raw / class_sum,
            class2_prob_raw / class_sum,
        ], dim=-1)
        # Emit 2-column logits for backward compat (from compositional probs)
        mhc_class_logits = torch.log(inferred_class_probs.clamp(min=1e-8))
        outputs["mhc_class_logits"] = mhc_class_logits
        outputs["mhc_class_probs_inferred"] = inferred_class_probs
        outputs["mhc_class_pred"] = torch.argmax(mhc_class_logits, dim=-1)

        # Emit combined species logits for backward compat (average of per-chain)
        inferred_species_probs = 0.5 * (mhc_a_species_probs + mhc_b_species_probs)
        mhc_species_logits = torch.log(inferred_species_probs.clamp(min=1e-8))
        outputs["mhc_species_logits"] = mhc_species_logits
        outputs["mhc_species_probs_inferred"] = inferred_species_probs
        outputs["mhc_species_pred"] = torch.argmax(mhc_species_logits, dim=-1)

        # Chain compatibility (design S5.3)
        chain_compat_logit = self.chain_compat_head(torch.cat([
            mhc_a_vec, mhc_b_vec,
            mhc_a_type_probs, mhc_b_type_probs,
            mhc_a_species_probs, mhc_b_species_probs,
        ], dim=-1))
        chain_compat_prob = torch.sigmoid(chain_compat_logit)
        outputs["chain_compat_logit"] = chain_compat_logit
        outputs["chain_compat_prob"] = chain_compat_prob

        # Override with user-provided class/species
        if isinstance(mhc_class, torch.Tensor):
            class_probs = _class_probs_from_input(
                batch_size=batch_size,
                device=pep_tok.device,
                class_probs=mhc_class,
            )
        elif mhc_class is not None:
            class_probs = _class_probs_from_input(
                batch_size=batch_size,
                device=pep_tok.device,
                mhc_class=mhc_class,
            )
        else:
            class_probs = inferred_class_probs

        species_override_active = mhc_species_input is not None
        if isinstance(mhc_species_input, torch.Tensor):
            species_probs = _species_probs_from_input(
                batch_size=batch_size,
                device=pep_tok.device,
                species_probs=mhc_species_input,
            )
        elif mhc_species_input is not None:
            species_probs = _species_probs_from_input(
                batch_size=batch_size,
                device=pep_tok.device,
                species=mhc_species_input,
            )
        else:
            species_probs = inferred_species_probs

        outputs["mhc_class_probs"] = class_probs
        outputs["mhc_species_probs"] = species_probs
        outputs["mhc_is_class1_prob"] = class_probs[:, :1]
        outputs["mhc_is_class2_prob"] = class_probs[:, 1:2]
        outputs["species_probs"] = species_probs

        # ------------------------------------------------------------------
        # 3) Latent query DAG with segmented attention
        # ------------------------------------------------------------------
        context_mhc_a_species = species_probs if species_override_active else mhc_a_species_probs
        context_mhc_b_species = species_probs if species_override_active else mhc_b_species_probs
        apc_cell_type_context = self.apc_cell_type_context_proj(
            torch.cat(
                [class_probs, context_mhc_a_species, context_mhc_b_species, chain_compat_prob],
                dim=-1,
            )
        ).unsqueeze(1)
        groove_vec = self._compute_groove_vec(
            h=h,
            seg_masks=seg_masks,
            offsets=offsets,
            class_probs=class_probs,
        )
        outputs["apc_cell_type_context_vec"] = apc_cell_type_context.squeeze(1)
        outputs["groove_vec"] = groove_vec

        # Which latents receive APC context token.
        _gets_apc_context = {"processing", "pmhc_interaction"}
        _gets_groove = {"pmhc_interaction"}

        # Variant C: pMHC interaction block (enriched representations for pMHC interaction only)
        if self.use_pmhc_interaction_block:
            h_binding = self._run_pmhc_interaction(h, seg_masks, offsets)
        else:
            h_binding = h

        latent_vals: Dict[str, torch.Tensor] = {}
        latent_store: Dict[str, torch.Tensor] = {}
        binding_attention: Dict[str, List[torch.Tensor]] = {}
        for name in self.CROSS_ATTN_LATENTS:
            seg_names = self.LATENT_SEGMENTS[name]
            allowed = torch.zeros_like(valid_mask)
            for seg_name in seg_names:
                allowed = allowed | seg_masks[seg_name]

            extra_tokens: List[torch.Tensor] = []
            if name in _gets_apc_context:
                extra_tokens.append(apc_cell_type_context)
            if name in _gets_groove:
                extra_tokens.append(groove_vec.unsqueeze(1))

            is_binding = name in self.BINDING_LATENT_NAMES
            collect = bool(
                return_binding_attention and is_binding
            )
            attn_layers = None

            if is_binding:
                latent_vec, latent_tokens, binding_diag = self._binding_latent_query(
                    name=name,
                    h=h_binding,
                    seg_masks=seg_masks,
                    offsets=offsets,
                    latent_store=latent_store,
                    dep_names=self.LATENT_DEPS[name],
                    class_probs=class_probs,
                    extra_tokens=extra_tokens if extra_tokens else None,
                    collect_attn=collect,
                )
                latent_store[name] = latent_tokens
                outputs.update(
                    {
                        key: value
                        for key, value in binding_diag.items()
                        if key != "attn_layers"
                    }
                )
                if binding_diag.get("attn_layers"):
                    binding_attention[name] = list(binding_diag["attn_layers"])
            else:
                latent_vec, attn_layers = self._latent_query(
                    name=name,
                    h=h,
                    allowed_mask=allowed,
                    latent_store=latent_store,
                    dep_names=self.LATENT_DEPS[name],
                    extra_tokens=extra_tokens if extra_tokens else None,
                    collect_attn=collect,
                )
                latent_store[name] = latent_vec
            latent_vals[name] = latent_vec
            if attn_layers:
                binding_attention[name] = list(attn_layers)

            # After species_of_origin: optional latent override, then foreignness.
            if name == "species_of_origin":
                if species_of_origin_override is not None:
                    species_idx_t = _species_idx_tensor(
                        species_of_origin_override,
                        batch_size,
                        latent_vec.device,
                    )
                    latent_vals["species_of_origin"] = self.species_override_embed(species_idx_t)
                    latent_store["species_of_origin"] = latent_vals["species_of_origin"]
                latent_vals["foreignness"] = self.foreignness_proj(
                    F.gelu(latent_vals["species_of_origin"])
                )
                latent_store["foreignness"] = latent_vals["foreignness"]

        processing_vec = latent_vals["processing"]
        interaction_vec = self.pmhc_interaction_vec_norm(latent_vals["pmhc_interaction"])
        latent_vals["pmhc_interaction"] = interaction_vec
        recognition_vec = latent_vals["recognition"]

        # Simplified binding path: interaction_vec → proj → binding_affinity_vec
        binding_affinity_vec = self.binding_affinity_readout_proj(interaction_vec)
        binding_stability_vec = self.binding_stability_readout_proj(interaction_vec)
        direct_segment_input = torch.cat([pep_vec, mhc_a_vec, mhc_b_vec], dim=-1)
        direct_affinity_vec = self.binding_direct_segment_affinity_proj(direct_segment_input)
        direct_stability_vec = self.binding_direct_segment_stability_proj(direct_segment_input)
        if self.binding_direct_segment_mode == "affinity_residual":
            binding_affinity_vec = binding_affinity_vec + direct_affinity_vec
        elif self.binding_direct_segment_mode == "affinity_stability_residual":
            binding_affinity_vec = binding_affinity_vec + direct_affinity_vec
            binding_stability_vec = binding_stability_vec + direct_stability_vec
        elif self.binding_direct_segment_mode == "gated_affinity":
            gate = torch.sigmoid(
                self.binding_direct_segment_gate(
                    torch.cat([binding_affinity_vec, direct_affinity_vec], dim=-1)
                )
            )
            binding_affinity_vec = (1.0 - gate) * binding_affinity_vec + gate * direct_affinity_vec
            outputs["binding_direct_segment_gate_mean"] = gate.mean()
        outputs["binding_direct_segment_mode"] = self.binding_direct_segment_mode
        outputs["binding_direct_affinity_vec"] = direct_affinity_vec
        outputs["binding_direct_stability_vec"] = direct_stability_vec

        # Class-specific processing projections
        processing_class1_vec = self.processing_class1_proj(processing_vec)
        processing_class2_vec = self.processing_class2_proj(processing_vec)

        # Class-specific presentation MLPs
        presentation_class1_vec = self.presentation_class1_vec_norm(
            self.presentation_class1_mlp(
                torch.cat([processing_class1_vec, interaction_vec], dim=-1)
            )
        )
        presentation_class2_vec = self.presentation_class2_vec_norm(
            self.presentation_class2_mlp(
                torch.cat([processing_class2_vec, interaction_vec], dim=-1)
            )
        )

        # Lineage-specific immunogenicity MLPs
        immunogenicity_cd8_vec = self.immunogenicity_cd8_vec_norm(
            self.immunogenicity_cd8_mlp(
                torch.cat([interaction_vec, recognition_vec], dim=-1)
            )
        )
        immunogenicity_cd4_vec = self.immunogenicity_cd4_vec_norm(
            self.immunogenicity_cd4_mlp(
                torch.cat([interaction_vec, recognition_vec], dim=-1)
            )
        )

        latent_vals["binding_affinity"] = binding_affinity_vec
        latent_vals["binding_stability"] = binding_stability_vec
        latent_vals["processing_class1"] = processing_class1_vec
        latent_vals["processing_class2"] = processing_class2_vec
        latent_vals["presentation_class1"] = presentation_class1_vec
        latent_vals["presentation_class2"] = presentation_class2_vec
        latent_vals["recognition_cd8"] = recognition_vec
        latent_vals["recognition_cd4"] = recognition_vec
        latent_vals["immunogenicity_cd8"] = immunogenicity_cd8_vec
        latent_vals["immunogenicity_cd4"] = immunogenicity_cd4_vec
        # Compat aliases for training code
        latent_vals["processing_mixed"] = processing_vec
        latent_vals["recognition_mixed"] = recognition_vec
        latent_vals["immunogenicity_mixed"] = immunogenicity_cd8_vec
        latent_vals["presentation_mixed"] = presentation_class1_vec

        outputs["latent_vecs"] = latent_vals
        if return_binding_attention and binding_attention:
            mhc_mask = torch.cat(
                [
                    seg_masks["mhc_a"][:, offsets["mhc_a"]],
                    seg_masks["mhc_b"][:, offsets["mhc_b"]],
                ],
                dim=1,
            )
            outputs.update(self._binding_attention_stats(binding_attention, mhc_mask))

        # Species of origin and foreignness readouts
        outputs["species_of_origin_logits"] = self.species_of_origin_head(
            latent_vals["species_of_origin"]
        )
        foreignness_logit = self.foreignness_head(latent_vals["foreignness"])
        outputs["foreignness_logit"] = foreignness_logit
        outputs["foreignness_prob"] = torch.sigmoid(foreignness_logit)

        # pmhc_vec = interaction_vec (backward compat alias)
        outputs["pmhc_vec"] = interaction_vec
        outputs["pmhc_interaction_vec"] = interaction_vec
        trunk_state = self._trunk_state_from_outputs(outputs)

        # ------------------------------------------------------------------
        # 6) Processing logits (from proc latents)
        # ------------------------------------------------------------------
        outputs.update(self.predict_processing_from_trunk(trunk_state))

        # ------------------------------------------------------------------
        # 7) Binding latents and calibrated binding logit
        # ------------------------------------------------------------------
        outputs.update(
            self.predict_affinity_from_trunk(
                trunk_state,
                mhc_class=mhc_class,
                binding_context=binding_context,
                species_probs=outputs.get("species_probs"),
            )
        )
        binding_class1_logit = outputs["binding_class1_logit"]
        binding_class2_logit = outputs["binding_class2_logit"]

        # ------------------------------------------------------------------
        # 8) Presentation logits from the shared presentation vector
        # ------------------------------------------------------------------
        outputs.update(self.predict_presentation_from_trunk(trunk_state))
        class1_pres_logit = outputs["presentation_class1_logit"]
        class2_pres_logit = outputs["presentation_class2_logit"]
        pres_logit = outputs["presentation_logit"]

        # ------------------------------------------------------------------
        # 9) Presentation-linked observables
        # ------------------------------------------------------------------
        ms_detectability_logit = self.ms_detectability_head(latent_vals["ms_detectability"])
        outputs["ms_detectability_logit"] = ms_detectability_logit

        # Elution logit = f(pres_logit, ms_detectability_logit) — no pmhc_vec (S9.3)
        elution_logit = self.elution_head(pres_logit, ms_detectability_logit)
        outputs["elution_logit"] = elution_logit
        outputs["elution_prob"] = torch.sigmoid(elution_logit)
        # ms_logit == elution_logit per S9.3 (same tensor, no separate head)
        outputs["ms_logit"] = elution_logit
        outputs["ms_prob"] = torch.sigmoid(elution_logit)

        # ------------------------------------------------------------------
        # 10) Recognition and immunogenicity readouts
        # ------------------------------------------------------------------
        recognition_cd8_logit = self.recognition_cd8_head(recognition_vec)
        recognition_cd4_logit = self.recognition_cd4_head(recognition_vec)
        outputs["recognition_cd8_logit"] = recognition_cd8_logit
        outputs["recognition_cd4_logit"] = recognition_cd4_logit
        outputs["recognition_cd8_prob"] = torch.sigmoid(recognition_cd8_logit)
        outputs["recognition_cd4_prob"] = torch.sigmoid(recognition_cd4_logit)

        # Repertoire logit: class-weighted mixture of cd8/cd4 recognition (S9.4)
        recognition_repertoire_logit = (
            class_probs[:, :1] * recognition_cd8_logit
            + class_probs[:, 1:2] * recognition_cd4_logit
        )
        outputs["recognition_repertoire_logit"] = recognition_repertoire_logit
        outputs["recognition_repertoire_prob"] = torch.sigmoid(recognition_repertoire_logit)
        outputs["recognition_mixed_logit"] = recognition_repertoire_logit
        outputs["recognition_mixed_prob"] = outputs["recognition_repertoire_prob"]

        # Immunogenicity readout from lineage-specific latent vecs (design S9.5)
        immunogenicity_cd8_vec = trunk_state.immunogenicity_cd8_vec
        immunogenicity_cd4_vec = trunk_state.immunogenicity_cd4_vec
        immunogenicity_cd8_logit = self.immunogenicity_cd8_latent_head(immunogenicity_cd8_vec)
        immunogenicity_cd4_logit = self.immunogenicity_cd4_latent_head(immunogenicity_cd4_vec)
        outputs["immunogenicity_cd8_logit"] = immunogenicity_cd8_logit
        outputs["immunogenicity_cd4_logit"] = immunogenicity_cd4_logit
        outputs["immunogenicity_cd8_prob"] = torch.sigmoid(immunogenicity_cd8_logit)
        outputs["immunogenicity_cd4_prob"] = torch.sigmoid(immunogenicity_cd4_logit)

        immunogenicity_mixture_logit = (
            class_probs[:, :1] * immunogenicity_cd8_logit
            + class_probs[:, 1:2] * immunogenicity_cd4_logit
        )
        outputs["immunogenicity_mixture_logit"] = immunogenicity_mixture_logit
        outputs["immunogenicity_mixed_logit"] = immunogenicity_mixture_logit
        outputs["immunogenicity_logit"] = immunogenicity_mixture_logit.squeeze(-1)
        outputs["immunogenicity_prob"] = torch.sigmoid(outputs["immunogenicity_logit"])
        outputs["immunogenicity_mixed_prob"] = torch.sigmoid(immunogenicity_mixture_logit)

        # ------------------------------------------------------------------
        # 11) Context-conditioned T-cell assay output (S10.3)
        # ------------------------------------------------------------------
        context = tcell_context or {}
        tcell_logit = self.tcell_assay_head(
            immunogenicity_cd8_vec=immunogenicity_cd8_vec,
            immunogenicity_cd4_vec=immunogenicity_cd4_vec,
            presentation_class1_logit=class1_pres_logit,
            presentation_class2_logit=class2_pres_logit,
            binding_class1_logit=binding_class1_logit,
            binding_class2_logit=binding_class2_logit,
            class_probs=class_probs,
            assay_method_idx=context.get("assay_method_idx"),
            assay_readout_idx=context.get("assay_readout_idx"),
            apc_type_idx=context.get("apc_type_idx"),
            culture_context_idx=context.get("culture_context_idx"),
            stim_context_idx=context.get("stim_context_idx"),
            peptide_format_idx=context.get("peptide_format_idx"),
            culture_duration_hours=context.get("culture_duration_hours"),
        )
        tcell_panel_logits = self.tcell_assay_head.predict_panel(
            immunogenicity_cd8_vec=immunogenicity_cd8_vec,
            immunogenicity_cd4_vec=immunogenicity_cd4_vec,
            presentation_class1_logit=class1_pres_logit,
            presentation_class2_logit=class2_pres_logit,
            binding_class1_logit=binding_class1_logit,
            binding_class2_logit=binding_class2_logit,
            class_probs=class_probs,
            assay_method_idx=context.get("assay_method_idx"),
            assay_readout_idx=context.get("assay_readout_idx"),
            apc_type_idx=context.get("apc_type_idx"),
            culture_context_idx=context.get("culture_context_idx"),
            stim_context_idx=context.get("stim_context_idx"),
            peptide_format_idx=context.get("peptide_format_idx"),
            culture_duration_hours=context.get("culture_duration_hours"),
        )
        outputs["tcell_logit"] = tcell_logit
        outputs["tcell_prob"] = torch.sigmoid(tcell_logit)
        outputs["tcell_context_logits"] = tcell_panel_logits
        outputs["tcell_panel_logits"] = tcell_panel_logits

        # ------------------------------------------------------------------
        # 12) pMHC-only receptor-evidence outputs
        # ------------------------------------------------------------------
        tcr_evidence_logit = self.tcr_evidence_head(interaction_vec)
        tcr_evidence_method_logits = self.tcr_evidence_method_head(interaction_vec)
        outputs["tcr_evidence_logit"] = tcr_evidence_logit
        outputs["tcr_evidence_prob"] = torch.sigmoid(tcr_evidence_logit)
        outputs["tcr_evidence_method_logits"] = tcr_evidence_method_logits
        outputs["tcr_evidence_method_probs"] = torch.sigmoid(tcr_evidence_method_logits)

        return outputs

    def forward_affinity_only(
        self,
        pep_tok: torch.Tensor,
        mhc_a_tok: torch.Tensor,
        mhc_b_tok: torch.Tensor,
        mhc_class: Optional[Any] = None,
        species: Optional[Any] = None,
        mhc_species: Optional[Any] = None,
        immune_species: Optional[Any] = None,
        species_of_origin: Optional[Any] = None,
        flank_n_tok: Optional[torch.Tensor] = None,
        flank_c_tok: Optional[torch.Tensor] = None,
        peptide_species: Optional[Any] = None,
        binding_context: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, Any]:
        """Affinity-only forward under the canonical sequence-only input contract."""
        del binding_context
        outputs = self.forward(
            pep_tok=pep_tok,
            mhc_a_tok=mhc_a_tok,
            mhc_b_tok=mhc_b_tok,
            mhc_class=mhc_class,
            species=species,
            mhc_species=mhc_species,
            immune_species=immune_species,
            species_of_origin=species_of_origin,
            flank_n_tok=flank_n_tok,
            flank_c_tok=flank_c_tok,
            peptide_species=peptide_species,
            binding_context=None,
        )
        affinity_keys = {
            "pep_vec",
            "mhc_a_vec",
            "mhc_b_vec",
            "groove_vec",
            "pmhc_vec",
            "pmhc_interaction_vec",
            "mhc_class_probs",
            "mhc_is_class1_prob",
            "mhc_is_class2_prob",
            "species_probs",
            "assays",
            "binding_latents",
            "binding_affinity_probe_kd_raw",
            "binding_affinity_probe_kd",
            "binding_affinity_score_raw",
            "binding_affinity_score",
            "binding_stability_score_raw",
            "binding_stability_score",
            "binding_assay_context_vec",
            "binding_logit_from_core",
            "binding_kd_bias_raw",
            "binding_kd_bias",
            "binding_kd_bias_cap",
            "binding_probe_mix_weight",
            "binding_core_kd_log10",
            "binding_mixed_kd_log10",
            "binding_base_logit",
            "binding_class1_logit",
            "binding_class2_logit",
            "binding_logit",
            "binding_mixed_logit",
            "binding_class1_prob",
            "binding_class2_prob",
            "binding_prob",
            "binding_mixed_prob",
            "latent_vecs",
            "core_window_mask",
            "core_window_start",
            "core_window_length",
            "core_window_prior_logit",
            "core_window_score_logit",
            "core_window_logit",
            "core_window_posterior_prob",
            "core_start_logit",
            "core_start_prob",
            "core_start_probs",
            "core_membership_prob",
            "core_relative_position_index",
            "core_length",
            "npfr_length",
            "cpfr_length",
            "core_length_norm",
            "npfr_length_norm",
            "cpfr_length_norm",
            "binding_mhc_attention_effective_residues",
            "binding_mhc_attention_mass",
            "binding_mhc_attention_valid_mask",
            "binding_mhc_attention_token_count",
        }
        return {key: value for key, value in outputs.items() if key in affinity_keys}

    def forward_presentation_only(
        self,
        pep_tok: torch.Tensor,
        mhc_a_tok: torch.Tensor,
        mhc_b_tok: torch.Tensor,
        mhc_class: Optional[Any] = None,
        species: Optional[Any] = None,
        mhc_species: Optional[Any] = None,
        immune_species: Optional[Any] = None,
        species_of_origin: Optional[Any] = None,
        flank_n_tok: Optional[torch.Tensor] = None,
        flank_c_tok: Optional[torch.Tensor] = None,
        peptide_species: Optional[Any] = None,
    ) -> Dict[str, Any]:
        outputs = self.forward(
            pep_tok=pep_tok,
            mhc_a_tok=mhc_a_tok,
            mhc_b_tok=mhc_b_tok,
            mhc_class=mhc_class,
            species=species,
            mhc_species=mhc_species,
            immune_species=immune_species,
            species_of_origin=species_of_origin,
            flank_n_tok=flank_n_tok,
            flank_c_tok=flank_c_tok,
            peptide_species=peptide_species,
        )
        presentation_keys = {
            "pep_vec",
            "mhc_a_vec",
            "mhc_b_vec",
            "groove_vec",
            "pmhc_vec",
            "pmhc_interaction_vec",
            "mhc_class_probs",
            "mhc_is_class1_prob",
            "mhc_is_class2_prob",
            "species_probs",
            "processing_logit",
            "processing_class1_logit",
            "processing_class2_logit",
            "processing_prob",
            "presentation_logit",
            "presentation_class1_logit",
            "presentation_class2_logit",
            "presentation_prob",
            "presentation_mixed_logit",
            "ms_detectability_logit",
            "elution_logit",
            "elution_prob",
            "ms_logit",
            "ms_prob",
            "latent_vecs",
        }
        return {key: value for key, value in outputs.items() if key in presentation_keys}

    def encode_pmhc(
        self,
        pep_tok: torch.Tensor,
        mhc_a_tok: torch.Tensor,
        mhc_b_tok: torch.Tensor,
        mhc_class: str = "I",
    ) -> torch.Tensor:
        """Encode pMHC into a fixed vector representation.

        Returns interaction_vec (pmhc_interaction_vec_dim) from the latent DAG.
        """
        outputs = self.forward(
            pep_tok=pep_tok,
            mhc_a_tok=mhc_a_tok,
            mhc_b_tok=mhc_b_tok,
            mhc_class=mhc_class,
        )
        return outputs["pmhc_vec"]
