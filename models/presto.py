"""Presto: Full unified model for pMHC presentation, recognition, and immunogenicity.

Canonical forward path:
- Single token stream encoder over peptide/flanks/MHCa/MHCb.
- Segmented latent-query attention DAG produces biologic latent vectors.
- Assay and task outputs are readouts of those shared latent vectors.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .pmhc import (
    BindingModule,
    PresentationBottleneck,
    PROCESSING_SPECIES_BUCKETS,
    _class_probs_from_input,
    _species_probs_from_input,
)
from .tcr import (
    TCREncoder,
    TCRpMHCMatcher,
    ChainClassifier,
    ChainAttributeClassifier,
    CellTypeClassifier,
)
from .heads import AssayHeads, TCellAssayHead, ElutionHead
from .affinity import (
    DEFAULT_MAX_AFFINITY_NM,
    DEFAULT_BINDING_MIDPOINT_NM,
    DEFAULT_BINDING_LOG10_SCALE,
    max_log10_nM,
    binding_logit_from_kd_log10,
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
    SEG_MHC_A = 3
    SEG_MHC_B = 4

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
        enable_tcr: bool = False,
        n_categories: Optional[int] = None,
        max_affinity_nM: float = DEFAULT_MAX_AFFINITY_NM,
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
    ):
        """Initialize Presto."""
        super().__init__()
        self.d_model = d_model
        self.enable_tcr = bool(enable_tcr)
        self.n_categories = None  # deprecated; kept for backward compat
        self.max_affinity_nM = float(max_affinity_nM)
        self.max_log10_nM = max_log10_nM(self.max_affinity_nM)
        self.binding_midpoint_nM = float(binding_midpoint_nM)
        self.binding_midpoint_log10_nM = math.log10(max(self.binding_midpoint_nM, 1e-12))
        self.binding_log10_scale = max(float(binding_log10_scale), 1e-6)
        self.missing_token_idx = int(AA_TO_IDX["<MISSING>"])
        self.x_token_idx = int(AA_TO_IDX["X"])
        aux_layers = max(1, n_layers // 2)

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
        self.core_window_size = 9
        self.max_pfr_length = 50
        self.pfr_length_dim = 32
        self.use_groove_prior = use_groove_prior
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
        self.pep_frac_mlp = nn.Sequential(
            nn.Linear(1, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        # Flanks: distance-from-cleavage
        self.nflank_dist_pos = nn.Embedding(25, d_model)
        self.cflank_dist_pos = nn.Embedding(25, d_model)
        # MHC: per-chain sequential
        self.mhc_a_pos = nn.Embedding(400, d_model)
        self.mhc_b_pos = nn.Embedding(400, d_model)
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

        # pmhc_vec from interaction, presentation, peptide, and direct MHC summaries.
        self.pmhc_vec_proj = nn.Linear(
            self.pmhc_interaction_vec_dim + 4 * d_model,
            d_model,
        )
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
        self.presentation_mlp = nn.Sequential(
            nn.Linear(2 * d_model + self.pmhc_interaction_vec_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.presentation_vec_norm = nn.LayerNorm(d_model)
        self.immunogenicity_mlp = nn.Sequential(
            nn.Linear(self.pmhc_interaction_vec_dim + d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.immunogenicity_vec_norm = nn.LayerNorm(d_model)
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

        # Variant D: groove attention prior
        if use_groove_prior:
            # Sigmoid decay: ~1.0 before cutoff, smooth falloff, ~0.0 past it.
            # Accounts for signal peptide (~24aa) shifting groove start.
            def _groove_decay(length: int, cutoff: int, width: float) -> torch.Tensor:
                pos = torch.arange(length, dtype=torch.float)
                return torch.sigmoid((cutoff - pos) / width)

            # MHC-A (alpha chain): SP~24 + α1+α2~182 = ~206, cutoff=210, width=15
            # Positions 0-180: ~1.0, 180-240: smooth decay, 240+: ~0.0
            self.groove_bias_a = nn.Parameter(_groove_decay(400, cutoff=210, width=15.0))
            # MHC-B (beta chain): SP~26 + β1~94 = ~120, cutoff=120, width=10
            # Positions 0-100: ~1.0, 100-140: smooth decay, 140+: ~0.0
            self.groove_bias_b = nn.Parameter(_groove_decay(400, cutoff=120, width=10.0))

        # Immunogenicity MLPs (design S7.4 Level 3: MLP, not cross-attention)
        self.immunogenicity_cd8_mlp = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.immunogenicity_cd4_mlp = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Context vector MLP (design S5.4): class_probs + a_species + b_species + compat
        self.context_token_proj = nn.Sequential(
            nn.Linear(2 + n_species * 2 + 1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # ------------------------------------------------------------------
        # Readout heads from latent vectors
        # ------------------------------------------------------------------
        self.processing_class1_head = nn.Linear(d_model, 1)
        self.processing_class2_head = nn.Linear(d_model, 1)

        self.binding_affinity_probe = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )
        self.binding = BindingModule(d_model=self.pmhc_interaction_vec_dim)

        self.presentation_class1_latent_head = nn.Linear(d_model, 1)
        self.presentation_class2_latent_head = nn.Linear(d_model, 1)
        # Keep latent residual positive and non-trivial from step 1.
        # softplus(-1.5) ~= 0.20
        self.w_presentation_class1_latent = nn.Parameter(torch.tensor(-1.5))
        self.w_presentation_class2_latent = nn.Parameter(torch.tensor(-1.5))

        self.recognition_cd8_head = nn.Linear(d_model, 1)
        self.recognition_cd4_head = nn.Linear(d_model, 1)
        self.immunogenicity_cd8_latent_head = nn.Linear(d_model, 1)
        self.immunogenicity_cd4_latent_head = nn.Linear(d_model, 1)

        # Presentation bottleneck (kept as canonical additive-logit path).
        self.presentation = PresentationBottleneck()
        self.w_class1_presentation_stability = nn.Parameter(torch.tensor(0.3))
        self.w_class2_presentation_stability = nn.Parameter(torch.tensor(0.3))
        self.w_class1_presentation_class = nn.Parameter(torch.tensor(0.2))
        self.w_class2_presentation_class = nn.Parameter(torch.tensor(0.2))
        self.w_binding_class1_calibration = nn.Parameter(torch.tensor(0.2))
        self.w_binding_class2_calibration = nn.Parameter(torch.tensor(0.2))

        # MS detectability readout from latent (design S7.4)
        self.ms_detectability_head = nn.Linear(d_model, 1)

        # Elution head (S9.3: pres_logit + ms_detect_logit, no pmhc_vec).
        self.elution_head = ElutionHead()

        # TCR encoding/matching.
        self.tcr_encoder = TCREncoder(d_model=d_model, n_layers=n_layers, n_heads=n_heads)
        self.matcher = TCRpMHCMatcher(d_model=d_model)

        # Chain classifiers.
        self.chain_classifier = ChainClassifier(d_model=d_model, n_layers=aux_layers, n_heads=n_heads)
        self.chain_attribute_classifier = ChainAttributeClassifier(d_model=d_model, n_layers=aux_layers, n_heads=n_heads)
        self.cell_classifier = CellTypeClassifier(d_model=d_model)

        # Assay outputs.
        self.assay_heads = AssayHeads(
            d_model=d_model,
            max_log10_nM=self.max_log10_nM,
        )
        self.kd_assay_bias = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )
        self.kd_assay_bias_scale = nn.Parameter(torch.tensor(0.5))
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
        # Drop old CategoryHead keys from pre-foreignness checkpoints
        # Drop old DAG-violating head keys (elution context_head, MSDetectionHead,
        # RepertoireHead, old TCellAssayHead keys)
        drop_prefixes = [
            f"{prefix}category_head.",
            f"{prefix}elution_head.context_head.",
            f"{prefix}ms_detection.",
            f"{prefix}repertoire.",
            f"{prefix}tcell_assay_head.base_with_tcr.",
            f"{prefix}tcell_assay_head.base_without_tcr.",
            f"{prefix}tcell_assay_head.context_bias.",
            f"{prefix}tcell_assay_head.lineage_projection.",
            f"{prefix}tcell_assay_head.assay_method_classifier.",
            f"{prefix}tcell_assay_head.assay_readout_classifier.",
            f"{prefix}tcell_assay_head.apc_type_classifier.",
            f"{prefix}tcell_assay_head.culture_context_classifier.",
            f"{prefix}tcell_assay_head.stim_context_classifier.",
        ]
        drop_exact = [
            f"{prefix}elution_head.w_context",
            f"{prefix}elution_head.w_processing",
            f"{prefix}tcell_assay_head.w_bio",
            f"{prefix}tcell_assay_head.w_base",
            f"{prefix}tcell_assay_head.w_ctx",
            f"{prefix}tcell_assay_head.w_lineage",
            f"{prefix}tcell_assay_head.bias",
        ]
        for key in list(state_dict.keys()):
            if any(key.startswith(dp) for dp in drop_prefixes):
                state_dict.pop(key)
            elif key in drop_exact:
                state_dict.pop(key)
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
        has_tcr: bool = False,
        has_tcr_paired: bool = False,
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

        # Build segment-specific positional embeddings (design S3.2.3)
        pos_embed = torch.zeros(batch_size, seq_len, self.d_model, device=device)

        # Peptide: triple-frame (N-term distance, C-term distance, fractional)
        pep_sl = offsets["peptide"]
        pep_len_per = (tokens[:, pep_sl] != 0).sum(dim=1).clamp(min=1)  # (B,)
        pep_idx = torch.arange(pep_sl.stop - pep_sl.start, device=device).unsqueeze(0).expand(batch_size, -1)
        nterm_idx = pep_idx.clamp(max=self.pep_nterm_pos.num_embeddings - 1)
        cterm_dist = (pep_len_per.unsqueeze(1) - 1 - pep_idx).clamp(min=0)
        cterm_idx = cterm_dist.clamp(max=self.pep_cterm_pos.num_embeddings - 1)
        frac_pos = pep_idx.float() / (pep_len_per.unsqueeze(1) - 1).clamp(min=1).float()
        pos_embed[:, pep_sl, :] = (
            self.pep_nterm_pos(nterm_idx)
            + self.pep_cterm_pos(cterm_idx)
            + self.pep_frac_mlp(frac_pos.unsqueeze(-1))
        )

        # N-flank: distance-from-cleavage (reversed: last position = closest to cleavage)
        nfl_sl = offsets["nflank"]
        nfl_len_per = (tokens[:, nfl_sl] != 0).sum(dim=1).clamp(min=1)
        nfl_idx = torch.arange(nfl_sl.stop - nfl_sl.start, device=device).unsqueeze(0).expand(batch_size, -1)
        nfl_dist = (nfl_len_per.unsqueeze(1) - 1 - nfl_idx).clamp(min=0)
        pos_embed[:, nfl_sl, :] = self.nflank_dist_pos(
            nfl_dist.clamp(max=self.nflank_dist_pos.num_embeddings - 1)
        )

        # C-flank: distance-from-cleavage (first position = closest to cleavage)
        cfl_sl = offsets["cflank"]
        cfl_idx = torch.arange(cfl_sl.stop - cfl_sl.start, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_embed[:, cfl_sl, :] = self.cflank_dist_pos(
            cfl_idx.clamp(max=self.cflank_dist_pos.num_embeddings - 1)
        )

        # MHC alpha: sequential
        mhc_a_sl = offsets["mhc_a"]
        mhc_a_idx = torch.arange(mhc_a_sl.stop - mhc_a_sl.start, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_embed[:, mhc_a_sl, :] = self.mhc_a_pos(
            mhc_a_idx.clamp(max=self.mhc_a_pos.num_embeddings - 1)
        )

        # MHC beta: sequential
        mhc_b_sl = offsets["mhc_b"]
        mhc_b_idx = torch.arange(mhc_b_sl.stop - mhc_b_sl.start, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_embed[:, mhc_b_sl, :] = self.mhc_b_pos(
            mhc_b_idx.clamp(max=self.mhc_b_pos.num_embeddings - 1)
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
        has_tcr_vec = torch.full((batch_size,), bool(has_tcr), dtype=torch.bool, device=device)
        has_tcr_paired_vec = torch.full(
            (batch_size,),
            bool(has_tcr_paired),
            dtype=torch.bool,
            device=device,
        )
        completeness_id = (
            (has_nflank.long() << 0)
            | (has_cflank.long() << 1)
            | (has_mhc_a.long() << 2)
            | (has_mhc_b.long() << 3)
            | (has_tcr_vec.long() << 4)
            | (has_tcr_paired_vec.long() << 5)
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

        h = self.stream_encoder(
            x,
            mask=seg_block_mask,
            src_key_padding_mask=key_padding_mask,
        )
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

        # Fixed structural penalty for pep→MHC attention: positions past 250
        # in either sub-chain are alpha3/TM/cytoplasmic and never contact peptide.
        _HARD_CUTOFF = 250
        mhc_a_len = mhc_a_sl.stop - mhc_a_sl.start
        mhc_b_len = mhc_b_sl.stop - mhc_b_sl.start
        mhc_total = mhc_a_len + mhc_b_len
        pmhc_hard_bias: Optional[torch.Tensor] = None
        if mhc_a_len > _HARD_CUTOFF or mhc_b_len > _HARD_CUTOFF:
            pmhc_hard_bias = torch.zeros(mhc_total, device=h.device)
            if mhc_a_len > _HARD_CUTOFF:
                pmhc_hard_bias[_HARD_CUTOFF:mhc_a_len] = -10.0
            if mhc_b_len > _HARD_CUTOFF:
                pmhc_hard_bias[mhc_a_len + _HARD_CUTOFF:mhc_total] = -10.0
            # Shape for MHA attn_mask: (query_len, kv_len)
            # Broadcast across query positions (peptide length)
            pmhc_hard_bias = pmhc_hard_bias.unsqueeze(0)

        for layer in self.pmhc_interaction:
            # Peptide cross-attends to MHC (with structural position penalty)
            # Convert key_padding_mask to float when using attn_mask
            mhc_kpm = mhc_pad
            if pmhc_hard_bias is not None:
                mhc_kpm = mhc_pad.float().masked_fill(mhc_pad, float("-inf"))
            pep_out, _ = layer["pep_to_mhc_attn"](
                layer["pep_norm1"](pep_h), mhc_h, mhc_h,
                key_padding_mask=mhc_kpm,
                attn_mask=pmhc_hard_bias,
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

        Class I uses the alpha1/alpha2 region of chain A only.
        Class II uses the alpha1 region of chain A and beta1 region of chain B.
        When class is uncertain, mix the two summaries by `class_probs`.
        """
        mhc_a_slice = offsets["mhc_a"]
        mhc_b_slice = offsets["mhc_b"]
        mhc_a_h = h[:, mhc_a_slice, :]
        mhc_b_h = h[:, mhc_b_slice, :]
        mhc_a_mask = seg_masks["mhc_a"][:, mhc_a_slice]
        mhc_b_mask = seg_masks["mhc_b"][:, mhc_b_slice]
        mhc_h = torch.cat([mhc_a_h, mhc_b_h], dim=1)

        batch_size = h.shape[0]
        a_pos = torch.arange(mhc_a_h.shape[1], device=h.device).view(1, -1)
        b_pos = torch.arange(mhc_b_h.shape[1], device=h.device).view(1, -1)

        class1_mask = torch.cat(
            [
                mhc_a_mask & (a_pos < 180),
                torch.zeros_like(mhc_b_mask),
            ],
            dim=1,
        )
        class2_mask = torch.cat(
            [
                mhc_a_mask & (a_pos < 90),
                mhc_b_mask & (b_pos < 90),
            ],
            dim=1,
        )

        query = self.groove_query.unsqueeze(0).expand(batch_size, -1, -1)

        def _attend(valid_mask: torch.Tensor) -> torch.Tensor:
            valid_mask = self._ensure_nonempty_kv_mask(valid_mask)
            groove_out, _ = self.groove_attn(
                query,
                mhc_h,
                mhc_h,
                key_padding_mask=~valid_mask,
                need_weights=False,
            )
            return groove_out.squeeze(1)

        groove_class1 = _attend(class1_mask)
        groove_class2 = _attend(class2_mask)
        return (
            class_probs[:, :1] * groove_class1
            + class_probs[:, 1:2] * groove_class2
        )

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

        core_len = torch.minimum(
            pep_len,
            pep_len.new_full((batch_size,), self.core_window_size),
        )
        num_candidates = (pep_len - core_len + 1).clamp(min=1)
        max_candidates = int(num_candidates.max().item())

        starts = torch.arange(max_candidates, device=h.device).view(1, -1).expand(
            batch_size,
            -1,
        )
        candidate_mask = starts < num_candidates.unsqueeze(1)
        ends = starts + core_len.unsqueeze(1)

        core_offsets = torch.arange(self.core_window_size, device=h.device).view(1, 1, -1)
        core_positions = starts.unsqueeze(-1) + core_offsets
        core_token_mask = (
            candidate_mask.unsqueeze(-1)
            & (core_offsets < core_len.view(batch_size, 1, 1))
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
        cpfr_len = (pep_len.unsqueeze(1) - ends).clamp(min=0)
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

        attn_mask = None
        if self.use_groove_prior:
            kv_len = kv.shape[1]
            attn_bias = torch.zeros(kv_len, device=h.device)
            mhc_offset = self.core_window_size
            mhc_a_len = mhc_a_h.shape[1]
            mhc_b_len = mhc_b_h.shape[1]
            attn_bias[mhc_offset : mhc_offset + mhc_a_len] = self.groove_bias_a[:mhc_a_len]
            attn_bias[
                mhc_offset + mhc_a_len : mhc_offset + mhc_a_len + mhc_b_len
            ] = self.groove_bias_b[:mhc_b_len]
            hard_cutoff = 250
            if mhc_a_len > hard_cutoff:
                attn_bias[mhc_offset + hard_cutoff : mhc_offset + mhc_a_len] += -10.0
            if mhc_b_len > hard_cutoff:
                start = mhc_offset + mhc_a_len + hard_cutoff
                stop = mhc_offset + mhc_a_len + mhc_b_len
                attn_bias[start:stop] += -10.0
            n_q = 1 if self.latent_queries[name].ndim == 1 else self.latent_queries[name].shape[0]
            attn_mask = attn_bias.unsqueeze(0).expand(n_q, -1)

        candidate_interaction_flat, candidate_tokens_flat, attn_layers = self._run_binding_query(
            name=name,
            kv=kv,
            kv_valid=kv_valid,
            collect_attn=collect_attn,
            attn_mask=attn_mask,
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
        core_len_expanded = core_len.unsqueeze(1).expand(-1, max_candidates)
        prior_features = torch.cat(
            [
                core_len_expanded.float().unsqueeze(-1) / pep_len_f.unsqueeze(-1),
                npfr_len.float().unsqueeze(-1) / pep_len_f.unsqueeze(-1),
                cpfr_len.float().unsqueeze(-1) / pep_len_f.unsqueeze(-1),
                class_probs.unsqueeze(1).expand(-1, max_candidates, -1),
            ],
            dim=-1,
        )
        core_window_prior_logit = self.core_window_prior(prior_features).squeeze(-1)
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
            core_window_posterior * core_len.unsqueeze(1).float(),
            dim=1,
            keepdim=True,
        )
        expected_cpfr = torch.sum(
            core_window_posterior * cpfr_len.float(),
            dim=1,
            keepdim=True,
        )

        diagnostics: Dict[str, torch.Tensor] = {
            "core_window_mask": candidate_mask,
            "core_window_start": starts,
            "core_window_length": core_len.unsqueeze(1).expand_as(starts),
            "core_window_prior_logit": core_window_prior_logit,
            "core_window_score_logit": core_window_score_logit,
            "core_window_logit": core_window_logit,
            "core_window_posterior_prob": core_window_posterior,
            "core_start_logit": core_start_logit,
            "core_start_prob": core_start_prob,
            "core_start_probs": core_start_prob,
            "core_membership_prob": core_membership,
            "core_relative_position_index": core_relative_position_index,
            "core_length": core_len.unsqueeze(1).expand_as(pos),
            "npfr_length": pos,
            "cpfr_length": (pep_len.view(-1, 1) - pos - core_len.view(-1, 1)).clamp(min=0),
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
        tcr_a_tok: Optional[torch.Tensor] = None,
        tcr_b_tok: Optional[torch.Tensor] = None,
        flank_n_tok: Optional[torch.Tensor] = None,
        flank_c_tok: Optional[torch.Tensor] = None,
        tcell_context: Optional[Dict[str, torch.Tensor]] = None,
        return_binding_attention: bool = False,
        peptide_species: Optional[Any] = None,  # deprecated alias for species_of_origin
    ) -> Dict[str, Any]:
        """Forward pass through full model."""
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

        has_tcr_input = tcr_a_tok is not None or tcr_b_tok is not None
        has_tcr = self.enable_tcr and has_tcr_input
        has_tcr_paired = self.enable_tcr and tcr_a_tok is not None and tcr_b_tok is not None

        stream = self._build_single_stream(
            pep_tok=pep_tok,
            mhc_a_tok=mhc_a_tok,
            mhc_b_tok=mhc_b_tok,
            flank_n_tok=flank_n_tok,
            flank_c_tok=flank_c_tok,
            species_id=species_id,
            has_tcr=has_tcr,
            has_tcr_paired=has_tcr_paired,
        )
        h = stream["states"]
        valid_mask = stream["valid_mask"]
        seg_masks = stream["segment_masks"]
        offsets = stream["offsets"]

        pep_vec = self._masked_mean(h, seg_masks["peptide"])
        mhc_a_vec = self._masked_mean(h, seg_masks["mhc_a"])
        mhc_b_vec = self._masked_mean(h, seg_masks["mhc_b"])

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

        # Compositional class probabilities (design S5.1)
        # Fine type indices: MHC_I=0, MHC_IIa=1, MHC_IIb=2, B2M=3, unknown=4
        # Class I = MHC_I × B2M
        # Class II = MHC_IIa × MHC_IIb
        class1_prob_raw = mhc_a_type_probs[:, 0] * mhc_b_type_probs[:, 3]
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
        # 3) Optional TCR encoding (future feature; disabled by default)
        # ------------------------------------------------------------------
        tcr_vec = None
        if self.enable_tcr and has_tcr_input:
            tcr_vec = self.tcr_encoder(tcr_a_tok, tcr_b_tok)
            outputs["tcr_vec"] = tcr_vec
            outputs["cell_type_logits"] = self.cell_classifier(tcr_vec)

            chain_tok_for_attr = None
            if tcr_b_tok is not None and tcr_a_tok is not None:
                has_beta = (tcr_b_tok != 0).any(dim=1, keepdim=True)
                chain_tok_for_attr = torch.where(has_beta, tcr_b_tok, tcr_a_tok)
            elif tcr_b_tok is not None:
                chain_tok_for_attr = tcr_b_tok
            else:
                chain_tok_for_attr = tcr_a_tok

            chain_attr_logits = self.chain_attribute_classifier(chain_tok_for_attr)
            outputs["chain_species_logits"] = chain_attr_logits["species_logits"]
            outputs["chain_type_logits"] = chain_attr_logits["chain_logits"]
            outputs["chain_phenotype_logits"] = chain_attr_logits["phenotype_logits"]

        # ------------------------------------------------------------------
        # 4) Latent query DAG with segmented attention
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
        # Which latents receive tcr_vec (design S7.5)
        _gets_tcr: set[str] = set()

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
            if name in _gets_tcr and tcr_vec is not None:
                extra_tokens.append(tcr_vec.unsqueeze(1))

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

        binding_affinity_vec = self.binding_affinity_readout_proj(interaction_vec)
        binding_stability_vec = self.binding_stability_readout_proj(interaction_vec)
        presentation_vec = self.presentation_vec_norm(
            self.presentation_mlp(
                torch.cat([processing_vec, interaction_vec, groove_vec], dim=-1)
            )
        )
        immunogenicity_vec = self.immunogenicity_vec_norm(
            self.immunogenicity_mlp(torch.cat([interaction_vec, recognition_vec], dim=-1))
        )

        latent_vals["presentation"] = presentation_vec
        latent_vals["immunogenicity"] = immunogenicity_vec

        # Compatibility aliases for downstream training/inference code while the
        # canonical latent DAG moves to unified concept vectors.
        latent_vals["binding_affinity"] = binding_affinity_vec
        latent_vals["binding_stability"] = binding_stability_vec
        latent_vals["processing_class1"] = processing_vec
        latent_vals["processing_class2"] = processing_vec
        latent_vals["presentation_class1"] = presentation_vec
        latent_vals["presentation_class2"] = presentation_vec
        latent_vals["recognition_cd8"] = recognition_vec
        latent_vals["recognition_cd4"] = recognition_vec
        latent_vals["immunogenicity_cd8"] = immunogenicity_vec
        latent_vals["immunogenicity_cd4"] = immunogenicity_vec
        latent_vals["processing_mixed"] = processing_vec
        latent_vals["presentation_mixed"] = presentation_vec
        latent_vals["recognition_mixed"] = recognition_vec
        latent_vals["immunogenicity_mixed"] = immunogenicity_vec

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

        # pmhc_vec from latent vectors (design S9.7)
        pmhc_vec = self.pmhc_vec_proj(torch.cat([
            interaction_vec,
            presentation_vec,
            pep_vec,
            mhc_a_vec,
            mhc_b_vec,
        ], dim=-1))
        outputs["pmhc_vec"] = pmhc_vec
        outputs["pmhc_interaction_vec"] = interaction_vec

        # ------------------------------------------------------------------
        # 6) Processing logits (from proc latents)
        # ------------------------------------------------------------------
        proc_class1_logit = self.processing_class1_head(processing_vec)
        proc_class2_logit = self.processing_class2_head(processing_vec)
        proc_logit = class_probs[:, :1] * proc_class1_logit + class_probs[:, 1:2] * proc_class2_logit

        outputs["processing_class1_logit"] = proc_class1_logit
        outputs["processing_class2_logit"] = proc_class2_logit
        outputs["processing_logit"] = proc_logit
        outputs["processing_mixed_logit"] = proc_logit
        outputs["processing_class1_prob"] = torch.sigmoid(proc_class1_logit)
        outputs["processing_class2_prob"] = torch.sigmoid(proc_class2_logit)
        outputs["processing_prob"] = torch.sigmoid(proc_logit)
        outputs["processing_mixed_prob"] = outputs["processing_prob"]

        # ------------------------------------------------------------------
        # 7) Binding latents and calibrated binding logit
        # ------------------------------------------------------------------
        # Auxiliary probe: shortcut gradient path from binding_affinity latent → KD
        probe_kd = self.binding_affinity_probe(binding_affinity_vec)
        outputs["binding_affinity_probe_kd"] = torch.clamp(probe_kd, min=-3.0, max=8.0)

        binding_latents = self.binding(
            interaction_vec,
            mhc_class=mhc_class,
            class_probs=class_probs,
        )
        outputs["binding_latents"] = binding_latents
        stability_score = -binding_latents["log_koff"]

        log_kd_per_sample = self.binding.derive_kd(binding_latents).squeeze(-1)
        binding_logit_from_core = binding_logit_from_kd_log10(
            log_kd_per_sample,
            midpoint_nM=self.binding_midpoint_nM,
            log10_scale=self.binding_log10_scale,
        )
        binding_logit_from_core = torch.clamp(binding_logit_from_core, min=-20.0, max=20.0)
        outputs["binding_logit_from_core"] = binding_logit_from_core

        # Canonical assay-calibrated KD path.
        kd_from_binding = (
            self.binding_midpoint_log10_nM
            - self.binding_log10_scale * binding_logit_from_core
        ).unsqueeze(-1)
        kd_bias = torch.tanh(self.kd_assay_bias(binding_affinity_vec)) * F.softplus(self.kd_assay_bias_scale)

        assays = self.assay_heads(binding_affinity_vec, binding_stability_vec, binding_latents=binding_latents)
        # Keep only a lower clamp here. Upper capping is handled by
        # `derive_affinity_observables` via a smooth bound; hard max-clamping
        # here would collapse all very weak binders to one constant value.
        kd_log10 = torch.clamp(
            kd_from_binding + kd_bias,
            min=-3.0,
        )
        affinity_obs = self.assay_heads.derive_affinity_observables(binding_affinity_vec, kd_log10)
        assays["KD_nM"] = affinity_obs["KD_nM"]
        assays["IC50_nM"] = affinity_obs["IC50_nM"]
        assays["EC50_nM"] = affinity_obs["EC50_nM"]
        outputs["assays"] = assays

        binding_base_logit = binding_logit_from_kd_log10(
            assays["KD_nM"].squeeze(-1),
            midpoint_nM=self.binding_midpoint_nM,
            log10_scale=self.binding_log10_scale,
        )
        binding_base_logit = torch.clamp(binding_base_logit, min=-20.0, max=20.0)
        class_margin = class_probs[:, :1] - class_probs[:, 1:2]
        binding_class1_logit = (
            binding_base_logit.unsqueeze(-1)
            + F.softplus(self.w_binding_class1_calibration) * class_margin
        )
        binding_class2_logit = (
            binding_base_logit.unsqueeze(-1)
            - F.softplus(self.w_binding_class2_calibration) * class_margin
        )
        binding_logit = (
            class_probs[:, :1] * binding_class1_logit
            + class_probs[:, 1:2] * binding_class2_logit
        ).squeeze(-1)
        outputs["binding_base_logit"] = binding_base_logit
        outputs["binding_class1_logit"] = binding_class1_logit
        outputs["binding_class2_logit"] = binding_class2_logit
        outputs["binding_logit"] = binding_logit
        outputs["binding_mixed_logit"] = binding_logit.unsqueeze(-1)
        outputs["binding_class1_prob"] = torch.sigmoid(binding_class1_logit)
        outputs["binding_class2_prob"] = torch.sigmoid(binding_class2_logit)
        outputs["binding_prob"] = torch.sigmoid(binding_logit)
        outputs["binding_mixed_prob"] = outputs["binding_prob"].unsqueeze(-1)

        # ------------------------------------------------------------------
        # 8) Presentation logits from the shared presentation vector
        # ------------------------------------------------------------------
        class1_pres_logit = self.presentation_class1_latent_head(presentation_vec)
        class2_pres_logit = self.presentation_class2_latent_head(presentation_vec)
        pres_logit = class_probs[:, :1] * class1_pres_logit + class_probs[:, 1:2] * class2_pres_logit

        outputs["presentation_class1_logit"] = class1_pres_logit
        outputs["presentation_class2_logit"] = class2_pres_logit
        outputs["presentation_logit"] = pres_logit
        outputs["presentation_mixed_logit"] = pres_logit
        outputs["presentation_class1_prob"] = torch.sigmoid(class1_pres_logit)
        outputs["presentation_class2_prob"] = torch.sigmoid(class2_pres_logit)
        outputs["presentation_prob"] = torch.sigmoid(pres_logit)
        outputs["presentation_mixed_prob"] = outputs["presentation_prob"]

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

        # Optional TCR-specific matching
        recognition_evidence = recognition_repertoire_logit
        if tcr_vec is not None:
            match_logit = self.matcher(tcr_vec, pmhc_vec)
            outputs["match_logit"] = match_logit
            outputs["match_prob"] = torch.sigmoid(match_logit)
            recognition_evidence = match_logit

        if recognition_evidence.ndim == 1:
            recognition_evidence = recognition_evidence.unsqueeze(-1)

        # Immunogenicity readout from MLP-computed latent vecs (design S9.5)
        immunogenicity_cd8_logit = self.immunogenicity_cd8_latent_head(immunogenicity_vec)
        immunogenicity_cd4_logit = self.immunogenicity_cd4_latent_head(immunogenicity_vec)
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
            immunogenicity_cd8_vec=immunogenicity_vec,
            immunogenicity_cd4_vec=immunogenicity_vec,
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
            immunogenicity_cd8_vec=immunogenicity_vec,
            immunogenicity_cd4_vec=immunogenicity_vec,
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

        return outputs

    def encode_pmhc(
        self,
        pep_tok: torch.Tensor,
        mhc_a_tok: torch.Tensor,
        mhc_b_tok: torch.Tensor,
        mhc_class: str = "I",
    ) -> torch.Tensor:
        """Encode pMHC into a fixed vector representation.

        Runs the full forward pass and returns pmhc_vec derived from latent
        vectors (design S9.7).
        """
        outputs = self.forward(
            pep_tok=pep_tok,
            mhc_a_tok=mhc_a_tok,
            mhc_b_tok=mhc_b_tok,
            mhc_class=mhc_class,
        )
        return outputs["pmhc_vec"]

    def encode_tcr(
        self,
        tcr_a_tok: torch.Tensor,
        tcr_b_tok: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode TCR into a fixed vector representation."""
        return self.tcr_encoder(tcr_a_tok, tcr_b_tok)

    def classify_chain(self, chain_tok: torch.Tensor) -> torch.Tensor:
        """Classify chain type from sequence."""
        return self.chain_classifier(chain_tok)

    def classify_chain_attributes(self, chain_tok: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Classify chain attributes (species, chain type, phenotype)."""
        return self.chain_attribute_classifier(chain_tok)

    def predict_chain_attributes(self, chain_tok: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Predict chain attributes with probabilities."""
        return self.chain_attribute_classifier.predict(chain_tok)
