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
    MSDetectionHead,
    PROCESSING_SPECIES_BUCKETS,
    _class_probs_from_input,
    _species_probs_from_input,
)
from .tcr import (
    TCREncoder,
    TCRpMHCMatcher,
    RepertoireHead,
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
    ORGANISM_TO_IDX,
    TCELL_APC_TYPES,
    TCELL_ASSAY_METHODS,
    TCELL_ASSAY_READOUTS,
    TCELL_CULTURE_CONTEXTS,
    TCELL_STIM_CONTEXTS,
)


def _species_idx_tensor(
    peptide_species: Any,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Convert peptide_species (str or list of str) to index tensor."""
    if isinstance(peptide_species, str):
        idx = ORGANISM_TO_IDX.get(peptide_species, 0)
        return torch.full((batch_size,), idx, dtype=torch.long, device=device)
    if isinstance(peptide_species, (list, tuple)):
        indices = [ORGANISM_TO_IDX.get(s, 0) for s in peptide_species]
        return torch.tensor(indices, dtype=torch.long, device=device)
    if isinstance(peptide_species, torch.Tensor):
        return peptide_species.to(device=device, dtype=torch.long)
    return torch.zeros(batch_size, dtype=torch.long, device=device)


class Presto(nn.Module):
    """Presto unified model with single token stream + latent DAG."""

    SEG_NFLANK = 0
    SEG_PEPTIDE = 1
    SEG_CFLANK = 2
    SEG_MHC_A = 3
    SEG_MHC_B = 4

    LATENT_ORDER = [
        "processing_class1",
        "processing_class2",
        "ms_detectability",
        "species_of_origin",
        "binding_affinity",
        "binding_stability",
        "presentation_class1",
        "presentation_class2",
        "recognition_cd8",
        "recognition_cd4",
        "immunogenicity_cd8",
        "immunogenicity_cd4",
    ]

    # Per design S7.5: segment access table
    LATENT_SEGMENTS = {
        "processing_class1": ["nflank", "peptide", "cflank"],
        "processing_class2": ["nflank", "peptide", "cflank"],
        "ms_detectability": ["peptide"],
        "species_of_origin": ["peptide"],  # peptide-only cross-attention
        "binding_affinity": ["peptide", "mhc_a", "mhc_b"],
        "binding_stability": ["peptide", "mhc_a", "mhc_b"],
        "presentation_class1": [],       # pure bottleneck: no token access
        "presentation_class2": [],       # pure bottleneck: no token access
        "recognition_cd8": ["peptide"],  # peptide + foreignness as dep
        "recognition_cd4": ["peptide"],  # peptide + foreignness as dep
        "immunogenicity_cd8": [],        # MLP only, no cross-attention
        "immunogenicity_cd4": [],        # MLP only, no cross-attention
    }

    # Per design S7.2: DAG dependencies
    LATENT_DEPS = {
        "processing_class1": [],
        "processing_class2": [],
        "ms_detectability": [],
        "species_of_origin": [],
        "binding_affinity": [],
        "binding_stability": [],
        "presentation_class1": ["processing_class1", "binding_affinity", "binding_stability"],
        "presentation_class2": ["processing_class2", "binding_affinity", "binding_stability"],
        "recognition_cd8": ["foreignness"],   # foreignness as extra KV token
        "recognition_cd4": ["foreignness"],   # foreignness as extra KV token
        "immunogenicity_cd8": ["binding_affinity", "binding_stability", "recognition_cd8"],
        "immunogenicity_cd4": ["binding_affinity", "binding_stability", "recognition_cd4"],
    }

    # Latents computed via cross-attention (excludes MLP-only immunogenicity)
    CROSS_ATTN_LATENTS = [n for n in LATENT_ORDER if not n.startswith("immunogenicity_")]
    N_LATENT_LAYERS = 2

    # Binding latent names that use enhanced query path when available
    BINDING_LATENT_NAMES = {"binding_affinity", "binding_stability"}

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
        binding_n_queries: int = 1,                  # B: multi-token queries
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

        # pmhc_vec from latent vectors (design S9.7)
        self.pmhc_vec_proj = nn.Linear(d_model * 3, d_model)

        # ------------------------------------------------------------------
        # Per-chain MHC inference heads (design S5.1-S5.3)
        # ------------------------------------------------------------------
        n_species = N_ORGANISM_CATEGORIES
        # Per-chain fine type (6 classes):
        # {MHC_Ia, MHC_Ib, MHC_IIa, MHC_IIb, B2M, unknown}
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

        # Variant D: groove attention prior
        if use_groove_prior:
            # MHC-A (alpha chain): favor α1+α2 domains (pos 0-180) for Class I,
            # α1 domain (pos 0-90) for Class II — learnable, so model adapts.
            groove_init_a = torch.zeros(400)
            groove_init_a[:180] = 1.0  # α1+α2 groove residues
            self.groove_bias_a = nn.Parameter(groove_init_a)
            # MHC-B (beta chain): favor β1 domain (pos 0-90) for Class II groove.
            # For Class I, beta = β2m (~99 residues, does not contact peptide
            # directly); model can learn to zero this out for Class I.
            groove_init_b = torch.zeros(400)
            groove_init_b[:90] = 1.0  # β1 groove residues
            self.groove_bias_b = nn.Parameter(groove_init_b)

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

        self.binding_fuse = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.binding_affinity_probe = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )
        self.binding = BindingModule(d_model=d_model)

        self.presentation_class1_latent_head = nn.Linear(d_model, 1)
        self.presentation_class2_latent_head = nn.Linear(d_model, 1)
        self.w_presentation_class1_latent = nn.Parameter(torch.tensor(0.0))
        self.w_presentation_class2_latent = nn.Parameter(torch.tensor(0.0))

        self.recognition_cd8_head = nn.Linear(d_model, 1)
        self.recognition_cd4_head = nn.Linear(d_model, 1)
        self.immunogenicity_cd8_latent_head = nn.Linear(d_model, 1)
        self.immunogenicity_cd4_latent_head = nn.Linear(d_model, 1)

        self.core_start_head = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.core_width_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

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

        # MS/elution heads.
        self.ms_detection = MSDetectionHead(d_model=d_model)
        self.elution_head = ElutionHead(d_model=d_model)

        # TCR encoding/matching.
        self.tcr_encoder = TCREncoder(d_model=d_model, n_layers=n_layers, n_heads=n_heads)
        self.matcher = TCRpMHCMatcher(d_model=d_model)
        self.repertoire = RepertoireHead(d_model=d_model)

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
        )

        # Species of origin and foreignness heads (replaces old CategoryHead)
        self.foreignness_proj = nn.Linear(d_model, d_model)
        self.species_of_origin_head = nn.Linear(d_model, N_ORGANISM_CATEGORIES)
        self.foreignness_head = nn.Linear(d_model, 1)
        self.species_override_embed = nn.Embedding(N_ORGANISM_CATEGORIES, d_model)

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
        drop_prefixes = [f"{prefix}category_head."]
        for key in list(state_dict.keys()):
            if any(key.startswith(dp) for dp in drop_prefixes):
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

        # Chain completeness bitfield
        has_nflank = flank_n_tok is not None and (flank_n_tok != 0).any(dim=1).any().item()
        has_cflank = flank_c_tok is not None and (flank_c_tok != 0).any(dim=1).any().item()
        has_mhc_a = (mhc_a_tok != 0).any(dim=1).any().item() if mhc_a_tok is not None else False
        has_mhc_b = (mhc_b_tok != 0).any(dim=1).any().item() if mhc_b_tok is not None else False
        completeness_bits = (
            int(has_nflank) << 0
            | int(has_cflank) << 1
            | int(has_mhc_a) << 2
            | int(has_mhc_b) << 3
            | int(has_tcr) << 4
            | int(has_tcr_paired) << 5
        )
        completeness_id = torch.full((batch_size,), completeness_bits, dtype=torch.long, device=device)

        species_cond = self.species_cond_embed(species_id)

        global_cond = (
            species_cond
            + self.chain_completeness_embed(completeness_id)
        )  # (batch_size, d_model)

        x = (
            self.aa_embedding(tok_ids)
            + self.segment_embedding(seg_ids).unsqueeze(0)
            + pos_embed
            + global_cond.unsqueeze(1)  # broadcast to all tokens
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

        kv = h
        kv_valid = allowed_mask

        dep_tokens: List[torch.Tensor] = []
        for dep in dep_names:
            dep_tokens.append(latent_store[dep].unsqueeze(1))
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

        empty = ~kv_valid.any(dim=1)
        if empty.any():
            kv_valid = kv_valid.clone()
            kv_valid[empty, 0] = True

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
            # Peptide cross-attends to MHC
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
        parts = []
        prev_end = 0
        # Ordered segment slices: nflank, peptide, cflank, mhc_a, mhc_b
        replacements = {
            pep_sl: pep_h,
            mhc_a_sl: mhc_h[:, :mhc_a_len, :],
            mhc_b_sl: mhc_h[:, mhc_a_len:, :],
        }
        for sl in sorted(replacements.keys(), key=lambda s: s.start):
            if sl.start > prev_end:
                parts.append(h[:, prev_end:sl.start, :])
            parts.append(replacements[sl])
            prev_end = sl.stop
        if prev_end < h.shape[1]:
            parts.append(h[:, prev_end:, :])
        return torch.cat(parts, dim=1)

    def _binding_latent_query(
        self,
        name: str,
        h: torch.Tensor,
        allowed_mask: torch.Tensor,
        latent_store: Dict[str, torch.Tensor],
        dep_names: List[str],
        mhc_a_mask: torch.Tensor,
        offsets: Dict[str, slice],
        extra_tokens: Optional[List[torch.Tensor]] = None,
        collect_attn: bool = False,
    ) -> tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """Compute binding latent with enhanced architecture (Variants A/B/D).

        Supports:
        - Deeper cross-attention (Variant A via binding_n_latent_layers)
        - Multi-token queries with optional self-attention (Variant B)
        - Groove attention prior as additive bias (Variant D)
        """
        batch_size = h.shape[0]

        kv = h
        kv_valid = allowed_mask

        dep_tokens: List[torch.Tensor] = []
        for dep in dep_names:
            dep_tokens.append(latent_store[dep].unsqueeze(1))
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

        empty = ~kv_valid.any(dim=1)
        if empty.any():
            kv_valid = kv_valid.clone()
            kv_valid[empty, 0] = True

        # Build groove attention bias mask (Variant D)
        attn_mask = None
        if self.use_groove_prior:
            # Additive attention bias for MHC-A (α chain) and MHC-B (β chain)
            kv_len = kv.shape[1]
            attn_bias = torch.zeros(kv_len, device=h.device)
            # Alpha chain groove bias
            mhc_a_sl = offsets["mhc_a"]
            mhc_a_len = mhc_a_sl.stop - mhc_a_sl.start
            attn_bias[mhc_a_sl] = self.groove_bias_a[:mhc_a_len]
            # Beta chain groove bias (β1 domain for Class II, β2m for Class I)
            mhc_b_sl = offsets["mhc_b"]
            mhc_b_len = mhc_b_sl.stop - mhc_b_sl.start
            attn_bias[mhc_b_sl] = self.groove_bias_b[:mhc_b_len]
            # Expand for multi-head attention: (n_queries, kv_len)
            n_q = self.binding_n_queries
            attn_mask = attn_bias.unsqueeze(0).expand(n_q, -1)

        # Initialize query tokens
        query_param = self.latent_queries[name]
        if query_param.ndim == 1:
            # Single query token: (d_model,) -> (B, 1, d_model)
            q = query_param.view(1, 1, -1).expand(batch_size, 1, -1)
        else:
            # Multi-token queries: (K, d_model) -> (B, K, d_model)
            q = query_param.unsqueeze(0).expand(batch_size, -1, -1)

        collected_attn: List[torch.Tensor] = []
        layers = self.latent_layers[name]
        has_self_attn = (
            self.binding_n_queries > 1
            and self.binding_use_decoder_layers
            and hasattr(self, "binding_query_self_attn")
        )

        for layer_idx, layer in enumerate(layers):
            # Optional self-attention among query tokens (Variant B)
            if has_self_attn:
                sa_layer = self.binding_query_self_attn[name][layer_idx]
                sa_out, _ = sa_layer["self_attn"](
                    sa_layer["norm"](q), sa_layer["norm"](q), sa_layer["norm"](q),
                    need_weights=False,
                )
                q = q + sa_out

            # Cross-attention to KV
            # When groove prior is active, convert key_padding_mask to float
            # to match attn_mask type and avoid PyTorch deprecation warning.
            kpm = ~kv_valid
            if attn_mask is not None:
                kpm = kpm.float().masked_fill(kpm, float("-inf"))
            attn_out, attn_weights = layer["attn"](
                layer["norm1"](q),
                kv,
                kv,
                key_padding_mask=kpm,
                attn_mask=attn_mask,
                need_weights=collect_attn,
                average_attn_weights=False,
            )
            if collect_attn and isinstance(attn_weights, torch.Tensor):
                collected_attn.append(attn_weights)
            q = q + attn_out
            q = q + layer["ffn"](layer["norm2"](q))

        # Pool multi-token queries down to single vector
        if self.binding_n_queries > 1 and q.shape[1] > 1:
            if self.binding_query_pool == "attention":
                # Attention-weighted pooling
                pool_weights = F.softmax(
                    self.binding_pool_attn(q).squeeze(-1), dim=1
                )  # (B, K)
                q = (q * pool_weights.unsqueeze(-1)).sum(dim=1, keepdim=True)
            else:
                # Mean pooling
                q = q.mean(dim=1, keepdim=True)

        return q.squeeze(1), (collected_attn if collect_attn else None)

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
                attn_per_layer.append(layer_attn.mean(dim=1).squeeze(1))
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
        tcr_a_tok: Optional[torch.Tensor] = None,
        tcr_b_tok: Optional[torch.Tensor] = None,
        flank_n_tok: Optional[torch.Tensor] = None,
        flank_c_tok: Optional[torch.Tensor] = None,
        tcell_context: Optional[Dict[str, torch.Tensor]] = None,
        return_binding_attention: bool = False,
        peptide_species: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Forward pass through full model."""
        outputs: Dict[str, Any] = {}

        # ------------------------------------------------------------------
        # 1) Build single token stream and pooled representations
        # ------------------------------------------------------------------
        # Resolve species_id for global conditioning (7 species_cond_embed slots)
        _species_map = {s: i for i, s in enumerate(PROCESSING_SPECIES_BUCKETS)}
        if isinstance(species, str):
            _sp = normalize_processing_species_label(species, default=None)
            species_id = torch.full((pep_tok.shape[0],), _species_map.get(_sp, 0), dtype=torch.long, device=pep_tok.device)
        else:
            species_id = None  # defaults to 0 in _build_single_stream

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
        # Fine type indices: MHC_Ia=0, MHC_Ib=1, MHC_IIa=2, MHC_IIb=3, B2M=4, unknown=5
        # Class I = (MHC_Ia + MHC_Ib) × B2M
        # Class II = MHC_IIa × MHC_IIb
        class1_prob_raw = (mhc_a_type_probs[:, 0] + mhc_a_type_probs[:, 1]) * mhc_b_type_probs[:, 4]
        class2_prob_raw = mhc_a_type_probs[:, 2] * mhc_b_type_probs[:, 3]
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

        if isinstance(species, torch.Tensor):
            species_probs = _species_probs_from_input(
                batch_size=batch_size,
                device=pep_tok.device,
                species_probs=species,
            )
        elif species is not None:
            species_probs = _species_probs_from_input(
                batch_size=batch_size,
                device=pep_tok.device,
                species=species,
            )
        else:
            species_probs = inferred_species_probs

        outputs["mhc_class_probs"] = class_probs
        outputs["mhc_species_probs"] = species_probs
        outputs["mhc_is_class1_prob"] = class_probs[:, :1]
        outputs["mhc_is_class2_prob"] = class_probs[:, 1:2]
        outputs["species_probs"] = species_probs

        # ------------------------------------------------------------------
        # 3) Core summary from peptide stream
        # ------------------------------------------------------------------
        pep_slice = offsets["peptide"]
        pep_h = h[:, pep_slice, :]
        pep_valid = valid_mask[:, pep_slice]

        pep_len = pep_valid.sum(dim=1).clamp(min=1)
        pos = torch.arange(pep_h.shape[1], device=pep_h.device).view(1, -1).expand(batch_size, -1)

        mhc_pair_vec = torch.cat([mhc_a_vec, mhc_b_vec], dim=-1)
        mhc_expanded = mhc_pair_vec.unsqueeze(1).expand(-1, pep_h.shape[1], -1)
        core_start_logit = self.core_start_head(
            torch.cat([pep_h, mhc_expanded], dim=-1)
        ).squeeze(-1)
        core_start_logit = core_start_logit.masked_fill(~pep_valid, -1e4)

        core_position_mask = pep_valid.clone()
        empty_core = ~core_position_mask.any(dim=1)
        if empty_core.any():
            core_position_mask[empty_core, 0] = True
            core_start_logit = core_start_logit.clone()
            core_start_logit[empty_core, 0] = 0.0

        core_start_prob = F.softmax(core_start_logit, dim=1)
        core_start_prob = core_start_prob * core_position_mask.float()
        core_start_prob = core_start_prob / core_start_prob.sum(dim=1, keepdim=True).clamp(min=1e-8)

        core_width = 8.0 + 4.0 * torch.sigmoid(self.core_width_head(mhc_pair_vec))
        expected_start = (core_start_prob * pos.float()).sum(dim=1, keepdim=True)
        pep_len_f = pep_len.float().unsqueeze(-1)
        core_length_norm = core_width / pep_len_f
        npfr_length_norm = expected_start / pep_len_f
        cpfr_length_norm = (pep_len_f - core_width - expected_start).clamp(min=0.0) / pep_len_f

        core_context_vec = (pep_h * core_start_prob.unsqueeze(-1)).sum(dim=1)

        outputs["core_position_mask"] = core_position_mask
        outputs["core_start_logit"] = core_start_logit.masked_fill(~core_position_mask, -1e4)
        outputs["core_start_prob"] = core_start_prob
        outputs["core_start_index"] = pos
        outputs["core_length"] = torch.round(core_width).long().expand(-1, pep_h.shape[1])
        outputs["npfr_length"] = pos
        outputs["cpfr_length"] = (
            pep_len.view(-1, 1) - pos - outputs["core_length"]
        ).clamp(min=0)
        outputs["core_length_norm"] = core_length_norm
        outputs["npfr_length_norm"] = npfr_length_norm
        outputs["cpfr_length_norm"] = cpfr_length_norm
        outputs["core_context_vec"] = core_context_vec

        # ------------------------------------------------------------------
        # 4) Optional TCR encoding (future feature; disabled by default)
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
        # 5) Latent query DAG with segmented attention
        # ------------------------------------------------------------------
        # Per design S5.4 / S7.5: context_vec goes to processing, binding,
        # and presentation latents. core_context_vec goes to presentation_class2
        # and recognition_cd4 only. tcr_vec goes to recognition only.
        context_token = self.context_token_proj(
            torch.cat([class_probs, mhc_a_species_probs, mhc_b_species_probs, chain_compat_prob], dim=-1)
        ).unsqueeze(1)
        core_context_token = core_context_vec.unsqueeze(1)

        # Which latents receive context_token (design S7.5)
        _gets_context = {
            "processing_class1", "processing_class2",
            "binding_affinity", "binding_stability",
            "presentation_class1", "presentation_class2",
        }
        # Which latents receive core_context_token (design S6.3)
        _gets_core_context = {"presentation_class2", "recognition_cd4"}
        # Which latents receive tcr_vec (design S7.5)
        _gets_tcr = {"recognition_cd8", "recognition_cd4"}

        # Variant C: pMHC interaction block (enriched representations for binding only)
        if self.use_pmhc_interaction_block:
            h_binding = self._run_pmhc_interaction(h, seg_masks, offsets)
        else:
            h_binding = h

        latent_vals: Dict[str, torch.Tensor] = {}
        binding_attention: Dict[str, List[torch.Tensor]] = {}
        for name in self.CROSS_ATTN_LATENTS:
            seg_names = self.LATENT_SEGMENTS[name]
            allowed = torch.zeros_like(valid_mask)
            for seg_name in seg_names:
                allowed = allowed | seg_masks[seg_name]

            extra_tokens: List[torch.Tensor] = []
            if name in _gets_context:
                extra_tokens.append(context_token)
            if name in _gets_core_context:
                extra_tokens.append(core_context_token)
            if name in _gets_tcr and tcr_vec is not None:
                extra_tokens.append(tcr_vec.unsqueeze(1))

            is_binding = name in self.BINDING_LATENT_NAMES
            collect = bool(
                return_binding_attention and is_binding
            )

            if is_binding and self._has_binding_enhancements:
                latent_vec, attn_layers = self._binding_latent_query(
                    name=name,
                    h=h_binding,
                    allowed_mask=allowed,
                    latent_store=latent_vals,
                    dep_names=self.LATENT_DEPS[name],
                    mhc_a_mask=seg_masks["mhc_a"],
                    offsets=offsets,
                    extra_tokens=extra_tokens if extra_tokens else None,
                    collect_attn=collect,
                )
            else:
                latent_vec, attn_layers = self._latent_query(
                    name=name,
                    h=h,
                    allowed_mask=allowed,
                    latent_store=latent_vals,
                    dep_names=self.LATENT_DEPS[name],
                    extra_tokens=extra_tokens if extra_tokens else None,
                    collect_attn=collect,
                )
            latent_vals[name] = latent_vec
            if attn_layers:
                binding_attention[name] = list(attn_layers)

            # After species_of_origin: compute foreignness and optionally override
            if name == "species_of_origin":
                if peptide_species is not None:
                    species_idx_t = _species_idx_tensor(
                        peptide_species, batch_size, latent_vec.device,
                    )
                    latent_vals["species_of_origin"] = self.species_override_embed(species_idx_t)
                latent_vals["foreignness"] = self.foreignness_proj(
                    F.gelu(latent_vals["species_of_origin"])
                )

        # Immunogenicity: MLP from upstream latent vecs (design S7.4 Level 3)
        latent_vals["immunogenicity_cd8"] = self.immunogenicity_cd8_mlp(torch.cat([
            latent_vals["binding_affinity"],
            latent_vals["binding_stability"],
            latent_vals["recognition_cd8"],
        ], dim=-1))
        latent_vals["immunogenicity_cd4"] = self.immunogenicity_cd4_mlp(torch.cat([
            latent_vals["binding_affinity"],
            latent_vals["binding_stability"],
            latent_vals["recognition_cd4"],
        ], dim=-1))

        outputs["latent_vecs"] = latent_vals
        if return_binding_attention and binding_attention:
            mhc_mask = seg_masks["mhc_a"] | seg_masks["mhc_b"]
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
            latent_vals["binding_affinity"],
            latent_vals["presentation_class1"],
            latent_vals["presentation_class2"],
        ], dim=-1))
        outputs["pmhc_vec"] = pmhc_vec

        # ------------------------------------------------------------------
        # 6) Processing logits (from proc latents)
        # ------------------------------------------------------------------
        proc_class1_logit = self.processing_class1_head(latent_vals["processing_class1"])
        proc_class2_logit = self.processing_class2_head(latent_vals["processing_class2"])
        proc_logit = class_probs[:, :1] * proc_class1_logit + class_probs[:, 1:2] * proc_class2_logit

        outputs["processing_class1_logit"] = proc_class1_logit
        outputs["processing_class2_logit"] = proc_class2_logit
        outputs["processing_logit"] = proc_logit
        outputs["processing_class1_prob"] = torch.sigmoid(proc_class1_logit)
        outputs["processing_class2_prob"] = torch.sigmoid(proc_class2_logit)
        outputs["processing_prob"] = torch.sigmoid(proc_logit)

        # ------------------------------------------------------------------
        # 7) Binding latents and calibrated binding logit
        # ------------------------------------------------------------------
        binding_vec = self.binding_fuse(
            torch.cat([latent_vals["binding_affinity"], latent_vals["binding_stability"]], dim=-1)
        )
        # Auxiliary probe: shortcut gradient path from binding_affinity latent → KD
        probe_kd = self.binding_affinity_probe(latent_vals["binding_affinity"])
        outputs["binding_affinity_probe_kd"] = torch.clamp(probe_kd, min=-3.0, max=8.0)

        binding_latents = self.binding(binding_vec, mhc_class=mhc_class, class_probs=class_probs)
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
        kd_bias = torch.tanh(self.kd_assay_bias(pmhc_vec)) * F.softplus(self.kd_assay_bias_scale)

        assays = self.assay_heads(pmhc_vec, binding_latents=binding_latents)
        # Keep only a lower clamp here. Upper capping is handled by
        # `derive_affinity_observables` via a smooth bound; hard max-clamping
        # here would collapse all very weak binders to one constant value.
        kd_log10 = torch.clamp(
            kd_from_binding + kd_bias,
            min=-3.0,
        )
        affinity_obs = self.assay_heads.derive_affinity_observables(pmhc_vec, kd_log10)
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
        outputs["binding_class1_prob"] = torch.sigmoid(binding_class1_logit)
        outputs["binding_class2_prob"] = torch.sigmoid(binding_class2_logit)
        outputs["binding_prob"] = torch.sigmoid(binding_logit)

        # ------------------------------------------------------------------
        # 8) Presentation logits (class-specific additive logit path)
        # ------------------------------------------------------------------
        class1_pres_base = self.presentation(
            proc_class1_logit,
            binding_class1_logit,
        )
        class2_pres_base = self.presentation(
            proc_class2_logit,
            binding_class2_logit,
        )

        # Latent residual is gated by explicitly initialized zero coefficients.
        class1_pres_base = class1_pres_base + self.w_presentation_class1_latent * self.presentation_class1_latent_head(latent_vals["presentation_class1"])
        class2_pres_base = class2_pres_base + self.w_presentation_class2_latent * self.presentation_class2_latent_head(latent_vals["presentation_class2"])

        class1_prob_logit = torch.logit(class_probs[:, :1].clamp(min=1e-4, max=1.0 - 1e-4))
        class2_prob_logit = torch.logit(class_probs[:, 1:2].clamp(min=1e-4, max=1.0 - 1e-4))
        class1_pres_logit = (
            class1_pres_base
            + F.softplus(self.w_class1_presentation_stability) * stability_score
            + F.softplus(self.w_class1_presentation_class) * class1_prob_logit
        )
        class2_pres_logit = (
            class2_pres_base
            + F.softplus(self.w_class2_presentation_stability) * stability_score
            + F.softplus(self.w_class2_presentation_class) * class2_prob_logit
        )
        pres_logit = class_probs[:, :1] * class1_pres_logit + class_probs[:, 1:2] * class2_pres_logit

        outputs["presentation_class1_logit"] = class1_pres_logit
        outputs["presentation_class2_logit"] = class2_pres_logit
        outputs["presentation_logit"] = pres_logit
        outputs["presentation_class1_prob"] = torch.sigmoid(class1_pres_logit)
        outputs["presentation_class2_prob"] = torch.sigmoid(class2_pres_logit)
        outputs["presentation_prob"] = torch.sigmoid(pres_logit)

        # ------------------------------------------------------------------
        # 9) Presentation-linked observables
        # ------------------------------------------------------------------
        ms_detectability_logit = self.ms_detectability_head(latent_vals["ms_detectability"])
        outputs["ms_detectability_logit"] = ms_detectability_logit
        outputs["ms_logit"] = self.ms_detection(
            pmhc_vec,
            pep_tok=pep_tok,
            proc_logit=proc_logit,
            pres_logit=pres_logit,
        )
        outputs["ms_prob"] = torch.sigmoid(outputs["ms_logit"])
        outputs["elution_logit"] = self.elution_head(
            pmhc_vec,
            presentation_logit=pres_logit,
            processing_logit=proc_logit,
        )
        outputs["elution_prob"] = torch.sigmoid(outputs["elution_logit"])

        recognition_repertoire_logit = self.repertoire(pmhc_vec)
        outputs["recognition_repertoire_logit"] = recognition_repertoire_logit
        outputs["recognition_repertoire_prob"] = torch.sigmoid(recognition_repertoire_logit)

        # ------------------------------------------------------------------
        # 10) Optional TCR-specific matching (future feature; disabled by default)
        # ------------------------------------------------------------------
        recognition_evidence = recognition_repertoire_logit
        if tcr_vec is not None:
            match_logit = self.matcher(tcr_vec, pmhc_vec)
            outputs["match_logit"] = match_logit
            outputs["match_prob"] = torch.sigmoid(match_logit)
            recognition_evidence = match_logit

        if recognition_evidence.ndim == 1:
            recognition_evidence = recognition_evidence.unsqueeze(-1)

        # ------------------------------------------------------------------
        # 11) Recognition and immunogenicity readouts
        # ------------------------------------------------------------------
        recognition_cd8_logit = self.recognition_cd8_head(latent_vals["recognition_cd8"])
        recognition_cd4_logit = self.recognition_cd4_head(latent_vals["recognition_cd4"])
        outputs["recognition_cd8_logit"] = recognition_cd8_logit
        outputs["recognition_cd4_logit"] = recognition_cd4_logit
        outputs["recognition_cd8_prob"] = torch.sigmoid(recognition_cd8_logit)
        outputs["recognition_cd4_prob"] = torch.sigmoid(recognition_cd4_logit)

        # Immunogenicity readout from MLP-computed latent vecs (design S9.5)
        immunogenicity_cd8_logit = self.immunogenicity_cd8_latent_head(latent_vals["immunogenicity_cd8"])
        immunogenicity_cd4_logit = self.immunogenicity_cd4_latent_head(latent_vals["immunogenicity_cd4"])
        outputs["immunogenicity_cd8_logit"] = immunogenicity_cd8_logit
        outputs["immunogenicity_cd4_logit"] = immunogenicity_cd4_logit
        outputs["immunogenicity_cd8_prob"] = torch.sigmoid(immunogenicity_cd8_logit)
        outputs["immunogenicity_cd4_prob"] = torch.sigmoid(immunogenicity_cd4_logit)

        immunogenicity_mixture_logit = (
            class_probs[:, :1] * immunogenicity_cd8_logit
            + class_probs[:, 1:2] * immunogenicity_cd4_logit
        )
        outputs["immunogenicity_mixture_logit"] = immunogenicity_mixture_logit
        outputs["immunogenicity_logit"] = immunogenicity_mixture_logit.squeeze(-1)
        outputs["immunogenicity_prob"] = torch.sigmoid(outputs["immunogenicity_logit"])

        # ------------------------------------------------------------------
        # 12) Context-conditioned T-cell assay output
        # ------------------------------------------------------------------
        context = tcell_context or {}
        tcell_logit, tcell_context_logits = self.tcell_assay_head(
            pmhc_vec=pmhc_vec,
            immunogenicity_logit=outputs["immunogenicity_logit"],
            tcr_vec=tcr_vec,
            immunogenicity_cd4_logit=immunogenicity_cd4_logit,
            immunogenicity_cd8_logit=immunogenicity_cd8_logit,
            class_probs=class_probs,
            assay_method_idx=context.get("assay_method_idx"),
            assay_readout_idx=context.get("assay_readout_idx"),
            apc_type_idx=context.get("apc_type_idx"),
            culture_context_idx=context.get("culture_context_idx"),
            stim_context_idx=context.get("stim_context_idx"),
        )
        outputs["tcell_logit"] = tcell_logit
        outputs["tcell_prob"] = torch.sigmoid(tcell_logit)
        outputs["tcell_context_logits"] = tcell_context_logits

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
