"""Shared Class-I/II processing, overhang geometry, and signed APC perturbation module.

Unifies antigen processing and presentation across MHC Class I and Class II via:
1. Always-active interior/exterior cut-aligned N- and C-boundary windows plus a
   continuous/RBF peptide-length basis (preserving interior P1/P2/P-omega motifs
   even when exterior flanking sequences are absent).
2. Per-register N/C peptide-flanking-region (PFR) overhang geometry for Class-II
   register pooling alongside Class-I full-length candidates.
3. Signed host gene perturbation axes (KO < KD < WT < KO+rescue < OE/transduction/
   transfection) and signed cytokine exposure axes with:
   - strict separation of pan-APM factors from HLA-A/B/C heavy-chain locus gates,
   - sequence-routed simplex HLA locus compatibility (preventing cross-locus KO bleed),
   - zero-mean multiplicative FiLM modulation + calibrated bounded yield gates,
   - direct detached-pMHC x APC candidate selectivity and chaperone editing.

Intrinsic pMHC binding endpoints never invoke this module and remain strictly
invariant to cellular/APC context and flanking sequence context.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from ..data.modular_context import (
    HLA_LOCUS_GENES,
    PERTURBATION_KINDS,
    canonical_perturbation_gene,
    is_supported_gene_perturbation,
    parse_gene_perturbation,
    supported_perturbation_genes,
)
from .apc_state import factor_classes

__all__ = [
    "HLA_LOCUS_GENES",
    "PERTURBATION_KINDS",
    "SharedProcessingModule",
    "build_shared_apc_columns",
    "canonical_perturbation_gene",
    "is_supported_gene_perturbation",
    "nonneg_bounded",
    "parse_gene_perturbation",
    "supported_perturbation_genes",
]

_HYDROPHOBIC = frozenset("LIVMFYW")
_BASIC = frozenset("KR")
_ACIDIC = frozenset("DE")
_LENGTH_RBF_BASIS = (
    (8.0, 1.2),
    (9.0, 1.0),
    (10.0, 1.2),
    (11.5, 1.5),
    (14.0, 2.0),
    (17.0, 2.5),
    (21.0, 3.5),
)


def nonneg_bounded(theta: torch.Tensor, cap: float = 1.0) -> torch.Tensor:
    """Zero at theta<=0, strictly in [0, cap) forward, full straight-through gradient."""
    raw = F.softplus(theta) - math.log(2.0)
    pos = raw.clamp_min(0.0)
    ste = pos + (raw - pos) - (raw - pos).detach()
    return cap * torch.tanh(ste)


def build_shared_apc_columns(config) -> tuple[tuple[str, str, str, tuple[str, ...]], ...]:
    """Ordered signed APC columns `(feature_name, family, target, classes)`."""
    columns = []
    for gene in supported_perturbation_genes(config.interventions):
        columns.append((f"shared_apc:gene:{gene}:signed", "gene", gene, factor_classes(gene)))
    for cytokine in config.cytokines:
        columns.append(
            (f"shared_apc:cytokine:{cytokine}:signed", "cytokine", cytokine, ("I", "II"))
        )
    return tuple(columns)


class SharedProcessingModule(nn.Module):
    """Unified Class-I/II boundary/length supply, register overhang, and signed APC module."""

    radius = 6
    pfr_window = 2
    apc_dim = 32
    reg_dim = 64
    edit_dim = 16

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.v2 = bool(getattr(config, "shared_processing_v2", False))
        self.state_width = config.state_bottleneck if config.state_mode == "harmonized" else 0
        all_columns = build_shared_apc_columns(config)
        self.pan_columns = tuple(
            col for col in all_columns if not (col[1] == "gene" and col[2] in HLA_LOCUS_GENES)
        )
        self.hla_columns = tuple(
            col for col in all_columns if col[1] == "gene" and col[2] in HLA_LOCUS_GENES
        )
        # Exposed APC encoder columns (pan-APM & cytokines; HLA locus gates route separately)
        self.apc_columns = self.pan_columns

        # Boundary: 6 interior + 6 exterior residue vectors + 6 exterior masks + 14 cut flags
        self.boundary_width = 2 * self.radius * self.d_model + self.radius + 14
        self.length_width = len(_LENGTH_RBF_BASIS) + 5
        self.register_width = 2 * self.d_model + 8

        self.boundary_input = nn.Linear(self.boundary_width, self.d_model)
        self.length_input = nn.Linear(self.length_width, self.d_model)
        if self.v2:
            half_d = self.d_model // 2
            self.junction_conv3 = nn.Conv1d(self.d_model, half_d, kernel_size=3, padding=1)
            self.junction_conv5 = nn.Conv1d(
                self.d_model, self.d_model - half_d, kernel_size=5, padding=2
            )
            self.junction_conv_proj = nn.Linear(2 * self.d_model, self.d_model)
            self.boundary_mlp = nn.Sequential(
                nn.Linear(self.d_model, self.d_model),
                nn.GELU(),
                nn.Linear(self.d_model, self.d_model),
            )
        self.supply_norm = nn.LayerNorm(self.d_model, elementwise_affine=False)
        self.supply_readout = nn.Linear(self.d_model, 1, bias=False)
        nn.init.zeros_(self.supply_readout.weight)

        self.register_input = nn.Linear(self.register_width, self.reg_dim)
        self.register_norm = nn.LayerNorm(self.reg_dim, elementwise_affine=False)
        self.register_readout = nn.Linear(self.reg_dim, 1, bias=False)
        nn.init.zeros_(self.register_readout.weight)
        self.logit_temp = nn.Parameter(torch.zeros(2))

        if self.state_width:
            self.state_film = nn.Linear(self.state_width, self.d_model, bias=False)

        if self.pan_columns:
            for cls, key in (("I", "apc_mask_i"), ("II", "apc_mask_ii")):
                self.register_buffer(
                    key, torch.tensor([float(cls in col[3]) for col in self.pan_columns])
                )
            self.apc_encoder = nn.Linear(len(self.pan_columns), self.apc_dim, bias=False)
            self.apc_film = nn.Linear(self.apc_dim, self.d_model, bias=False)
            self.register_apc_film = nn.Linear(self.apc_dim, self.reg_dim, bias=False)

            self.yield_theta = nn.Parameter(torch.zeros(len(self.pan_columns)))
            pan_caps = []
            for _, family, name, _ in self.pan_columns:
                if family == "gene" and name in {"B2M", "CIITA"}:
                    pan_caps.append(0.80)
                elif family == "cytokine" or name in {"TAP1", "TAP2", "TAPBP", "HLA-DM"}:
                    pan_caps.append(0.65)
                else:
                    pan_caps.append(0.65)
            self.register_buffer("pan_yield_caps", torch.tensor(pan_caps))

            self.pmhc_proj = nn.Linear(self.d_model, self.apc_dim, bias=False)
            self.pmhc_apc_proj = nn.Linear(self.apc_dim, self.apc_dim, bias=False)
            nn.init.zeros_(self.pmhc_apc_proj.weight)

            self.groove_proj = nn.Linear(2 * self.d_model, self.apc_dim, bias=False)
            self.groove_apc_proj = nn.Linear(self.apc_dim, self.apc_dim, bias=False)
            nn.init.zeros_(self.groove_apc_proj.weight)

            self.edit_groove = nn.Linear(2 * self.d_model, self.edit_dim, bias=False)
            self.edit_apc = nn.Linear(self.apc_dim, self.edit_dim, bias=False)
            nn.init.zeros_(self.edit_apc.weight)

        if self.hla_columns:
            self.hla_locus_router = nn.Linear(2 * self.d_model, len(self.hla_columns), bias=True)
            nn.init.zeros_(self.hla_locus_router.weight)
            nn.init.zeros_(self.hla_locus_router.bias)
            self.hla_locus_gain = nn.Parameter(torch.zeros(len(self.hla_columns)))

    def boundary_feature_names(self) -> list[str]:
        interior = [
            f"shared_boundary:interior:{pos}:rep:{i}"
            for pos in range(self.radius)
            for i in range(self.d_model)
        ]
        exterior = [
            f"shared_boundary:exterior:{pos}:rep:{i}"
            for pos in range(self.radius)
            for i in range(self.d_model)
        ]
        ext_present = [f"shared_boundary:exterior:{pos}:present" for pos in range(self.radius)]
        flags = [
            "shared_boundary:term_hydrophobic",
            "shared_boundary:term_basic",
            "shared_boundary:term_acidic",
            "shared_boundary:term_proline",
            "shared_boundary:penult_proline",
            "shared_boundary:penult_acidic",
            "shared_boundary:penult_hydrophobic",
            "shared_boundary:flank_known",
            "shared_boundary:side_N",
            "shared_boundary:side_C",
            "shared_boundary:protein_terminus_value",
            "shared_boundary:protein_terminus_known",
            "shared_boundary:class_I",
            "shared_boundary:class_II",
        ]
        return interior + exterior + ext_present + flags

    def length_feature_names(self) -> list[str]:
        return [
            "shared_length:log_ratio_9",
            "shared_length:centered_12",
            "shared_length:inv_ratio_9",
            *[f"shared_length:rbf_{int(center * 10)}" for center, _ in _LENGTH_RBF_BASIS],
            "shared_length:class_I",
            "shared_length:class_II",
        ]

    def register_feature_names(self) -> list[str]:
        return (
            [f"shared_register:left_pfr:rep:{i}" for i in range(self.d_model)]
            + [f"shared_register:right_pfr:rep:{i}" for i in range(self.d_model)]
            + [
                "shared_register:log1p_left_len",
                "shared_register:log1p_right_len",
                "shared_register:min_overhang",
                "shared_register:abs_overhang_diff",
                "shared_register:left_pfr_proline",
                "shared_register:right_pfr_proline",
                "shared_register:class_I",
                "shared_register:class_II",
            ]
        )

    def apc_feature_names(self) -> list[str]:
        return [col[0] for col in self.pan_columns]

    def _effective_interventions(self, cell) -> dict[str, str]:
        merged: dict[str, str] = {}
        for source in (
            cell.interventions,
            cell.supplied_interventions,
            cell.intervention_overrides,
        ):
            for key, assertion in source.items():
                merged[key] = assertion.state
        return merged

    def _encode_apc_columns(
        self, cell, columns: tuple[tuple[str, str, str, tuple[str, ...]], ...]
    ) -> torch.Tensor:
        if not columns:
            return self.supply_readout.weight.new_zeros(0)
        claims = self._effective_interventions(cell)
        by_gene: dict[str, list[tuple[str, str]]] = {}
        for key, state in claims.items():
            parsed = parse_gene_perturbation(key)
            if parsed is not None:
                by_gene.setdefault(parsed[1], []).append((parsed[0], state))

        values = []
        for _, family, name, _ in columns:
            if family == "gene":
                loss = 0.0
                gain = 0.0
                for kind, state in by_gene.get(name, ()):
                    if state != "present":
                        continue
                    if kind == "knockout":
                        loss = min(loss, -1.0)
                    elif kind == "knockdown":
                        loss = min(loss, -0.5) if loss < 0 else -0.5
                    elif kind in ("overexpression", "transduction", "transfection"):
                        gain = max(gain, 1.0)
                if loss < 0.0 and gain > 0.0:
                    # Endogenous KO + exogenous rescue/transduction restores active expression
                    values.append(0.6)
                elif loss < 0.0:
                    values.append(loss)
                else:
                    values.append(gain)
            else:
                exposure = cell.cytokines[name]
                if exposure.state == "present":
                    values.append(1.0 + 0.1 * math.log1p(exposure.dose or 0.0))
                elif exposure.state == "absent":
                    values.append(-0.35)
                else:
                    values.append(0.0)
        return self.supply_readout.weight.new_tensor(values)

    def apc_features(self, cell) -> torch.Tensor:
        return self._encode_apc_columns(cell, self.pan_columns)

    def _length_features(self, pep_len: int, mhc_class: str, ref: torch.Tensor) -> torch.Tensor:
        vals = [
            math.log(pep_len / 9.0),
            (pep_len - 12.0) / 8.0,
            9.0 / pep_len,
            *[
                math.exp(-0.5 * ((pep_len - center) / sigma) ** 2)
                for center, sigma in _LENGTH_RBF_BASIS
            ],
            float(mhc_class == "I"),
            float(mhc_class == "II"),
        ]
        return ref.new_tensor(vals)

    def _boundary_features(
        self, row, token_states: dict[str, torch.Tensor], mhc_class: str, side: str
    ) -> torch.Tensor:
        pep = row.peptide
        pep_states = token_states[pep]
        context = row.context
        flank = context.source_n if side == "N" else context.source_c

        if side == "N":
            interior = pep_states[: self.radius]
            term_aa = pep[0]
            penult_aa = pep[1] if len(pep) > 1 else pep[0]
        else:
            interior = pep_states[-self.radius :]
            term_aa = pep[-1]
            penult_aa = pep[-2] if len(pep) > 1 else pep[-1]
        if len(interior) < self.radius:
            interior = torch.cat(
                [interior, pep_states.new_zeros((self.radius - len(interior), self.d_model))], 0
            )

        if flank:
            f_states = token_states[flank]
            ext_raw = f_states[-self.radius :] if side == "N" else f_states[: self.radius]
            pad_len = self.radius - len(ext_raw)
            if side == "N":
                exterior = torch.cat([pep_states.new_zeros((pad_len, self.d_model)), ext_raw], 0)
                ext_present = [0.0] * pad_len + [1.0] * len(ext_raw)
            else:
                exterior = torch.cat([ext_raw, pep_states.new_zeros((pad_len, self.d_model))], 0)
                ext_present = [1.0] * len(ext_raw) + [0.0] * pad_len
        else:
            exterior = pep_states.new_zeros((self.radius, self.d_model))
            ext_present = [0.0] * self.radius

        terminus = context.n_terminus if side == "N" else context.c_terminus
        flags = pep_states.new_tensor(
            [
                *ext_present,
                float(term_aa in _HYDROPHOBIC),
                float(term_aa in _BASIC),
                float(term_aa in _ACIDIC),
                float(term_aa == "P"),
                float(penult_aa == "P"),
                float(penult_aa in _ACIDIC),
                float(penult_aa in _HYDROPHOBIC),
                float(flank is not None),
                float(side == "N"),
                float(side == "C"),
                float(terminus or False),
                float(terminus is not None),
                float(mhc_class == "I"),
                float(mhc_class == "II"),
            ]
        )
        return torch.cat([interior.flatten(), exterior.flatten(), flags])

    def _register_features(
        self,
        peptide: str,
        pep_states: torch.Tensor,
        register_candidates: list[tuple[int, str, list[float]]],
        mhc_class: str,
    ) -> torch.Tensor:
        pep_len = len(peptide)
        rows = []
        zero_vec = pep_states.new_zeros(self.d_model)
        for start, core, _ in register_candidates:
            end = start + len(core)
            left_len = start
            right_len = pep_len - end
            left_pfr = (
                pep_states[max(0, start - self.pfr_window) : start].mean(0)
                if left_len > 0
                else zero_vec
            )
            right_pfr = (
                pep_states[end : min(pep_len, end + self.pfr_window)].mean(0)
                if right_len > 0
                else zero_vec
            )
            left_seq = peptide[max(0, start - self.pfr_window) : start]
            right_seq = peptide[end : min(pep_len, end + self.pfr_window)]
            scalars = pep_states.new_tensor(
                [
                    math.log1p(left_len),
                    math.log1p(right_len),
                    min(left_len, right_len) / 6.0,
                    abs(left_len - right_len) / 8.0,
                    float("P" in left_seq),
                    float("P" in right_seq),
                    float(mhc_class == "I"),
                    float(mhc_class == "II"),
                ]
            )
            rows.append(torch.cat([left_pfr, right_pfr, scalars]))
        return torch.stack(rows)

    def _junction_sequence(
        self, row, token_states: dict[str, torch.Tensor], side: str
    ) -> torch.Tensor:
        """Contiguous `[2*radius, d_model]` sequence crossing the cleavage cut."""
        pep = row.peptide
        pep_states = token_states[pep]
        flank = row.context.source_n if side == "N" else row.context.source_c
        interior = pep_states[: self.radius] if side == "N" else pep_states[-self.radius :]
        if len(interior) < self.radius:
            pad = pep_states.new_zeros((self.radius - len(interior), self.d_model))
            interior = (
                torch.cat([interior, pad], 0) if side == "N" else torch.cat([pad, interior], 0)
            )
        if flank:
            f_states = token_states[flank]
            ext_raw = f_states[-self.radius :] if side == "N" else f_states[: self.radius]
            pad_len = self.radius - len(ext_raw)
            pad = pep_states.new_zeros((pad_len, self.d_model))
            exterior = torch.cat([pad, ext_raw], 0) if side == "N" else torch.cat([ext_raw, pad], 0)
        else:
            exterior = pep_states.new_zeros((self.radius, self.d_model))
        return (
            torch.cat([exterior, interior], 0)
            if side == "N"
            else torch.cat([interior, exterior], 0)
        )

    def forward(
        self,
        row,
        mhc,
        encoded: dict[str, torch.Tensor],
        token_states: dict[str, torch.Tensor],
        register_candidates: list[tuple[int, str, list[float]]],
        mhc_class: str,
        cell,
        affinity_scores: torch.Tensor,
        affinity_direction: int,
        apc_state: torch.Tensor | None = None,
        pmhc_hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Return `(supply_shift, register_selection, diagnostics)`.

        `supply_shift` is strictly MHC-candidate-independent; `register_selection`
        carries register overhang/PFR scores, pMHC-specific APC compatibility,
        sequence-routed HLA locus gating, and chaperone editing selectivity.
        """
        pep_states = token_states[row.peptide]
        feat_n = self._boundary_features(row, token_states, mhc_class, "N")
        feat_c = self._boundary_features(row, token_states, mhc_class, "C")
        feat_len = self._length_features(len(row.peptide), mhc_class, pep_states)

        # Stack N and C through boundary_input in one call so feature_inputs sees both sides
        b_hidden = self.boundary_input(torch.stack([feat_n, feat_c])).sum(0)
        l_hidden = self.length_input(feat_len)
        if self.v2:
            j_seq = torch.stack(
                [
                    self._junction_sequence(row, token_states, "N"),
                    self._junction_sequence(row, token_states, "C"),
                ]
            ).transpose(1, 2)  # [2, d_model, 2*radius]
            conv_feat = torch.cat(
                [F.gelu(self.junction_conv3(j_seq)), F.gelu(self.junction_conv5(j_seq))], dim=1
            )
            cut_pool = conv_feat[:, :, self.radius - 2 : self.radius + 2].mean(-1)
            all_pool = conv_feat.mean(-1)
            conv_hidden = self.junction_conv_proj(torch.cat([cut_pool, all_pool], dim=-1)).sum(0)
            base_in = F.gelu(b_hidden + l_hidden + conv_hidden)
            base_hidden = self.supply_norm(base_in + self.boundary_mlp(base_in))
        else:
            base_hidden = self.supply_norm(F.gelu(b_hidden + l_hidden))

        gamma = pep_states.new_zeros(self.d_model)
        if self.state_width and apc_state is not None:
            gamma = gamma + 0.35 * torch.tanh(self.state_film(apc_state))

        yield_shift = pep_states.new_zeros(())
        apc_vec = None
        masked_pan = None
        if self.pan_columns:
            raw_pan = self.apc_features(cell)
            mask = self.apc_mask_i if mhc_class == "I" else self.apc_mask_ii
            masked_pan = raw_pan * mask
            apc_vec = torch.tanh(self.apc_encoder(masked_pan))
            gamma = gamma + 0.15 * torch.tanh(self.apc_film(apc_vec))
            pan_weights = nonneg_bounded(self.yield_theta, cap=1.0) * self.pan_yield_caps
            yield_shift = (pan_weights * masked_pan).sum()

        supply_cap = 2.25 if self.v2 else 1.35
        supply_core = supply_cap * torch.tanh(
            self.supply_readout((1.0 + gamma) * base_hidden).squeeze(-1)
        )
        supply_shift = supply_core + yield_shift

        # Register PFR / overhang selection + pMHC/groove x APC selection + chaperone editing
        reg_feats = self._register_features(row.peptide, pep_states, register_candidates, mhc_class)
        reg_hidden = self.register_norm(F.gelu(self.register_input(reg_feats)))
        if apc_vec is not None:
            reg_hidden = (1.0 + 0.30 * torch.tanh(self.register_apc_film(apc_vec))) * reg_hidden
        reg_scores = 0.65 * torch.tanh(self.register_readout(reg_hidden).squeeze(-1))

        groove_repr = torch.cat([encoded[mhc.groove1].detach(), encoded[mhc.groove2].detach()], -1)
        groove_centered = F.layer_norm(groove_repr, (2 * self.d_model,))

        pmhc_apc_scores = reg_scores.new_zeros(reg_scores.shape)
        groove_shift = pep_states.new_zeros(())
        edit_alpha = pep_states.new_zeros(())
        if apc_vec is not None:
            if pmhc_hidden is not None:
                pmhc_centered = F.layer_norm(pmhc_hidden.detach(), (self.d_model,))
                pmhc_apc_scores = 0.25 * torch.tanh(
                    (self.pmhc_proj(pmhc_centered) * self.pmhc_apc_proj(apc_vec)).sum(-1)
                )
            groove_shift = 0.15 * torch.tanh(
                (self.groove_proj(groove_centered) * self.groove_apc_proj(apc_vec)).sum(-1)
            )
            edit_alpha = 0.25 * torch.tanh(
                (self.edit_groove(groove_centered) * self.edit_apc(apc_vec)).sum(-1)
            )

        locus_shift = pep_states.new_zeros(())
        if self.hla_columns and mhc_class == "I":
            raw_hla = self._encode_apc_columns(cell, self.hla_columns)
            locus_gates = torch.sigmoid(self.hla_locus_router(groove_centered))
            locus_gains = nonneg_bounded(self.hla_locus_gain, cap=0.60)
            locus_shift = (locus_gates * locus_gains * raw_hla).sum(-1)

        edit_scores = edit_alpha * (affinity_direction * affinity_scores.detach())
        total_selection = reg_scores + pmhc_apc_scores + groove_shift + locus_shift + edit_scores

        diagnostics = {
            "supply_core": supply_core,
            "apc_yield_shift": yield_shift,
            "register_overhang_scores": reg_scores,
            "pmhc_apc_scores": pmhc_apc_scores,
            "groove_apc_shift": groove_shift,
            "hla_locus_shift": locus_shift,
            "chaperone_edit_alpha": edit_alpha,
        }
        if masked_pan is not None:
            diagnostics["signed_apc_features"] = masked_pan
        return supply_shift, total_selection, diagnostics

    def calibrate_presentation(self, score: torch.Tensor, p_i: float, p_ii: float) -> torch.Tensor:
        """Per-class rank-preserving temperature (`1.0` at zero init, strictly positive)."""
        scales = 1.0 + 0.50 * torch.tanh(self.logit_temp)
        return (p_i * scales[0] + p_ii * scales[1]) * score
