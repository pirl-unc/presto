"""Factorized fixed-panel T-cell outputs downstream of cellular pMHC biology."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from ..data.tcell_assay import AXES

# Kyte-Doolittle hydropathy and Zamyatnin side-chain volume (A^3) for TCR-facing bulge priors.
_KD_HYDRO: dict[str, float] = {
    "A": 1.8,
    "R": -4.5,
    "N": -3.5,
    "D": -3.5,
    "C": 2.5,
    "Q": -3.5,
    "E": -3.5,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "L": 3.8,
    "K": -3.9,
    "M": 1.9,
    "F": 2.8,
    "P": -1.6,
    "S": -0.8,
    "T": -0.7,
    "W": -0.9,
    "Y": -1.3,
    "V": 4.2,
}
_VOLUME: dict[str, float] = {
    "A": 88.6,
    "R": 173.4,
    "N": 114.1,
    "D": 111.1,
    "C": 108.5,
    "Q": 143.8,
    "E": 138.4,
    "G": 60.1,
    "H": 153.2,
    "I": 166.7,
    "L": 166.7,
    "K": 168.6,
    "M": 162.9,
    "F": 189.9,
    "P": 112.7,
    "S": 89.0,
    "T": 116.1,
    "W": 227.8,
    "Y": 193.6,
    "V": 140.0,
}
_AROMATIC = frozenset("FWY")
_LARGE_HYDRO = frozenset("ILVMFWY")
_POS_CHARGE = frozenset("KRH")
_NEG_CHARGE = frozenset("DE")
_POLAR = frozenset("STNQC")
_PRO_GLY = frozenset("PG")


def tcr_facing_indices(core_length: int, mhc_class: str) -> tuple[int, ...]:
    """Return 0-based positions within a core sequence that face the TCR.

    - Class I: central solvent-exposed bulge P4..P(Omega-2) (indices 3..L-2),
      mirroring structural TCR-pMHC-I contacts and PRIME 2.0 / Calis 2013.
    - Class II: upward-pointing non-anchor core positions P2, P3, P5, P8
      (indices 1, 2, 4, 7 of the 9-mer binding core).
    """
    if mhc_class == "II":
        idxs = tuple(i for i in (1, 2, 4, 7) if i < core_length)
        return idxs or (0,)
    if core_length >= 6:
        return tuple(range(3, core_length - 1))
    if core_length >= 4:
        return tuple(range(2, core_length - 1))
    return tuple(range(core_length))


def tcr_bulge_physicochem(
    register_candidates: list[tuple[int, str, list[float]]],
    mhc_class: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Compute an 8-dim deterministic physicochemical summary of TCR-facing bulge residues."""
    rows = []
    for _, core, _ in register_candidates:
        idxs = tcr_facing_indices(len(core), mhc_class)
        aas = [core[i] for i in idxs]
        n = float(max(1, len(aas)))
        arom = sum(1.0 for a in aas if a in _AROMATIC) / n
        lhyd = sum(1.0 for a in aas if a in _LARGE_HYDRO) / n
        kd = sum(_KD_HYDRO.get(a, 0.0) for a in aas) / (n * 4.5)
        vol = sum((_VOLUME.get(a, 140.0) - 140.0) / 60.0 for a in aas) / n
        pos = sum(1.0 for a in aas if a in _POS_CHARGE) / n
        neg = sum(1.0 for a in aas if a in _NEG_CHARGE) / n
        pol = sum(1.0 for a in aas if a in _POLAR) / n
        pg = sum(1.0 for a in aas if a in _PRO_GLY) / n
        rows.append([arom, lhyd, kd, vol, pos, neg, pol, pg])
    return torch.tensor(rows, device=device, dtype=dtype)


class TCellAssayLayer(nn.Module):
    """Emit positivity and conditional ordered intensity for every declared track."""

    def __init__(
        self,
        input_width,
        tracks,
        embedding_dim=16,
        hidden_width=None,
        *,
        include_positivity: bool = True,
    ):
        super().__init__()
        self.tracks = tracks
        self.embedding_dim = embedding_dim
        self.include_positivity = bool(include_positivity)
        hidden_width = hidden_width or input_width
        self.vocabulary = {
            axis: tuple(sorted({getattr(track, axis) for track in tracks})) for axis in AXES
        }
        self.factors = nn.ModuleDict(
            {
                axis: nn.ParameterDict(
                    {value: nn.Parameter(torch.randn(embedding_dim) * 0.02) for value in values}
                )
                for axis, values in self.vocabulary.items()
            }
        )
        decoder_width = input_width + len(AXES) * embedding_dim
        if self.include_positivity:
            self.positivity_decoder = nn.Sequential(
                nn.Linear(decoder_width, hidden_width),
                nn.GELU(),
                nn.Linear(hidden_width, 1),
            )
        self.intensity_decoder = nn.Sequential(
            nn.Linear(decoder_width, hidden_width),
            nn.GELU(),
            nn.Linear(hidden_width, 1),
        )
        # threshold[1] is strictly greater than threshold[0] by construction.
        self.ordinal_threshold_base = nn.Parameter(torch.tensor(-0.5))
        self.ordinal_threshold_delta = nn.Parameter(torch.tensor(0.541324854612918))

    @property
    def ordinal_thresholds(self):
        first = self.ordinal_threshold_base
        return torch.stack(
            (
                first,
                first + torch.nn.functional.softplus(self.ordinal_threshold_delta) + 1e-4,
            )
        )

    def _features(self, hidden, track):
        vectors = [hidden]
        for axis in AXES:
            factor = self.factors[axis][getattr(track, axis)]
            vectors.append(factor.expand(*hidden.shape[:-1], self.embedding_dim))
        return torch.cat(vectors, dim=-1)

    def score(self, hidden, track):
        if not self.include_positivity:
            raise RuntimeError("positivity_decoder is disabled when tcell_staged=True")
        return self.positivity_decoder(self._features(hidden, track)).squeeze(-1)

    def intensity_logits(self, hidden, track):
        location = self.intensity_decoder(self._features(hidden, track)).squeeze(-1)
        return location.unsqueeze(-1) - self.ordinal_thresholds

    def forward(self, hidden):
        return {
            track.key: {
                **({"positivity": self.score(hidden, track)} if self.include_positivity else {}),
                "conditional_intensity": self.intensity_logits(hidden, track),
            }
            for track in self.tracks
        }

    def dependencies(self, track=None, *, ordinal=True):
        selected = self.tracks if track is None else (track,)
        names: set[str] = set()
        if self.include_positivity:
            names.update(
                f"tcell_assay_layer.positivity_decoder.{name}"
                for name, _ in self.positivity_decoder.named_parameters()
            )
        if ordinal:
            names.update(
                f"tcell_assay_layer.intensity_decoder.{name}"
                for name, _ in self.intensity_decoder.named_parameters()
            )
            names.update(
                {
                    "tcell_assay_layer.ordinal_threshold_base",
                    "tcell_assay_layer.ordinal_threshold_delta",
                }
            )
        if self.include_positivity or ordinal:
            for candidate in selected:
                names.update(
                    f"tcell_assay_layer.factors.{axis}.{getattr(candidate, axis)}" for axis in AXES
                )
        return names

    def trained(self, track, evidence, *, ordinal=False):
        names = self.dependencies(track, ordinal=ordinal)
        return all(name in evidence and bool(evidence[name].any()) for name in names)


class TCellStagedHead(nn.Module):
    """PRIME-inspired staged T-cell immunogenicity head for Class I and Class II.

    Decomposes T-cell assay positivity into:
      1. Detached pMHC presentation/binding + stability supply (class-gated),
      2. Track-free TCR-facing bulge recognition (`P4..P(Omega-2)` for Class I,
         `P2, P3, P5, P8` for Class II) combining post-encoder residue embeddings
         with 8 deterministic physicochemical descriptors, and
      3. Factorized additive assay-track intercepts (+ small context residual) so
         cross-track prevalence differences cannot invert biological ranking.
    """

    def __init__(
        self,
        d_model: int,
        context_width: int,
        tracks,
        *,
        mhc_classes: tuple[str, ...] = ("I", "II"),
        dropout: float = 0.1,
        has_affinity: bool = True,
        has_stability: bool = True,
        has_presentation: bool = True,
        class_specific_tcr: bool = False,
    ):
        super().__init__()
        self.tracks = tracks
        self.mhc_classes = tuple(mhc_classes)
        self.has_affinity = bool(has_affinity)
        self.has_stability = bool(has_stability)
        self.has_presentation = bool(has_presentation)
        self.class_specific_tcr = bool(class_specific_tcr)
        self.vocabulary = {
            axis: tuple(sorted({getattr(track, axis) for track in tracks})) for axis in AXES
        }
        # PRIME-style orthogonality: TCR bulge MLP sees ONLY peptide-intrinsic TCR-facing
        # bulge features [bulge_mean, bulge_max, phys8, cls_oh] (NO pmhc_hidden!), preventing
        # collinear anti-correlation against pMHC supply.
        in_width = 2 * d_model + 8 + 2
        self.tcr_norm = nn.LayerNorm(in_width)
        self.tcr_mlp = nn.Sequential(
            nn.Linear(in_width, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
        )
        # Zero-init final linear only (exact 0.0 step-0 residual over calibrated 3-supply
        # floor, with full aligned O(1) step-1 gradient and zero scalar-gate saddle point).
        nn.init.zeros_(self.tcr_mlp[-1].weight)
        nn.init.zeros_(self.tcr_mlp[-1].bias)
        if self.class_specific_tcr and "II" in self.mhc_classes:
            self.tcr_norm_ii = nn.LayerNorm(in_width)
            self.tcr_mlp_ii = nn.Sequential(
                nn.Linear(in_width, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.LayerNorm(d_model),
                nn.Linear(d_model, 1),
            )
            nn.init.zeros_(self.tcr_mlp_ii[-1].weight)
            nn.init.zeros_(self.tcr_mlp_ii[-1].bias)
        # Strictly non-negative (softplus) class-specific biological cascade supply weights:
        # calibrated logistic slopes (~0.37 bind + ~0.37 stab + ~0.26 no-flank presentation).
        # Under class_specific_tcr=True, Class II omits unsupervised half_life so Class II
        # pmhc_supply relies cleanly on supervised Binding + No-Flank Presentation.
        if self.has_affinity:
            self.bind_weight_raw = nn.ParameterDict(
                {cls: nn.Parameter(torch.tensor(-0.8)) for cls in self.mhc_classes}
            )
        stab_classes = (
            tuple(cls for cls in self.mhc_classes if cls == "I")
            if self.class_specific_tcr
            else self.mhc_classes
        )
        self.stab_classes = stab_classes
        if self.has_stability and stab_classes:
            self.stab_weight_raw = nn.ParameterDict(
                {cls: nn.Parameter(torch.tensor(-0.8)) for cls in stab_classes}
            )
        if self.has_presentation:
            self.elut_weight_raw = nn.ParameterDict(
                {cls: nn.Parameter(torch.tensor(-1.2)) for cls in self.mhc_classes}
            )
        self.global_bias = nn.Parameter(torch.zeros(1))
        self.track_biases = nn.ModuleDict(
            {
                axis: nn.ParameterDict({value: nn.Parameter(torch.zeros(1)) for value in values})
                for axis, values in self.vocabulary.items()
            }
        )
        self.context_proj = nn.Linear(context_width, 1, bias=False) if context_width > 0 else None
        if self.context_proj is not None:
            nn.init.zeros_(self.context_proj.weight)

    def track_offset(self, track) -> torch.Tensor:
        total = self.global_bias.squeeze(0)
        for axis in AXES:
            total = total + self.track_biases[axis][getattr(track, axis)].squeeze(0)
        return total

    def score_registers(
        self,
        hidden: torch.Tensor,
        bulge_mean: torch.Tensor,
        bulge_max: torch.Tensor,
        physicochem: torch.Tensor,
        *,
        mhc_class: str,
        detached_affinity: torch.Tensor | None,
        detached_stability: torch.Tensor | None,
        detached_presentation: torch.Tensor | None = None,
        context: torch.Tensor | None,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (per_track_positivity, recognition, tcr_propensity, pmhc_supply)."""
        cls_idx = 0 if mhc_class == "I" else 1
        cls_oh = hidden.new_zeros((len(hidden), 2))
        cls_oh[:, cls_idx] = 1.0
        feats = torch.cat([bulge_mean, bulge_max, physicochem, cls_oh], dim=-1)
        if self.class_specific_tcr and mhc_class == "II" and hasattr(self, "tcr_mlp_ii"):
            raw_tcr = self.tcr_mlp_ii(self.tcr_norm_ii(feats)).squeeze(-1)
        else:
            raw_tcr = self.tcr_mlp(self.tcr_norm(feats)).squeeze(-1)
        tcr_propensity = 2.0 * torch.tanh(0.5 * raw_tcr)

        supply = hidden.new_zeros(len(hidden))
        if self.has_affinity and detached_affinity is not None:
            bind_w = F.softplus(self.bind_weight_raw[mhc_class])
            supply = supply + bind_w * (-detached_affinity)
        if (
            self.has_stability
            and detached_stability is not None
            and hasattr(self, "stab_weight_raw")
            and mhc_class in self.stab_weight_raw
        ):
            stab_w = F.softplus(self.stab_weight_raw[mhc_class])
            supply = supply + stab_w * detached_stability
        if self.has_presentation and detached_presentation is not None:
            elut_w = F.softplus(self.elut_weight_raw[mhc_class])
            supply = supply + elut_w * detached_presentation

        recognition = supply + tcr_propensity
        ctx_delta = (
            self.context_proj(context).squeeze(-1)
            if (self.context_proj is not None and context is not None)
            else hidden.new_zeros(())
        )
        per_track = {
            track.key: recognition + self.track_offset(track) + ctx_delta for track in self.tracks
        }
        return per_track, recognition, tcr_propensity, supply

    def score_aligned(
        self,
        bulge_mean: torch.Tensor,
        bulge_max: torch.Tensor,
        physicochem: torch.Tensor,
        register_attention: torch.Tensor,
        *,
        mhc_class: str,
        pooled_affinity: torch.Tensor | None,
        pooled_stability: torch.Tensor | None,
        pooled_presentation: torch.Tensor | None,
        context: torch.Tensor | None,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        """Register-aligned peptide-MHC-level T-cell scoring (zero `log(K)` length artifact).

        Pools each detached biophysical supply (`-ic50`, Class I `half_life`, full no-flank
        `elution_presentation`) at its own calibrated scale BEFORE linear combination, and
        weights per-register `tcr_propensity` strictly by the thermodynamic binding-register
        probability `register_attention = softmax(-det_aff)`.
        """
        cls_idx = 0 if mhc_class == "I" else 1
        cls_oh = bulge_mean.new_zeros((len(bulge_mean), 2))
        cls_oh[:, cls_idx] = 1.0
        feats = torch.cat([bulge_mean, bulge_max, physicochem, cls_oh], dim=-1)
        if self.class_specific_tcr and mhc_class == "II" and hasattr(self, "tcr_mlp_ii"):
            raw_tcr = self.tcr_mlp_ii(self.tcr_norm_ii(feats)).squeeze(-1)
        else:
            raw_tcr = self.tcr_mlp(self.tcr_norm(feats)).squeeze(-1)
        tcr_per_reg = 2.0 * torch.tanh(0.5 * raw_tcr)
        attn = register_attention.detach().to(dtype=tcr_per_reg.dtype)
        tcr_propensity = (attn * tcr_per_reg).sum()

        supply = bulge_mean.new_zeros(())
        if self.has_affinity and pooled_affinity is not None:
            bind_w = F.softplus(self.bind_weight_raw[mhc_class])
            supply = supply + bind_w * (-pooled_affinity)
        if (
            self.has_stability
            and pooled_stability is not None
            and hasattr(self, "stab_weight_raw")
            and mhc_class in self.stab_weight_raw
        ):
            stab_w = F.softplus(self.stab_weight_raw[mhc_class])
            supply = supply + stab_w * pooled_stability
        if self.has_presentation and pooled_presentation is not None:
            elut_w = F.softplus(self.elut_weight_raw[mhc_class])
            supply = supply + elut_w * pooled_presentation

        recognition = supply + tcr_propensity
        ctx_delta = (
            self.context_proj(context).squeeze(-1)
            if (self.context_proj is not None and context is not None)
            else bulge_mean.new_zeros(())
        )
        per_track = {
            track.key: recognition + self.track_offset(track) + ctx_delta for track in self.tracks
        }
        return per_track, recognition, tcr_propensity, supply

    def dependencies(self, track=None) -> set[str]:
        selected = self.tracks if track is None else (track,)
        names = {
            f"tcell_staged_head.{name}"
            for name, _ in self.named_parameters()
            if not name.startswith("track_biases.")
        }
        for candidate in selected:
            names.update(
                f"tcell_staged_head.track_biases.{axis}.{getattr(candidate, axis)}" for axis in AXES
            )
        return names

    def trained(self, track, evidence) -> bool:
        names = self.dependencies(track)
        return all(name in evidence and bool(evidence[name].any()) for name in names)
