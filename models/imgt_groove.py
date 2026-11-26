"""Pan-class 202-slot IMGT structural groove coordinate mapping and direct pocket interaction.

Uses `mhcseqs >= 2.17.0` (`IMGT_PAN_CLASS_GROOVE_SLOTS`, 100 HALF1 + 102 HALF2 = 202 slots;
`IMGT_PAN_CLASS_POCKET_SLOTS`, exact 3D-verified 46-slot NetMHCpan-34 ∪ NetMHCIIpan-34 union;
`IMGT_PAN_CLASS_NETMHCPAN_SLOTS`, 34 Class I contacts; and `IMGT_PAN_CLASS_NETMHCIIPAN_SLOTS`,
34 Class II contacts) to:
  1. Map every residue of `(groove1, groove2, mhc_class)` onto its exact 3D-homologous IMGT
     slot (`0..201`, `202` overflow), eliminating indel frame shifts across HLA-A/B/C/E/F/G,
     HLA-DR/DP/DQ (e.g., `DQA1*05:01`'s `55->61` helix deletion), and non-human MHC.
  2. Provide `IMGTPocketCrossInteraction`, combining:
     (a) Unblurred 9-pocket x 202-IMGT-slot structural cross-attention initialized with a
         class-specific `+2.5` 3D contact prior over the exact NetMHCpan-34 (Class I) and
         NetMHCIIpan-34 (Class II) peptide-contacting IMGT G-DOMAIN slots,
     (b) A direct, softmax-free unblurred `[pep_9_aa, pocket_46_aa, cls_oh]` MLP bypass that
         directly resolves subtle 2-3 residue pocket polymorphisms (`DRB1*04:01 / *04:04 / *04:05`)
         without Transformer self-attention blur across 90+ conserved scaffolding residues, and
     (c) Zero-init final linear projections: exact `0.0` step-0 warm-start perturbation with
         full `O(1)` step-1 gradients.
  3. Provide `DirectPocketBilinearHead`: an unblurred `9 × 46` IMGT bilinear contact-energy +
     PFR scalar head added directly to per-register affinity scores before register pooling.
"""

from __future__ import annotations

from functools import lru_cache
import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

_LOGGER = logging.getLogger(__name__)
_WARNED_FALLBACK = False
_WARNED_POCKET_FALLBACK = False

IMGT_PAN_CLASS_NUM_SLOTS = 202
IMGT_OVERFLOW_SLOT = 202
IMGT_VOCAB_SIZE = 203  # 0..201 canonical pan-class IMGT slots + 202 overflow/overhang

_AA20 = "ACDEFGHIKLMNPQRSTVWY"
_AA_TO_ID: dict[str, int] = {aa: idx for idx, aa in enumerate(_AA20)}
_GAP_ID = 20
_UNK_ID = 21
IMGT_AA_VOCAB_SIZE = 22  # 20 canonical AAs + '-' gap (20) + 'X'/unknown (21)

POCKET_SLOT_SCHEME_UNION46 = "mhcseqs_union46"
POCKET_SLOT_SCHEME_LEGACY58 = "legacy58"
VALID_POCKET_SLOT_SCHEMES = (POCKET_SLOT_SCHEME_UNION46, POCKET_SLOT_SCHEME_LEGACY58)

# Static mirror of mhcseqs==2.17.0 verified 3D-aligned pocket contacts (used only if an older
# mhcseqs < 2.17.0 is installed; when mhcseqs >= 2.17.0 is present we import directly from it).
_FALLBACK_UNION_46_SLOTS: tuple[tuple[str, str], ...] = (
    ("HALF1", "7"),
    ("HALF1", "9"),
    ("HALF1", "24"),
    ("HALF1", "26"),
    ("HALF1", "33"),
    ("HALF1", "45"),
    ("HALF1", "56"),
    ("HALF1", "59"),
    ("HALF1", "61"),
    ("HALF1", "62"),
    ("HALF1", "63"),
    ("HALF1", "66"),
    ("HALF1", "67"),
    ("HALF1", "69"),
    ("HALF1", "70"),
    ("HALF1", "73"),
    ("HALF1", "74"),
    ("HALF1", "76"),
    ("HALF1", "77"),
    ("HALF1", "80"),
    ("HALF1", "81"),
    ("HALF1", "84"),
    ("HALF2", "5"),
    ("HALF2", "7"),
    ("HALF2", "9"),
    ("HALF2", "22"),
    ("HALF2", "24"),
    ("HALF2", "26"),
    ("HALF2", "28"),
    ("HALF2", "45"),
    ("HALF2", "55"),
    ("HALF2", "59"),
    ("HALF2", "61A"),
    ("HALF2", "63"),
    ("HALF2", "66"),
    ("HALF2", "67"),
    ("HALF2", "69"),
    ("HALF2", "70"),
    ("HALF2", "72A"),
    ("HALF2", "73"),
    ("HALF2", "76"),
    ("HALF2", "77"),
    ("HALF2", "80"),
    ("HALF2", "81"),
    ("HALF2", "84"),
    ("HALF2", "85"),
)
_FALLBACK_CLASS_I_34_SLOTS: tuple[tuple[str, str], ...] = (
    ("HALF1", "7"),
    ("HALF1", "9"),
    ("HALF1", "24"),
    ("HALF1", "45"),
    ("HALF1", "59"),
    ("HALF1", "62"),
    ("HALF1", "63"),
    ("HALF1", "66"),
    ("HALF1", "67"),
    ("HALF1", "69"),
    ("HALF1", "70"),
    ("HALF1", "73"),
    ("HALF1", "74"),
    ("HALF1", "76"),
    ("HALF1", "77"),
    ("HALF1", "80"),
    ("HALF1", "81"),
    ("HALF1", "84"),
    ("HALF2", "5"),
    ("HALF2", "7"),
    ("HALF2", "9"),
    ("HALF2", "24"),
    ("HALF2", "26"),
    ("HALF2", "28"),
    ("HALF2", "55"),
    ("HALF2", "59"),
    ("HALF2", "61A"),
    ("HALF2", "63"),
    ("HALF2", "67"),
    ("HALF2", "69"),
    ("HALF2", "70"),
    ("HALF2", "73"),
    ("HALF2", "77"),
    ("HALF2", "81"),
)
_FALLBACK_CLASS_II_34_SLOTS: tuple[tuple[str, str], ...] = (
    ("HALF1", "7"),
    ("HALF1", "9"),
    ("HALF1", "24"),
    ("HALF1", "26"),
    ("HALF1", "33"),
    ("HALF1", "56"),
    ("HALF1", "61"),
    ("HALF1", "66"),
    ("HALF1", "67"),
    ("HALF1", "69"),
    ("HALF1", "73"),
    ("HALF1", "74"),
    ("HALF1", "76"),
    ("HALF1", "80"),
    ("HALF1", "81"),
    ("HALF2", "5"),
    ("HALF2", "7"),
    ("HALF2", "9"),
    ("HALF2", "22"),
    ("HALF2", "24"),
    ("HALF2", "26"),
    ("HALF2", "45"),
    ("HALF2", "55"),
    ("HALF2", "63"),
    ("HALF2", "66"),
    ("HALF2", "67"),
    ("HALF2", "70"),
    ("HALF2", "72A"),
    ("HALF2", "73"),
    ("HALF2", "76"),
    ("HALF2", "80"),
    ("HALF2", "81"),
    ("HALF2", "84"),
    ("HALF2", "85"),
)

# Legacy pre-#165 58-slot list retained strictly for loading historical v2.8 / v2.9 / v3.0
# checkpoints (`pocket_slot_scheme="legacy58"`). Never used for new models.
_LEGACY_58_CONTACT_SPECS: tuple[tuple[str, str], ...] = (
    *[
        ("HALF1", p)
        for p in (
            "5",
            "7",
            "9",
            "11",
            "22",
            "24",
            "31",
            "32",
            "43",
            "45",
            "52",
            "53",
            "55",
            "58",
            "59",
            "62",
            "63",
            "65",
            "66",
            "67",
            "68",
            "69",
            "70",
            "72",
            "73",
            "74",
            "76",
            "77",
            "80",
            "81",
            "84",
        )
    ],
    *[
        ("HALF2", p)
        for p in (
            "5",
            "7",
            "9",
            "11",
            "13",
            "24",
            "26",
            "28",
            "30",
            "45",
            "53",
            "55",
            "57",
            "60",
            "62",
            "63",
            "66",
            "67",
            "68",
            "69",
            "70",
            "73",
            "77",
            "80",
            "81",
            "84",
            "85",
        )
    ],
)


@lru_cache(maxsize=1)
def _mhcseqs_pocket_specs() -> tuple[
    tuple[tuple[str, str], ...],
    tuple[tuple[str, str], ...],
    tuple[tuple[str, str], ...],
]:
    """Load canonical 3D-verified pan-class pocket slot tuples from `mhcseqs >= 2.17.0`."""
    try:
        from mhcseqs.imgt import (
            IMGT_PAN_CLASS_NETMHCIIPAN_SLOTS,
            IMGT_PAN_CLASS_NETMHCPAN_SLOTS,
            IMGT_PAN_CLASS_POCKET_SLOTS,
        )

        return (
            tuple(IMGT_PAN_CLASS_POCKET_SLOTS),
            tuple(IMGT_PAN_CLASS_NETMHCPAN_SLOTS),
            tuple(IMGT_PAN_CLASS_NETMHCIIPAN_SLOTS),
        )
    except Exception:  # noqa: BLE001
        global _WARNED_POCKET_FALLBACK
        if not _WARNED_POCKET_FALLBACK:
            _WARNED_POCKET_FALLBACK = True
            _LOGGER.warning(
                "mhcseqs.imgt.IMGT_PAN_CLASS_POCKET_SLOTS unavailable (install mhcseqs>=2.17.0); "
                "using verified static mirror of the 46-slot NetMHCpan/NetMHCIIpan union."
            )
        return (
            _FALLBACK_UNION_46_SLOTS,
            _FALLBACK_CLASS_I_34_SLOTS,
            _FALLBACK_CLASS_II_34_SLOTS,
        )


def pocket_slot_specs(scheme: str = POCKET_SLOT_SCHEME_UNION46) -> tuple[tuple[str, str], ...]:
    """Return the `(half, imgt_pos)` slot tuple for the requested pocket slot scheme."""
    if scheme == POCKET_SLOT_SCHEME_UNION46:
        return _mhcseqs_pocket_specs()[0]
    if scheme == POCKET_SLOT_SCHEME_LEGACY58:
        return _LEGACY_58_CONTACT_SPECS
    raise ValueError(
        f"unsupported pocket_slot_scheme {scheme!r}; expected one of {VALID_POCKET_SLOT_SCHEMES}"
    )


def class_contact_prior_specs(
    mhc_class: str, scheme: str = POCKET_SLOT_SCHEME_UNION46
) -> tuple[tuple[str, str], ...]:
    """Return class-specific 3D contact slots for initializing `contact_bias`."""
    if scheme == POCKET_SLOT_SCHEME_UNION46:
        _, c1, c2 = _mhcseqs_pocket_specs()
        return c1 if str(mhc_class).upper() == "I" else c2
    if scheme == POCKET_SLOT_SCHEME_LEGACY58:
        return _LEGACY_58_CONTACT_SPECS
    raise ValueError(
        f"unsupported pocket_slot_scheme {scheme!r}; expected one of {VALID_POCKET_SLOT_SCHEMES}"
    )


NUM_CANONICAL_POCKET_SLOTS = len(pocket_slot_specs(POCKET_SLOT_SCHEME_UNION46))  # 46
NUM_LEGACY_POCKET_SLOTS = len(_LEGACY_58_CONTACT_SPECS)  # 58


@lru_cache(maxsize=1)
def _slot_lookup() -> tuple[dict[tuple[str, str], int], object] | None:
    try:
        from mhcseqs.imgt import IMGT_PAN_CLASS_GROOVE_SLOTS, align_groove_to_imgt
    except Exception:  # noqa: BLE001
        return None
    lookup = {slot: idx for idx, slot in enumerate(IMGT_PAN_CLASS_GROOVE_SLOTS)}
    return lookup, align_groove_to_imgt


def _parse_imgt_pos_fallback(pos: str) -> int:
    digits = "".join(ch for ch in pos if ch.isdigit())
    return max(0, int(digits) - 1) if digits else 0


def _resolve_specs_to_indices(specs: tuple[tuple[str, str], ...]) -> tuple[int, ...]:
    loaded = _slot_lookup()
    if loaded is not None:
        slot_to_idx, _ = loaded
        resolved = tuple(slot_to_idx[s] for s in specs if s in slot_to_idx)
        if len(resolved) == len(specs):
            return resolved
    # Deterministic positional fallback when mhcseqs < 2.11 is installed (unit tests only)
    out = []
    for half, pos in specs:
        base = _parse_imgt_pos_fallback(pos)
        out.append(min(base, 99) if half == "HALF1" else 100 + min(base, 101))
    return tuple(out)


@lru_cache(maxsize=4)
def canonical_pocket_slot_indices(
    scheme: str = POCKET_SLOT_SCHEME_UNION46,
) -> tuple[int, ...]:
    """Return exact integer indices in `0..201` for canonical 3D peptide-contacting slots."""
    return _resolve_specs_to_indices(pocket_slot_specs(scheme))


@lru_cache(maxsize=8)
def canonical_class_pocket_slot_indices(
    mhc_class: str,
    scheme: str = POCKET_SLOT_SCHEME_UNION46,
) -> tuple[int, ...]:
    """Return class-specific 3D contact slot indices in `0..201`."""
    return _resolve_specs_to_indices(class_contact_prior_specs(mhc_class, scheme))


def pocket_slot_overlap_mapping(old_scheme: str, new_scheme: str) -> list[tuple[int, int]]:
    """Return `(old_idx, new_idx)` pairs for IMGT slots shared between two pocket schemes."""
    old_specs = pocket_slot_specs(old_scheme)
    new_specs = pocket_slot_specs(new_scheme)
    old_pos = {spec: idx for idx, spec in enumerate(old_specs)}
    return [(old_pos[spec], new_idx) for new_idx, spec in enumerate(new_specs) if spec in old_pos]


@lru_cache(maxsize=8192)
def _align_groove_cached(
    groove1: str, groove2: str, mhc_class: str = "I"
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Return cached `(half1_slots, half2_slots, pan_class_202_aa_ids)`."""
    n1, n2 = len(groove1), len(groove2)
    cls = "II" if str(mhc_class).upper() == "II" else "I"
    loaded = _slot_lookup()
    # Real MHC groove domains are >= 50 aa; tiny synthetic unit-test strings use clean fallback.
    if loaded is not None and n1 >= 50 and n2 >= 50:
        slot_to_idx, align_fn = loaded
        try:
            aln = align_fn(groove1, groove2, mhc_class=cls)
            q1 = aln.half1.query_index_to_imgt
            q2 = aln.half2.query_index_to_imgt
            ov = IMGT_OVERFLOW_SLOT
            s1 = tuple(slot_to_idx.get(("HALF1", q1[i]), ov) if i in q1 else ov for i in range(n1))
            s2 = tuple(slot_to_idx.get(("HALF2", q2[j]), ov) if j in q2 else ov for j in range(n2))
            pan_seq = aln.pan_class_fixed_seq
            if len(pan_seq) == IMGT_PAN_CLASS_NUM_SLOTS:
                pan_ids = tuple(
                    _GAP_ID if ch == "-" else _AA_TO_ID.get(ch, _UNK_ID) for ch in pan_seq
                )
                return s1, s2, pan_ids
        except Exception:  # noqa: BLE001
            pass
    # Deterministic positional fallback for synthetic/short test grooves or older mhcseqs.
    global _WARNED_FALLBACK
    if n1 >= 50 and n2 >= 50 and not _WARNED_FALLBACK:
        _WARNED_FALLBACK = True
        _LOGGER.warning(
            "mhcseqs.imgt.align_groove_to_imgt unavailable or failed on real MHC groove "
            "(len=%d,%d); falling back to raw positional slot indices. Install mhcseqs>=2.17.0 "
            "for exact pan-class IMGT alignment!",
            n1,
            n2,
        )
    s1 = tuple(min(i, 99) for i in range(n1))
    s2 = tuple(100 + min(j, 101) for j in range(n2))
    pan_list = [_GAP_ID] * IMGT_PAN_CLASS_NUM_SLOTS
    for i, ch in enumerate(groove1[:100]):
        pan_list[i] = _AA_TO_ID.get(ch, _UNK_ID)
    for j, ch in enumerate(groove2[:102]):
        pan_list[100 + j] = _AA_TO_ID.get(ch, _UNK_ID)
    return s1, s2, tuple(pan_list)


def imgt_groove_slot_indices(
    groove1: str, groove2: str, mhc_class: str = "I"
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return cached `(half1_slots, half2_slots)` in `0..202` for one MHC groove pair."""
    s1, s2, _ = _align_groove_cached(groove1, groove2, mhc_class)
    return s1, s2


def imgt_pan_class_aa_ids(groove1: str, groove2: str, mhc_class: str = "I") -> tuple[int, ...]:
    """Return cached 202-length tuple of raw AA/gap IDs (`0..21`) on the pan-class IMGT grid."""
    _, _, pan_ids = _align_groove_cached(groove1, groove2, mhc_class)
    return pan_ids


def peptide_core_aa_ids(core: str) -> tuple[int, ...]:
    """Map peptide core characters to `0..21` IDs matching `IMGT_AA_VOCAB_SIZE`."""
    return tuple(_AA_TO_ID.get(ch, _UNK_ID) for ch in core)


def _gather_nine_slots(x: torch.Tensor) -> torch.Tensor:
    """Gather canonical 9-pocket slots `[P1..P4, P5_mid, P6..P9]` from `(B, L, D)`."""
    length = x.size(1)
    p1 = x[:, 0]
    p2 = x[:, min(1, length - 1)]
    p3 = x[:, min(2, length - 1)]
    p4 = x[:, min(3, length - 1)]
    if length >= 9:
        p5 = x[:, 4 : length - 4].mean(1)
    elif length == 8:
        p5 = x[:, 3:5].mean(1)
    else:
        p5 = x.mean(1)
    p6 = x[:, max(0, length - 4)]
    p7 = x[:, max(0, length - 3)]
    p8 = x[:, max(0, length - 2)]
    p9 = x[:, length - 1]
    return torch.stack([p1, p2, p3, p4, p5, p6, p7, p8, p9], dim=1)


class IMGTPocketCrossInteraction(nn.Module):
    """Unblurred 3D IMGT pocket-contact cross-interaction + direct pocket MLP bypass.

    Combines:
      1. Multi-head 9-pocket x 202-IMGT-slot cross-attention with class-specific `+2.5` 3D
         structural prior on exact NetMHCpan-34 (Class I) and NetMHCIIpan-34 (Class II)
         peptide-contacting IMGT slots and bilinear `pep * pocket` readout,
      2. A direct, softmax-free unblurred `[pep_9_aa (9x16), pocket_S_aa (Sx16), cls_oh (2)]`
         MLP over the exact `mhcseqs.IMGT_PAN_CLASS_POCKET_SLOTS` 46-slot union (or 58 legacy
         slots when loading pre-#165 checkpoints) that directly separates 2-residue pocket
         polymorphisms (e.g. DRB1*04:01/04/05),
      3. Zero-init final linear projections: exact `0.0` step-0 perturbation with immediate
         `O(1)` step-1 gradients.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        dropout: float = 0.1,
        *,
        pocket_slot_scheme: str = POCKET_SLOT_SCHEME_UNION46,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.head_dim = self.d_model // self.n_heads
        self.pocket_slot_scheme = str(pocket_slot_scheme)
        d_pair = max(32, self.d_model // 2)
        d_direct_aa = 16

        pocket_idx = torch.tensor(
            canonical_pocket_slot_indices(self.pocket_slot_scheme), dtype=torch.long
        )
        c1_prior_idx = torch.tensor(
            canonical_class_pocket_slot_indices("I", self.pocket_slot_scheme), dtype=torch.long
        )
        c2_prior_idx = torch.tensor(
            canonical_class_pocket_slot_indices("II", self.pocket_slot_scheme), dtype=torch.long
        )
        self.n_pocket_slots = int(pocket_idx.numel())
        self.register_buffer("pocket_slot_idx", pocket_idx, persistent=False)

        self.groove_aa_emb = nn.Embedding(IMGT_AA_VOCAB_SIZE, self.d_model)
        self.slot_pos_emb = nn.Embedding(IMGT_PAN_CLASS_NUM_SLOTS, self.d_model)
        self.groove_norm = nn.LayerNorm(self.d_model)

        self.pep_aa_emb = nn.Embedding(IMGT_AA_VOCAB_SIZE, self.d_model)
        self.pocket_query_emb = nn.Parameter(torch.zeros(2, 9, self.d_model))
        self.pep_norm = nn.LayerNorm(self.d_model)

        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.attn_out = nn.Linear(self.d_model, self.d_model)
        # Initialize class-specific 3D structural contact bias to +2.5 on exact Class I (34)
        # and Class II (34) peptide-contacting IMGT slots.
        init_bias = torch.zeros(2, self.n_heads, 9, IMGT_PAN_CLASS_NUM_SLOTS)
        init_bias[0, :, :, c1_prior_idx] = 2.5
        init_bias[1, :, :, c2_prior_idx] = 2.5
        self.contact_bias = nn.Parameter(init_bias)
        self.attn_dropout = nn.Dropout(dropout)

        self.pair_norm = nn.LayerNorm(3 * self.d_model)
        self.pair_mlp = nn.Linear(3 * self.d_model, d_pair)
        self.pair_dropout = nn.Dropout(dropout)
        self.out_norm = nn.LayerNorm(9 * d_pair)
        self.out_proj = nn.Linear(9 * d_pair, self.d_model)
        # Zero-init final linear only (no 0.0 scalar gate saddle point):
        # exact 0.0 step-0 perturbation + full aligned O(1) step-1 gradient!
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        # Direct softmax-free unblurred pseudo-sequence pocket MLP:
        # [9 peptide core AAs x 16, S canonical IMGT pocket AAs x 16, 2 class one-hot]
        self.direct_pep_emb = nn.Embedding(IMGT_AA_VOCAB_SIZE, d_direct_aa)
        self.direct_pocket_emb = nn.Embedding(IMGT_AA_VOCAB_SIZE, d_direct_aa)
        direct_in = (9 + self.n_pocket_slots) * d_direct_aa + 2
        self.direct_mlp = nn.Sequential(
            nn.LayerNorm(direct_in),
            nn.Linear(direct_in, self.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model),
        )
        nn.init.zeros_(self.direct_mlp[-1].weight)
        nn.init.zeros_(self.direct_mlp[-1].bias)

    def forward(
        self,
        core_states: torch.Tensor,
        core_aa_ids: torch.Tensor,
        pan_groove_ids: torch.Tensor,
        mhc_class: str,
    ) -> torch.Tensor:
        """Compute `(B, d_model)` unblurred IMGT pocket-interaction residual."""
        bsz = core_states.size(0)
        cls_idx = 0 if mhc_class == "I" else 1

        # 1. Unblurred 9-pocket x 202-IMGT-slot cross-attention with 3D contact prior
        pep_raw = self.pep_aa_emb(core_aa_ids)
        pep_9 = self.pep_norm(
            _gather_nine_slots(core_states)
            + _gather_nine_slots(pep_raw)
            + self.pocket_query_emb[cls_idx].unsqueeze(0)
        )
        groove_raw = self.groove_norm(
            self.groove_aa_emb(pan_groove_ids) + self.slot_pos_emb.weight.unsqueeze(0)
        )

        q = self.q_proj(pep_9).view(bsz, 9, self.n_heads, self.head_dim).transpose(1, 2)
        k = (
            self.k_proj(groove_raw)
            .view(bsz, IMGT_PAN_CLASS_NUM_SLOTS, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(groove_raw)
            .view(bsz, IMGT_PAN_CLASS_NUM_SLOTS, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )
        scale = self.head_dim**-0.5
        bias = self.contact_bias[cls_idx].unsqueeze(0)
        logits = torch.matmul(q, k.transpose(-2, -1)) * scale + bias
        attn = self.attn_dropout(F.softmax(logits, dim=-1))
        ctx = torch.matmul(attn, v).transpose(1, 2).contiguous().view(bsz, 9, self.d_model)
        pocket_ctx = self.attn_out(ctx)

        pair_feat = torch.cat([pep_9, pocket_ctx, pep_9 * pocket_ctx], dim=-1)
        h = self.pair_dropout(F.gelu(self.pair_mlp(self.pair_norm(pair_feat))))
        cross_out = self.out_proj(self.out_norm(h.reshape(bsz, -1)))

        # 2. Direct softmax-free unblurred 9-pep x S-pocket pseudo-sequence MLP
        pep_9_direct = _gather_nine_slots(self.direct_pep_emb(core_aa_ids)).reshape(bsz, -1)
        pocket_ids = pan_groove_ids.index_select(1, self.pocket_slot_idx)
        pocket_direct = self.direct_pocket_emb(pocket_ids).reshape(bsz, -1)
        cls_oh = core_states.new_zeros((bsz, 2))
        cls_oh[:, cls_idx] = 1.0
        direct_out = self.direct_mlp(torch.cat([pep_9_direct, pocket_direct, cls_oh], dim=-1))

        return cross_out + direct_out


class _SingleClassPocketBilinear(nn.Module):
    """Unblurred 9-pocket x S-IMGT-slot bilinear contact-energy + PFR scalar head for one class."""

    def __init__(
        self,
        n_mhc_slots: int = NUM_CANONICAL_POCKET_SLOTS,
        d_rank: int = 24,
        d_hidden: int = 192,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.d_rank = int(d_rank)
        self.n_pep = 9
        self.n_mhc = int(n_mhc_slots)
        self.pep_u = nn.Parameter(torch.randn(self.n_pep, IMGT_AA_VOCAB_SIZE, self.d_rank) * 0.1)
        self.mhc_v = nn.Parameter(torch.randn(self.n_mhc, IMGT_AA_VOCAB_SIZE, self.d_rank) * 0.1)
        self.pep_bias = nn.Parameter(torch.zeros(self.n_pep, IMGT_AA_VOCAB_SIZE))
        self.mhc_bias = nn.Parameter(torch.zeros(self.n_mhc, IMGT_AA_VOCAB_SIZE))
        n_pairs = self.n_pep * self.n_mhc
        in_dim = n_pairs + 3
        self.linear_energy = nn.Linear(in_dim, 1)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden // 2, 1),
        )
        # Zero-init final projections -> exact 0.0 step-0 warm-start perturbation
        nn.init.zeros_(self.linear_energy.weight)
        nn.init.zeros_(self.linear_energy.bias)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(
        self,
        core_aa_ids: torch.Tensor,
        pocket_ids: torch.Tensor,
        pfr: torch.Tensor,
    ) -> torch.Tensor:
        """core_aa_ids: (B, L>=8), pocket_ids: (B, S), pfr: (B, 3) -> scalar delta_G (B,)."""
        bsz, length = core_aa_ids.shape
        dev = core_aa_ids.device
        # Gather canonical 9 peptide pocket factors [P1..P4, P5_mid, P6..P9] without losing
        # N/C anchors on 8-14mer Class I peptides; exact 1:1 on all 9-mer cores.
        if length == 9:
            p_idx = torch.arange(9, device=dev).unsqueeze(0).expand(bsz, -1)
            u = self.pep_u[p_idx, core_aa_ids]  # (B, 9, R)
            pb = self.pep_bias[p_idx, core_aa_ids].sum(-1)
        else:
            u_slots = [
                self.pep_u[0, core_aa_ids[:, 0]],
                self.pep_u[1, core_aa_ids[:, min(1, length - 1)]],
                self.pep_u[2, core_aa_ids[:, min(2, length - 1)]],
                self.pep_u[3, core_aa_ids[:, min(3, length - 1)]],
            ]
            pb_slots = [
                self.pep_bias[0, core_aa_ids[:, 0]],
                self.pep_bias[1, core_aa_ids[:, min(1, length - 1)]],
                self.pep_bias[2, core_aa_ids[:, min(2, length - 1)]],
                self.pep_bias[3, core_aa_ids[:, min(3, length - 1)]],
            ]
            if length >= 9:
                mid_ids = core_aa_ids[:, 4 : length - 4]
                u_slots.append(self.pep_u[4, mid_ids].mean(dim=1))
                pb_slots.append(self.pep_bias[4, mid_ids].mean(dim=1))
            elif length == 8:
                mid_ids = core_aa_ids[:, 3:5]
                u_slots.append(self.pep_u[4, mid_ids].mean(dim=1))
                pb_slots.append(self.pep_bias[4, mid_ids].mean(dim=1))
            else:
                u_slots.append(self.pep_u[4, core_aa_ids].mean(dim=1))
                pb_slots.append(self.pep_bias[4, core_aa_ids].mean(dim=1))
            for slot_i, pos_i in (
                (5, max(0, length - 4)),
                (6, max(0, length - 3)),
                (7, max(0, length - 2)),
                (8, length - 1),
            ):
                u_slots.append(self.pep_u[slot_i, core_aa_ids[:, pos_i]])
                pb_slots.append(self.pep_bias[slot_i, core_aa_ids[:, pos_i]])
            u = torch.stack(u_slots, dim=1)
            pb = torch.stack(pb_slots, dim=1).sum(-1)

        s_idx = torch.arange(self.n_mhc, device=dev).unsqueeze(0).expand(bsz, -1)
        v = self.mhc_v[s_idx, pocket_ids]  # (B, S, R)
        mb = self.mhc_bias[s_idx, pocket_ids].sum(-1)
        e_mat = torch.einsum("bpr,bsr->bps", u, v) / math.sqrt(self.d_rank)
        flat = torch.cat([e_mat.reshape(bsz, -1), pfr.to(dtype=u.dtype)], dim=-1)
        return pb + mb + self.linear_energy(flat).squeeze(-1) + self.mlp(flat).squeeze(-1)


class DirectPocketBilinearHead(nn.Module):
    """Pan-class, class-specific unblurred `9 × S` IMGT contact-energy scalar head.

    Uses the exact 46-slot 3D-verified `mhcseqs.IMGT_PAN_CLASS_POCKET_SLOTS` union by default
    (or 58 legacy slots strictly when loading pre-#165 checkpoints). Added directly to
    per-register affinity scores (`assay_layer.score`) BEFORE register pooling — zero
    intervening Transformer or `LayerNorm` blur — so 2-3 residue MHC polymorphisms
    (`DRB1*04:01/04/05`, `DQ`, `HLA-A/B/C`) produce direct `O(1)` free-energy shifts `ΔG_contact`.
    """

    def __init__(
        self,
        mhc_classes: tuple[str, ...] = ("I", "II"),
        *,
        d_rank: int = 24,
        d_hidden: int = 192,
        dropout: float = 0.15,
        pocket_slot_scheme: str = POCKET_SLOT_SCHEME_UNION46,
    ):
        super().__init__()
        self.mhc_classes = tuple(mhc_classes)
        self.pocket_slot_scheme = str(pocket_slot_scheme)
        pocket_idx = torch.tensor(
            canonical_pocket_slot_indices(self.pocket_slot_scheme), dtype=torch.long
        )
        self.n_pocket_slots = int(pocket_idx.numel())
        self.register_buffer("pocket_slot_idx", pocket_idx, persistent=False)
        self.by_class = nn.ModuleDict(
            {
                cls: _SingleClassPocketBilinear(
                    n_mhc_slots=self.n_pocket_slots,
                    d_rank=d_rank,
                    d_hidden=d_hidden,
                    dropout=dropout,
                )
                for cls in self.mhc_classes
            }
        )

    def forward(
        self,
        core_aa_ids: torch.Tensor,
        pan_groove_ids: torch.Tensor,
        pfr: torch.Tensor,
        mhc_class: str,
    ) -> torch.Tensor:
        pocket_ids = pan_groove_ids.index_select(1, self.pocket_slot_idx)
        return self.by_class[mhc_class](core_aa_ids, pocket_ids, pfr)
