"""Antigen-processing component state (see tasks/apm_state_vector_spec.md).

A sample's processing machinery is a vector ``s in R^G`` of log-fold changes over a
fixed gene registry, rather than one severity-collapsed category. Knockouts clamp a
gene to ``KO_FLOOR``, partial loss (knockdown, inhibitor) is a smaller negative,
overexpression is positive. Simultaneous interventions set several entries. A
separate observed-mask keeps "not reported" distinct from "reported unperturbed".

Pathways are static incidence over the registry, so the same vector yields a class I
direct, class II and cross-presentation view.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import torch

#: Value a clean knockout is clamped to (log2 fold vs reference).
KO_FLOOR = -4.0
#: Default partial loss for knockdown / inhibitor.
PARTIAL_LOSS = -2.0
#: Default overexpression when no fold is reported.
OVEREXPRESSION = 1.5

#: gene -> pathway steps. Steps are the units heads act on.
APM_GENE_STEPS: Dict[str, Tuple[str, ...]] = {
    # class I direct: proteasomal supply
    "PSMB5": ("c_cleavage",),
    "PSMB8": ("c_cleavage",),
    "PSMB9": ("c_cleavage",),
    "PSMB10": ("c_cleavage",),
    "PSME1": ("c_cleavage",),
    "PSME2": ("c_cleavage",),
    # transport and trimming
    "TAP1": ("transport",),
    "TAP2": ("transport",),
    "ERAP1": ("n_trimming",),
    "ERAP2": ("n_trimming",),
    # loading / assembly
    "TAPBP": ("loading",),
    "CALR": ("loading",),
    "CANX": ("loading",),
    "PDIA3": ("loading",),
    "GANAB": ("loading",),
    "SPPL3": ("loading",),
    "TAPBPL": ("loading",),
    "IRF2": ("other_apm",),
    "B2M": ("class_i_gate",),
    "NLRC5": ("class_i_gate",),
    # class II
    "CIITA": ("class_ii_gate",),
    "CD74": ("class_ii_gate", "class_ii_loading"),
    "CTSS": ("class_ii_cleavage", "xp_cleavage"),
    "CTSL": ("class_ii_cleavage",),
    "CTSB": ("class_ii_cleavage",),
    "LGMN": ("class_ii_cleavage",),
    "IFI30": ("class_ii_cleavage",),
    "HLA_DM": ("class_ii_loading",),
    "HLA_DO": ("class_ii_loading",),
    # cross-presentation (vacuolar / cytosolic routes)
    "CYBB": ("xp_cleavage",),
    "SEC22B": ("xp_transfer",),
    "SEC61": ("xp_transfer",),
    "BATF3": ("xp_transfer",),
}

APM_GENES: List[str] = list(APM_GENE_STEPS)
APM_GENE_TO_IDX: Dict[str, int] = {g: i for i, g in enumerate(APM_GENES)}
N_APM_GENES = len(APM_GENES)

STEPS: List[str] = sorted({s for steps in APM_GENE_STEPS.values() for s in steps})
STEP_TO_IDX: Dict[str, int] = {s: i for i, s in enumerate(STEPS)}

#: pathway -> the steps it consists of.
PATHWAY_STEPS: Dict[str, Tuple[str, ...]] = {
    "class_i": ("c_cleavage", "n_trimming", "transport", "loading", "class_i_gate", "other_apm"),
    "class_ii": ("class_ii_gate", "class_ii_cleavage", "class_ii_loading"),
    "cross_presentation": ("xp_cleavage", "xp_transfer", "transport", "c_cleavage"),
}

#: Aliases hitlist / IEDB use, normalised to registry symbols.
_ALIASES = {
    "tap1": "TAP1",
    "tap2": "TAP2",
    "tap": None,
    "psmb5": "PSMB5",
    "psmb8": "PSMB8",
    "psmb9": "PSMB9",
    "psmb10": "PSMB10",
    "erap1": "ERAP1",
    "erap2": "ERAP2",
    "tapbp": "TAPBP",
    "calr": "CALR",
    "canx": "CANX",
    "pdia3": "PDIA3",
    "ganab": "GANAB",
    "sppl3": "SPPL3",
    "tapbpr": "TAPBPL",
    "irf2": "IRF2",
    "b2m": "B2M",
    "nlrc5": "NLRC5",
    "ciita": "CIITA",
    "cd74": "CD74",
    "cathepsin": "CTSS",
    "ctss": "CTSS",
    "ctsl": "CTSL",
    "ctsb": "CTSB",
    "lgmn": "LGMN",
    "ifi30": "IFI30",
    "hla_dm": "HLA_DM",
    "hla_do": "HLA_DO",
    "cybb": "CYBB",
    "sec22b": "SEC22B",
    "sec61": "SEC61",
    "batf3": "BATF3",
    # tool compounds act on a whole family
    "tap_inhibitor": ("TAP1", "TAP2"),
    "tap_deficient_line": ("TAP1", "TAP2"),
    "erap_inhibitor": ("ERAP1", "ERAP2"),
    "proteasome_inhibitor": ("PSMB5", "PSMB8", "PSMB9", "PSMB10"),
    "cathepsin_inhibitor": ("CTSS", "CTSL", "CTSB"),
}

KIND_VALUE = {"ko": KO_FLOOR, "partial": PARTIAL_LOSS, "overexpression": OVEREXPRESSION}


def _genes_for(token: str) -> Tuple[str, ...]:
    key = token.strip().lower()
    hit = _ALIASES.get(key)
    if hit is None:
        return ()
    return (hit,) if isinstance(hit, str) else tuple(hit)


def encode_state(
    interventions: Optional[Iterable[Tuple[str, str]]] = None,
    fold_changes: Optional[Mapping[str, float]] = None,
    observed: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build ``(state, observed_mask)`` for one sample.

    ``interventions`` is ``(gene_or_compound, kind)`` with kind in ``ko`` / ``partial``
    / ``overexpression``. Compound names expand to their targets and default to
    ``partial``. ``fold_changes`` sets measured log2 folds directly. ``observed=False``
    marks the arm as unresolved, which is NOT the same as an unperturbed control.
    """
    state = torch.zeros(N_APM_GENES)
    mask = torch.full((N_APM_GENES,), 1.0 if observed else 0.0)
    for token, kind in interventions or ():
        for gene in _genes_for(token):
            idx, value = APM_GENE_TO_IDX[gene], KIND_VALUE[kind]
            current = float(state[idx])
            # Order-independent: the more severe loss wins; overexpression only
            # applies to a gene with no reported loss.
            if value < 0:
                state[idx] = min(current, value) if current < 0 else value
            elif current >= 0:
                state[idx] = max(current, value)
    for gene, fold in (fold_changes or {}).items():
        if gene in APM_GENE_TO_IDX:
            state[APM_GENE_TO_IDX[gene]] = float(fold)
    return state, mask


def encode_apm_genes(genes: Optional[object], observed: bool = True):
    """hitlist ``apm_genes_perturbed`` string -> state. Names without an explicit
    kind are treated as knockouts, except compound inhibitors (partial)."""
    if not genes:
        return encode_state(observed=observed)
    if isinstance(genes, str):
        tokens = [t.strip() for t in genes.replace(",", ";").split(";") if t.strip()]
    else:
        tokens = [str(t).strip() for t in genes if str(t).strip()]
    pairs = []
    for tok in tokens:
        kind = "partial" if "inhibitor" in tok.lower() else "ko"
        pairs.append((tok, kind))
    return encode_state(pairs, observed=observed)


def step_incidence() -> torch.Tensor:
    """``(G, S)`` 0/1 matrix: gene g participates in step s."""
    inc = torch.zeros(N_APM_GENES, len(STEPS))
    for gene, steps in APM_GENE_STEPS.items():
        for step in steps:
            inc[APM_GENE_TO_IDX[gene], STEP_TO_IDX[step]] = 1.0
    return inc


def pathway_incidence(pathway: str) -> torch.Tensor:
    """``(G,)`` 0/1 mask of genes belonging to a pathway."""
    inc = step_incidence()
    cols = [STEP_TO_IDX[s] for s in PATHWAY_STEPS[pathway]]
    return (inc[:, cols].sum(dim=1) > 0).float()


class PathwayVectors(torch.nn.Module):
    """Learned pathway vectors ``v_p = sum_g mask[p,g] * s_g * e_g``.

    Output-side only: consumers are processing/presentation heads, never the encoder
    trunk or intrinsic binding. Unobserved entries contribute zero.
    """

    def __init__(self, d_state: int = 16):
        super().__init__()
        self.d_state = d_state
        self.gene_embed = torch.nn.Parameter(torch.randn(N_APM_GENES, d_state) * 0.02)
        for name in PATHWAY_STEPS:
            self.register_buffer(f"mask_{name}", pathway_incidence(name))

    def forward(self, state: torch.Tensor, observed: Optional[torch.Tensor] = None):
        if observed is not None:
            state = state * observed
        out = {}
        for name in PATHWAY_STEPS:
            mask = getattr(self, f"mask_{name}")
            out[name] = (state * mask).unsqueeze(-1).mul(self.gene_embed).sum(dim=1)
        return out


class ApmStateEffects(torch.nn.Module):
    """State-linear shifts to an excision logit.

    ``shift = sum_g s_g * (pref_c[g](window_c) + pref_n[g](window_n) + bias_g)``
    -- linear in the state, so KO, partial loss and overexpression interpolate and
    unseen combinations compose additively. Gate genes (B2M/NLRC5 for class I,
    CIITA/CD74 for class II) add a loss-only floor: ``-softplus(w_g) * relu(-s_g)``.

    Profiles are zero-initialised: with no data the module is the identity.
    """

    def __init__(
        self, n_aa: int, junction_window: int = 5, pathway: str = "class_i", n_extra: int = 0
    ):
        super().__init__()
        # `n_extra` appends scalar-token positions (e.g. peptide length) that are
        # indexed like residues but sit outside the junction window.
        self.window = 2 * junction_window + n_extra
        self.pathway = pathway
        # Effect of gene g = shared profile of each step it belongs to + a small
        # gene-specific residual. The shared part is what lets a held-out gene
        # borrow strength from its siblings; the residual is regularised.
        self.register_buffer("incidence", step_incidence())
        n_steps = len(STEPS)
        self.step_c = torch.nn.Parameter(torch.zeros(n_steps, self.window, n_aa))
        self.step_n = torch.nn.Parameter(torch.zeros(n_steps, self.window, n_aa))
        self.resid_c = torch.nn.Parameter(torch.zeros(N_APM_GENES, self.window, n_aa))
        self.resid_n = torch.nn.Parameter(torch.zeros(N_APM_GENES, self.window, n_aa))
        self.step_bias = torch.nn.Parameter(torch.zeros(n_steps))
        self.resid_bias = torch.nn.Parameter(torch.zeros(N_APM_GENES))
        self.gate_logit = torch.nn.Parameter(torch.zeros(N_APM_GENES))
        gate_steps = {"class_i": "class_i_gate", "class_ii": "class_ii_gate"}.get(pathway)
        inc = step_incidence()
        gate = inc[:, STEP_TO_IDX[gate_steps]] if gate_steps else torch.zeros(N_APM_GENES)
        self.register_buffer("gate_mask", gate)
        self.register_buffer("gene_mask", pathway_incidence(pathway))

    @property
    def delta_c(self):
        return torch.einsum("gs,swa->gwa", self.incidence, self.step_c) + self.resid_c

    @property
    def delta_n(self):
        return torch.einsum("gs,swa->gwa", self.incidence, self.step_n) + self.resid_n

    @property
    def bias(self):
        return self.incidence @ self.step_bias + self.resid_bias

    def residual_penalty(self):
        return self.resid_c.pow(2).sum() + self.resid_n.pow(2).sum() + self.resid_bias.pow(2).sum()

    @staticmethod
    def _pref(delta: torch.Tensor, window_idx: torch.Tensor) -> torch.Tensor:
        batch = window_idx.shape[0]
        g, w, _ = delta.shape
        expanded = delta.unsqueeze(0).expand(batch, g, w, -1)
        index = window_idx.unsqueeze(1).unsqueeze(-1).expand(batch, g, w, 1)
        return expanded.gather(3, index).squeeze(-1).sum(dim=2)  # (batch, G)

    def forward(self, window_c, window_n, state, observed=None):
        if observed is not None:
            state = state * observed
        s = state * self.gene_mask
        per_gene = self._pref(self.delta_c, window_c.long()) + self._pref(
            self.delta_n, window_n.long()
        )
        shift = (s * (per_gene + self.bias)).sum(dim=1)
        gate = -(
            torch.nn.functional.softplus(self.gate_logit) * self.gate_mask * torch.relu(-s)
        ).sum(dim=1)
        return shift + gate
