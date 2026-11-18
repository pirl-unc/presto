"""Output-side antigen-processing state bottleneck for the modular Presto.

Every state source (KO / overexpression via ``context.interventions``; cytokines via
``context.cytokines``) becomes one gene-state vector ``s``; step-shared gene embeddings map
it to a small bottleneck ``z`` (class-gated by pathway); the term added to the staged
elution ``processing`` score is ``z . U(peptide_vector)`` with ``U`` zero-initialised, so a
warm-started checkpoint is unchanged at step 0 and the trunk is never conditioned.
"""

from __future__ import annotations

import torch
from torch import nn

from . import apm_state as S

IFN_TARGETS = "PSMB8 PSMB9 PSMB10 PSME1 PSME2 TAP1 TAP2 TAPBP B2M NLRC5 ERAP1 ERAP2".split()
STATE_MODES = ("structured", "free")
FREE_SLOTS = 128


def row_state(context):
    """(gene state (G,), perturbed-token list) from a Context; unknown/absent-evidence = 0."""
    state = torch.zeros(S.N_APM_GENES)
    tokens = []
    interventions = {**context.interventions, **context.intervention_overrides}
    for name, claim in interventions.items():
        gene = name.upper()
        if gene not in S.APM_GENE_TO_IDX:
            continue
        index = S.APM_GENE_TO_IDX[gene]
        if claim.state == "absent":
            state[index] = min(float(state[index]), S.KO_FLOOR)
            tokens.append((gene, -1.0))
        elif claim.state == "present" and float(state[index]) == 0.0:
            state[index] = S.OVEREXPRESSION
            tokens.append((gene, 1.0))
    cytokines = {**context.cytokines, **context.cytokine_overrides}
    ifn = cytokines.get("ifn_gamma")
    if ifn is not None and ifn.state == "present":
        for gene in IFN_TARGETS:
            index = S.APM_GENE_TO_IDX[gene]
            if float(state[index]) == 0.0:
                state[index] = S.OVEREXPRESSION
        tokens.append(("IFN_GAMMA", 1.0))
    if "lps" in cytokines and cytokines["lps"].state == "present":
        tokens.append(("LPS", 1.0))
    return state, tokens


class ApmStateBottleneck(nn.Module):
    def __init__(self, d_z: int, d_model: int, mode: str = "structured"):
        super().__init__()
        if mode not in STATE_MODES:
            raise ValueError(f"state_mode must be one of {STATE_MODES}")
        self.mode, self.d_z = mode, int(d_z)
        # Only the parameters a mode actually uses exist: the trainer's coverage gate fails on
        # any parameter that never receives a gradient, and a dead table would be a silent no-op.
        if mode == "structured":
            self.register_buffer("incidence", S.step_incidence().float())
            for pathway, key in (("class_i", "mask_i"), ("class_ii", "mask_ii")):
                self.register_buffer(key, S.pathway_incidence(pathway).float())
            self.step_embedding = nn.Parameter(torch.randn(self.incidence.shape[1], self.d_z) * 0.1)
            self.gene_residual = nn.Parameter(torch.zeros(self.incidence.shape[0], self.d_z))
        else:
            # Non-zero init: the readout is zero-initialised, so a zero table would sit at the
            # saddle z = 0 = readout and never receive a gradient.
            self.free_embedding = nn.Parameter(torch.randn(FREE_SLOTS, self.d_z) * 0.1)
        self.readout = nn.Linear(d_model, self.d_z, bias=False)
        nn.init.zeros_(self.readout.weight)

    def residual_penalty(self):
        if self.mode == "structured":
            return self.gene_residual.pow(2).sum()
        return self.free_embedding.pow(2).sum()

    def z(self, state, tokens, mhc_class):
        if self.mode == "structured":
            mask = self.mask_i if mhc_class == "I" else self.mask_ii
            return (state.to(self.step_embedding.device) * mask) @ (
                self.incidence @ self.step_embedding + self.gene_residual
            )
        out = self.free_embedding.new_zeros(self.d_z)
        for token, sign in tokens:
            slot = int.from_bytes(token.encode()[:4].ljust(4, b"\0"), "little") % FREE_SLOTS
            out = out + sign * self.free_embedding[slot]
        return out

    def forward(self, context, mhc_class, peptide_vector):
        state, tokens = row_state(context)
        if not tokens:
            return peptide_vector.new_zeros(())
        return (self.z(state, tokens, mhc_class) * self.readout(peptide_vector)).sum()
