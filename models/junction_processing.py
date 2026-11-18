"""Shared, cut-aligned source-boundary correction to presentation supply.

The same ordered residue representation and MLP serve class I/II and N/C
boundaries. Missing reference context contributes zero; a known protein end is
a valid boundary with an empty exterior. This score is not a cleavage rate.
"""

import torch
from torch import nn


class JunctionProcessing(nn.Module):
    radius = 6

    def __init__(self, d_model, state_width=0):
        super().__init__()
        self.d_model = d_model
        self.state_width = state_width
        width = 2 * self.radius * (d_model + 1) + 6 + state_width
        self.input = nn.Linear(width, d_model)
        self.activation = nn.GELU()
        self.readout = nn.Linear(d_model, 1, bias=False)
        nn.init.zeros_(self.readout.weight)

    def feature_names(self):
        positions = list(range(-self.radius, 0)) + list(range(1, self.radius + 1))
        return (
            [f"junction:{p}:representation:{i}" for p in positions for i in range(self.d_model)]
            + [f"junction:{p}:present" for p in positions]
            + [
                "junction:class:I",
                "junction:class:II",
                "junction:side:N",
                "junction:side:C",
                "junction:protein_terminus:value",
                "junction:protein_terminus:known",
            ]
            + [f"junction:apc_state:{i}" for i in range(self.state_width)]
        )

    def boundary_features(self, row, states, mhc_class, side, state=None):
        if mhc_class not in ("I", "II") or side not in ("N", "C"):
            raise ValueError("resolved MHC class and N/C boundary required")
        context = row.context
        flank = context.source_n if side == "N" else context.source_c
        if flank is None:
            return None
        peptide = states[row.peptide]
        empty = peptide.new_zeros((0, self.d_model))
        flank_states = states[flank] if flank else empty
        left, right = (flank_states, peptide) if side == "N" else (peptide, flank_states)
        left, right = left[-self.radius :], right[: self.radius]

        def padding(n):
            return peptide.new_zeros((n, self.d_model))

        ordered = torch.cat(
            [padding(self.radius - len(left)), left, right, padding(self.radius - len(right))], 0
        )
        present = (
            [0.0] * (self.radius - len(left))
            + [1.0] * (len(left) + len(right))
            + [0.0] * (self.radius - len(right))
        )
        terminus = context.n_terminus if side == "N" else context.c_terminus
        flags = peptide.new_tensor(
            present
            + [
                float(mhc_class == "I"),
                float(mhc_class == "II"),
                float(side == "N"),
                float(side == "C"),
                float(terminus or False),
                float(terminus is not None),
            ]
        )
        values = [ordered.flatten(), flags]
        if self.state_width:
            if state is None or state.shape != (self.state_width,):
                raise ValueError("junction correction requires the declared shared APC state")
            values.append(state)
        return torch.cat(values)

    def forward(self, row, states, mhc_class, state=None):
        corrections = {}
        for side in ("N", "C"):
            features = self.boundary_features(row, states, mhc_class, side, state)
            corrections[side] = (
                self.readout(self.activation(self.input(features))).squeeze(-1)
                if features is not None
                else self.readout.weight.new_zeros(())
            )
        return sum(corrections.values()), corrections
