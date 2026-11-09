"""Literal external boundaries, averaging every overlapping parent occurrence."""

import numpy as np

from features import AA, raw_features

TOKENS = [*AA, "OTHER", "BOUNDARY"]
POSITIONS = ["N-2", "N-1", "C+1", "C+2"]


def occurrences(parent, peptide):
    starts = []
    start = parent.find(peptide)
    while start >= 0:
        starts.append(start)
        start = parent.find(peptide, start + 1)
    assert starts, "No exact peptide occurrence; do not drop or remap"
    return starts


def endpoint(parent, peptide):
    starts = occurrences(parent, peptide)
    context = np.zeros((4, len(TOKENS)))
    records = []
    for start in starts:
        end = start + len(peptide)
        tokens = []
        for j, index in enumerate((start - 2, start - 1, end, end + 1)):
            token = (
                "BOUNDARY"
                if not 0 <= index < len(parent)
                else parent[index]
                if parent[index] in AA
                else "OTHER"
            )
            context[j, TOKENS.index(token)] += 1 / len(starts)
            tokens.append(token)
        records.append(dict(start_1based=start + 1, end_1based=end, tokens=tokens))
    return np.r_[raw_features(peptide, "sequence"), context.ravel()], records


def vectors(parent, peptide, reference):
    a, _ = endpoint(parent, peptide)
    b, _ = endpoint(parent, reference)
    return dict(
        length=np.subtract(
            raw_features(peptide, "length"), raw_features(reference, "length")
        ).tolist(),
        sequence=(a[:61] - b[:61]).tolist(),
        boundary=(a - b).tolist(),
    )


def names():
    baseline = (
        ["ln_length"]
        + ["fraction_" + a for a in AA]
        + ["first_" + a for a in AA]
        + ["last_" + a for a in AA]
    )
    return baseline + [
        position + "_" + token for position in POSITIONS for token in TOKENS
    ]
