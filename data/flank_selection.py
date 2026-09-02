"""Choosing which source protein's flanks to train on.

A peptide usually occurs in more than one protein. Measured on hitlist 1.55.2
for HLA-A*02:01 alone: **80.5%** of evidence rows map to more than one protein
(the worst maps to 2,985), and **21.6%** have mappings that *disagree on a
flank* -- 157,423 rows where the junction context is genuinely ambiguous.

The excision head scores a cleavage from its junction residues, so for those
rows we are training on one candidate junction as though it were the observed
one. The proteasome cut some specific protein; we do not know which.

`map_source_proteins=True` returns one row per (evidence row, protein mapping)
and the collapse happens here, which means every candidate is in hand at the
moment the choice is made. What was missing was a place to make that choice
deliberately, and any record that a choice was made at all.

Ranking, strongest evidence first:

1. **Expression.** If the sample's transcript/gene abundance is known, the
   protein that is actually expressed is the one the peptide most likely came
   from. hitlist computes this properly -- `compute_peptide_origin` scores
   candidate genes by TPM and, where transcript rows exist, by the summed TPM
   of the isoforms whose translation actually contains the peptide. Callers
   pass the resulting per-gene scores in as `expression`.
2. **Canonical transcript.** The previous behaviour, and a reasonable
   tie-break, but not evidence: a peptide's canonical-transcript occurrence is
   not necessarily the one that was processed.
3. **Deterministic order.** Sorted by protein identifier so a run is
   reproducible rather than dependent on frame order.

The choice is reported alongside the result. A downstream consumer can then
down-weight ambiguous rows, or marginalise over candidates the way the binding
core already marginalises over registers, instead of rediscovering the
ambiguity by measuring it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

#: How the winning mapping was chosen, strongest first.
SELECTION_BASES = (
    "expression",
    "canonical_transcript",
    "resolved_flank",
    "deterministic_order",
)

#: The residue code for "a residue is here, identity unresolved".
#:
#: A flank carrying one is a worse training example than an equivalent flank
#: without: the junction residue the excision head reads is exactly the
#: unresolved one, since every N-side X sits at protein position 0. Where the
#: candidates are otherwise tied, prefer the resolved flank.
UNRESOLVED_RESIDUE = "X"


@dataclass(frozen=True)
class FlankChoice:
    """One chosen mapping, plus what was given up to choose it."""

    mapping: Mapping[str, Any]
    #: How many source proteins the peptide mapped to.
    n_candidates: int
    #: Whether every candidate agreed on both flanks. When False the chosen
    #: junction is one of several and should not be treated as observed.
    flanks_agree: bool
    #: Which rule in SELECTION_BASES decided it.
    basis: str
    #: Expression score of the winner, when expression was used.
    expression_score: Optional[float] = None

    @property
    def is_ambiguous(self) -> bool:
        return self.n_candidates > 1 and not self.flanks_agree


def _flank_pair(mapping: Mapping[str, Any]) -> tuple:
    return (
        str(mapping.get("n_flank") or ""),
        str(mapping.get("c_flank") or ""),
    )


def _has_unresolved(mapping: Mapping[str, Any]) -> bool:
    """Whether either flank carries an unresolved residue."""
    return any(UNRESOLVED_RESIDUE in part for part in _flank_pair(mapping))


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "t"}
    return bool(value)


def select_source_mapping(
    candidates: Sequence[Mapping[str, Any]],
    *,
    expression: Optional[Mapping[str, float]] = None,
    gene_field: str = "gene_name",
    protein_field: str = "protein_id",
) -> FlankChoice:
    """Pick the mapping whose flanks should be trained on.

    Parameters
    ----------
    candidates
        The mappings for one evidence row -- one per source protein. Each is a
        row-like mapping carrying at least ``n_flank`` and ``c_flank``.
    expression
        Optional ``gene_name -> score`` for the sample this peptide came from,
        e.g. TPM, or the output of hitlist's `compute_peptide_origin`. Higher
        wins. Genes absent from the mapping score as unknown rather than zero,
        so a partial expression table degrades to the canonical rule instead of
        silently preferring whatever happens to be listed.
    gene_field, protein_field
        Column names, so this works against either a hitlist frame or a test
        fixture.

    Returns
    -------
    FlankChoice
        The winner, how many candidates there were, whether they agreed, and
        which rule decided.
    """
    if not candidates:
        raise ValueError("select_source_mapping needs at least one candidate")

    rows = list(candidates)
    distinct_flanks = {_flank_pair(row) for row in rows}
    agree = len(distinct_flanks) == 1

    # Deterministic baseline order, so every rule below breaks ties the same
    # way and a run is reproducible regardless of frame order.
    rows.sort(key=lambda row: str(row.get(protein_field) or ""))

    if expression:
        scored = [
            (score, row)
            for row, score in (
                (row, expression.get(str(row.get(gene_field) or ""))) for row in rows
            )
            if score is not None
        ]
        if scored:
            best_score, best_row = max(scored, key=lambda pair: pair[0])
            return FlankChoice(
                mapping=best_row,
                n_candidates=len(rows),
                flanks_agree=agree,
                basis="expression",
                expression_score=float(best_score),
            )

    canonical = [row for row in rows if _truthy(row.get("is_canonical_transcript"))]
    if canonical:
        # Within canonical candidates, still prefer a resolved flank.
        resolved_canonical = [row for row in canonical if not _has_unresolved(row)]
        return FlankChoice(
            mapping=(resolved_canonical or canonical)[0],
            n_candidates=len(rows),
            flanks_agree=agree,
            basis="canonical_transcript",
        )

    # No canonical candidate. Before falling back to an arbitrary-but-stable
    # order, prefer a mapping whose flanks are actually resolved.
    #
    # This is worth a rule of its own: measured on the corpus, 525 evidence
    # rows reach training with an `X` in the chosen N-flank, and 514 of them
    # had a non-X alternative sitting right there. The junction residue the
    # excision head reads is precisely the unresolved one -- every N-side X is
    # at protein position 0 -- so those were the worst available choice.
    resolved = [row for row in rows if not _has_unresolved(row)]
    if resolved and len(resolved) != len(rows):
        return FlankChoice(
            mapping=resolved[0],
            n_candidates=len(rows),
            flanks_agree=agree,
            basis="resolved_flank",
        )

    return FlankChoice(
        mapping=rows[0],
        n_candidates=len(rows),
        flanks_agree=agree,
        basis="deterministic_order",
    )
