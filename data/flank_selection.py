"""Choosing which source protein's flanks to train on.

A peptide usually occurs in more than one protein. Measured on hitlist 1.55.2
for HLA-A*02:01 alone: **80.5%** of evidence rows map to more than one protein
(the worst maps to 2,985), and **21.6%** have mappings that *disagree on a
flank* -- 157,423 rows where the junction context is genuinely ambiguous.

The excision head scores a cleavage from its junction residues. Presto now
preserves a flank pair only when candidates agree, expression resolves a gene,
or a unique canonical source resolves within one gene. Otherwise the context
is explicitly unknown; the peptide/MHC measurement remains usable.

`map_source_proteins=True` returns one row per (evidence row, protein mapping)
and the collapse happens here, which means every candidate is in hand at the
moment the choice is made. What was missing was a place to make that choice
deliberately, and any record that a choice was made at all.

Resolution evidence, strongest first:

1. **Expression.** If the sample's transcript/gene abundance is known, the
   protein that is actually expressed is the one the peptide most likely came
   from. hitlist computes this properly -- `compute_peptide_origin` scores
   candidate genes by TPM and, where transcript rows exist, by the summed TPM
   of the isoforms whose translation actually contains the peptide. Callers
   pass the resulting per-gene scores in as `expression`.
2. **Canonical transcript within one gene.** A reasonable isoform tie-break,
   but never evidence for choosing between genes: each gene normally brings
   its own canonical transcript.
3. **Agreement.** When every candidate has the same junction, which source row
   represents it is immaterial.

Flank *cleanliness* is deliberately absent from that evidence -- see
`UNRESOLVED_RESIDUE`. A row whose chosen flank contains an unresolved amino
acid is dropped rather than reassigned to whichever other protein happens to
offer a confident-looking residue.

The choice is reported alongside the result. A downstream consumer must replace
the flank pair with unknown context when ``flank_context_resolved`` is false;
candidate marginalization remains a possible later model extension.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

#: How the winning mapping was chosen, strongest first.
SELECTION_BASES = (
    "expression",
    "canonical_transcript",
    "deterministic_order",
)

#: The residue code hitlist emits for "a residue is here, identity unresolved".
#:
#: Deliberately *not* a ranking signal. Preferring a resolved candidate sounds
#: harmless but is not: measured on the corpus, 91% of the rows such a rule
#: flips also change which **gene** the peptide is attributed to. That trades a
#: known unknown for a possibly-wrong protein of origin, which is the worse
#: error -- the flank is only meaningful if the source protein is right.
#:
#: So an unresolved flank is a reason to drop the row, not to substitute a
#: different protein's. See `has_unresolved_flank` and, in `hitlist_source`,
#: `drop_unresolved_flank_rows`.
UNRESOLVED_RESIDUE = "X"

MAPPING_CATEGORY_SINGLE = "single"
MAPPING_CATEGORY_FLANKS_AGREE = "flanks_agree"
MAPPING_CATEGORY_WITHIN_GENE_CANONICAL = "within_gene_canonical"
MAPPING_CATEGORY_EXPRESSION_RESOLVED = "expression_resolved"
MAPPING_CATEGORY_CROSS_GENE_UNRESOLVED = "cross_gene_unresolved"
MAPPING_CATEGORY_WITHIN_GENE_UNRESOLVED = "within_gene_unresolved"
MAPPING_CATEGORY_UNMAPPED = "unmapped"

#: Whether each category means "we know which junction this peptide came
#: from". One table, because this fact was previously written down three
#: times -- once per branch of `select_source_mapping`, once as a set
#: comprehension in `hitlist_source`, and once as
#: `UNRESOLVED_MAPPING_CATEGORIES` -- and the three had already drifted:
#: the `hitlist_source` copy omitted `expression_resolved`, and `unmapped`
#: was in neither the resolved set nor the unresolved one.
MAPPING_CATEGORY_RESOLVED = {
    MAPPING_CATEGORY_SINGLE: True,
    MAPPING_CATEGORY_FLANKS_AGREE: True,
    MAPPING_CATEGORY_WITHIN_GENE_CANONICAL: True,
    MAPPING_CATEGORY_EXPRESSION_RESOLVED: True,
    MAPPING_CATEGORY_CROSS_GENE_UNRESOLVED: False,
    MAPPING_CATEGORY_WITHIN_GENE_UNRESOLVED: False,
    MAPPING_CATEGORY_UNMAPPED: False,
}

MAPPING_CATEGORIES = tuple(MAPPING_CATEGORY_RESOLVED)

RESOLVED_MAPPING_CATEGORIES = frozenset(
    category for category, resolved in MAPPING_CATEGORY_RESOLVED.items() if resolved
)

#: Categories whose junction is not known. `unmapped` belongs here: a peptide
#: with no source protein at all has no junction context, and masking it is
#: the same statement as masking an ambiguous one.
UNRESOLVED_MAPPING_CATEGORIES = frozenset(
    category for category, resolved in MAPPING_CATEGORY_RESOLVED.items() if not resolved
)

#: Production behavior and its explicit experimental comparator. The legacy
#: policy reproduces the old semantics--global canonical preference followed
#: by one arbitrary source--but makes the last-resort ordering deterministic.
#: The bulk path formerly fell through to frame order because it did not
#: project protein_id, so byte-for-byte historical replay is neither possible
#: nor desirable in a controlled comparison.
SOURCE_MAPPING_POLICY_MASK_UNRESOLVED = "mask_unresolved"
SOURCE_MAPPING_POLICY_LEGACY = "legacy_global_canonical"
SOURCE_MAPPING_POLICIES = (
    SOURCE_MAPPING_POLICY_MASK_UNRESOLVED,
    SOURCE_MAPPING_POLICY_LEGACY,
)


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
    n_genes: int = 0
    n_flank_pairs: int = 0
    category: str = ""
    flank_context_resolved: bool = False

    @property
    def is_ambiguous(self) -> bool:
        return self.n_candidates > 1 and not self.flanks_agree


def _flank_pair(mapping: Mapping[str, Any]) -> tuple:
    return (
        str(mapping.get("n_flank") or ""),
        str(mapping.get("c_flank") or ""),
    )


def has_unresolved_flank(mapping: Mapping[str, Any]) -> bool:
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
    genes = {str(row.get(gene_field) or "").strip() for row in rows} - {""}
    n_genes = len(genes)

    def _source_key(row: Mapping[str, Any]) -> str:
        return str(row.get("transcript_id") or row.get(protein_field) or "").strip()

    def _position(row: Mapping[str, Any]) -> float:
        try:
            return float(row.get("position"))
        except (TypeError, ValueError):
            return float("inf")

    def _usable_canonical(subset: Sequence[Mapping[str, Any]]):
        canonical_rows = [row for row in subset if _truthy(row.get("is_canonical_transcript"))]
        sources = {_source_key(row) for row in canonical_rows} - {""}
        pairs = {_flank_pair(row) for row in canonical_rows}
        if len(sources) == 1 and len(pairs) == 1:
            return canonical_rows[0]
        return None

    def _result(
        row: Mapping[str, Any],
        *,
        basis: str,
        category: str,
        expression_score: Optional[float] = None,
    ) -> FlankChoice:
        # Resolution follows from the category; it is not a second, separately
        # asserted fact that a call site could get wrong.
        return FlankChoice(
            mapping=row,
            n_candidates=len(rows),
            flanks_agree=agree,
            basis=basis,
            expression_score=expression_score,
            n_genes=n_genes,
            n_flank_pairs=len(distinct_flanks),
            category=category,
            flank_context_resolved=MAPPING_CATEGORY_RESOLVED[category],
        )

    # Deterministic baseline order, so every rule below breaks ties the same
    # way and a run is reproducible regardless of frame order.
    rows.sort(
        key=lambda row: (
            str(row.get(protein_field) or ""),
            str(row.get("transcript_id") or ""),
            _position(row),
            _flank_pair(row),
        )
    )

    if len(rows) == 1:
        return _result(
            rows[0],
            basis="deterministic_order",
            category=MAPPING_CATEGORY_SINGLE,
        )

    if agree:
        canonical = _usable_canonical(rows)
        return _result(
            canonical or rows[0],
            basis="canonical_transcript" if canonical else "deterministic_order",
            category=MAPPING_CATEGORY_FLANKS_AGREE,
        )

    if expression:
        scored = [
            (score, row)
            for row, score in (
                (row, expression.get(str(row.get(gene_field) or ""))) for row in rows
            )
            if score is not None
        ]
        if scored:
            best_score = max(score for score, _ in scored)
            best_genes = {
                str(row.get(gene_field) or "").strip()
                for score, row in scored
                if score == best_score
            } - {""}
            if len(best_genes) == 1:
                best_gene = next(iter(best_genes))
                gene_rows = [
                    row for row in rows if str(row.get(gene_field) or "").strip() == best_gene
                ]
                gene_pairs = {_flank_pair(row) for row in gene_rows}
                canonical = _usable_canonical(gene_rows)
                if len(gene_pairs) == 1 or canonical is not None:
                    return _result(
                        canonical or gene_rows[0],
                        basis="expression",
                        category=MAPPING_CATEGORY_EXPRESSION_RESOLVED,
                        expression_score=float(best_score),
                    )

    if n_genes == 1:
        canonical = _usable_canonical(rows)
        if canonical is not None:
            return _result(
                canonical,
                basis="canonical_transcript",
                category=MAPPING_CATEGORY_WITHIN_GENE_CANONICAL,
            )
        category = MAPPING_CATEGORY_WITHIN_GENE_UNRESOLVED
    elif n_genes > 1:
        category = MAPPING_CATEGORY_CROSS_GENE_UNRESOLVED
    else:
        category = MAPPING_CATEGORY_WITHIN_GENE_UNRESOLVED
    return _result(
        rows[0],
        basis="deterministic_order",
        category=category,
    )
