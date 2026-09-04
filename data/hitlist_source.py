"""Load Presto training records from the ``hitlist`` curated indexes.

``hitlist`` is the upstream curation layer for MHC ligand mass spectrometry and
in-vitro binding evidence (IEDB + CEDAR + supplementary deposits, harmonized
against curated per-sample metadata). It is preferred over the legacy
``merged_deduped.tsv`` path for one decisive reason:

    **it supplies flanking sequence.**

The merged TSV has no flank columns, so every Presto model trained through that
path saw a single ``<MISSING>`` token in the ``nflank`` / ``cflank`` segments
(``data/collate.py`` only tokenizes flanks when at least one sample has them).
``hitlist.generate_training_table(map_source_proteins=True)`` attaches
``n_flank`` / ``c_flank`` from ``peptide_mappings.parquet``, which is ~98%
populated. Junction context is the substrate for the antigen-processing
pathway, so this is a prerequisite for any excision modeling.

Coverage note: hitlist's training table covers **MS/elution and in-vitro
binding evidence only**. T-cell response, TCR evidence, and IEDB processing
exports are not in it, so those modalities still come from the merged TSV. See
``load_records_from_hitlist`` for what this module does and does not return.

The column contract
-------------------

What Presto asks hitlist for is public, because it is a cross-repository
interface rather than a detail of this module:

``SHARED_COLUMNS``
    Peptide, MHC context, provenance, flanks. Needed by every evidence family.
``BINDING_COLUMNS``, ``MS_COLUMNS``
    ``SHARED_COLUMNS`` plus what is specific to in-vitro binding and to
    elution, respectively. ``MS_COLUMNS`` is where sample provenance and
    antigen-processing state enter the model, and it carries the version
    constraint: the per-sample / study-level APM split needs hitlist >= 1.46.0.
``PROTEIN_MAPPING_COLUMNS``
    The subset that exists only under ``map_source_proteins=True``.
``training_columns(evidence, include_flanks=...)``
    The accessor. Use it rather than the constants directly, so the
    flank-dropping rule lives in exactly one place.
``assert_columns_present(frame, evidence, include_flanks=...)``
    The guard, run on every load.

Requiring these to be named from the outside is not ceremony. hitlist projects
columns by intersection, so asking for a column it no longer has is not an
error there and not an exception here -- the frame simply comes back narrower
and whatever was built from that column quietly becomes a constant. A rename
upstream is therefore invisible at the call site by default, which is why the
set is public, asserted on load, and pinned in
``tests/test_hitlist_source.py``.
"""

from __future__ import annotations

import random
import sys
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Tuple, Union

from .flank_selection import (
    MAPPING_CATEGORIES,
    MAPPING_CATEGORY_CROSS_GENE_UNRESOLVED,
    MAPPING_CATEGORY_FLANKS_AGREE,
    MAPPING_CATEGORY_SINGLE,
    MAPPING_CATEGORY_UNMAPPED,
    MAPPING_CATEGORY_WITHIN_GENE_CANONICAL,
    MAPPING_CATEGORY_WITHIN_GENE_UNRESOLVED,
    RESOLVED_MAPPING_CATEGORIES,
    UNRESOLVED_MAPPING_CATEGORIES,
    SOURCE_MAPPING_POLICIES,
    SOURCE_MAPPING_POLICY_LEGACY,
    SOURCE_MAPPING_POLICY_MASK_UNRESOLVED,
    UNRESOLVED_RESIDUE,
)
from .vocab import (
    apm_group_for_row,
    drop_unencodable_sequence,
    is_unmapped_condition,
    stimulus_for_condition,
    is_encodable_sequence,
    is_missing_scalar,
    normalize_sequence_text,
)
from .loaders import (
    BindingRecord,
    ElutionRecord,
    KineticsRecord,
    ProcessingRecord,
    StabilityRecord,
    TCellRecord,
    TcrEvidenceRecord,
)

# ---------------------------------------------------------------------------
# The hitlist column contract
#
# These names are the whole of what Presto asks hitlist for. They are public
# because they are an interface between two repositories, not an implementation
# detail of this module: an upstream rename or semantic change to any one of
# them lands here as silently wrong training data, so the set has to be
# nameable from the outside -- by tests, by a schema check, and by an issue
# filed against either repo.
#
# The projection is kept explicit rather than taking the whole frame because
# hitlist's training table is 95 columns wide and several million rows;
# projecting at the source is the difference between a few hundred MB and an
# OOM.
# ---------------------------------------------------------------------------

#: What every evidence family needs regardless of assay: the peptide, its MHC
#: context, where the row came from, and the flanking sequence that is the
#: reason this loader exists at all (see the module docstring).
SHARED_COLUMNS: Tuple[str, ...] = (
    "peptide",
    "mhc_restriction",
    "mhc_allele_set",
    "mhc_class",
    "host",
    "source_organism",
    "source",
    # Source-observation identity. These remain attached after mapping
    # expansion/collapse so a model prediction can be traced to the raw row.
    "assay_iri",
    "reference_iri",
    "pmid",
    "n_flank",
    "c_flank",
    # Offset of the peptide in its source protein. The only thing separating
    # "this peptide sits at the protein's own terminus" from "we never mapped
    # it" -- both otherwise arrive as a short or empty flank.
    "position",
    "evidence_row_id",
    # These identifiers are required by the collapse policy below. Without
    # them the advertised protein-id fallback is really input frame order, and
    # canonical status cannot be restricted to within-gene ambiguity.
    "gene_name",
    "gene_id",
    "protein_id",
    "transcript_id",
    "is_canonical_transcript",
    "proteome",
    "proteome_source",
)

#: `SHARED_COLUMNS` plus the in-vitro measurement itself. `response_measured`
#: carries IEDB's controlled vocabulary and decides which record type a row
#: becomes; see `_AFFINITY_RESPONSES` and friends below.
BINDING_COLUMNS: Tuple[str, ...] = SHARED_COLUMNS + (
    "response_measured",
    "quantitative_value",
    "measurement_units",
    "measurement_inequality",
    "assay_method",
)

#: `SHARED_COLUMNS` plus the biological sample an eluted peptide came out of.
#:
#: Sample provenance is requested as orthogonal axes rather than one flat
#: class. A single label derived from `cell_line_name` cannot express the space
#: that actually matters -- solid tissue, solid and haematological cancers,
#: donor blood, PBMC, sorted immune cells, and cell lines from all of those
#: origins. That space is a product of lineage x malignancy x immortalization x
#: site, and collapsing it put a primary AML blast and an AML cell line in the
#: same bucket, which is the distinction most likely to matter: lines drift and
#: routinely lose immunoproteasome and TAP. The four `src_*` booleans are
#: computed by hitlist at 100% coverage, versus 58.3% for `cell_line_name`.
#:
#: Tier 3 is cellular state. `condition_category` is the per-sample condition
#: and `apm_genes_perturbed` names the perturbed antigen-processing genes.
#: The study-level roll-up (`study_apm_perturbed` / `study_apm_genes`) is
#: deliberately not requested: it is ORed across a deposit, so a WT control
#: inside a knockout study inherits the flag and the arm contrast cancels.
#:
#: That split requires **hitlist >= 1.46.0** (pirl-unc/hitlist#353). Earlier
#: releases had no study-level column at all, because `apm_genes_perturbed`
#: *was* the ORed one -- see `presto#13`. Two further caveats are tracked
#: rather than fixed here: `apm_genes_perturbed` is empty both for a genuine
#: control and for a row whose arm could not be resolved (`presto#15`,
#: `pirl-unc/hitlist#392`), and the category vocabulary drifts with hitlist
#: releases (`presto#14`).
MS_COLUMNS: Tuple[str, ...] = SHARED_COLUMNS + (
    "cell_line_name",
    "source_tissue",
    "cell_type",
    "src_cell_line",
    "src_healthy_tissue",
    "src_cancer",
    "src_adjacent_to_tumor",
    "condition_category",
    "apm_genes_perturbed",
    # Arm identity. `apm_genes_perturbed` is only a *claim* where an arm was
    # actually resolved: hitlist emits an empty gene list both for a genuine
    # control and for a row it could not attribute (hitlist#392).
    # `sample_attribution` is what separates them, and it doubles as an
    # evidence tier -- `allele_exact` / `elution_conditions` are per-peptide,
    # while `class_pool` / `pmid_ambiguous` resolve only to a pool or a
    # deposit. `sample_label` is the biological sample, which is the right
    # grouping key for a leakage-safe split: peptide-disjoint splits do not
    # stop the same sample appearing on both sides. See presto#15.
    "sample_attribution",
    "is_control_arm",
    "sample_label",
)

#: Columns hitlist populates only under `map_source_proteins=True`, by joining
#: `peptide_mappings.parquet`.
#:
#: Subtracting them when flanks are skipped is hygiene, not necessity:
#: `hitlist.export._project_training_columns` intersects the request with what
#: the frame actually has, so an unavailable column is dropped in silence
#: rather than raising. Keeping the request honest about what it expects is
#: what lets `assert_columns_present` tell a deliberate omission apart from an
#: upstream rename.
PROTEIN_MAPPING_COLUMNS: FrozenSet[str] = frozenset(
    # `position` is what separates "this peptide sits at the protein's own
    # terminus" from "we never mapped this peptide". Both arrive as a short or
    # empty flank; only the first is a fact about the biology. See
    # HITLIST_FLANK_WIDTH.
    {
        "n_flank",
        "c_flank",
        "gene_name",
        "gene_id",
        "protein_id",
        "transcript_id",
        "is_canonical_transcript",
        "position",
        "proteome",
        "proteome_source",
    }
)

#: The `include_evidence` value hitlist expects, mapped to what we ask it for.
COLUMNS_BY_EVIDENCE: Dict[str, Tuple[str, ...]] = {
    "binding": BINDING_COLUMNS,
    "ms": MS_COLUMNS,
}


#: Oldest hitlist whose column semantics match what this module assumes.
#:
#: 1.53.0 for the arm-attribution columns (`sample_attribution`,
#: `is_control_arm`, `sample_label`), which landed across
#: pirl-unc/hitlist#354/#356/#367 between 1.46.0 and 1.53.0. Without them a row
#: whose arm could not be resolved is indistinguishable from a control.
#:
#: Before 1.46.0 (pirl-unc/hitlist#353) there was no study-level APM column,
#: because `apm_genes_perturbed` *was* the study-level roll-up: it ORed the
#: parent study's knockout panel onto every sample, so a WT control inside a KO
#: study carried the KO flag. This module reads `apm_genes_perturbed` as the
#: *per-sample* truth, which is only correct from 1.46.0 on.
#:
#: The column set resolves on 1.41 as well, so an older install trains happily
#: and reports nothing -- 816,023 observations (18.4%) carry the wrong
#: perturbation label, 716,992 of them genuinely-unperturbed rows wearing their
#: study's panel. That is why this is an assertion and not a comment.
MINIMUM_HITLIST_VERSION = (1, 53, 0)


def _parse_version(raw: Optional[str]) -> Optional[Tuple[int, ...]]:
    """Leading numeric components of a version string, or None if unreadable."""
    if not raw:
        return None
    parts: List[int] = []
    for chunk in str(raw).split(".")[:3]:
        digits = ""
        for character in chunk:
            if not character.isdigit():
                break
            digits += character
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts) or None


def require_supported_hitlist(raw_version: Optional[str]) -> None:
    """Fail loudly on a hitlist too old for the semantics this module assumes.

    An unreadable or absent version is allowed through: editable installs and
    source checkouts often have no usable `__version__`, and refusing to run
    there would be worse than the risk. A version we *can* read and that is too
    old is a hard error, because the failure it causes is silent.
    """
    parsed = _parse_version(raw_version)
    if parsed is None:
        return
    if parsed < MINIMUM_HITLIST_VERSION:
        wanted = ".".join(str(part) for part in MINIMUM_HITLIST_VERSION)
        raise RuntimeError(
            f"hitlist {raw_version} is too old for this loader; need >= {wanted}. "
            "Before 1.46.0 `apm_genes_perturbed` was ORed across a study, so "
            "genuinely unperturbed control rows carry their study's knockout "
            "panel -- 18.4% of observations on the corpus this was measured "
            "against. Before 1.53.0 the arm-attribution columns do not exist, "
            "so a row whose arm could not be resolved is indistinguishable "
            "from a control. Training would succeed and the labels would be "
            "wrong either way. "
            "Upgrade with `pip install -U 'hitlist>=" + wanted + "'`."
        )


#: Residues hitlist extracts on each side when a peptide is mapped.
#:
#: `hitlist.DEFAULT_FLANK`, 15 as of 1.55.2 (it was 10 before). A *mapped* row
#: whose flank is shorter than this ran out of protein -- the peptide sits at
#: the protein's N- or C-terminus -- rather than being unmapped. That is the
#: whole basis for distinguishing <TERMINUS> from <MISSING>, so
#: `tests/test_terminus_context.py` checks the assumption against live data
#: instead of trusting this number.
HITLIST_FLANK_WIDTH = 15


def flank_context(
    flank: Optional[str], position: Optional[float], *, width: int = HITLIST_FLANK_WIDTH
) -> Tuple[str, bool]:
    """``(sequence, is_terminus)`` for one side of a peptide.

    A flank shorter than ``width`` on a *mapped* row means the protein ended
    there. On an unmapped row (no ``position``) a short flank means only that
    nothing is known, which is a different statement and must not be encoded
    as though the protein terminated.
    """
    raw = normalize_sequence_text(flank)
    text = drop_unencodable_sequence(flank)
    position_text = str(position).strip() if position is not None else ""
    mapped = (
        position is not None
        and position_text not in {"", "nan", "NaN", "<NA>", "None"}
        and position == position  # NaN-safe
    )
    # Measured on the RAW length, not the cleaned one. `drop_unencodable_sequence`
    # blanks a flank entirely when it carries one unrepresentable residue
    # (selenocysteine, annotation junk), and a blanked flank on a mapped row
    # would otherwise be read as "the protein ended here" -- inventing a
    # terminus out of a tokenizer limitation.
    return text, bool(mapped and not is_missing_scalar(flank) and len(raw) < int(width))


def _flank_fields(row) -> Dict[str, Any]:
    """The four flank fields for a record, from one mapping row."""
    # Keep the selected mapping position as provenance even when the policy
    # deliberately hides its junction context from the model.  The dedicated
    # marker controls terminus inference; overloading a missing `position`
    # erased useful lineage and made otherwise mapped rows untraceable.
    context_masked = str(row.get("source_mapping_context_masked", False)).lower() == "true"
    position = None if context_masked else row.get("position")
    n_text, n_terminus = flank_context(row.get("n_flank"), position)
    c_text, c_terminus = flank_context(row.get("c_flank"), position)
    return {
        "flank_n": n_text,
        "flank_c": c_text,
        "flank_n_is_terminus": n_terminus,
        "flank_c_is_terminus": c_terminus,
    }


def _mapping_fields(row) -> Dict[str, Any]:
    """Row-level mapping diagnostics carried to records and prediction dumps."""

    def _count(name: str) -> int:
        value = row.get(name, 0)
        try:
            number = float(value)
        except (TypeError, ValueError):
            return 0
        return int(number) if number == number else 0

    def _flag(name: str) -> bool:
        # NaN-safe like `_count`. An unmatched join leaves an object column
        # holding NaN, and `bool(float("nan"))` is True -- which would stamp a
        # row as having a resolved junction precisely when nothing is known
        # about it, the opposite of the safe default.
        value = row.get(name, False)
        return bool(value) if value == value else False

    return {
        "source_mapping_category": _clean(row.get("source_mapping_category")),
        "source_mapping_n_candidates": _count("source_mapping_n_candidates"),
        "source_mapping_n_genes": _count("source_mapping_n_genes"),
        "source_mapping_n_flank_pairs": _count("source_mapping_n_flank_pairs"),
        "flank_context_resolved": _flag("flank_context_resolved"),
    }


def _lineage_fields(row) -> Dict[str, Any]:
    """Stable observation and selected-mapping identity for one record."""

    position = row.get("position")
    try:
        numeric_position = float(position)
    except (TypeError, ValueError):
        mapping_position = None
    else:
        mapping_position = int(numeric_position) if numeric_position == numeric_position else None

    canonical = row.get("is_canonical_transcript")
    if canonical is None or _clean(canonical) == "":
        canonical_value = None
    else:
        canonical_value = str(canonical).strip().lower() in {"1", "1.0", "true", "yes", "t"}

    return {
        "evidence_row_id": _clean(row.get("evidence_row_id")),
        "assay_iri": _clean(row.get("assay_iri")),
        "reference_iri": _clean(row.get("reference_iri")),
        "pmid": _clean(row.get("pmid")),
        "mapping_gene_name": _clean(row.get("gene_name")),
        "mapping_gene_id": _clean(row.get("gene_id")),
        "mapping_protein_id": _clean(row.get("protein_id")),
        "mapping_transcript_id": _clean(row.get("transcript_id")),
        "mapping_position": mapping_position,
        "mapping_proteome": _clean(row.get("proteome")),
        "mapping_proteome_source": _clean(row.get("proteome_source")),
        "mapping_is_canonical_transcript": canonical_value,
    }


def training_columns(evidence: str, *, include_flanks: bool) -> List[str]:
    """Columns to request from ``hitlist.generate_training_table``.

    Parameters
    ----------
    evidence
        A key of :data:`COLUMNS_BY_EVIDENCE` -- ``"binding"`` or ``"ms"``.
        These are hitlist's own ``include_evidence`` values, so the caller
        passes the same string to both arguments.
    include_flanks
        Whether the caller is running with ``map_source_proteins=True``. When
        False, :data:`PROTEIN_MAPPING_COLUMNS` are dropped, because hitlist
        only materializes them via the protein-mapping join.

    Returns
    -------
    list of str
        A fresh list, safe for the caller to mutate. The module constants are
        tuples so that a caller cannot append to the contract by accident.

    Raises
    ------
    KeyError
        If ``evidence`` is not a known family. Deliberately not tolerant: a
        typo here would silently project the wrong columns for a whole
        training run.
    """
    columns = COLUMNS_BY_EVIDENCE[evidence]
    if include_flanks:
        return list(columns)
    return [c for c in columns if c not in PROTEIN_MAPPING_COLUMNS]


def assert_columns_present(frame, evidence: str, *, include_flanks: bool) -> None:
    """Fail loudly if hitlist did not return every column we asked for.

    hitlist projects with an intersection -- ``_project_training_columns``
    keeps ``[c for c in requested if c in df.columns]`` -- so a column that has
    been renamed, moved behind a flag, or dropped upstream does not raise. It
    just is not there, and the feature built from it silently becomes a
    constant. That is the failure mode this whole contract exists to catch, so
    it is worth one set difference per load.

    Raises
    ------
    RuntimeError
        Naming the missing columns and the installed hitlist version, which is
        almost always the thing that changed.
    """
    expected = set(training_columns(evidence, include_flanks=include_flanks))
    missing = sorted(expected - set(frame.columns))
    if not missing:
        return
    # Read the version off the already-imported module rather than importing
    # it: hitlist is an optional extra, and anything holding a frame to check
    # has necessarily imported it already. A bare `import hitlist` here would
    # raise ModuleNotFoundError over the top of the real diagnostic wherever
    # the frame came from somewhere else -- a fixture, a cache, a stub.
    hitlist = sys.modules.get("hitlist")
    version = getattr(hitlist, "__version__", "unknown") if hitlist else "not imported"

    raise RuntimeError(
        f"hitlist returned {len(frame.columns)} columns for evidence={evidence!r} "
        f"but {len(missing)} requested column(s) are absent: {', '.join(missing)}. "
        f"Installed hitlist is {version}; these "
        "columns are dropped silently by hitlist's projection, so this is most "
        "likely an upstream rename. Reconcile MS_COLUMNS / BINDING_COLUMNS "
        "against the current hitlist export before training on this frame."
    )


# ``response_measured`` is the same controlled vocabulary the merged TSV carries
# in ``value_type``, so these strings flow through to
# ``BindingRecord.measurement_type`` unchanged and land in the existing assay
# families used by ``data/collate.py::_categorize_binding_assay_type``.
_AFFINITY_RESPONSES = frozenset(
    {
        "half maximal inhibitory concentration (IC50)",
        "half maximal effective concentration (EC50)",
        "dissociation constant KD",
        "dissociation constant KD (~IC50)",
        "dissociation constant KD (~EC50)",
    }
)
_T_HALF_RESPONSES = frozenset({"half life"})
_TM_RESPONSES = frozenset({"50% dissociation temperature"})
_KOFF_RESPONSES = frozenset({"off rate"})
_KON_RESPONSES = frozenset({"on rate"})

# Units are consistent per response family in the current index (every affinity
# response is nM, half life is min, Tm is degrees C). Guard anyway: a unit swap
# upstream would otherwise be ingested silently at the wrong scale, which is
# invisible in the loss and very expensive to debug later.
_EXPECTED_UNITS = {
    **{response: "nM" for response in _AFFINITY_RESPONSES},
    "half life": "min",
    "50% dissociation temperature": "°C",
}

# Everything else (qualitative binding, ligand presentation, MHC binding, 3D
# structure) carries no numeric value. The merged-TSV loader drops those rows
# (``if value is None: continue``), so we drop them too and count them, rather
# than silently changing the training contract while migrating sources.
_MINUTES_PER_HOUR = 60.0


def _qualifier_from_inequality(text: Any) -> int:
    """Map hitlist's ``measurement_inequality`` to Presto's censor code."""
    token = _clean(text)
    if token.startswith("<"):
        return -1
    if token.startswith(">"):
        return 1
    return 0


def _split_allele_set(value: Any) -> List[str]:
    """Split a hitlist allele-set cell into a list of allele names."""
    if isinstance(value, (list, tuple, set)):
        return [token for item in value if (token := _clean(item))]
    token = _clean(value)
    if not token:
        return []
    for sep in (";", ","):
        if sep in token:
            return [part.strip() for part in token.split(sep) if part.strip()]
    return [token]


def _clean(value: Any) -> str:
    """Normalize a possibly-NaN cell to a plain string."""
    if value is None:
        return ""
    token = str(value)
    if token in {"nan", "NaN", "<NA>", "None"}:
        return ""
    return token.strip()


def _text_column(frame, name: str):
    """Whitespace-normalized string column, or an empty aligned series."""
    import pandas as pd

    if name not in frame.columns:
        return pd.Series("", index=frame.index, dtype=str)
    return frame[name].fillna("").astype(str).str.strip()


def _canonical_mask(frame):
    """Boolean canonical flag tolerant of bool and serialized fixtures."""
    if "is_canonical_transcript" not in frame.columns:
        return _text_column(frame, "is_canonical_transcript").eq("true")
    values = frame["is_canonical_transcript"]
    if str(values.dtype) in {"bool", "boolean"}:
        return values.fillna(False).astype(bool)
    # Numeric or textual, whichever the column turns out to carry. A bool
    # column with nulls arrives from parquet or CSV as float64, and
    # stringifying that yields "1.0", which is in no truthy set -- so every
    # canonical row would read as non-canonical and silently degrade
    # `within_gene_canonical` to `within_gene_unresolved`. Taking the union
    # rather than branching also keeps a mixed object column correct, and
    # keeps this agreeing with the scalar `flank_selection._truthy`.
    import pandas as pd

    numeric = pd.to_numeric(values, errors="coerce").fillna(0).ne(0)
    # Converting first maps missing object values to non-truthy strings and
    # avoids pandas' deprecated silent downcast in object.fillna(False/"").
    text = values.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "t"})
    return numeric | text


def _mapping_present_mask(frame):
    """Rows produced by a real peptide mapping rather than a left-join miss."""
    import pandas as pd

    present = pd.Series(False, index=frame.index)
    if "position" in frame.columns:
        present = present | frame["position"].notna()
    for column in (
        "gene_name",
        "gene_id",
        "protein_id",
        "transcript_id",
        "n_flank",
        "c_flank",
    ):
        if column in frame.columns:
            present = present | _text_column(frame, column).ne("")
    return present


def _mapping_resolution_summary(frame):
    """One row of source-junction evidence metadata per evidence row.

    Candidate multiplicity is not itself ambiguity: most multi-mappings agree
    on both flanks. Disagreement is resolved only when it stays within one
    known gene and exactly one canonical source contributes exactly one flank
    pair. Canonical status is deliberately not allowed to choose between
    genes, because every gene normally contributes its own canonical source.
    """
    import pandas as pd

    if "evidence_row_id" not in frame.columns or not len(frame):
        return pd.DataFrame(
            columns=[
                "source_mapping_n_candidates",
                "source_mapping_n_genes",
                "source_mapping_n_flank_pairs",
                "source_mapping_category",
                "flank_context_resolved",
            ]
        )

    evidence_ids = frame["evidence_row_id"]
    summary = pd.DataFrame(index=pd.Index(evidence_ids.drop_duplicates(), name="evidence_row_id"))
    summary["source_mapping_n_candidates"] = 0
    summary["source_mapping_n_genes"] = 0
    summary["source_mapping_n_flank_pairs"] = 0
    summary["source_mapping_n_canonical_sources"] = 0
    summary["source_mapping_n_canonical_flank_pairs"] = 0

    present = _mapping_present_mask(frame)
    if present.any():
        mapped = frame.loc[present].copy()
        mapped["_gene_key"] = _text_column(mapped, "gene_name")
        gene_id = _text_column(mapped, "gene_id")
        mapped.loc[mapped["_gene_key"].eq(""), "_gene_key"] = gene_id
        mapped["_gene_key"] = mapped["_gene_key"].where(mapped["_gene_key"].ne(""), None)
        mapped["_flank_pair_key"] = (
            _text_column(mapped, "n_flank") + "\x1f" + _text_column(mapped, "c_flank")
        )
        grouped = mapped.groupby("evidence_row_id", sort=False, dropna=False)
        mapped_summary = pd.DataFrame(
            {
                "source_mapping_n_candidates": grouped.size(),
                "source_mapping_n_genes": grouped["_gene_key"].nunique(dropna=True),
                "source_mapping_n_flank_pairs": grouped["_flank_pair_key"].nunique(dropna=False),
            }
        )
        summary.update(mapped_summary)

        canonical = mapped.loc[_canonical_mask(mapped)].copy()
        if len(canonical):
            canonical["_source_key"] = _text_column(canonical, "transcript_id")
            protein_id = _text_column(canonical, "protein_id")
            canonical.loc[canonical["_source_key"].eq(""), "_source_key"] = protein_id
            canonical["_source_key"] = canonical["_source_key"].where(
                canonical["_source_key"].ne(""), None
            )
            canonical_grouped = canonical.groupby("evidence_row_id", sort=False, dropna=False)
            canonical_summary = pd.DataFrame(
                {
                    "source_mapping_n_canonical_sources": canonical_grouped["_source_key"].nunique(
                        dropna=True
                    ),
                    "source_mapping_n_canonical_flank_pairs": canonical_grouped[
                        "_flank_pair_key"
                    ].nunique(dropna=False),
                }
            )
            summary.update(canonical_summary)

    for column in (
        "source_mapping_n_candidates",
        "source_mapping_n_genes",
        "source_mapping_n_flank_pairs",
        "source_mapping_n_canonical_sources",
        "source_mapping_n_canonical_flank_pairs",
    ):
        summary[column] = summary[column].fillna(0).astype("int64")

    n_candidates = summary["source_mapping_n_candidates"]
    n_genes = summary["source_mapping_n_genes"]
    n_pairs = summary["source_mapping_n_flank_pairs"]
    usable_canonical = (summary["source_mapping_n_canonical_sources"] == 1) & (
        summary["source_mapping_n_canonical_flank_pairs"] == 1
    )

    category = pd.Series(
        MAPPING_CATEGORY_WITHIN_GENE_UNRESOLVED,
        index=summary.index,
        dtype=str,
    )
    category.loc[n_candidates == 0] = MAPPING_CATEGORY_UNMAPPED
    category.loc[n_candidates == 1] = MAPPING_CATEGORY_SINGLE
    category.loc[(n_candidates > 1) & (n_pairs == 1)] = MAPPING_CATEGORY_FLANKS_AGREE
    category.loc[(n_pairs > 1) & (n_genes > 1)] = MAPPING_CATEGORY_CROSS_GENE_UNRESOLVED
    category.loc[(n_pairs > 1) & (n_genes == 1) & usable_canonical] = (
        MAPPING_CATEGORY_WITHIN_GENE_CANONICAL
    )
    summary["source_mapping_category"] = category
    # One table, in `flank_selection`. Re-listing the resolved categories here
    # is how this copy came to omit `expression_resolved`.
    summary["flank_context_resolved"] = category.isin(RESOLVED_MAPPING_CATEGORIES)
    return summary


def _mapping_stats_from_summary(summary) -> Dict[str, Any]:
    if not len(summary):
        return {
            "evidence_rows": 0,
            "rows_with_multiple_proteins": 0,
            "rows_with_disagreeing_flanks": 0,
            "max_proteins_for_one_row": 0,
            "rows_with_resolved_flank_context": 0,
            "category_counts": {},
        }
    n_candidates = summary["source_mapping_n_candidates"]
    n_pairs = summary["source_mapping_n_flank_pairs"]
    return {
        "evidence_rows": int(len(summary)),
        "rows_with_multiple_proteins": int((n_candidates > 1).sum()),
        "rows_with_disagreeing_flanks": int((n_pairs > 1).sum()),
        "max_proteins_for_one_row": int(n_candidates.max()),
        "rows_with_resolved_flank_context": int(summary["flank_context_resolved"].sum()),
        "category_counts": {
            str(category): int(count)
            for category, count in summary["source_mapping_category"].value_counts().items()
        },
    }


def _collapse_source_mappings(
    frame,
    *,
    source_mapping_policy: str = SOURCE_MAPPING_POLICY_MASK_UNRESOLVED,
):
    """Collapse mappings under one explicit junction-context policy.

    Both policies classify rows identically. ``mask_unresolved`` clears
    genuinely ambiguous junctions to an absent flank; ``legacy_global_canonical``
    retains the historical global-canonical / arbitrary-source semantics as an
    experiment-only comparator, with a stable candidate order replacing the
    former frame-order accident.
    """
    if source_mapping_policy not in SOURCE_MAPPING_POLICIES:
        raise ValueError(
            f"source_mapping_policy must be one of {SOURCE_MAPPING_POLICIES}; "
            f"got {source_mapping_policy!r}"
        )
    if "evidence_row_id" not in frame.columns:
        return frame, _mapping_stats_from_summary(_mapping_resolution_summary(frame))

    summary = _mapping_resolution_summary(frame)
    # A real mapping outranks a left-join miss. Without this key the blank row
    # wins: every identifier is `""`, which sorts first ascending, so an
    # evidence row that mapped to exactly one protein could still collapse to
    # the empty candidate and then be stamped `single` / resolved by a summary
    # that had excluded that row from classification. Sorting rather than
    # filtering keeps an evidence row whose candidates are *all* misses.
    ordered = frame.assign(
        _mapping_present=_mapping_present_mask(frame),
        _canonical_priority=_canonical_mask(frame),
    )
    sort_columns = ["_mapping_present", "_canonical_priority"]
    ascending = [False, False]
    for column in (
        "gene_name",
        "gene_id",
        "protein_id",
        "transcript_id",
        "position",
        "n_flank",
        "c_flank",
    ):
        if column in ordered.columns:
            sort_columns.append(column)
            ascending.append(True)
    ordered = ordered.sort_values(
        sort_columns,
        ascending=ascending,
        kind="stable",
        na_position="last",
    )
    collapsed = ordered.drop_duplicates(subset=["evidence_row_id"], keep="first").drop(
        columns=["_mapping_present", "_canonical_priority"]
    )
    collapsed = collapsed.join(summary, on="evidence_row_id", validate="one_to_one")

    if source_mapping_policy == SOURCE_MAPPING_POLICY_MASK_UNRESOLVED:
        collapsed = _mask_unresolved_mapping_context(collapsed)
    stats = _mapping_stats_from_summary(summary)
    stats["source_mapping_policy"] = source_mapping_policy
    return collapsed, stats


def _mask_unresolved_mapping_context(frame):
    """Clear unresolved selected junctions without changing row membership.

    An unresolved junction is written as an *absent* flank, not as a sentinel
    residue. The model already distinguishes the three states from the tensor
    side: real residues tokenize normally, a terminus pads with `X`, and an
    absent flank pads with `?` (`Presto._pad_for_side`). Writing a marker into
    the sequence string was a third, redundant channel for the same fact -- and
    the one the collator's optional-sequence cleaner erased anyway.

    Returns a new frame. The caller passes the output of
    `drop_unresolved_flank_rows`, which is a slice when rows were dropped and
    the caller's own object when none were, so mutating in place would either
    be chained assignment or a silent rewrite of the caller's data.
    """
    if "source_mapping_category" not in frame.columns:
        # `_collapse_source_mappings` returns the frame untouched when there is
        # no `evidence_row_id` to group on, so there is nothing to mask.
        return frame
    masked = frame.copy()
    unresolved = masked["source_mapping_category"].isin(UNRESOLVED_MAPPING_CATEGORIES)
    masked["source_mapping_context_masked"] = False
    for column in ("n_flank", "c_flank"):
        if column in masked.columns:
            masked.loc[unresolved, column] = ""
    # Do not erase the selected mapping position: it is source lineage.  This
    # separate marker tells `_flank_fields` that the cleared context is absent,
    # rather than a true protein terminus, without changing provenance.
    masked.loc[unresolved, "source_mapping_context_masked"] = True
    return masked


def _select_best_mapping(frame, *, source_mapping_policy):
    """The collapsed frame under one explicit policy.

    The policy is required rather than defaulted. `load_records_from_hitlist`
    deliberately collapses under the legacy policy, drops rows whose selected
    flank carries an `X`, and only then masks -- because masking first would
    clear a selected `X` before the filter sees it, so the masked arm would
    retain rows the legacy arm drops and the comparison would stop being
    paired. A caller that got `mask_unresolved` by default and then ran the
    filter would silently reproduce that confound.
    """
    collapsed, _ = _collapse_source_mappings(frame, source_mapping_policy=source_mapping_policy)
    return collapsed


def unresolved_flank_mask(frame):
    """Rows whose chosen flanks carry an unresolved residue (`X`).

    Computed on the flanks as hitlist emits them, before any cleaning, so a
    flank blanked for some other reason is not mistaken for a resolved one.
    """
    import pandas as pd

    present = [column for column in ("n_flank", "c_flank") if column in frame.columns]
    if not present:
        return pd.Series(False, index=frame.index)
    mask = pd.Series(False, index=frame.index)
    for column in present:
        text = frame[column].fillna("").astype(str)
        mask = mask | text.str.contains(UNRESOLVED_RESIDUE, regex=False)
    return mask


def drop_unresolved_flank_rows(frame):
    """Discard rows whose chosen flank has a residue of unknown identity.

    `X` means "a residue is here and we do not know which". hitlist emits it
    where a transcript's 5' CDS is incomplete, so the first codon is partial;
    the rows are overwhelmingly non-canonical models (44,485 non-canonical vs
    1,507 canonical, ~10x the corpus-wide ratio). The mechanism is the standard
    explanation and fits the evidence, but the CDS-completeness flag is not
    exposed in the training table, so the correlation is verified and the cause
    is not.

    Dropping rather than substituting is the point. The obvious alternative --
    prefer whichever candidate protein has a clean flank -- reattributes the
    peptide to a **different gene** in 91% of the rows it changes, which is a
    worse error than an honest unknown: a flank only means anything if the
    source protein is right.

    The cost is small enough to state exactly, measured on hitlist 1.55.2 after
    the collapse: **862 of 4,418,352** MS evidence rows (0.0195%) and **86 of
    891,685** binding rows (0.0096%). Both verified to leave zero `X` in the
    surviving flanks. In exchange `X` stops appearing in training data at all,
    leaving it free to mean an unknown residue and nothing else.
    """
    if not {"n_flank", "c_flank"} & set(frame.columns):
        return frame, 0
    mask = unresolved_flank_mask(frame)
    dropped = int(mask.sum())
    if not dropped:
        return frame, 0
    return frame.loc[~mask], dropped


def mapping_ambiguity_stats(frame) -> Dict[str, Any]:
    """How much of the flank context is a choice rather than an observation.

    Computed before the collapse, because afterwards the alternatives are gone.
    Reported in the ingest stats so the number is visible in a training log
    rather than rediscovered by measuring the corpus.
    """
    return _mapping_stats_from_summary(_mapping_resolution_summary(frame))


def _iter_row_dicts(frame, chunk_size: int = 50_000):
    """Yield row dicts without materializing the whole frame at once.

    `frame.to_dict("records")` builds one dict per row for the entire table
    before the loop starts -- on the full corpus that is ~750k dicts held
    simultaneously, which undoes the column pruning done upstream to keep the
    load within memory. Chunking caps the peak at `chunk_size` dicts while
    keeping the `.get()` access the callers are written against.
    """
    n_rows = len(frame)
    for start in range(0, n_rows, chunk_size):
        chunk = frame.iloc[start : start + chunk_size]
        for row in chunk.to_dict("records"):
            yield row


# Peptide and flank admissibility both defer to the tokenizer vocab via
# `presto.data.vocab`, so there is one source of truth rather than a second,
# subtly different residue set here. An earlier version of this guard used the
# 20 canonical residues and so rejected `X`, which the tokenizer represents
# perfectly well -- needlessly dropping ~51 usable elution rows.
def normalize_ingested_peptide(peptide: Optional[str]) -> str:
    """Upper-case a peptide and return "" if it cannot be tokenized.

    Upper-casing first is the point: the tokenizer upper-cases internally, so a
    lowercase peptide encodes perfectly well, but testing the raw string
    against an upper-case residue set rejected it and dropped the whole row.
    The flank path already normalized case; peptides did not, so the two
    disagreed about identical input.

    Peptides are targets, so an unrepresentable residue drops the row: unlike
    a flank, the epitope cannot degrade to "absent".
    """
    text = normalize_sequence_text(peptide)
    return text if is_encodable_sequence(text) else ""


def load_records_from_hitlist(
    *,
    max_binding: Optional[int] = None,
    max_stability: Optional[int] = None,
    max_kinetics: Optional[int] = None,
    max_elution: Optional[int] = None,
    mhc_class: Optional[str] = None,
    species: Optional[str] = None,
    mhc_allele: Optional[Union[str, Sequence[str]]] = None,
    length_min: int = 7,
    length_max: int = 30,
    include_flanks: bool = True,
    source_mapping_policy: str = SOURCE_MAPPING_POLICY_MASK_UNRESOLVED,
    sampling_seed: int = 42,
) -> Tuple[
    List[BindingRecord],
    List[KineticsRecord],
    List[StabilityRecord],
    List[ProcessingRecord],
    List[ElutionRecord],
    List[TCellRecord],
    List[TcrEvidenceRecord],
    Dict[str, Any],
]:
    """Build Presto records from the hitlist curated indexes.

    Returns the same 8-tuple shape as
    ``scripts/train_iedb.py::load_records_from_merged_tsv`` so the two sources
    are interchangeable at the call site.

    Modalities covered here: ``binding``, ``stability``, ``kinetics``,
    ``elution``. Returned empty because hitlist's training table does not carry
    them: ``processing``, ``tcell``, ``tcr_evidence`` — callers that need those
    should union with the merged-TSV loader.

    ``include_flanks=False`` skips the protein-mapping join, which is much
    faster and is how the Stage 0a parity check reproduces merged-TSV behavior.
    ``source_mapping_policy`` controls only disagreeing mapped junctions; it
    never changes the assay observation or target.
    """
    try:
        import hitlist
    except ImportError as exc:  # pragma: no cover - exercised via install docs
        raise SystemExit(
            "hitlist is required for --data-source hitlist. Install it with "
            "`pip install hitlist`, or use --data-source merged_tsv."
        ) from exc

    require_supported_hitlist(getattr(hitlist, "__version__", None))
    if source_mapping_policy not in SOURCE_MAPPING_POLICIES:
        raise ValueError(
            f"source_mapping_policy must be one of {SOURCE_MAPPING_POLICIES}; "
            f"got {source_mapping_policy!r}"
        )

    shared_filters = dict(
        mhc_class=mhc_class,
        species=species,
        mhc_allele=list(mhc_allele) if isinstance(mhc_allele, (list, tuple)) else mhc_allele,
        length_min=length_min,
        length_max=length_max,
        map_source_proteins=include_flanks,
    )

    # Empty unless flanks were requested; the ambiguity only exists in the
    # exploded protein-mapping frame.
    mapping_ambiguity: Dict[str, Any] = {}

    binding_frame = hitlist.generate_training_table(
        include_evidence="binding",
        columns=training_columns("binding", include_flanks=include_flanks),
        **shared_filters,
    )
    ms_frame = hitlist.generate_training_table(
        include_evidence="ms",
        columns=training_columns("ms", include_flanks=include_flanks),
        **shared_filters,
    )
    assert_columns_present(binding_frame, "binding", include_flanks=include_flanks)
    assert_columns_present(ms_frame, "ms", include_flanks=include_flanks)
    source_rows_before_mapping_collapse = {
        "binding": len(binding_frame),
        "ms": len(ms_frame),
    }

    if include_flanks:
        # Select and filter before either experimental policy transforms the
        # flanks. Otherwise masking turns a selected X into ? before the X
        # filter runs, so the masked arm retains rows the legacy arm drops and
        # the comparison is no longer paired. Classification and collapse still
        # share one groupby pass per frame.
        binding_frame, binding_mapping_stats = _collapse_source_mappings(
            binding_frame, source_mapping_policy=SOURCE_MAPPING_POLICY_LEGACY
        )
        ms_frame, ms_mapping_stats = _collapse_source_mappings(
            ms_frame, source_mapping_policy=SOURCE_MAPPING_POLICY_LEGACY
        )
        # Only after the collapse: a row is dropped for the flank it actually
        # trains on, not for one some rejected candidate happened to carry.
        binding_frame, binding_unresolved = drop_unresolved_flank_rows(binding_frame)
        ms_frame, ms_unresolved = drop_unresolved_flank_rows(ms_frame)
        if source_mapping_policy == SOURCE_MAPPING_POLICY_MASK_UNRESOLVED:
            binding_frame = _mask_unresolved_mapping_context(binding_frame)
            ms_frame = _mask_unresolved_mapping_context(ms_frame)
        binding_mapping_stats["source_mapping_policy"] = source_mapping_policy
        ms_mapping_stats["source_mapping_policy"] = source_mapping_policy
        mapping_ambiguity.update(
            {
                "binding": binding_mapping_stats,
                "ms": ms_mapping_stats,
            }
        )
        mapping_ambiguity["rows_dropped_unresolved_flank"] = {
            "binding": binding_unresolved,
            "ms": ms_unresolved,
        }

    binding_records: List[BindingRecord] = []
    kinetics_records: List[KineticsRecord] = []
    stability_records: List[StabilityRecord] = []
    elution_records: List[ElutionRecord] = []

    skipped_no_value = 0
    skipped_unroutable = 0
    skipped_bad_unit = 0
    skipped_noncanonical_peptide = 0
    skipped_missing_mhc_allele = 0
    unmapped_conditions: Dict[str, int] = {}

    for row in _iter_row_dicts(binding_frame):
        response = _clean(row.get("response_measured"))
        raw_value = row.get("quantitative_value")
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            value = None
        # NB: float(nan) succeeds, so an explicit NaN test is required here.
        # Without it, missing measurements silently enter training as NaN
        # targets and poison the loss.
        if value is None or value != value:
            # Qualitative binding / ligand presentation / structural rows.
            # Dropped for parity with the merged-TSV contract.
            skipped_no_value += 1
            continue

        expected_unit = _EXPECTED_UNITS.get(response)
        if expected_unit is not None:
            unit = _clean(row.get("measurement_units"))
            if unit != expected_unit:
                skipped_bad_unit += 1
                continue

        peptide = normalize_ingested_peptide(row.get("peptide"))
        if not peptide:
            skipped_noncanonical_peptide += 1
            continue

        allele = _clean(row.get("mhc_restriction"))
        allele_set = _split_allele_set(row.get("mhc_allele_set"))
        qualifier = _qualifier_from_inequality(row.get("measurement_inequality"))
        assay_method = _clean(row.get("assay_method")) or None
        common = dict(
            peptide=peptide,
            mhc_allele=allele,
            mhc_class=_clean(row.get("mhc_class")) or None,
            species=_clean(row.get("host")) or None,
            antigen_species=_clean(row.get("source_organism")) or None,
            source=_clean(row.get("source")) or "hitlist",
            alleles=allele_set or None,
            **_mapping_fields(row),
            **_lineage_fields(row),
        )

        if response in _AFFINITY_RESPONSES:
            binding_records.append(
                BindingRecord(
                    value=value,
                    qualifier=qualifier,
                    measurement_type=response,
                    assay_type=response,
                    assay_method=assay_method,
                    **_flank_fields(row),
                    **common,
                )
            )
        elif response in _T_HALF_RESPONSES:
            # hitlist reports half life in minutes; StabilityRecord.t_half is
            # in hours (data/collate.py converts back to minutes downstream).
            stability_records.append(
                StabilityRecord(
                    t_half=value / _MINUTES_PER_HOUR,
                    t_half_qualifier=qualifier,
                    assay_type=response,
                    assay_method=assay_method,
                    **common,
                )
            )
        elif response in _TM_RESPONSES:
            stability_records.append(
                StabilityRecord(
                    tm=value,
                    tm_qualifier=qualifier,
                    assay_type=response,
                    assay_method=assay_method,
                    **common,
                )
            )
        elif response in _KOFF_RESPONSES:
            kinetics_records.append(
                KineticsRecord(
                    koff=value,
                    koff_qualifier=qualifier,
                    assay_type=response,
                    assay_method=assay_method,
                    **common,
                )
            )
        elif response in _KON_RESPONSES:
            kinetics_records.append(
                KineticsRecord(
                    kon=value,
                    kon_qualifier=qualifier,
                    assay_type=response,
                    assay_method=assay_method,
                    **common,
                )
            )
        else:
            skipped_unroutable += 1

    for row in _iter_row_dicts(ms_frame):
        alleles = _split_allele_set(row.get("mhc_allele_set"))
        if not alleles:
            single = _clean(row.get("mhc_restriction"))
            if single:
                alleles = [single]
        if not alleles:
            # Matches the merged-TSV loader, which drops elution rows with no
            # resolvable allele.
            skipped_missing_mhc_allele += 1
            continue
        peptide = normalize_ingested_peptide(row.get("peptide"))
        if not peptide:
            skipped_noncanonical_peptide += 1
            continue
        # An unmapped-but-recorded condition means hitlist grew a treatment
        # category CONDITION_TO_STIMULUS does not know. It still falls back to
        # `none`, but counting it keeps that from being invisible: silently
        # scoring genuinely stimulated samples as unstimulated is the failure
        # mode worth catching.
        condition_category = _clean(row.get("condition_category"))
        if is_unmapped_condition(condition_category):
            unmapped_conditions[condition_category] = (
                unmapped_conditions.get(condition_category, 0) + 1
            )
        elution_records.append(
            ElutionRecord(
                peptide=peptide,
                alleles=alleles,
                detected=True,
                **_flank_fields(row),
                stimulus=stimulus_for_condition(condition_category),
                apm_perturbation=apm_group_for_row(
                    _clean(row.get("apm_genes_perturbed")),
                    _clean(row.get("sample_attribution")),
                ),
                sample_label=_clean(row.get("sample_label")),
                sample_attribution=_clean(row.get("sample_attribution")),
                cell_type=_clean(row.get("cell_type")) or _clean(row.get("cell_line_name")) or None,
                is_cell_line=row.get("src_cell_line"),
                is_healthy_tissue=row.get("src_healthy_tissue"),
                is_cancer=row.get("src_cancer"),
                is_tumor_adjacent=row.get("src_adjacent_to_tumor"),
                tissue=_clean(row.get("source_tissue")) or None,
                mhc_class=_clean(row.get("mhc_class")) or None,
                species=_clean(row.get("host")) or None,
                antigen_species=_clean(row.get("source_organism")) or None,
                source=_clean(row.get("source")) or "hitlist",
                **_mapping_fields(row),
                **_lineage_fields(row),
            )
        )

    counts_before_cap = {
        "binding": len(binding_records),
        "kinetics": len(kinetics_records),
        "stability": len(stability_records),
        "elution": len(elution_records),
    }
    binding_records = list(_cap_list(binding_records, max_binding, sampling_seed))
    stability_records = list(_cap_list(stability_records, max_stability, sampling_seed))
    kinetics_records = list(_cap_list(kinetics_records, max_kinetics, sampling_seed))
    elution_records = list(_cap_list(elution_records, max_elution, sampling_seed))

    def _flank_coverage(records: Sequence[Any]) -> float:
        """Fraction of records that actually carry flanking residues.

        Coverage, not resolution. Gating this on `flank_context_resolved`
        made it report the resolution rate instead, which masking does not
        change -- so both arms of the source-mapping comparison printed the
        same number and the stat could not audit the thing it was added for.
        `_flank_resolution` reports resolution separately.
        """
        if not records:
            return 0.0
        with_flank = sum(
            1
            for r in records
            if (getattr(r, "flank_n", "") or "") or (getattr(r, "flank_c", "") or "")
        )
        return with_flank / len(records)

    def _flank_resolution(records: Sequence[Any]) -> float:
        """Fraction of records whose source mapping resolved the junction."""
        if not records:
            return 0.0
        resolved = sum(1 for r in records if bool(getattr(r, "flank_context_resolved", False)))
        return resolved / len(records)

    stats: Dict[str, Any] = {
        "source": "hitlist",
        "sampling_seed": int(sampling_seed),
        "source_rows_before_mapping_collapse": source_rows_before_mapping_collapse,
        "source_rows_after_mapping_collapse": {
            "binding": len(binding_frame),
            "ms": len(ms_frame),
        },
        "counts_before_cap": counts_before_cap,
        "counts": {
            "binding": len(binding_records),
            "kinetics": len(kinetics_records),
            "stability": len(stability_records),
            "processing": 0,
            "elution": len(elution_records),
            "tcell": 0,
            "tcr_evidence": 0,
        },
        "rows_dropped_by_cap": {
            name: counts_before_cap[name] - count
            for name, count in {
                "binding": len(binding_records),
                "kinetics": len(kinetics_records),
                "stability": len(stability_records),
                "elution": len(elution_records),
            }.items()
        },
        "requested_caps": {
            "binding": max_binding,
            "kinetics": max_kinetics,
            "stability": max_stability,
            "elution": max_elution,
        },
        "skipped_no_numeric_value": skipped_no_value,
        "skipped_unroutable_response": skipped_unroutable,
        "skipped_unexpected_unit": skipped_bad_unit,
        "skipped_noncanonical_peptide": skipped_noncanonical_peptide,
        "skipped_missing_mhc_allele": skipped_missing_mhc_allele,
        "unmapped_condition_categories": dict(unmapped_conditions),
        # How often the flank we trained on was a choice among several. See
        # presto#34 -- for HLA-A*02:01, 21.6% of rows have mappings that
        # disagree on a flank.
        "mapping_ambiguity": mapping_ambiguity,
        "stability_assay_methods": _method_counts(stability_records),
        "kinetics_assay_methods": _method_counts(kinetics_records),
        "flank_coverage": {
            "binding": _flank_coverage(binding_records),
            "elution": _flank_coverage(elution_records),
        },
        "flank_resolution": {
            "binding": _flank_resolution(binding_records),
            "elution": _flank_resolution(elution_records),
        },
    }

    return (
        binding_records,
        kinetics_records,
        stability_records,
        [],  # processing: not in hitlist's training table
        elution_records,
        [],  # tcell: not in hitlist's training table
        [],  # tcr_evidence: not in hitlist's training table
        stats,
    )


def _method_counts(records: Sequence[Any]) -> Dict[str, int]:
    """Count assay methods in a record list, for run provenance.

    Half-life in particular spans several non-comparable methods, so the mix is
    worth recording in the run stats rather than discovering it later.
    """
    counts: Dict[str, int] = {}
    for record in records:
        key = getattr(record, "assay_method", None) or "unspecified"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def _cap_list(records: List[Any], cap: Optional[int], seed: int) -> List[Any]:
    """Deterministically subsample a record list to ``cap`` entries."""
    if cap is None or cap <= 0 or len(records) <= cap:
        return records
    rng = random.Random(seed)
    return rng.sample(records, cap)


__all__ = [
    "SHARED_COLUMNS",
    "BINDING_COLUMNS",
    "MS_COLUMNS",
    "PROTEIN_MAPPING_COLUMNS",
    "COLUMNS_BY_EVIDENCE",
    "MAPPING_CATEGORIES",
    "UNRESOLVED_MAPPING_CATEGORIES",
    "training_columns",
    "assert_columns_present",
    "normalize_ingested_peptide",
    "load_records_from_hitlist",
]
