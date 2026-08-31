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
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Tuple, Union

from .vocab import (
    apm_group_for_genes,
    drop_unencodable_sequence,
    is_unmapped_condition,
    stimulus_for_condition,
    is_encodable_sequence,
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
    "n_flank",
    "c_flank",
    "evidence_row_id",
    "is_canonical_transcript",
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
    {"n_flank", "c_flank", "is_canonical_transcript"}
)

#: The `include_evidence` value hitlist expects, mapped to what we ask it for.
COLUMNS_BY_EVIDENCE: Dict[str, Tuple[str, ...]] = {
    "binding": BINDING_COLUMNS,
    "ms": MS_COLUMNS,
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
    import hitlist

    raise RuntimeError(
        f"hitlist returned {len(frame.columns)} columns for evidence={evidence!r} "
        f"but {len(missing)} requested column(s) are absent: {', '.join(missing)}. "
        f"Installed hitlist is {getattr(hitlist, '__version__', 'unknown')}; these "
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
    token = str(text or "").strip()
    if token.startswith("<"):
        return -1
    if token.startswith(">"):
        return 1
    return 0


def _split_allele_set(value: Any) -> List[str]:
    """Split a hitlist allele-set cell into a list of allele names."""
    if isinstance(value, (list, tuple, set)):
        return [str(v).strip() for v in value if str(v).strip()]
    token = str(value or "").strip()
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


def _select_best_mapping(frame):
    """Collapse exploded protein mappings to one row per evidence row.

    ``map_source_proteins=True`` emits one row per (evidence row, protein
    mapping), which more than doubles the row count and would train on the same
    measurement several times. Prefer the canonical-transcript mapping so the
    flanks come from the dominant isoform; fall back to the first mapping.
    """
    if "evidence_row_id" not in frame.columns:
        return frame
    ordered = frame
    if "is_canonical_transcript" in frame.columns:
        ordered = frame.sort_values(
            "is_canonical_transcript", ascending=False, kind="stable"
        )
    return ordered.drop_duplicates(subset=["evidence_row_id"], keep="first")


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
    text = str(peptide or "").strip().upper()
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
    """
    try:
        import hitlist
    except ImportError as exc:  # pragma: no cover - exercised via install docs
        raise SystemExit(
            "hitlist is required for --data-source hitlist. Install it with "
            "`pip install hitlist`, or use --data-source merged_tsv."
        ) from exc

    shared_filters = dict(
        mhc_class=mhc_class,
        species=species,
        mhc_allele=list(mhc_allele) if isinstance(mhc_allele, (list, tuple)) else mhc_allele,
        length_min=length_min,
        length_max=length_max,
        map_source_proteins=include_flanks,
    )

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

    if include_flanks:
        binding_frame = _select_best_mapping(binding_frame)
        ms_frame = _select_best_mapping(ms_frame)

    binding_records: List[BindingRecord] = []
    kinetics_records: List[KineticsRecord] = []
    stability_records: List[StabilityRecord] = []
    elution_records: List[ElutionRecord] = []

    skipped_no_value = 0
    skipped_unroutable = 0
    skipped_bad_unit = 0
    skipped_noncanonical_peptide = 0
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
        )

        if response in _AFFINITY_RESPONSES:
            binding_records.append(
                BindingRecord(
                    value=value,
                    qualifier=qualifier,
                    measurement_type=response,
                    assay_type=response,
                    assay_method=assay_method,
                    flank_n=drop_unencodable_sequence(row.get("n_flank")),
                    flank_c=drop_unencodable_sequence(row.get("c_flank")),
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
                flank_n=drop_unencodable_sequence(row.get("n_flank")),
                flank_c=drop_unencodable_sequence(row.get("c_flank")),
                stimulus=stimulus_for_condition(condition_category),
                apm_perturbation=apm_group_for_genes(
                    _clean(row.get("apm_genes_perturbed"))
                ),
                cell_type=_clean(row.get("cell_type"))
                or _clean(row.get("cell_line_name"))
                or None,
                is_cell_line=row.get("src_cell_line"),
                is_healthy_tissue=row.get("src_healthy_tissue"),
                is_cancer=row.get("src_cancer"),
                is_tumor_adjacent=row.get("src_adjacent_to_tumor"),
                tissue=_clean(row.get("source_tissue")) or None,
                mhc_class=_clean(row.get("mhc_class")) or None,
                species=_clean(row.get("host")) or None,
                antigen_species=_clean(row.get("source_organism")) or None,
                source=_clean(row.get("source")) or "hitlist",
            )
        )

    binding_records = list(_cap_list(binding_records, max_binding, sampling_seed))
    stability_records = list(_cap_list(stability_records, max_stability, sampling_seed))
    kinetics_records = list(_cap_list(kinetics_records, max_kinetics, sampling_seed))
    elution_records = list(_cap_list(elution_records, max_elution, sampling_seed))

    def _flank_coverage(records: Sequence[Any]) -> float:
        if not records:
            return 0.0
        with_flank = sum(1 for r in records if getattr(r, "flank_n", "") or getattr(r, "flank_c", ""))
        return with_flank / len(records)

    stats: Dict[str, Any] = {
        "source": "hitlist",
        "counts": {
            "binding": len(binding_records),
            "kinetics": len(kinetics_records),
            "stability": len(stability_records),
            "processing": 0,
            "elution": len(elution_records),
            "tcell": 0,
            "tcr_evidence": 0,
        },
        "skipped_no_numeric_value": skipped_no_value,
        "skipped_unroutable_response": skipped_unroutable,
        "skipped_unexpected_unit": skipped_bad_unit,
        "skipped_noncanonical_peptide": skipped_noncanonical_peptide,
        "unmapped_condition_categories": dict(unmapped_conditions),
        "stability_assay_methods": _method_counts(stability_records),
        "kinetics_assay_methods": _method_counts(kinetics_records),
        "flank_coverage": {
            "binding": _flank_coverage(binding_records),
            "elution": _flank_coverage(elution_records),
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
    "training_columns",
    "assert_columns_present",
    "normalize_ingested_peptide",
    "load_records_from_hitlist",
]
