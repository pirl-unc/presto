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
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from .vocab import (
    apm_group_for_genes,
    drop_unencodable_sequence,
    inducer_for_condition,
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

# Columns projected out of the training table. Kept explicit because the full
# table is 95 columns wide and several million rows; projecting early is the
# difference between a few hundred MB and an OOM.
_SHARED_COLUMNS = [
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
]

_BINDING_COLUMNS = _SHARED_COLUMNS + [
    "response_measured",
    "quantitative_value",
    "measurement_units",
    "measurement_inequality",
    "assay_method",
]

_MS_COLUMNS = _SHARED_COLUMNS + [
    "cell_line_name",
    "source_tissue",
    "cell_type",
    # Tier 3 cellular state. `condition_category` is the per-sample truth;
    # `apm_genes_perturbed` names the genes. The study-level roll-up
    # (`study_apm_perturbed`) is deliberately not used: it is ORed across a
    # study, so a WT control inside a KO study inherits the flag.
    "condition_category",
    "apm_genes_perturbed",
]

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


# The 20 canonical residues. Anything else -- `X` for an unknown residue, or an
# IEDB annotation such as `SXPSGGXGV + INDIST(X2, X7)` / `ILAETVAXV + OTH(X8)` --
# describes chemistry this model has no representation for. Roughly 0.007% of
# rows, but they used to reach the tokenizer and abort training mid-epoch, so
# they are dropped explicitly at ingest and counted.
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
def is_canonical_peptide(peptide: str) -> bool:
    """True when the peptide can be tokenized.

    Peptides are targets, so an unrepresentable residue drops the row: unlike
    a flank, the epitope cannot degrade to "absent".
    """
    return is_encodable_sequence(peptide)


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
        columns=_BINDING_COLUMNS if include_flanks else
        [c for c in _BINDING_COLUMNS if c not in {"n_flank", "c_flank", "is_canonical_transcript"}],
        **shared_filters,
    )
    ms_frame = hitlist.generate_training_table(
        include_evidence="ms",
        columns=_MS_COLUMNS if include_flanks else
        [c for c in _MS_COLUMNS if c not in {"n_flank", "c_flank", "is_canonical_transcript"}],
        **shared_filters,
    )

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

        peptide = _clean(row.get("peptide"))
        if not is_canonical_peptide(peptide):
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
        peptide = _clean(row.get("peptide"))
        if not is_canonical_peptide(peptide):
            skipped_noncanonical_peptide += 1
            continue
        elution_records.append(
            ElutionRecord(
                peptide=peptide,
                alleles=alleles,
                detected=True,
                flank_n=drop_unencodable_sequence(row.get("n_flank")),
                flank_c=drop_unencodable_sequence(row.get("c_flank")),
                inducer=inducer_for_condition(_clean(row.get("condition_category"))),
                apm_perturbation=apm_group_for_genes(
                    _clean(row.get("apm_genes_perturbed"))
                ),
                cell_type=_clean(row.get("cell_line_name")) or None,
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
