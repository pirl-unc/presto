"""Positive MS evidence with explicit candidate pairs and scoped source conditions.

This adapter does not create negatives, training partitions, active pathways or
measured machinery activity. Contributor joining remains a cohort-level gate.
"""

from collections import defaultdict
from dataclasses import replace
from functools import lru_cache
import re

from .modular import AA, Context, ContextAssertion, Exposure, Observation
from .modular_hitlist import AdmissionError, HumanMHCCatalog, human_dr_reference_assumption


UNIQUE_ARM_ATTRIBUTIONS = frozenset(
    {
        "curated_sample_label",
        "elution_conditions",
        "allele_exact",
        "discriminated",
        "single_sample_pmid",
    }
)
INTERVENTION_FIELDS = {
    "condition_knockout_genes": "knockout",
    "condition_knockdown_genes": "knockdown",
    "condition_overexpression_genes": "overexpression",
    "condition_transfection": "transfection",
    "condition_transduction": "transduction",
    "condition_drugs": "drug",
}
FACTOR_FIELDS = (*INTERVENTION_FIELDS, "condition_cytokines")
UNKNOWN_TOKENS = frozenset(
    {"unknown", "unspecified", "not_reported", "unreported", "not_applicable"}
)


def condition_tokens(value):
    """Use canonical semicolon tokens without interpreting a blank as `none`."""
    if value is None or value == "":
        return ()
    if not isinstance(value, str):
        raise AdmissionError("nontext_condition_tokens", deferred=True)
    tokens = tuple(dict.fromkeys(part.strip() for part in value.split(";") if part.strip()))
    if "none" in tokens and len(tokens) != 1:
        raise AdmissionError("contradictory_none_condition", deferred=True)
    return tokens


def has_resolved_condition(raw):
    return (
        raw.get("sample_attribution") in UNIQUE_ARM_ATTRIBUTIONS
        and bool(raw.get("condition_id"))
        and raw.get("condition_status") == "annotated"
        and raw.get("condition_evidence") in {"curated_text", "primary_source"}
        and (
            raw.get("condition_evidence") != "primary_source"
            or bool(raw.get("condition_reference"))
        )
    )


def study_factor_vocabulary(records):
    """Collect declared factors, never outcomes, for explicit `none` assertions."""
    values = defaultdict(lambda: defaultdict(set))
    for raw in records:
        if has_resolved_condition(raw):
            for field in FACTOR_FIELDS:
                values[str(raw.get("pmid") or "")][field].update(
                    token
                    for token in condition_tokens(raw.get(field))
                    if token != "none" and token not in UNKNOWN_TOKENS
                )
    return {
        pmid: {field: tuple(sorted(tokens)) for field, tokens in fields.items()}
        for pmid, fields in values.items()
    }


def source_condition(raw, vocabulary):
    """Keep the source's independent intervention and cytokine claims."""
    admitted = has_resolved_condition(raw)
    provenance = {
        "condition_admitted": admitted,
        "condition_id": raw.get("condition_id"),
        "sample_attribution": raw.get("sample_attribution"),
        "evidence_kind": raw.get("condition_evidence"),
        "reference": raw.get("condition_reference"),
        "policy": "unique_arm_explicit_tokens_v1",
    }
    if not admitted:
        return Context(provenance=provenance)
    interventions, cytokines = {}, {}
    for field in FACTOR_FIELDS:
        tokens = condition_tokens(raw.get(field))
        absent = tokens == ("none",)
        names = vocabulary.get(field, ()) if absent else tokens
        for name in names:
            if name in UNKNOWN_TOKENS:
                continue
            evidence = {
                **provenance,
                "pmid": raw.get("pmid"),
                "source_field": field,
                "source_value": raw.get(field),
                "scope": "curated_condition",
                "absence_basis": "explicit_none_in_same_family" if absent else None,
            }
            state = "absent" if absent else "present"
            if field == "condition_cytokines":
                cytokines[name] = Exposure(state=state, evidence=evidence)
            else:
                interventions[f"{INTERVENTION_FIELDS[field]}:{name}"] = ContextAssertion(
                    state, evidence
                )
    cell_type = raw.get("cell_type") or None
    if cell_type and (not isinstance(cell_type, str) or ";" in cell_type):
        cell_type = None
    return Context(
        cell_type=cell_type,
        cytokines=cytokines,
        interventions=interventions,
        provenance=provenance,
    )


@lru_cache(maxsize=4096)
def _candidate_pairs(value, catalog, mhc_class="II", dra_reference=None):
    if not isinstance(value, str) or not value.strip():
        raise AdmissionError("missing_candidate_set", deferred=True)
    separator = ";" if ";" in value else ","
    names = tuple(dict.fromkeys(part.strip() for part in value.split(separator) if part.strip()))
    pairs, reported = {}, defaultdict(list)
    for name in names:
        pair = catalog.resolve(
            name,
            mhc_class,
            reference_assumption=human_dr_reference_assumption(
                name, reference=dra_reference if mhc_class == "II" else None
            ),
        )
        key = (pair.groove1, pair.groove2, pair.species1, pair.species2)
        pairs[key] = pair
        reported[key].append(name)
    if not pairs:
        raise AdmissionError("missing_candidate_set", deferred=True)
    return tuple(
        replace(pair, provenance={**pair.provenance, "reported_candidates": reported[key]})
        for key, pair in pairs.items()
    )


def class_ii_ms_observation(
    raw, catalog: HumanMHCCatalog, *, source_sha256, ordinal, study_factors
):
    """Historical class-II-only admission; use the shared adapter for both classes."""
    if (
        raw.get("mhc_class") != "II"
        or raw.get("is_ms_observation") is not True
        or raw.get("assay_modality") != "ms"
        or raw.get("evidence_kind") != "ms"
    ):
        raise AdmissionError("outside_class_ii_ms")
    return ms_observation(
        raw, catalog, source_sha256=source_sha256, ordinal=ordinal, study_factors=study_factors
    )


def ms_observation(
    raw,
    catalog: HumanMHCCatalog,
    *,
    source_sha256,
    ordinal,
    study_factors,
    dra_reference=None,
    allow_reported_negatives=False,
):
    """Admit source outcomes; a reported nondetection is scoped to its assay.

    Unknown cellular context stays unknown for either outcome. Negative admission
    never claims an unobserved peptide is a non-binder or absent in other assays.
    """
    if (
        not isinstance(source_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", source_sha256) is None
        or type(ordinal) is not int
        or ordinal < 0
    ):
        raise ValueError("source SHA256 and nonnegative original ordinal required")
    reported_negative = (
        allow_reported_negatives
        and raw.get("ms_outcome_status") == "negative"
        and isinstance(raw.get("qualitative_measurement"), str)
        and raw["qualitative_measurement"].strip().lower().startswith("negative")
        and raw.get("is_ms_observation") is False
    )
    if (
        raw.get("mhc_class") not in {"I", "II"}
        or not (raw.get("is_ms_observation") is True or reported_negative)
        or raw.get("assay_modality") != "ms"
        or raw.get("evidence_kind") != "ms"
    ):
        raise AdmissionError("outside_mhc_ms")
    if not reported_negative and raw.get("ms_outcome_status") not in {
        "positive_result",
        "curated_ligand_table",
    }:
        raise AdmissionError("not_explicit_positive_ms")
    if raw.get("mhc_species") != "Homo sapiens":
        raise AdmissionError("outside_human_mhc")
    if raw.get("has_ptm") is True or raw.get("peptide_modifications"):
        raise AdmissionError("modified_peptidoform")
    if raw.get("has_ptm") is not False:
        raise AdmissionError("unspecified_peptidoform", deferred=True)
    if raw.get("is_non_peptide_ligand") is not False:
        raise AdmissionError("nonpeptide_or_unspecified_ligand")
    if raw.get("is_potential_contaminant") is True:
        raise AdmissionError("flagged_contaminant")
    peptide = raw.get("peptide")
    minimum = 9 if raw["mhc_class"] == "II" else 1
    if not isinstance(peptide, str) or not minimum <= len(peptide) <= 512 or set(peptide) - set(AA):
        raise AdmissionError("unsupported_peptide")
    if raw.get("provenance_status") != "indexed" or any(
        not isinstance(raw.get(name), str) or not raw[name].strip()
        for name in ("evidence_row_id", "evidence_source_id", "provenance_id")
    ):
        raise AdmissionError("missing_indexed_source_identity", deferred=True)
    if raw.get("restriction_evidence") == "predicted":
        raise AdmissionError("predicted_restriction", deferred=True)
    provenance = raw.get("mhc_allele_provenance")
    if provenance != "exact" and not (
        provenance in {"sample_allele_match", "sample_locus_match"}
        and raw.get("sample_attribution") in UNIQUE_ARM_ATTRIBUTIONS
    ):
        raise AdmissionError("unresolved_candidate_scope", deferred=True)
    pairs = _candidate_pairs(raw.get("mhc_allele_set"), catalog, raw["mhc_class"], dra_reference)
    context = source_condition(raw, study_factors.get(str(raw.get("pmid") or ""), {}))
    result = Observation(
        observation_id=raw["evidence_row_id"],
        source="hitlist_native_ms",
        endpoint="elution_observation",
        peptide=peptide,
        mhc=pairs,
        value=0.0 if reported_negative else 1.0,
        unit="binary",
        split="predict",
        group_id="unpartitioned",
        context=context,
        provenance={
            "measurement_namespace": "hitlist_evidence_row_id",
            "original_measurement_id": raw["evidence_row_id"],
            "source_sha256": source_sha256,
            "source_ordinal": ordinal,
            "source_contributor_id": raw["provenance_id"],
            "raw_record": dict(raw),
            "label_kind": "reported_nondetection" if reported_negative else "reported_positive",
            "interpretation": (
                "Source-reported nondetection in this assay; "
                "not global non-presentation or non-binding"
                if reported_negative
                else "reported_positive_ms_ligand; no negative or detection calibration"
            ),
            "partition_status": "unassigned; contributor and dependency qualification required",
        },
    )
    result.validate()
    return result


def indexed_ms_observation(
    raw, catalog: HumanMHCCatalog, *, source_sha256, ordinal, dra_reference=None
):
    """Admit literal indexed MS outcomes without inventing sample enrichment.

    The canonical other-assay index retains reported MS negatives. Its assay and
    contributor IDs suffice for source lineage; candidate sets must already have
    exact scope because no donor/condition attribution is supplied by this route.
    """
    assay = raw.get("assay_iri")
    if not isinstance(assay, str) or not assay.strip() or not raw.get("provenance_id"):
        raise AdmissionError("missing_indexed_source_identity", deferred=True)
    if raw.get("mhc_allele_provenance") != "exact":
        raise AdmissionError("unresolved_candidate_scope", deferred=True)
    evidence_id = "ms:" + assay
    adapted = {
        **raw,
        "evidence_kind": "ms",
        "evidence_row_id": evidence_id,
        "evidence_source_id": evidence_id,
        "provenance_status": "indexed",
    }
    result = ms_observation(
        adapted,
        catalog,
        source_sha256=source_sha256,
        ordinal=ordinal,
        study_factors={},
        dra_reference=dra_reference,
        allow_reported_negatives=True,
    )
    return replace(
        result,
        provenance={
            **result.provenance,
            "raw_record": dict(raw),
            "indexed_identity_basis": "canonical assay_iri and contributor provenance_id",
            "sample_enrichment": "none; only exact candidate scope admitted",
        },
    )
