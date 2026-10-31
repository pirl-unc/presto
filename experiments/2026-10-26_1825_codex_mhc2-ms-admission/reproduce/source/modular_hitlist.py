"""Strict raw Hitlist IC50 admission against a caller-pinned human catalog.

This adapter preserves physical measurements and source identity. Partitioning,
original-outcome deduplication and experiment qualification remain explicit steps.
"""

from collections import Counter, defaultdict
import math
import re

from .modular import AA, MHCInput, Observation


class AdmissionError(ValueError):
    """A source disposition, distinct from an unexpected implementation failure."""

    def __init__(self, reason: str, *, deferred: bool = False):
        super().__init__(reason)
        self.reason = reason
        self.disposition = "deferred" if deferred else "excluded"


def _reference_assignment(row):
    if row.get("source") != "imgt" or row.get("species") != "Homo sapiens":
        raise AdmissionError("catalog_outside_human_imgt")
    if not row.get("source_id") or not row.get("two_field_allele"):
        raise AdmissionError("catalog_missing_identifier")
    if row.get("is_functional") != "True" or any(
        row.get(name) != "False" for name in ("is_null", "is_questionable", "is_pseudogene")
    ):
        raise AdmissionError("catalog_functional_flags")
    if row.get("groove_status") != "ok" or row.get("groove_flags"):
        raise AdmissionError("catalog_groove_flags")
    cls, chain = row.get("mhc_class"), row.get("chain")
    slots = {
        ("I", "alpha"): ("groove1", "groove2"),
        ("II", "alpha"): ("groove1",),
        ("II", "beta"): ("groove2",),
    }.get((cls, chain))
    if slots is None:
        raise AdmissionError("catalog_class_chain")
    protein = row.get("sequence")
    if not isinstance(protein, str):
        raise AdmissionError("catalog_missing_protein")
    end = 0
    for slot in ("groove1", "groove2"):
        value = row.get(slot)
        if slot not in slots:
            if value:
                raise AdmissionError("catalog_unexpected_domain")
            continue
        if not isinstance(value, str) or not 60 <= len(value) <= 130 or set(value) - set(AA):
            raise AdmissionError("catalog_noncanonical_domain")
        try:
            declared_length = int(row.get(slot + "_len", ""))
        except (TypeError, ValueError) as error:
            raise AdmissionError("catalog_domain_length") from error
        if declared_length != len(value):
            raise AdmissionError("catalog_domain_length")
        position = protein.find(value, end)
        if position < 0:
            raise AdmissionError("catalog_domain_not_in_ordered_protein")
        end = position + len(value)
    return cls, chain, row.get("groove1") or "", row.get("groove2") or ""


class HumanMHCCatalog:
    """Resolve explicit two-field names only when all eligible aliases agree."""

    def __init__(self, rows):
        self.assignments = defaultdict(lambda: defaultdict(list))
        self.dispositions = Counter()
        self.used = set()
        for ordinal, raw in enumerate(rows):
            row = dict(raw)
            try:
                assignment = _reference_assignment(row)
                name = row["two_field_allele"]
                if not name.startswith("HomoSapiens-"):
                    raise AdmissionError("catalog_noncanonical_name")
                name = name.replace("HomoSapiens-", "HLA-", 1)
            except AdmissionError as error:
                self.dispositions[error.reason] += 1
                continue
            self.assignments[name][assignment].append({"source_row": ordinal, **row})
            self.dispositions["eligible_alias"] += 1

    def resolve(self, restriction: str, mhc_class: str) -> MHCInput:
        if not isinstance(restriction, str) or not restriction.startswith("HLA-"):
            raise AdmissionError("not_explicit_human_restriction")
        names = [
            part if part.startswith("HLA-") else "HLA-" + part for part in restriction.split("/")
        ]
        if len(names) != (1 if mhc_class == "I" else 2):
            raise AdmissionError("missing_explicit_mhc_partner", deferred=True)
        assignments = []
        for name in names:
            values = self.assignments.get(name)
            if not values:
                raise AdmissionError("unresolved_catalog_name", deferred=True)
            if len(values) != 1:
                raise AdmissionError("ambiguous_catalog_name", deferred=True)
            assignment = next(iter(values))
            if assignment[0] != mhc_class:
                raise AdmissionError("source_catalog_class_disagreement", deferred=True)
            assignments.append(assignment)
        by_chain = {assignment[1]: assignment for assignment in assignments}
        if mhc_class == "I":
            if set(by_chain) != {"alpha"}:
                raise AdmissionError("class_i_not_heavy_chain", deferred=True)
            first, second = by_chain["alpha"][2:]
        else:
            if set(by_chain) != {"alpha", "beta"}:
                raise AdmissionError("class_ii_not_alpha_beta_pair", deferred=True)
            first, second = by_chain["alpha"][2], by_chain["beta"][3]
        pair = MHCInput(
            first,
            second,
            mhc_class,
            species1="human",
            species2="human",
            provenance={
                "reported_restriction": restriction,
                "catalog_alleles": names,
                "partner_imputation": False,
            },
        )
        pair.validate()
        self.used.update(names)
        return pair

    def aliases(self, *, used_only=True):
        names = self.used if used_only else self.assignments.keys()
        return {
            name: [row for rows in self.assignments[name].values() for row in rows]
            for name in sorted(names)
        }


_NUMBER = re.compile(r"\s*(<=|>=|=|<|>)?\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*")


def purified_ic50_observation(raw, catalog: HumanMHCCatalog, *, source_sha256, ordinal):
    """Convert one actual export row; do not assign groups or invent missing facts."""
    if raw.get("source") not in ("iedb", "cedar") or raw.get("response_measured") != (
        "half maximal inhibitory concentration (IC50)"
    ):
        raise AdmissionError("outside_direct_iedb_cedar_ic50")
    if not isinstance(raw.get("assay_method"), str) or not raw["assay_method"].startswith(
        "purified MHC/"
    ):
        raise AdmissionError("not_explicit_purified_mhc")
    if raw.get("has_ptm") is not False or raw.get("peptide_modifications"):
        raise AdmissionError("modified_or_unspecified_peptidoform")
    if raw.get("is_non_peptide_ligand"):
        raise AdmissionError("non_peptide_ligand")
    peptide, cls = raw.get("peptide"), raw.get("mhc_class")
    if (
        not isinstance(peptide, str)
        or set(peptide) - set(AA)
        or cls not in ("I", "II")
        or not (9 if cls == "II" else 1) <= len(peptide) <= 512
    ):
        raise AdmissionError("unsupported_peptide_or_class")
    unit, value = raw.get("measurement_units"), raw.get("quantitative_value")
    if unit not in ("pM", "nM", "uM", "mM", "M"):
        raise AdmissionError("unsupported_concentration_unit")
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise AdmissionError("nonpositive_or_nonfinite_measurement")
    qualifier = raw.get("measurement_inequality")
    if qualifier in (None, ""):
        qualifier = "="
    if qualifier not in ("=", "<", "<=", ">", ">="):
        raise AdmissionError("unsupported_measurement_qualifier")
    text = raw.get("quantitative_measurement")
    match = _NUMBER.fullmatch(text) if isinstance(text, str) else None
    if match is None or float(match[2]) != value or (match[1] and match[1] != qualifier):
        raise AdmissionError("raw_numeric_or_qualifier_disagreement", deferred=True)
    assay = raw.get("assay_iri")
    if not isinstance(assay, str) or not assay.strip():
        raise AdmissionError("missing_original_assay_iri", deferred=True)
    pair = catalog.resolve(raw.get("mhc_restriction"), cls)
    result = Observation(
        f"hitlist-v4-row:{ordinal}",
        f"hitlist-v4-{raw['source']}",
        "ic50",
        peptide,
        (pair,),
        value=float(value),
        unit=unit,
        qualifier=qualifier,
        provenance={
            "measurement_namespace": "assay_iri",
            "original_measurement_id": assay,
            "source_sha256": source_sha256,
            "source_ordinal": ordinal,
            "target_policy": "physical_unclipped_v1",
            "missing_qualifier_convention": "source export exact; raw missingness retained",
            "raw_record": dict(raw),
        },
    )
    result.validate()
    return result
