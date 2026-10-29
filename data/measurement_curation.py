"""Source-preserving inequalities and study-scoped floor/ceiling review.

Panel rules identify statistical censoring assumptions, not confirmed protocols.
Their frozen decisions can be applied by the same admission adapter as explicit
source qualifiers. No rule uses model predictions or universal nM thresholds.
"""

from collections import defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import re


POLICY = "presto.measurement-curation.v1"
CONCENTRATION_RESPONSES = frozenset(
    {
        "half maximal inhibitory concentration (IC50)",
        "dissociation constant KD",
        "half maximal effective concentration (EC50)",
        "dissociation constant KD (~IC50)",
        "dissociation constant KD (~EC50)",
    }
)
NM = {"pM": 0.001, "nM": 1.0, "uM": 1000.0, "mM": 1e6, "M": 1e9}
_NUMBER = re.compile(r"\s*(<=|>=|=|<|>)?\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*")
_FINGERPRINT_FIELDS = (
    "assay_iri",
    "reference_iri",
    "pmid",
    "peptide",
    "mhc_restriction",
    "mhc_class",
    "assay_method",
    "response_measured",
    "measurement_units",
    "quantitative_value",
    "quantitative_measurement",
    "measurement_inequality",
    "qualitative_measurement",
)


class MeasurementCurationError(ValueError):
    pass


def source_fingerprint(raw):
    return hashlib.sha256(
        json.dumps(
            {key: raw.get(key) for key in _FINGERPRINT_FIELDS},
            sort_keys=True,
            allow_nan=False,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def reported_measurement(raw):
    """Reconcile a literal value and qualifier without assuming blank means exact."""
    value = raw.get("quantitative_value")
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise MeasurementCurationError("nonfinite_or_nonnumeric_measurement")
    text = raw.get("quantitative_measurement")
    match = (
        _NUMBER.fullmatch(text.replace("≤", "<=").replace("≥", ">="))
        if isinstance(text, str)
        else None
    )
    if match is None or float(match[2]) != value:
        raise MeasurementCurationError("raw_numeric_or_qualifier_disagreement")
    original = raw.get("measurement_inequality")
    reported = (
        original.strip().replace("≤", "<=").replace("≥", ">=")
        if isinstance(original, str)
        else original
    )
    if reported not in (None, "", "=", "<", "<=", ">", ">="):
        raise MeasurementCurationError("unsupported_measurement_qualifier")
    embedded = match[1]
    if reported and embedded and reported != embedded:
        raise MeasurementCurationError("conflicting_reported_qualifiers")
    effective = reported or embedded or "="
    return dict(
        policy=POLICY,
        reported_qualifier=original,
        numeric_text_qualifier=embedded,
        effective_qualifier=effective,
        status="reported" if reported or embedded else "unspecified_assumed_exact",
        basis="structured_qualifier"
        if reported
        else "numeric_text"
        if embedded
        else "missing_qualifier_convention",
    )


@dataclass(frozen=True)
class CensorDecision:
    assay_iri: str
    source_sha256: str
    qualifier: str | None
    status: str
    evidence: dict
    policy: str = POLICY

    def __post_init__(self):
        if not self.assay_iri or not re.fullmatch(r"[0-9a-f]{64}", self.source_sha256):
            raise ValueError("censor decision requires source identity and fingerprint")
        if self.policy != POLICY or self.status not in (
            "source_confirmed",
            "inferred_panel_limit",
            "needs_review",
        ):
            raise ValueError("unsupported censor decision policy/status")
        if (self.status == "needs_review") != (self.qualifier is None):
            raise ValueError("unresolved censor review cannot assert a qualifier")
        if self.qualifier is not None and self.qualifier not in ("<", "<=", ">", ">="):
            raise ValueError("censor review requires an inequality")
        if not isinstance(self.evidence, dict) or not self.evidence:
            raise ValueError("censor review requires evidence")
        if self.status == "source_confirmed" and any(
            not isinstance(self.evidence.get(k), str) or not self.evidence[k].strip()
            for k in ("reference", "locator")
        ):
            raise ValueError("confirmed censoring requires a source reference and locator")


def curated_measurement(raw, decision=None):
    result = reported_measurement(raw)
    if decision is None:
        return result
    if not isinstance(decision, CensorDecision):
        raise MeasurementCurationError("untyped_censor_decision")
    if decision.assay_iri != raw.get("assay_iri") or decision.source_sha256 != source_fingerprint(
        raw
    ):
        raise MeasurementCurationError("censor_review_source_changed")
    # Preserve explicit inequalities, including their original strictness.
    if result["effective_qualifier"] != "=":
        raise MeasurementCurationError("censor_review_changes_reported_bound")
    if decision.status == "needs_review":
        raise MeasurementCurationError("suspected_assay_limit_requires_review")
    return {
        **result,
        "effective_qualifier": decision.qualifier,
        "status": decision.status,
        "basis": "frozen_censor_review",
        "review": asdict(decision),
    }


def assay_method_context(raw):
    """Reported quantity and unreviewed method are independent provenance axes."""
    method = raw.get("assay_method") or ""
    parts = method.split("/")
    conflict = "12135423" in (raw.get("assay_comments") or "") and parts[1:] == [
        "competitive",
        "fluorescence",
    ]
    return dict(
        reported_method=method,
        reported_quantity=raw.get("response_measured"),
        preparation=parts[0] or "unknown",
        mechanism=parts[1] if len(parts) > 1 else "unknown",
        detector="/".join(parts[2:]) if len(parts) > 2 else "unknown",
        review_status="protocol_citation_conflict" if conflict else "source_tag_unreviewed",
        issue="https://github.com/pirl-unc/hitlist/issues/871" if conflict else None,
        quantity_establishes_direct_method=False,
    )


@dataclass(frozen=True)
class CensorScreenPolicy:
    min_panel_rows: int = 20
    min_heap_rows: int = 5
    min_heap_fraction: float = 0.1
    inference_heap_fraction: float = 0.2
    min_qualitative_agreement: float = 0.8
    min_distinct_values: int = 3
    floor_gap_ratio: float = 4.0

    def __post_init__(self):
        if self.min_panel_rows < 1 or self.min_heap_rows < 1 or self.min_distinct_values < 2:
            raise ValueError("invalid censor screen support")
        if not (0 < self.min_heap_fraction <= self.inference_heap_fraction <= 1):
            raise ValueError("invalid heap fractions")
        if not 0.5 < self.min_qualitative_agreement <= 1 or self.floor_gap_ratio <= 1:
            raise ValueError("invalid limit inference support")


def screen_censoring(records, policy=None):
    """Freeze source-fingerprinted decisions using a caller-declared source scope.

    Returns decisions and a compact panel audit. An isolated repeated internal
    value is retained as exact. Extreme heaps with insufficient directional
    evidence require review. Strong statistical floors/ceilings use inclusive
    inequalities and remain explicitly inferred. Explicit bounds are unchanged.
    """
    policy = CensorScreenPolicy() if policy is None else policy
    panels = defaultdict(list)
    seen = set()
    for raw in records:
        if (
            raw.get("response_measured") not in CONCENTRATION_RESPONSES
            or raw.get("measurement_units") not in NM
        ):
            continue
        parsed = reported_measurement(raw)
        value = raw["quantitative_value"] * NM[raw["measurement_units"]]
        if value <= 0:
            continue
        identity = raw.get("assay_iri")
        study = raw.get("reference_iri") or raw.get("pmid")
        if (
            not identity
            or not study
            or not raw.get("mhc_restriction")
            or not raw.get("assay_method")
        ):
            continue
        if identity in seen:
            raise MeasurementCurationError("duplicate_assay_identity_in_censor_screen")
        seen.add(identity)
        key = (
            str(study),
            raw.get("mhc_class"),
            raw["mhc_restriction"],
            raw["assay_method"],
            raw["response_measured"],
        )
        panels[key].append(
            dict(
                assay_iri=identity,
                source_sha256=source_fingerprint(raw),
                value=value,
                qualifier=parsed["effective_qualifier"],
                qualitative=(raw.get("qualitative_measurement") or "").lower(),
            )
        )
    decisions, audit = {}, []
    for panel, rows in panels.items():
        if len(rows) < policy.min_panel_rows:
            continue
        values = sorted({r["value"] for r in rows})
        if len(values) < policy.min_distinct_values:
            continue
        for direction, value in (("floor", values[0]), ("ceiling", values[-1])):
            at_value = [r for r in rows if r["value"] == value]
            exact = [r for r in at_value if r["qualifier"] == "="]
            fraction = len(exact) / len(rows)
            if len(exact) < policy.min_heap_rows or fraction < policy.min_heap_fraction:
                continue
            same = ("<", "<=") if direction == "floor" else (">", ">=")
            opposite = (">", ">=") if direction == "floor" else ("<", "<=")
            same_bounds = sum(r["qualifier"] in same for r in at_value)
            opposite_bounds = sum(r["qualifier"] in opposite for r in at_value)
            label = "positive" if direction == "floor" else "negative"
            agree = sum(r["qualitative"].startswith(label) for r in exact) / len(exact)
            gap = values[1] / value if direction == "floor" else value / values[-2]
            qualitative_support = (
                fraction >= policy.inference_heap_fraction
                and agree >= policy.min_qualitative_agreement
                and (direction == "ceiling" or gap >= policy.floor_gap_ratio)
            )
            inferred = not opposite_bounds and (same_bounds > 0 or qualitative_support)
            evidence = dict(
                panel=list(panel),
                direction=direction,
                value_nM=value,
                panel_rows=len(rows),
                exact_heap_rows=len(exact),
                heap_fraction=fraction,
                same_direction_explicit_bounds=same_bounds,
                opposite_direction_bounds=opposite_bounds,
                qualitative_agreement=agree,
                neighbor_gap_ratio=gap,
                screen_policy=asdict(policy),
                interpretation="Statistical source-panel inference; not a verified assay protocol.",
            )
            status = "inferred_panel_limit" if inferred else "needs_review"
            qualifier = ("<=" if direction == "floor" else ">=") if inferred else None
            audit.append({**evidence, "status": status, "effective_qualifier": qualifier})
            for row in exact:
                decisions[row["assay_iri"]] = CensorDecision(
                    row["assay_iri"],
                    row["source_sha256"],
                    qualifier,
                    status,
                    evidence,
                )
    return decisions, audit


@dataclass(frozen=True)
class SharedLimitPolicy:
    min_source_components: int = 3
    min_explicit_bound_components: int = 2
    min_affected_rows: int = 100
    min_directional_agreement: float = 0.95

    def __post_init__(self):
        if (
            self.min_source_components < 3
            or self.min_explicit_bound_components < 2
            or self.min_affected_rows < 100
            or not 0.95 <= self.min_directional_agreement <= 1
        ):
            raise ValueError("shared-limit evidence must be stricter than local screening")


def summarize_shared_limits(panels, source_components, verified_protocols=None, policy=None):
    """Summarize independent support; never issue or transfer a censor decision.

    Maps use the complete local panel tuple as key. A verified protocol mapping
    supplies ``family``, ``reference`` and ``locator``; raw assay tags alone do
    not establish equivalence. Missing component identities cannot count as
    independent sources. Explicit bounds in a panel support its direction but
    do not establish that every exact observation there was censored.
    """
    policy = SharedLimitPolicy() if policy is None else policy
    verified_protocols = verified_protocols or {}
    grouped = defaultdict(list)
    for panel in panels:
        key = tuple(panel["panel"])
        review = verified_protocols.get(key)
        verified = isinstance(review, dict) and all(
            isinstance(review.get(k), str) and review[k].strip()
            for k in ("family", "reference", "locator")
        )
        family = review["family"] if verified else key[3]
        group = (panel["direction"], panel["value_nM"], family, key[4], verified)
        grouped[group].append((panel, source_components.get(key)))
    result = []
    for (direction, value, family, quantity, verified), entries in sorted(grouped.items()):
        sources = {component for _, component in entries if component}
        explicit = {
            component
            for panel, component in entries
            if component and panel["same_direction_explicit_bounds"] > 0
        }
        affected = sum(p["exact_heap_rows"] for p, _ in entries)
        concordance = (
            sum(p["exact_heap_rows"] * p["qualitative_agreement"] for p, _ in entries) / affected
        )
        conflicts = sum(p["opposite_direction_bounds"] for p, _ in entries)
        gates = dict(
            independent_sources=len(sources) >= policy.min_source_components,
            explicit_bound_sources=len(explicit) >= policy.min_explicit_bound_components,
            affected_rows=affected >= policy.min_affected_rows,
            directional_agreement=concordance >= policy.min_directional_agreement,
            no_opposing_bounds=conflicts == 0,
            verified_protocol_family=verified,
            complete_independence_metadata=all(c for _, c in entries),
        )
        result.append(
            dict(
                direction=direction,
                value_nM=value,
                assay_family=family,
                quantity=quantity,
                local_panels=len(entries),
                affected_rows=affected,
                independent_source_components=len(sources),
                source_components_with_explicit_bounds=len(explicit),
                directional_agreement=concordance,
                gates=gates,
                status="shared_review_candidate" if all(gates.values()) else "local_only",
                policy=asdict(policy),
                transfers_applied=0,
            )
        )
    return result
