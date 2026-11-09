from copy import deepcopy

import pytest

from presto.data.measurement_curation import (
    CensorDecision,
    MeasurementCurationError,
    assay_method_context,
    curated_measurement,
    reported_measurement,
    screen_censoring,
    source_fingerprint,
)


def row(identity, value, qualifier=None, qualitative="Positive", unit="nM", study="study"):
    return dict(
        assay_iri=f"assay:{identity}",
        reference_iri=study,
        mhc_restriction="HLA-A*02:01",
        mhc_class="I",
        peptide="AAAAAAAAA",
        assay_method="purified MHC/competitive/radioactivity",
        response_measured="half maximal inhibitory concentration (IC50)",
        quantitative_value=float(value),
        quantitative_measurement=str(value),
        measurement_units=unit,
        measurement_inequality=qualifier,
        qualitative_measurement=qualitative,
    )


def panel():
    return (
        [row(i, 1) for i in range(8)]
        + [row(i + 8, 50000, qualitative="Negative") for i in range(8)]
        + [row(i + 16, 10 * (i + 1)) for i in range(14)]
    )


@pytest.mark.parametrize(
    "text,expected", [("≤ 1", "<="), ("<1", "<"), (">= 50000", ">="), (">50000", ">")]
)
def test_embedded_inequality_is_used_when_column_missing(text, expected):
    raw = row(1, 1 if "1" in text else 50000)
    raw["quantitative_measurement"] = text
    assert reported_measurement(raw)["effective_qualifier"] == expected
    assert reported_measurement(raw)["basis"] == "numeric_text"


def test_explicit_exact_conflicting_with_numeric_text_is_not_overridden():
    raw = {**row(1, 1, "="), "quantitative_measurement": "<1"}
    with pytest.raises(MeasurementCurationError, match="conflicting_reported"):
        reported_measurement(raw)


def test_missing_qualifier_is_preserved_as_an_assumption():
    result = reported_measurement(row(1, 20))
    assert result["reported_qualifier"] is None
    assert result["effective_qualifier"] == "="
    assert result["status"] == "unspecified_assumed_exact"


def test_upper_and_lower_plateaus_yield_inferred_bounds_without_mutating_source():
    rows = panel()
    original = deepcopy(rows)
    decisions, audit = screen_censoring(iter(rows))
    assert len(decisions) == 16 and len(audit) == 2
    assert {d.status for d in decisions.values()} == {"inferred_panel_limit"}
    assert {d.qualifier for d in decisions.values()} == {"<=", ">="}
    for raw in rows[:16]:
        result = curated_measurement(raw, decisions[raw["assay_iri"]])
        assert result["effective_qualifier"] == ("<=" if raw["quantitative_value"] == 1 else ">=")
        assert result["reported_qualifier"] is None
    assert rows == original


def test_inference_is_not_tied_to_round_numbers():
    rows = panel()
    for raw in rows:
        raw["quantitative_value"] *= 1.55886
        raw["quantitative_measurement"] = str(raw["quantitative_value"])
    decisions, _ = screen_censoring(rows)
    assert len(decisions) == 16


def test_units_are_normalized_with_source_fingerprints_retained():
    rows = panel()
    for raw in rows[8:16]:
        raw.update(quantitative_value=50.0, quantitative_measurement="50.0", measurement_units="uM")
    decisions, _ = screen_censoring(rows)
    assert decisions["assay:8"].evidence["value_nM"] == 50000
    assert curated_measurement(rows[8], decisions["assay:8"])["effective_qualifier"] == ">="


def test_different_studies_and_alleles_do_not_supply_panel_support():
    rows = panel()
    for i, raw in enumerate(rows):
        raw["reference_iri"] = f"study:{i % 3}"
    assert screen_censoring(rows) == ({}, [])


def test_unresolved_extreme_is_deferred_and_internal_heap_is_not_a_limit():
    rows = panel()
    for raw in rows:
        raw["qualitative_measurement"] = ""
    decisions, _ = screen_censoring(rows)
    assert {d.status for d in decisions.values()} == {"needs_review"}
    with pytest.raises(MeasurementCurationError, match="requires_review"):
        curated_measurement(rows[0], decisions["assay:0"])
    for raw in rows[:8]:
        raw.update(quantitative_value=25.0, quantitative_measurement="25")
    decisions, _ = screen_censoring(rows)
    assert all(f"assay:{i}" not in decisions for i in range(8))


def test_conflicting_bounds_prevent_statistical_override():
    rows = panel() + [row(40, 50000, ">"), row(41, 50000, "<")]
    decisions, _ = screen_censoring(rows)
    assert decisions["assay:8"].status == "needs_review"
    assert "assay:40" not in decisions and "assay:41" not in decisions


def test_source_change_invalidates_frozen_decision():
    rows = panel()
    decisions, _ = screen_censoring(rows)
    raw = {**rows[0], "peptide": "CCCCCCCCC"}
    with pytest.raises(MeasurementCurationError, match="source_changed"):
        curated_measurement(raw, decisions["assay:0"])


def test_confirmed_limit_requires_reference_and_preserves_explicit_bounds():
    raw = row(1, 20000)
    with pytest.raises(ValueError, match="reference"):
        CensorDecision(
            "assay:1", source_fingerprint(raw), ">=", "source_confirmed", {"reason": "heap"}
        )
    raw["measurement_inequality"] = ">"
    decision = CensorDecision(
        "assay:1",
        source_fingerprint(raw),
        ">=",
        "source_confirmed",
        {"reference": "fixture-paper", "locator": "fixture-table"},
    )
    with pytest.raises(MeasurementCurationError, match="reported_bound"):
        curated_measurement(raw, decision)


def test_quantity_does_not_establish_direct_method():
    raw = {
        **row(1, 20),
        "response_measured": "dissociation constant KD",
        "assay_method": "purified MHC/competitive/fluorescence",
        "assay_comments": "See PMID:12135423",
    }
    context = assay_method_context(raw)
    assert not context["quantity_establishes_direct_method"]
    assert context["review_status"] == "protocol_citation_conflict"
    assert context["mechanism"] == "competitive"  # preserve the original tag


def test_shared_thresholds_need_independent_sources_and_verified_protocols():
    from presto.data.measurement_curation import summarize_shared_limits

    panels = []
    for index in range(3):
        panels.append(
            dict(
                panel=[f'ref:{index}', 'I', f'MHC:{index}', 'reported/method', 'IC50'],
                direction='ceiling',
                value_nM=20000,
                exact_heap_rows=40,
                qualitative_agreement=1.0,
                same_direction_explicit_bounds=1,
                opposite_direction_bounds=0,
            )
        )
    same_study = {tuple(p['panel']): 'study:1' for p in panels}
    independent = {tuple(p['panel']): f'study:{i}' for i, p in enumerate(panels)}
    protocols = {
        tuple(p['panel']): dict(
            family='verified fixture protocol', reference='fixture', locator='method section'
        )
        for p in panels
    }
    a = summarize_shared_limits(panels, same_study, protocols)[0]
    assert a['independent_source_components'] == 1 and a['status'] == 'local_only'
    b = summarize_shared_limits(panels, independent)[0]
    assert b['status'] == 'local_only' and not b['gates']['verified_protocol_family']
    c = summarize_shared_limits(panels, independent, protocols)[0]
    assert c['status'] == 'shared_review_candidate' and c['transfers_applied'] == 0
    panels[0]['opposite_direction_bounds'] = 1
    assert summarize_shared_limits(panels, independent, protocols)[0]['status'] == 'local_only'


def test_nearby_thresholds_remain_distinct_and_missing_scope_is_not_replication():
    from presto.data.measurement_curation import summarize_shared_limits, SharedLimitPolicy

    panels = [
        dict(
            panel=[f'ref:{i}', 'I', 'MHC', 'method', 'quantity'],
            direction='ceiling',
            value_nM=v,
            exact_heap_rows=100,
            qualitative_agreement=1,
            same_direction_explicit_bounds=5,
            opposite_direction_bounds=0,
        )
        for i, v in enumerate([78100, 78125])
    ]
    result = summarize_shared_limits(panels, {})
    assert len(result) == 2
    assert all(r['independent_source_components'] == 0 for r in result)
    with pytest.raises(ValueError, match='stricter'):
        SharedLimitPolicy(min_source_components=1)
