"""Declared output identities and source evidence survive real model/data paths."""

import random
import sqlite3
from copy import deepcopy
from dataclasses import asdict, replace

import pytest
import torch

from presto.data.bulk_ms import BulkMSRecord
from presto.data.collate import PrestoCollator, PrestoSample, TCR_EVIDENCE_METHOD_BINS
from presto.data.label_provenance import record_target_provenance
from presto.data.loaders import PrestoDataset
from presto.legacy.presto import Presto
from presto.training.data_support import audit_split_support
from presto.training.output_contract import (
    RESIDUAL_MODES,
    OutputConfiguration,
    build_output_contract,
    declared_parameter_rows,
    label_evidence,
)
from presto.training.output_coverage import OutputCoverageCensus, write_output_coverage_artifacts
from presto.training.supported_outputs import evaluate_supported_outputs, require_supported_outputs


@pytest.fixture(scope="module", autouse=True)
def single_thread_model_checks():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def sample(**kwargs):
    values = dict(
        peptide="ACDEFGHIKLM",
        mhc_a="ACDEFGHIK",
        mhc_b="LMNPQRSTV",
        mhc_class="I",
        species="human",
        sample_id="first",
        evidence_row_id="assay-1",
        sample_source="iedb",
        primary_allele="HLA-A*02:01",
        source_mhc_alleles=("HLA-A*02:01",),
        resolved_mhc_alleles=("HLA-A*02:01",),
        target_provenance=record_target_provenance(
            "iedb", "binding", "elution", "tcell", "tcr_evidence"
        ),
    )
    values.update(kwargs)
    return PrestoSample(**values)


@pytest.mark.parametrize("topology", ["collapsed", "expanded"])
@pytest.mark.parametrize("residual", RESIDUAL_MODES)
@pytest.mark.parametrize("grouping", ["merged_kd", "split_kd_proxy"])
def test_actual_model_declares_every_output_and_alias(topology, residual, grouping):
    config = OutputConfiguration(
        latent_topology=topology, affinity_assay_residual_mode=residual, kd_grouping_mode=grouping
    )
    model = Presto(d_model=32, n_layers=1, n_heads=4, **asdict(config)).eval()
    batch = PrestoCollator()([sample()])
    with torch.no_grad():
        outputs = model(**batch.model_inputs())
    contract = build_output_contract(config)
    contract.validate_outputs(outputs)
    assert declared_parameter_rows(contract, model)
    assert len(contract.objectives) == 48
    assert contract.objectives["mil:ms"].endpoint == contract.objectives["mil:elution"].endpoint
    assert "ms_logit" not in contract.outputs
    if residual == "pooled_single_output":
        assert contract.objectives["row:binding_ic50"].endpoint == "assays.KD_nM"


def test_alternate_encoding_context_and_multiple_core_lengths():
    config = OutputConfiguration(
        affinity_target_encoding="mhcflurry",
        core_window_lengths=(8, 9, 10),
        binding_direct_segment_mode="gated_affinity",
        binding_kinetic_input_mode="fused",
    )
    model = Presto(d_model=32, n_layers=1, n_heads=4, **asdict(config)).eval()
    with torch.no_grad():
        outputs = model(**PrestoCollator()([sample()]).model_inputs())
    build_output_contract(config).validate_outputs(outputs)
    # Caller routing hints are allowed to disagree with the inferred class.
    assert not torch.equal(outputs["mhc_class_probs"], outputs["mhc_class_probs_inferred"])


def test_output_additions_and_alias_drift_fail():
    contract = build_output_contract()
    model = Presto(d_model=32, n_layers=1, n_heads=4).eval()
    with torch.no_grad():
        outputs = model(**PrestoCollator()([sample()]).model_inputs())
    outputs["assays"]["new_measurement"] = torch.zeros(1)
    with pytest.raises(ValueError, match="undeclared=.*new_measurement"):
        contract.validate_outputs(outputs)
    del outputs["assays"]["new_measurement"]
    outputs["ms_logit"] = outputs["ms_logit"] + 1
    with pytest.raises(ValueError, match="ms_logit: differs"):
        contract.validate_outputs(outputs)


def add(census, samples, split="train"):
    census.add_batch(split, samples, PrestoCollator()(samples))


def test_aliases_repeated_kd_losses_and_expanded_source_rows_do_not_inflate_unique_evidence(
    tmp_path,
):
    samples = [
        sample(bind_value=25, bind_measurement_type="KD", binding_assay_type="KD", elution_label=1)
    ]
    samples.append(
        replace(samples[0], sample_id="second-mapping", resolved_mhc_alleles=("HLA-A*03:01",))
    )
    with OutputCoverageCensus(build_output_contract()) as census:
        add(census, samples)
        counts = census.counts("train", "assays.KD_nM")
        assert counts["observations"] == counts["training_rows"] == 2
        assert counts["unique_observations"] == counts["source_rows"] == 1
        assert counts["unique_exact"] == counts["distinct_peptides"] == 1
        assert counts["distinct_resolved_alleles"] == 2
        stored_targets = census.db.execute(
            "SELECT target FROM observations WHERE endpoint='assays.KD_nM'"
        ).fetchall()
        assert [row[0] for row in stored_targets] == pytest.approx([1.39794001, 1.39794001])
        assert census.objective_hits["train"]["row:binding"]["observations"] == 2
        assert census.objective_hits["train"]["row:binding_kd"]["observations"] == 2
        assert census.counts("train", "elution_logit")["observations"] == 2
        assert census.counts("train", "elution_logit", roles=["direct"])["observations"] == 2
        assert census.counts("train", "presentation_logit", roles=["direct"])["observations"] == 0
        report = census.report()
        assert report["splits"]["train"]["outputs"]["recognition_cd4_logit"]["observations"] == 0
        artifacts = write_output_coverage_artifacts(tmp_path, report)
        assert "recognition_cd4_logit" in artifacts["output_coverage_csv"].read_text()


def test_censoring_vector_components_and_class_incidence():
    samples = [
        sample(
            bind_value=25,
            bind_qual=-1,
            binding_assay_type="IC50",
            tcr_evidence_label=1,
            tcr_evidence_method_bins=(TCR_EVIDENCE_METHOD_BINS[0],),
        ),
        sample(
            sample_id="second",
            evidence_row_id="assay-2",
            bind_value=100,
            bind_qual=1,
            binding_assay_type="KD",
            mhc_class="II",
        ),
    ]
    with OutputCoverageCensus(build_output_contract()) as census:
        add(census, samples)
        kd = census.counts("train", "assays.KD_nM")
        assert kd["left_censored"] == kd["right_censored"] == 1
        assert kd["exact"] == 0
        assert census.counts("train", "assays.KD_nM", roles=["direct"])["observations"] == 1
        assert census.counts("train", "assays.IC50_nM", roles=["direct"])["observations"] == 1
        vector = census.counts("train", "tcr_evidence_method_logits")
        assert vector["observations"] == vector["unique_observations"] == 3
        for index, name in enumerate(TCR_EVIDENCE_METHOD_BINS):
            counts = census.counts("train", "tcr_evidence_method_logits", name)
            assert counts["observations"] == 1
            assert counts["positive"] == (index == 0)
        for name in ("I", "II"):
            counts = census.counts("train", "mhc_class_logits", name)
            assert counts["observations"] == 2
            assert counts["positive"] == counts["negative"] == 1


BALANCE_FIELDS = ("positive", "negative", "unique_positive", "unique_negative")


def test_class_columns_reuse_invariant_counts_only_within_a_reuse_block():
    with OutputCoverageCensus(build_output_contract()) as census:
        add(census, [sample()])
        queries = []
        census.db.set_trace_callback(queries.append)
        with census.reusing_counts():
            base = census.counts("train", "mhc_class_logits")
            first = census.counts("train", "mhc_class_logits", "I")
            second = census.counts("train", "mhc_class_logits", "II")
            reused = len(queries)
            assert first["positive"] == second["negative"] == 1
            assert first["negative"] == second["positive"] == 0
            assert first["facets"] == second["facets"] == base["facets"]
            first["facets"].clear()
            base["observations"] = -1
            assert census.counts("train", "mhc_class_logits")["observations"] == 1
            assert census.counts("train", "mhc_class_logits", "I")["facets"]
        queries.clear()
        census.counts("train", "mhc_class_logits", "I")
        census.db.set_trace_callback(None)
        # Inside the block: one population aggregate, two facet queries and one grouped
        # balance for every class. Outside it nothing is cached, and a class column costs
        # the population pass and its facet queries, as before reuse.
        assert (reused, len(queries)) == (4, 3)

        add(census, [sample(sample_id="second", evidence_row_id="assay-2", mhc_class="II")])
        updated = census.counts("train", "mhc_class_logits", "I")
        assert updated["observations"] == 2
        assert updated["positive"] == updated["negative"] == 1


def test_cold_class_column_counts_its_class_in_the_population_pass():
    with OutputCoverageCensus(build_output_contract()) as census:
        add(
            census,
            [sample(), sample(sample_id="second", evidence_row_id="assay-2", mhc_class="II")],
        )
        queries = []
        census.db.set_trace_callback(queries.append)
        with census.reusing_counts():
            second = census.counts("train", "mhc_class_logits", "II", sources=["iedb"])
            cold = len(queries)
            first = census.counts("train", "mhc_class_logits", "I", sources=["iedb"])
            census.db.set_trace_callback(None)
            # The cached population keeps binary balance fields, not the requested class's.
            population = census.counts("train", "mhc_class_logits", sources=["iedb"])
        # A cold column needs only the population pass and its two facet queries, as
        # before reuse; the next column adds one grouped balance for every class.
        assert (cold, len(queries)) == (3, 4)
        assert (second["positive"], second["negative"]) == (1, 1)
        assert (first["positive"], first["negative"]) == (1, 1)
        assert population["positive"] == 0


def test_class_response_uniqueness_preserves_repeated_and_conflicting_source_rows():
    with OutputCoverageCensus(build_output_contract()) as census:
        add(
            census,
            [sample(), sample(sample_id="repeat"), sample(sample_id="conflicting", mhc_class="II")],
        )
        for column in ("I", "II"):
            counts = census.counts("train", "mhc_class_logits", column)
            assert counts["observations"] == 3
            assert counts["unique_observations"] == 2
            assert counts["unique_positive"] == counts["unique_negative"] == 1
            assert counts["positive"] == (2 if column == "I" else 1)
            assert counts["negative"] == (1 if column == "I" else 2)


@pytest.mark.parametrize(
    "filters,expected",
    [
        ({}, 2),
        ({"sources": ["iedb"]}, 1),
        ({"sources": ["other"]}, 1),
        ({"sources": []}, 0),
        ({"roles": ["auxiliary"]}, 2),
        ({"roles": ["direct"]}, 0),
        ({"families": ["absent"]}, 0),
    ],
)
@pytest.mark.parametrize("include_facets", [False, True])
def test_class_count_reuse_keeps_filters_and_facet_options_separate(
    filters, expected, include_facets
):
    with OutputCoverageCensus(build_output_contract()) as census:
        add(census, [sample(), sample(sample_id="other", sample_source="other", mhc_class="II")])
        # Prime the unfiltered, facet-bearing result before requesting another selection.
        census.counts("train", "mhc_class_logits")
        counts = census.counts(
            "train", "mhc_class_logits", "I", include_facets=include_facets, **filters
        )
        assert counts["observations"] == expected
        assert ("facets" in counts) is include_facets
        assert counts["positive"] + counts["negative"] == expected
        assert census.counts("val", "mhc_class_logits", "I")["observations"] == 0


def test_class_balances_match_per_class_definitions_for_every_selection():
    """One grouped balance query equals the per-class SQL it replaces.

    Rows are written directly so keys can carry conflicting targets and targets
    can match no class index, which collated samples never produce.
    """
    rng = random.Random(0)
    contract = build_output_contract()
    endpoint = "species_of_origin_logits"
    classes = contract.outputs[endpoint].columns
    with OutputCoverageCensus(contract) as census:
        for row_id in range(1, 61):
            census.db.execute(
                "INSERT INTO samples VALUES (?,?,?,?,?,?,?)",
                (
                    row_id,
                    "train",
                    f"s{row_id}",
                    rng.choice(["iedb", "other"]),
                    f"o{row_id}",
                    1,
                    "P",
                ),
            )
            for occurrence in range(rng.choice([1, 2, 3])):
                target = rng.choice([*range(len(classes)), -1, 0.5, len(classes)])
                census.db.execute(
                    "INSERT INTO observations(row_id,endpoint,column_name,role,family,origin,"
                    "observation_key,target,qualifier,loss_type,instances,scope,occurrence) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        row_id,
                        endpoint,
                        "",
                        rng.choice(["direct", "proxy"]),
                        "measured",
                        "x",
                        f"k{rng.randrange(20)}",
                        float(target),
                        None,
                        "ce",
                        1,
                        "row",
                        occurrence,
                    ),
                )
        census.db.commit()
        conflicting = census.db.execute(
            "SELECT COUNT(*) FROM (SELECT observation_key FROM observations "
            "GROUP BY observation_key HAVING COUNT(DISTINCT target) > 1)"
        ).fetchone()[0]
        assert conflicting > 0
        for filters in (
            {},
            {"roles": ["direct"]},
            {"sources": ["iedb", "other"]},
            {"sources": ["other"]},
            {"roles": []},
        ):
            where, args = census._selection(
                "train", endpoint, "", filters.get("roles"), None, filters.get("sources")
            )
            expected = {
                column: census.db.execute(
                    f"""SELECT COALESCE(SUM(o.target={index}),0),
                        COALESCE(SUM(o.target!={index}),0),
                        COUNT(DISTINCT CASE WHEN o.target={index} THEN o.observation_key END),
                        COUNT(DISTINCT CASE WHEN o.target!={index} THEN o.observation_key END)
                    FROM observations o JOIN samples s ON s.id=o.row_id WHERE {where}""",
                    args,
                ).fetchone()
                for index, column in enumerate(classes)
            }
            options = dict(include_facets=False, **filters)
            # Outside a reuse block every class is counted in the population pass.
            folded = {c: census.counts("train", endpoint, c, **options) for c in classes}
            # Inside one, a cached population sends every class through the grouped balance.
            with census.reusing_counts():
                census.counts("train", endpoint, **options)
                grouped = {c: census.counts("train", endpoint, c, **options) for c in classes}
            for column in classes:
                for path in (folded, grouped):
                    balance = tuple(path[column][field] for field in BALANCE_FIELDS)
                    assert balance == expected[column], (filters, column)


@pytest.mark.parametrize("sources", [("iedb", "iedb"), ("iedb", "other")])
def test_report_evidence_groups_equal_filtered_counts_and_stay_independent(sources):
    contract = build_output_contract()
    contract.outputs = {
        name: contract.outputs[name] for name in ("assays.KD_nM", "mhc_class_logits")
    }
    with OutputCoverageCensus(contract) as census:
        add(
            census,
            [
                sample(
                    sample_id=str(i),
                    evidence_row_id=str(i),
                    sample_source=source,
                    mhc_class=("I", "II")[i],
                    bind_value=25,
                    bind_measurement_type="KD",
                    binding_assay_type="KD",
                )
                for i, source in enumerate(sources)
            ],
        )
        queries = []
        census.db.set_trace_callback(queries.append)
        outputs = census.report()["splits"]["train"]["outputs"]
        census.db.set_trace_callback(None)
        groups = len(set(sources))
        # Each endpoint runs a population aggregate, two facet queries and the group
        # query; the class endpoint adds one balance. A sole group reuses those counts,
        # while every group of a mixed endpoint repeats them under its own filters.
        expected = 4 + 5 if groups == 1 else (4 + 2 * 3) + (5 + 2 * 4)
        assert len(queries) == expected
        for endpoint, entry in outputs.items():
            columns = contract.outputs[endpoint].columns
            assert len(entry["evidence"]) == groups
            for group in entry["evidence"]:
                filters = dict(
                    roles=[group["role"]], families=[group["family"]], sources=[group["source"]]
                )
                assert "evidence" not in group
                own = {k: v for k, v in group.items() if k not in ("role", "family", "source")}
                assert own.pop("columns") == {
                    column: census.counts("train", endpoint, column, **filters)
                    for column in columns
                }
                assert own == census.counts("train", endpoint, **filters)
            totals = deepcopy({k: v for k, v in entry.items() if k != "evidence"})
            for group in entry["evidence"]:
                group["facets"].clear()
                for column in columns:
                    group["columns"][column]["facets"].clear()
            assert {k: v for k, v in entry.items() if k != "evidence"} == totals


def table_rows(census):
    return tuple(
        census.db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        for table in ("samples", "facets", "observations")
    )


def failing_batch(census):
    """Add a batch whose second sample is rejected after observations were written."""
    statements = []
    census.db.set_trace_callback(statements.append)
    with pytest.raises(ValueError, match="nonfinite active observation"):
        add(
            census,
            [
                sample(sample_id="good", evidence_row_id="good", mhc_class="II"),
                sample(sample_id="bad", evidence_row_id="bad", elution_label=float("nan")),
            ],
        )
    census.db.set_trace_callback(None)
    assert any("INTO observations" in statement for statement in statements)


def test_failed_batch_leaves_census_and_counts_unchanged():
    with OutputCoverageCensus(build_output_contract()) as census:
        add(census, [sample()])
        before = {c: census.counts("train", "mhc_class_logits", c) for c in ("", "I", "II")}
        state = (dict(census.splits), census.sha256, deepcopy(census.objective_hits))
        rows = table_rows(census)
        failing_batch(census)
        assert table_rows(census) == rows
        assert (dict(census.splits), census.sha256, census.objective_hits) == state
        assert {c: census.counts("train", "mhc_class_logits", c) for c in ("", "I", "II")} == before
        add(census, [sample(sample_id="good", evidence_row_id="good", mhc_class="II")])
        assert census.counts("train", "mhc_class_logits", "II")["positive"] == 1


def test_failed_batch_undoes_only_its_own_writes():
    with OutputCoverageCensus(build_output_contract()) as census:
        add(
            census,
            [sample(), sample(sample_id="second", evidence_row_id="assay-2", mhc_class="II")],
        )
        census.db.execute("DELETE FROM observations WHERE endpoint='mhc_class_logits' AND target=1")
        rows = table_rows(census)
        failing_batch(census)
        # The earlier uncommitted delete survives the failed batch and still rolls back.
        assert table_rows(census) == rows
        census.db.rollback()
        assert table_rows(census)[2] == rows[2] + 1


def test_failed_batch_restores_affinity_audit_counts_and_clipped_rows():
    with OutputCoverageCensus(build_output_contract()) as census:
        add(census, [sample(bind_value=100000, bind_measurement_type="IC50")])
        audit = census.affinity_target_audit
        counts = deepcopy(audit.counts)
        changes = census.db.execute("SELECT * FROM affinity_target_changes").fetchall()
        assert changes
        rows = [
            sample(
                sample_id="clipped",
                evidence_row_id="clipped",
                bind_value=200000,
                bind_measurement_type="IC50",
                elution_label=float("nan"),
            )
        ]
        with pytest.raises(ValueError, match="nonfinite active observation"):
            add(census, rows)
        assert census.db.execute("SELECT * FROM affinity_target_changes").fetchall() == changes
        assert audit.counts == counts


def test_counts_after_a_report_follow_direct_writes_and_rollbacks():
    with OutputCoverageCensus(build_output_contract()) as census:
        add(
            census,
            [sample(), sample(sample_id="second", evidence_row_id="assay-2", mhc_class="II")],
        )
        census.report()
        assert census.counts("train", "mhc_class_logits", "I")["observations"] == 2
        census.db.execute("DELETE FROM observations WHERE endpoint='mhc_class_logits' AND target=1")
        removed = census.counts("train", "mhc_class_logits", "I")
        assert (removed["observations"], removed["positive"], removed["negative"]) == (1, 1, 0)
        census.db.rollback()
        restored = census.counts("train", "mhc_class_logits", "I")
        assert (restored["observations"], restored["positive"], restored["negative"]) == (2, 1, 1)


def test_counts_fail_on_a_closed_census_for_every_endpoint_shape():
    census = OutputCoverageCensus(build_output_contract())
    add(census, [sample()])
    census.report()
    census.close()
    for request in (("mhc_class_logits",), ("mhc_class_logits", "I"), ("elution_logit",)):
        with pytest.raises(sqlite3.ProgrammingError):
            census.counts("train", *request)


def test_class_count_reuse_distinguishes_values_sqlite_compares_differently():
    with OutputCoverageCensus(build_output_contract()) as census:
        add(census, [sample(sample_source="1")])
        with census.reusing_counts():
            # Against the TEXT source column SQLite compares 1 as '1' but 1.0 as '1.0'.
            assert census.counts("train", "mhc_class_logits", "I", sources=[1])["observations"] == 1
            assert (
                census.counts("train", "mhc_class_logits", "I", sources=[1.0])["observations"] == 0
            )


def test_class_columns_call_an_overridden_counts_once():
    class CountingCensus(OutputCoverageCensus):
        calls = 0

        def counts(self, *args, **kwargs):
            self.calls += 1
            return super().counts(*args, **kwargs)

    with CountingCensus(build_output_contract()) as census:
        add(census, [sample()])
        census.counts("train", "mhc_class_logits", "I")
        census.counts("train", "mhc_class_logits", "II")
        assert census.calls == 2


def test_actual_bulk_conversion_preserves_generated_and_proxy_origins():
    dataset = PrestoDataset(
        bulk_ms_records=[
            BulkMSRecord(
                peptide="ACDEFGHIK", detectability_label=0.7, excision_label=1, observed=True
            ),
            BulkMSRecord(
                peptide="ACDEFGHIK",
                machinery="lysc",
                excision_label=0,
                observed=False,
                generated_kind="bulk_wrong_enzyme",
            ),
        ]
    )
    assert [s.bulk_ms_observed for s in dataset] == [True, False]
    # Existing sampler grouping is preserved; provenance is separate metadata.
    assert all(s.synthetic_kind is None for s in dataset)
    batch = PrestoCollator()([*dataset])
    assert "target_provenance" not in batch.model_inputs()
    with OutputCoverageCensus(build_output_contract()) as census:
        census.add_batch("train", [*dataset], batch)
        assert census.counts("train", "excision_logit", roles=["proxy"])["positive"] == 1
        assert census.counts("train", "excision_logit", roles=["synthetic"])["negative"] == 1
        assert census.counts("train", "excision_logit", roles=["direct"])["observations"] == 0
        # An explicitly supplied label remains visible, but observing its
        # peptide does not establish a qualified detectability source.
        assert census.counts("train", "ms_detectability_logit", roles=["unknown"])["graded"] == 1
        assert (
            census.counts("train", "ms_detectability_logit", roles=["proxy"])["observations"] == 0
        )


@pytest.mark.parametrize(
    "generated", ["synthetic", "synthetic_negative_cascade", "synthetic_negative_peptide_scramble"]
)
def test_typed_generated_records_and_missing_provenance_are_not_direct(generated):
    objective = build_output_contract().objectives["row:binding"]
    row = sample(
        bind_value=25,
        binding_assay_type="KD",
        sample_source=generated,
        target_provenance=record_target_provenance(generated, "binding"),
    )
    assert label_evidence(objective, row).role == "synthetic"
    row = replace(row, sample_source="iedb", target_provenance={})
    assert label_evidence(objective, row).role == "unknown"


def manifest_for(contract, *, columns=None):
    from presto.training.target_policy import affinity_target_policies

    return {
        "schema_version": 1,
        "model_contract": asdict(contract.configuration),
        "source_contract": {
            "data_source": "merged_tsv",
            "data_seed": 7,
            "affinity_target_policies": affinity_target_policies(contract),
            "molecular_mhc_species_policy": contract.to_dict()["molecular_mhc_species_policy"],
            "processing_species_policy": contract.to_dict()["processing_species_policy"],
            "assay_context_policy": contract.to_dict()["assay_context_policy"],
        },
        "claims": [
            {
                "endpoint": "tcell_panel_logits.assay_method",
                "columns": columns or ["ELISPOT", "ICS"],
                "required_splits": ["train", "val", "test"],
                "evidence_roles": ["direct"],
                "source_families": ["tcell"],
                "raw_sources": ["iedb"],
                "minimums": {
                    "unique_observations": 2,
                    "distinct_peptides": 2,
                    "unique_positive": 1,
                    "unique_negative": 1,
                },
            }
        ],
    }


def test_prospective_gate_rejects_each_one_class_column_and_reports_global_zeros():
    contract = build_output_contract()
    manifest = manifest_for(contract)
    with OutputCoverageCensus(contract) as census:
        for split in ("train", "val", "test"):
            add(
                census,
                [
                    sample(tcell_label=1, tcell_assay_method="ELISPOT"),
                    sample(
                        sample_id="second",
                        evidence_row_id="assay-2",
                        peptide="LMNPQRSTV",
                        tcell_label=0,
                        tcell_assay_method="ICS",
                    ),
                ],
                split,
            )
        report = evaluate_supported_outputs(
            census, manifest, source_contract=manifest["source_contract"]
        )
        with pytest.raises(RuntimeError, match="unique_negative=0"):
            require_supported_outputs(report)
        decisions = {(x["endpoint"], x["column"]): x for x in report["outputs"]}
        assert (
            decisions[("tcell_panel_logits.assay_method", "ELISPOT")]["status"]
            == "requirements_not_met"
        )
        assert decisions[("core_start_logit", "")]["status"] == "undeclared"
        assert not evaluate_supported_outputs(census, None, source_contract={})["requirements_met"]


def test_passing_gate_is_scoped_and_missing_splits_or_alias_claims_fail():
    contract = build_output_contract()
    manifest = manifest_for(contract, columns=["ELISPOT"])
    with OutputCoverageCensus(contract) as census:
        for split in ("train", "val", "test"):
            add(
                census,
                [
                    sample(tcell_label=1, tcell_assay_method="ELISPOT"),
                    sample(
                        sample_id="second",
                        evidence_row_id="assay-2",
                        peptide="LMNPQRSTV",
                        tcell_label=0,
                        tcell_assay_method="ELISPOT",
                    ),
                ],
                split,
            )
        report = evaluate_supported_outputs(
            census, manifest, source_contract=manifest["source_contract"]
        )
        require_supported_outputs(report)
        assert sum(x["status"] == "requirements_met" for x in report["outputs"]) == 1
        manifest["claims"][0]["required_splits"].append("unprovided")
        report = evaluate_supported_outputs(
            census, manifest, source_contract=manifest["source_contract"]
        )
        with pytest.raises(RuntimeError, match="unprovided was not supplied"):
            require_supported_outputs(report)
        manifest["claims"][0]["endpoint"] = "ms_logit"
        with pytest.raises(ValueError, match="alias of elution_logit"):
            evaluate_supported_outputs(
                census, manifest, source_contract=manifest["source_contract"]
            )


def test_bag_observations_remain_bags_and_census_is_order_and_chunk_invariant():
    bag = sample(
        tcell_label=1,
        tcell_assay_method="ELISPOT",
        use_tcell_pathway_mil=True,
        tcell_mil_mhc_a_list=["ACDEFGHIK", "ACDEFGHIK"],
        tcell_mil_mhc_b_list=["LMNPQRSTV", "LMNPQRSTV"],
        tcell_mil_mhc_class_list=["I", "II"],
    )
    second = replace(bag, sample_id="second", evidence_row_id="assay-2", tcell_label=0)
    with (
        OutputCoverageCensus(build_output_contract()) as first,
        OutputCoverageCensus(build_output_contract()) as other,
    ):
        add(first, [bag, second])
        add(other, [second])
        add(other, [bag])
        assert first.sha256 == other.sha256
        counts = first.counts("train", "tcell_panel_logits.assay_method", "ELISPOT")
        assert counts["observations"] == 2
        assert counts["instances"] == 4
        assert (
            first.objective_hits["train"]["tcell_mil:tcell_assay_method_mil"]["observations"] == 2
        )


def test_census_rejects_nonfinite_active_observations():
    with OutputCoverageCensus(build_output_contract()) as census:
        with pytest.raises(ValueError, match="nonfinite active observation"):
            add(census, [sample(bind_value=float("nan"))])


def test_class_split_elution_bags_are_two_observations_from_one_source():
    row = sample(
        elution_label=1,
        mil_mhc_a_list=["ACDEFGHIK", "ACDEFGHIK"],
        mil_mhc_b_list=["LMNPQRSTV", "LMNPQRSTV"],
        mil_mhc_class_list=["I", "II"],
    )
    with OutputCoverageCensus(build_output_contract()) as census:
        add(census, [row])
        counts = census.counts("train", "elution_logit")
        assert counts["observations"] == counts["instances"] == 2
        assert (
            counts["unique_observations"] == counts["source_rows"] == counts["training_rows"] == 1
        )
        assert census.objective_hits["train"]["mil:elution"]["observations"] == 2
        assert census.objective_hits["train"]["mil:ms"]["observations"] == 2


def test_new_evidence_metadata_has_its_own_hash_without_changing_legacy_input_hashes():
    known = sample(bind_value=25, binding_assay_type="KD")
    unknown = replace(known, target_provenance={})
    before = audit_split_support({"train": [known]})
    after = audit_split_support({"train": [unknown]})
    assert before == after
    with (
        OutputCoverageCensus(build_output_contract()) as first,
        OutputCoverageCensus(build_output_contract()) as second,
    ):
        add(first, [known])
        add(second, [unknown])
        assert first.sha256 != second.sha256
