"""Declared output identities and source evidence survive real model/data paths."""

from dataclasses import asdict, replace

import pytest
import torch

from presto.data.bulk_ms import BulkMSRecord
from presto.data.collate import PrestoCollator, PrestoSample, TCR_EVIDENCE_METHOD_BINS
from presto.data.label_provenance import record_target_provenance
from presto.data.loaders import PrestoDataset
from presto.models.presto import Presto
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


def test_class_columns_reuse_invariant_counts_and_invalidate_after_appending():
    with OutputCoverageCensus(build_output_contract()) as census:
        add(census, [sample()])
        queries = []
        census.db.set_trace_callback(queries.append)
        base = census.counts("train", "mhc_class_logits")
        first = census.counts("train", "mhc_class_logits", "I")
        second = census.counts("train", "mhc_class_logits", "II")
        assert first["positive"] == second["negative"] == 1
        assert first["negative"] == second["positive"] == 0
        assert first["facets"] == second["facets"] == base["facets"]
        assert sum("GROUP BY f.field" in query for query in queries) == 2
        first["facets"].clear()
        base["observations"] = -1
        assert census.counts("train", "mhc_class_logits")["observations"] == 1
        assert census.counts("train", "mhc_class_logits", "I")["facets"]

        add(census, [sample(sample_id="second", evidence_row_id="assay-2", mhc_class="II")])
        updated = census.counts("train", "mhc_class_logits", "I")
        assert updated["observations"] == 2
        assert updated["positive"] == updated["negative"] == 1


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


@pytest.mark.parametrize("sources", [("iedb", "iedb"), ("iedb", "other")])
def test_report_reuses_only_single_group_counts_and_keeps_independent_results(sources):
    contract = build_output_contract()
    contract.outputs = {"assays.KD_nM": contract.outputs["assays.KD_nM"]}
    with OutputCoverageCensus(contract) as census:
        add(
            census,
            [
                sample(
                    sample_id=str(i),
                    evidence_row_id=str(i),
                    sample_source=source,
                    bind_value=25,
                    bind_measurement_type="KD",
                    binding_assay_type="KD",
                )
                for i, source in enumerate(sources)
            ],
        )
        queries = []
        census.db.set_trace_callback(queries.append)
        entry = census.report()["splits"]["train"]["outputs"]["assays.KD_nM"]
        groups = entry["evidence"]
        assert sum(group["observations"] for group in groups) == entry["observations"] == 2
        expected_scans = 1 if len(set(sources)) == 1 else 3
        assert sum("GROUP BY f.field,f.value" in query for query in queries) == expected_scans
        original_facets = dict(entry["facets"])
        groups[0]["facets"].clear()
        assert entry["facets"] == original_facets
        assert "evidence" not in groups[0]


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
        assert census.counts("train", "ms_detectability_logit", roles=["proxy"])["graded"] == 1


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
    return {
        "schema_version": 1,
        "model_contract": asdict(contract.configuration),
        "source_contract": {"data_source": "merged_tsv", "data_seed": 7},
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
