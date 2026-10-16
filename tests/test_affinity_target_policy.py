"""Clipped measurements remain explicit, source-linked and prospectively bound."""

import copy
import csv
import json
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest
import torch

from presto.data.collate import PrestoCollator
from presto.models.affinity import normalize_binding_target_log10
from presto.training.coverage_preflight import audit_training_coverage
from presto.training.output_contract import build_output_contract
from presto.training.output_coverage import OutputCoverageCensus
from presto.training.supervision import ROW_TASK_SPECS
from presto.training.supported_outputs import evaluate_supported_outputs
from presto.training.target_policy import AFFINITY_TARGET_POLICY, AffinityTargetPolicy
from test_output_coverage import sample


def test_direct_validator_cannot_omit_policy_from_both_contracts():
    from test_output_coverage import manifest_for

    with OutputCoverageCensus(build_output_contract()) as census:
        manifest = manifest_for(census.contract)
        del manifest["source_contract"]["affinity_target_policies"]
        with pytest.raises(ValueError, match="current canonical affinity target policies"):
            evaluate_supported_outputs(
                census, manifest, source_contract=manifest["source_contract"]
            )


def test_policy_preserves_historical_values_and_all_affinity_aliases():
    raw = torch.tensor([-1.0, 0.0, 0.0001, 0.001, 1.0, 50000.0, 50000.1, 1000000.0])
    historical = normalize_binding_target_log10(raw, max_affinity_nM=50000.0, assume_log10=False)
    assert torch.equal(AFFINITY_TARGET_POLICY(raw), historical)
    contract = build_output_contract()
    specs = [spec for spec in ROW_TASK_SPECS if spec.raw_unit == "nM"]
    assert len(specs) == 9
    for spec in specs:
        assert spec.target_transform is AFFINITY_TARGET_POLICY
        assert torch.equal(spec.target_transform(raw), historical)
        policy = contract.objectives["row:" + spec.name].target_policy
        assert policy["operation"] == "clip_then_log10"
        assert policy["minimum_nM"] == 0.001 and policy["maximum_nM"] == 50000.0
        assert policy["qualifier_policy"] == "preserve_original_direction"
    assert contract.to_dict()["schema_version"] == 4
    with pytest.raises(ValueError, match="positive and ordered"):
        AffinityTargetPolicy(minimum_nM=0)


def binding_sample(**kwargs):
    fields = dict(
        bind_value=1000000.0,
        bind_measurement_type="IC50",
        binding_assay_type="IC50",
        source_value="1000000.00000001",
        source_units="nM",
        source_inequality="=",
        source_endpoint="half maximal inhibitory concentration (IC50)",
        pmid="12345",
        doi="10.1/test",
        reference_iri="https://example.org/paper",
        assay_iri="https://example.org/assay/1",
    )
    fields.update(kwargs)
    return sample(**fields)


def test_census_distinguishes_exposures_clipped_identities_and_censor_directions(tmp_path):
    first = binding_sample()
    samples = [
        first,
        replace(first, sample_id="second-mapping"),
        binding_sample(
            sample_id="right",
            evidence_row_id="assay-2",
            bind_value=80000.0,
            bind_qual=1,
            source_value="80000",
            source_inequality=">",
        ),
        binding_sample(
            sample_id="left",
            evidence_row_id="assay-3",
            bind_value=0.0001,
            bind_qual=-1,
            source_value="0.0001",
            source_inequality="<",
        ),
        binding_sample(
            sample_id="normal", evidence_row_id="assay-4", bind_value=50000.0, source_value="50000"
        ),
        binding_sample(sample_id="inactive", evidence_row_id="assay-5"),
    ]
    batch = PrestoCollator()(samples)
    for name, mask in batch.target_masks.items():
        if name.startswith("binding"):
            mask[-1] = 0
    with OutputCoverageCensus(build_output_contract()) as census:
        census.add_batch("train", samples, batch)
        census.declare_split("val")
        audit = census.affinity_target_audit.report(census.splits)
        row = next(
            r
            for r in audit["counts"]
            if r["split"] == "train"
            and r["objective_id"] == "row:binding_ic50"
            and r["column"] == ""
        )
        assert row["n_observations"] == 5
        assert row["n_exact_observations"] == 3
        assert row["n_left_censored_observations"] == row["n_right_censored_observations"] == 1
        assert row["n_clipped_observations"] == 4
        assert row["n_clipped_unique_observations"] == 3
        assert row["n_above_maximum_observations"] == 3
        assert row["n_below_minimum_observations"] == 1
        assert row["n_clipped_exact_observations"] == 2
        assert row["n_clipped_left_censored_observations"] == 1
        assert row["n_clipped_right_censored_observations"] == 1
        assert row["n_untraceable_clipped_observations"] == 0
        assert all(r["n_observations"] == 0 for r in audit["counts"] if r["split"] == "val")
        panel = next(
            r
            for r in audit["counts"]
            if r["split"] == "train"
            and r["objective_id"] == "row:binding_assay_panel_assay_type"
            and r["column"] == "IC50"
        )
        assert panel["n_clipped_observations"] == 4
        paths = census.affinity_target_audit.write_artifacts(tmp_path, census.splits)
    with paths["affinity_target_changes"].open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert not any(r["sample_id"] in ("normal", "inactive") for r in rows)
    direct = {r["sample_id"]: r for r in rows if r["objective_id"] == "row:binding_ic50"}
    assert len(direct) == 4
    assert direct["first"]["source_value"] == "1000000.00000001"
    assert direct["first"]["collated_raw_target_nM"] == "1000000.0"
    assert float(direct["first"]["effective_target_log10_nM"]) == pytest.approx(4.698970)
    assert direct["left"]["qualifier"] == "-1" and direct["left"]["source_inequality"] == "<"
    assert direct["right"]["qualifier"] == "1" and direct["right"]["source_inequality"] == ">"
    assert direct["first"]["evidence_row_id"] == "assay-1"
    assert direct["first"]["pmid"] == "12345"
    assert direct["first"]["source_mhc_alleles"] == '["HLA-A*02:01"]'
    assert json.loads(paths["affinity_target_audit"].read_text()) == audit


def test_missing_source_identity_remains_untraceable():
    row = binding_sample(evidence_row_id="", sample_source="unknown")
    with OutputCoverageCensus(build_output_contract()) as census:
        census.add_batch("train", [row], PrestoCollator()([row]))
        entries = census.affinity_target_audit.report(census.splits)["counts"]
        direct = next(r for r in entries if r["objective_id"] == "row:binding_ic50")
        assert direct["n_untraceable_clipped_observations"] == 1
        assert direct["n_clipped_unique_observations"] == 1


def test_prospective_policy_is_bound_even_when_no_value_clips_and_failures_keep_reports(tmp_path):
    args = SimpleNamespace(data_source="hitlist", split_mode="peptide_group", seed=42)
    splits = {"train": [binding_sample(bind_value=50.0, source_value="50")]}
    audit_training_coverage(
        splits,
        collator=PrestoCollator(),
        args=args,
        data_seed=42,
        manifest=None,
        output_dir=tmp_path,
    )
    coverage = json.loads((tmp_path / "output_coverage.json").read_text())
    manifest = {
        "schema_version": 1,
        "model_contract": asdict(build_output_contract().configuration),
        "source_contract": coverage["source_contract"],
        "claims": [
            {
                "endpoint": "assays.IC50_nM",
                "columns": [""],
                "required_splits": ["train"],
                "evidence_roles": ["direct"],
                "source_families": ["binding:IC50"],
                "raw_sources": ["iedb"],
                "minimums": {"unique_observations": 1, "distinct_peptides": 1, "unique_exact": 1},
            }
        ],
    }
    _, passed = audit_training_coverage(
        splits,
        collator=PrestoCollator(),
        args=args,
        data_seed=42,
        manifest=manifest,
        output_dir=tmp_path / "valid",
    )
    assert passed["requirements_met"]
    for kind in ("missing", "changed"):
        invalid = copy.deepcopy(manifest)
        if kind == "missing":
            del invalid["source_contract"]["affinity_target_policies"]
        else:
            invalid["source_contract"]["affinity_target_policies"]["row:binding_ic50"][
                "maximum_nM"
            ] = 1000000.0
        output = tmp_path / kind
        with pytest.raises(ValueError, match="source_contract differs"):
            audit_training_coverage(
                splits,
                collator=PrestoCollator(),
                args=args,
                data_seed=42,
                manifest=invalid,
                output_dir=output,
            )
        audit = json.loads((output / "affinity_target_audit.json").read_text())
        assert all(row["n_clipped_observations"] == 0 for row in audit["counts"])
        assert (output / "affinity_target_changes.tsv").exists()
        assert (
            json.loads((output / "supported_outputs.json").read_text())["status"]
            == "invalid_manifest"
        )
