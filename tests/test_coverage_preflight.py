"""A prospective manifest gates the canonical preflight and survives configuration."""

import argparse
import csv
import json
import sys
from dataclasses import asdict
from types import ModuleType

import pytest

from presto.cli.main import create_parser
from presto.data.collate import PrestoCollator
from presto.data.loaders import BindingRecord, ElutionRecord
from presto.data.mhc_sequence_resolver import ExactMHCInput
from presto.models.presto import Presto
from presto.scripts import train_iedb
from presto.training.coverage_preflight import audit_training_coverage
from presto.training.output_contract import OutputConfiguration, build_output_contract
from presto.training.output_coverage import OutputCoverageCensus
from presto.training.supported_outputs import evaluate_supported_outputs
from test_output_coverage import sample, manifest_for


def split_samples():
    return {
        split: [
            sample(tcell_label=1, tcell_assay_method="ELISPOT"),
            sample(
                sample_id="second",
                evidence_row_id="assay-2",
                peptide="LMNPQRSTV",
                tcell_label=0,
                tcell_assay_method="ELISPOT",
            ),
        ]
        for split in ("train", "val", "test")
    }


def arguments(**values):
    return train_iedb._resolve_run_args(argparse.Namespace(**values))


def test_preflight_persists_complete_exploratory_and_failed_claim_reports(tmp_path):
    splits = split_samples()
    collator = PrestoCollator()
    args = arguments()
    _, exploratory = audit_training_coverage(
        splits, collator=collator, args=args, data_seed=7, manifest=None, output_dir=tmp_path
    )
    assert exploratory["mode"] == "exploratory"
    assert not exploratory["requirements_met"]
    for name in (
        "split_support.json",
        "output_coverage.json",
        "output_coverage.csv",
        "supported_outputs.json",
    ):
        assert (tmp_path / name).is_file()
    manifest = manifest_for(build_output_contract(), columns=["ELISPOT"])
    manifest["source_contract"] = exploratory["source_contract"]
    _, passed = audit_training_coverage(
        splits, collator=collator, args=args, data_seed=7, manifest=manifest, output_dir=tmp_path
    )
    assert passed["requirements_met"]
    manifest["claims"][0]["minimums"]["unique_negative"] = 3
    with pytest.raises(RuntimeError, match="unique_negative=1 requires >= 3"):
        audit_training_coverage(
            splits,
            collator=collator,
            args=args,
            data_seed=7,
            manifest=manifest,
            output_dir=tmp_path,
        )
    report = json.loads((tmp_path / "supported_outputs.json").read_text())
    assert not report["requirements_met"]
    assert any(row["status"] == "requirements_not_met" for row in report["outputs"])


def test_changed_curation_or_model_configuration_cannot_reuse_a_manifest(tmp_path):
    args = arguments()
    splits = split_samples()
    _, initial = audit_training_coverage(
        splits, collator=PrestoCollator(), args=args, data_seed=7, manifest=None
    )
    manifest = manifest_for(build_output_contract(), columns=["ELISPOT"])
    manifest["source_contract"] = initial["source_contract"]
    with pytest.raises(ValueError, match="source_contract differs"):
        audit_training_coverage(
            splits,
            collator=PrestoCollator(),
            args=args,
            data_seed=8,
            manifest=manifest,
            output_dir=tmp_path,
        )
    error = json.loads((tmp_path / "supported_outputs.json").read_text())
    assert error["status"] == "invalid_manifest"
    args.latent_topology = "collapsed"
    with pytest.raises(ValueError, match="model_contract differs"):
        audit_training_coverage(
            splits,
            collator=PrestoCollator(),
            args=args,
            data_seed=7,
            manifest=manifest,
            output_dir=tmp_path,
        )


def test_both_cli_entrypoints_and_config_merge_keep_the_same_coverage_flags(tmp_path, monkeypatch):
    argv = ["--supported-output-manifest", "claims.json", "--track-output-updates"]
    parsed = create_parser().parse_args(["train", "unified", *argv])
    captured = []
    monkeypatch.setattr(train_iedb, "run", lambda args: captured.append(args))
    train_iedb.main(argv)
    assert (
        parsed.supported_output_manifest == captured[0].supported_output_manifest == "claims.json"
    )
    assert parsed.track_output_updates is captured[0].track_output_updates is True
    defaults = create_parser().parse_args(["train", "unified"])
    for name in ("supported_output_manifest", "track_output_updates"):
        assert getattr(defaults, name) == train_iedb.IEDB_DEFAULTS[name]
    config = tmp_path / "train.json"
    config.write_text(
        json.dumps(
            {
                "train": {
                    "unified": {
                        "supported_output_manifest": "frozen.json",
                        "track_output_updates": True,
                        "kd_grouping_mode": "split_kd_proxy",
                        "affinity_assay_residual_mode": "dag_method_leaf",
                        "core_window_lengths": [8, 9, 10],
                    }
                }
            }
        )
    )
    resolved = arguments(config=str(config))
    assert resolved.supported_output_manifest == "frozen.json"
    assert resolved.track_output_updates is True
    assert OutputConfiguration.from_object(resolved).kd_grouping_mode == "split_kd_proxy"
    assert OutputConfiguration.from_object(resolved).core_window_lengths == (8, 9, 10)
    for name, default in asdict(OutputConfiguration()).items():
        assert train_iedb.IEDB_DEFAULTS[name] == default


def test_manifest_is_frozen_before_any_source_loader_and_cannot_be_replaced(tmp_path, monkeypatch):
    declaration = manifest_for(build_output_contract())
    source = tmp_path / "claim.json"
    source.write_text(json.dumps(declaration))
    run_dir = tmp_path / "run"
    monkeypatch.setattr(train_iedb, "_assert_mhcseqs_importable", lambda: None)
    (tmp_path / "merged_deduped.tsv").write_text("a competing input exists\n")

    def load(**kwargs):
        frozen = json.loads((run_dir / "supported_output_manifest.json").read_text())
        assert frozen == json.loads(json.dumps(declaration))
        raise RuntimeError("stop at the source boundary")

    monkeypatch.setattr(train_iedb, "load_records_from_merged_tsv", load)
    args = arguments(
        data_dir=str(tmp_path), run_dir=str(run_dir), supported_output_manifest=str(source)
    )
    with pytest.raises(RuntimeError, match="stop at the source boundary"):
        train_iedb.run(args)
    declaration["claims"][0]["minimums"]["unique_positive"] = 7
    source.write_text(json.dumps(declaration))
    with pytest.raises(ValueError, match="different supported-output manifest"):
        train_iedb.run(args)


@pytest.mark.parametrize("data_source", ["merged_tsv", "hitlist"])
@pytest.mark.parametrize("bulk", [False, True])
def test_selected_sources_reach_census_with_competing_inputs(
    tmp_path, monkeypatch, data_source, bulk
):
    """Use actual TSV parsing, dataset conversion, splitting, collation and census.

    Only external Hitlist records/frames and the exact MHC resolver are fixtures.
    Their presence must never turn source selection into an implicit union.
    """
    from presto.data import hitlist_source

    pd = pytest.importorskip("pandas")
    peptides = [f"ACDEFGHIK{suffix}" for suffix in "ACDEFGHIKLMN"]
    allele = "HLA-A*02:01"
    rows = [
        dict(
            peptide=peptide,
            mhc_allele=allele,
            source="iedb",
            record_type="binding",
            value=25 + index,
            value_type="IC50",
            qualifier=0,
            mhc_class="I",
            species="human",
        )
        for index, peptide in enumerate(peptides)
    ]
    rows += [
        dict(
            peptide=peptides[0],
            mhc_allele=allele,
            source="iedb",
            record_type="tcell",
            response="positive",
            assay_method="ELISPOT",
            mhc_class="I",
            species="human",
        )
    ]
    with (tmp_path / "merged_deduped.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, sorted({key for row in rows for key in row}), delimiter="\t"
        )
        writer.writeheader()
        writer.writerows(rows)
    (tmp_path / "iedb").mkdir()
    (tmp_path / "iedb" / "tcell_full_v3.csv").write_text("competing raw T-cell export\n")

    def unexpected_raw(*args, **kwargs):
        pytest.fail("Named input source unexpectedly loaded a competing raw export")

    monkeypatch.setattr(train_iedb, "find_iedb_export_file", unexpected_raw)
    calls = []

    def load_hitlist(**kwargs):
        calls.append("hitlist")
        binding = [
            BindingRecord(
                peptide,
                allele,
                30,
                measurement_type="KD",
                source="hitlist_fixture",
                evidence_row_id=f"hitlist:{index}",
            )
            for index, peptide in enumerate(peptides)
        ]
        elution = [ElutionRecord(peptides[0], [allele], source="hitlist_fixture")]
        return (
            binding,
            [],
            [],
            [],
            elution,
            [],
            [],
            {"counts": {"binding": 12, "elution": 1}, "flank_coverage": {}},
        )

    monkeypatch.setattr(hitlist_source, "load_records_from_hitlist", load_hitlist)
    if data_source == "hitlist":
        monkeypatch.setattr(train_iedb, "load_records_from_merged_tsv", unexpected_raw)
    provider = ModuleType("hitlist.bulk_proteomics")

    def load_bulk(**kwargs):
        calls.append("bulk")
        return pd.DataFrame(
            [
                dict(
                    peptide="LMNPQRSTK",
                    digestion_enzyme="trypsin",
                    n_fractions_in_run=12,
                    n_replicates_detected=3,
                )
            ]
        )

    provider.load_bulk_peptides = load_bulk
    monkeypatch.setitem(sys.modules, "hitlist.bulk_proteomics", provider)
    exact = ExactMHCInput(
        allele=allele,
        sequence="A" * 181,
        groove1="C" * 90,
        groove2="D" * 93,
        mhc_class="I",
        chain="alpha",
    )
    monkeypatch.setattr(train_iedb, "_assert_mhcseqs_importable", lambda: None)
    monkeypatch.setattr(
        train_iedb,
        "resolve_mhc_inputs_from_index",
        lambda *_: ({allele: exact}, {"resolved": 1, "total": 1}),
    )
    monkeypatch.setattr(train_iedb, "Presto", unexpected_raw)
    run_dir = tmp_path / "run"
    args = arguments(
        **({"data_source": "hitlist"} if data_source == "hitlist" else {}),
        data_dir=str(tmp_path),
        run_dir=str(run_dir),
        data_preflight_only=True,
        bulk_ms=bulk,
        device="cpu",
        test_frac=0.2,
        val_frac=0.2,
        synthetic_pmhc_negative_ratio=0,
        synthetic_elution_negative_ratio=0,
        synthetic_cascade_elution_negative_ratio=0,
        synthetic_cascade_tcell_negative_ratio=0,
        synthetic_processing_negative_ratio=0,
        synthetic_class_i_no_mhc_beta_negative_ratio=0,
        mhc_augmentation_samples=0,
        uniprot_negative_ratio=0,
    )
    train_iedb.run(args)
    assert calls == (["hitlist"] if data_source == "hitlist" else []) + (["bulk"] if bulk else [])
    coverage = json.loads((run_dir / "output_coverage.json").read_text())
    assert set(coverage["splits"]) == {"train", "val", "test"}
    assert coverage["source_contract"]["data_source"] == data_source
    assert coverage["source_contract"]["bulk_ms"] is bulk
    assert sum(split["rows"] for split in coverage["splits"].values()) == 13 + 2 * bulk
    tcell_count = sum(
        split["outputs"]["tcell_panel_logits.assay_method"]["observations"]
        for split in coverage["splits"].values()
    )
    assert tcell_count == (1 if data_source == "merged_tsv" else 0)
    evidence = [
        entry
        for split in coverage["splits"].values()
        for output in split["outputs"].values()
        for entry in output["evidence"]
    ]
    assert {entry["source"] for entry in evidence} == (
        {"iedb" if data_source == "merged_tsv" else "hitlist_fixture"}
        | ({"bulk_proteomics"} if bulk else set())
    )
    bulk_evidence = {
        (entry["family"], entry["role"])
        for entry in evidence
        if entry["source"] == "bulk_proteomics"
    }
    assert bulk_evidence == (
        {
            ("bulk_observed_product", "proxy"),
            ("bulk_depth_proxy", "proxy"),
            ("generated:bulk_wrong_enzyme", "synthetic"),
        }
        if bulk
        else set()
    )
    funnel = json.loads((run_dir / "data_funnel.json").read_text())
    if bulk:
        assert funnel["additions"]["bulk_ms"]["n_records"] == 2
        assert not funnel["additions"]["bulk_ms"]["in_silico_negatives_available"]
    else:
        assert "bulk_ms" not in funnel["additions"]
    if data_source == "merged_tsv" and not bulk:
        monkeypatch.setattr(train_iedb, "Presto", Presto)
        _verify_declared_training(args, coverage, tmp_path)


def _verify_declared_training(args, coverage, tmp_path):
    """The prospective gate must enable the tracker through the actual runner."""
    declaration = {
        "schema_version": 1,
        "model_contract": coverage["contract"]["configuration"],
        "source_contract": coverage["source_contract"],
        "claims": [
            {
                "endpoint": "assays.IC50_nM",
                "columns": [""],
                "required_splits": ["train", "val", "test"],
                "evidence_roles": ["direct"],
                "source_families": ["binding:IC50"],
                "raw_sources": ["iedb"],
                # Tiny fixture requirements exercise wiring, not scientific adequacy.
                "minimums": {"unique_observations": 1, "distinct_peptides": 1, "unique_exact": 1},
            }
        ],
    }
    manifest = tmp_path / "claims.json"
    manifest.write_text(json.dumps(declaration))
    run_dir = tmp_path / "declared"
    args.run_dir = str(run_dir)
    args.checkpoint = str(run_dir / "best.pt")
    args.supported_output_manifest = str(manifest)
    args.data_preflight_only = False
    args.epochs, args.max_batches, args.max_val_batches = 1, 1, 1
    args.d_model, args.n_layers, args.n_heads = 32, 1, 4
    args.batch_size, args.num_workers = 8, 0
    args.track_probe_affinity = args.track_probe_motif_scan = False
    args.track_pmhc_flow = args.track_output_latent_stats = False
    assert not args.track_output_updates  # A supplied manifest turns tracking on.
    train_iedb.run(args)
    support = json.loads((run_dir / "supported_outputs.json").read_text())
    assert support["requirements_met"]
    updates = json.loads((run_dir / "output_updates.json").read_text())
    assert updates["optimizer_steps"] == updates["batches_completed"] == 1
    assert (run_dir / "best.pt").is_file()
    for split in ("val", "test"):
        assert (run_dir / f"{split}_predictions.csv").is_file()
        assert (run_dir / f"{split}_loss_ledger.json").is_file()


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("endpoint", [], "canonical output name"),
        ("columns", ["ELISPOT", "ELISPOT"], "unique strings"),
        ("columns", ["not_a_method"], "unknown column"),
        ("evidence_roles", ["unknown"], "unknown provenance"),
        ("raw_sources", [" "], "must be explicit"),
        ("raw_sources", ["*"], "must be explicit"),
        ("source_families", ["binding:KD"], "undeclared source family"),
        ("source_families", ["generated:control"], "require the synthetic role"),
        ("minimums", {"unique_observations": True, "distinct_peptides": 1}, "integer counts"),
        ("minimums", {"unique_observations": -1, "distinct_peptides": 1}, "integer counts"),
        ("minimums", {"unique_observations": 0, "distinct_peptides": 1}, "positive observation"),
        ("minimums", {"unique_observations": 1, "distinct_peptides": 1}, "unique_positive"),
    ],
)
def test_malformed_claims_fail_explicitly(field, value, match):
    contract = build_output_contract()
    manifest = manifest_for(contract)
    manifest["claims"][0][field] = value
    with OutputCoverageCensus(contract) as census, pytest.raises(ValueError, match=match):
        evaluate_supported_outputs(census, manifest, source_contract=manifest["source_contract"])


def test_boolean_manifest_version_is_not_schema_one():
    contract = build_output_contract()
    manifest = manifest_for(contract)
    manifest["schema_version"] = True
    with (
        OutputCoverageCensus(contract) as census,
        pytest.raises(ValueError, match="schema_version"),
    ):
        evaluate_supported_outputs(census, manifest, source_contract=manifest["source_contract"])
