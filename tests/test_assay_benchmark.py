from dataclasses import replace
from types import SimpleNamespace

import pandas as pd
import pytest

from presto.data.assay_measurement import AssayMethod
from presto.data.modular import MHCInput, Observation
from presto.evaluation import assay_benchmark as benchmark
from presto.tests.test_external_benchmark import NETMHCPAN_OUT


def exported():
    obs = Observation(
        "id",
        "source",
        "ic50",
        "GILGFVFTL",
        (MHCInput("ACDEFGHIK", "LMNPQRSTV", "I", provenance={"catalog_alleles": ["HLA-A*02:01"]}),),
        value=100,
        unit="nM",
        provenance={"raw_record": {"pmid": "1234"}},
        assay_method=AssayMethod("purified", "competitive", "fluorescence", "verified"),
    )
    return dict(
        observation_id="id",
        endpoint="ic50",
        target=2.0,
        prediction=2.3,
        target_space="log10(nM)",
        observation=obs.to_dict(),
    )


def test_physical_units_and_raw_study_fallback():
    row = benchmark.affinity_record(exported())
    assert row["study"] == "1234" and row["raw_nM"] == 100
    assert row["tool_alleles"] == {"netmhcpan": "HLA-A0201", "mhcflurry": "HLA-A*02:01"}
    bad = exported()
    bad["target"] = 100
    with pytest.raises(ValueError, match="physical"):
        benchmark.affinity_record(bad)
    bad = exported()
    bad["observation"]["provenance"] = {}
    assert benchmark.affinity_record(bad)["exclusion"] == "unresolved_assay_study_or_mhc"


def test_allele_offsets_do_not_become_within_cell_accuracy():
    rows = []
    for study in ("s1", "s2", "s3"):
        for allele, offset in (("a", 0), ("b", 3)):
            for i in range(20):
                target = offset + i / 20
                rows.append(
                    dict(
                        mhc_class="I",
                        endpoint="ic50",
                        source="fixture",
                        study=study,
                        preparation="purified",
                        mechanism="competitive",
                        detector="fluorescence",
                        mhc_key=allele,
                        qualifier="=",
                        target_log10=target,
                        raw_nM=10**target,
                        presto=offset - i / 20,
                        netmhcpan=target,
                    )
                )
    cells = benchmark.paired_cells(pd.DataFrame(rows), "netmhcpan")
    assert set(cells.loc[cells.method == "presto", "spearman"].round(8)) == {-1}
    interval = benchmark.paired_study_interval(cells, "netmhcpan", "spearman")
    assert interval["mean"] == pytest.approx(-2)
    assert interval["ci_low"] == pytest.approx(-2)
    assert interval["studies"] == 3
    duplicated_source = pd.concat([cells, cells.assign(source="second_export")])
    assert (
        benchmark.paired_study_interval(duplicated_source, "netmhcpan", "spearman")["studies"] == 3
    )
    one = cells[cells.study == "s1"]
    assert (
        benchmark.paired_study_interval(one, "netmhcpan", "spearman")["status"]
        == "descriptive_only"
    )
    frame = pd.DataFrame(rows)
    frame.loc[0, "netmhcpan"] = float("nan")
    with pytest.raises(ValueError, match="identical rows"):
        benchmark.paired_cells(frame, "netmhcpan")
    constants = pd.DataFrame(rows).assign(presto=2.0)
    constant_cells = benchmark.paired_cells(constants, "netmhcpan")
    assert set(constant_cells.loc[constant_cells.method == "presto", "spearman"]) == {0.0}
    assert constant_cells.loc[constant_cells.method == "presto", "constant_prediction"].all()


def test_current_dtu_flags_and_version_are_verified(tmp_path, monkeypatch):
    executable = tmp_path / "binary"
    executable.write_text("fixture")
    tool = benchmark.DTUTool("netmhcpan", str(executable), "4.2c", str(tmp_path))
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(
            returncode=0, stdout="# NetMHCpan version 4.2c\n" + NETMHCPAN_OUT, stderr=""
        )

    monkeypatch.setattr(benchmark.subprocess, "run", run)
    result = benchmark.run_dtu(tool, "HLA-A0201", ["GILGFVFTL", "SIINFEKL"], tmp_path / "run")
    assert commands[0][1:3] == ["-inptype", "1"] and len(result) == 2
    assert {r["version"] for r in result} == {"4.2c"}
    with pytest.raises(RuntimeError, match="version"):
        benchmark.run_dtu(
            replace(tool, version="4.2e"),
            "HLA-A0201",
            ["GILGFVFTL", "SIINFEKL"],
            tmp_path / "wrong-version",
        )
    with pytest.raises(ValueError, match="version"):
        replace(tool, version="4.1b")


def test_fingerprint_detects_weight_changes_with_unchanged_version(tmp_path):
    (tmp_path / "version").write_text("4.2c")
    weights = tmp_path / "weights"
    weights.write_text("first")
    first = benchmark.fingerprint_package(tmp_path)
    weights.write_text("second")
    second = benchmark.fingerprint_package(tmp_path)
    assert first["files"]["version"] == second["files"]["version"]
    assert first["sha256"] != second["sha256"]
