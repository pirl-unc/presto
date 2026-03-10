"""Tests for data CLI helpers and command wiring."""

from pathlib import Path
from types import SimpleNamespace

from presto.cli import data as data_cli
from presto.cli.main import create_parser
from presto.data.mhc_index import MHCIndexError


def test_pyproject_has_mhcgnomes_dependency():
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    assert '"mhcgnomes"' in text


def test_parser_wires_mhc_index_build_command():
    parser = create_parser()
    args = parser.parse_args(
        [
            "data",
            "mhc-index",
            "build",
            "--imgt-fasta",
            "imgt.fasta",
            "--out-csv",
            "mhc_index.csv",
        ]
    )
    assert args.func is data_cli.cmd_data_mhc_index_build


def test_parser_wires_mhc_index_augment_command():
    parser = create_parser()
    args = parser.parse_args(
        [
            "data",
            "mhc-index",
            "augment",
            "--index-csv",
            "mhc_index.csv",
            "--out-csv",
            "mhc_index_augmented.csv",
        ]
    )
    assert args.func is data_cli.cmd_data_mhc_index_augment


def test_parser_wires_mhc_index_report_command():
    parser = create_parser()
    args = parser.parse_args(
        [
            "data",
            "mhc-index",
            "report",
            "--index-csv",
            "mhc_index.csv",
        ]
    )
    assert args.func is data_cli.cmd_data_mhc_index_report


def test_parser_wires_mhc_index_validate_command():
    parser = create_parser()
    args = parser.parse_args(
        [
            "data",
            "mhc-index",
            "validate",
            "--index-csv",
            "mhc_index.csv",
        ]
    )
    assert args.func is data_cli.cmd_data_mhc_index_validate


def test_parser_wires_mhc_index_refresh_command():
    parser = create_parser()
    args = parser.parse_args(
        [
            "data",
            "mhc-index",
            "refresh",
            "--datadir",
            "./data",
        ]
    )
    assert args.func is data_cli.cmd_data_mhc_index_refresh


def test_parser_wires_mhc_index_mouse_overlay_command():
    parser = create_parser()
    args = parser.parse_args(
        [
            "data",
            "mhc-index",
            "mouse-overlay",
            "--datadir",
            "./data",
        ]
    )
    assert args.func is data_cli.cmd_data_mhc_index_mouse_overlay


def test_parser_data_merge_supports_per_assay_csv_flag():
    parser = create_parser()
    args = parser.parse_args(
        [
            "data",
            "merge",
            "--datadir",
            "./data",
            "--assay-outdir",
            "./data/merged_assays",
            "--per-assay-csv",
        ]
    )
    assert args.func is data_cli.cmd_data_merge
    assert args.assay_outdir == "./data/merged_assays"
    assert args.per_assay_csv is True


def test_cmd_data_merge_passes_assay_outdir(tmp_path, monkeypatch):
    datadir = tmp_path / "data"
    datadir.mkdir()
    captured = {}

    def fake_deduplicate_all(**kwargs):
        captured.update(kwargs)
        return [], {"total_input": 0, "total_output": 0}

    monkeypatch.setattr(data_cli, "deduplicate_all", fake_deduplicate_all)
    args = SimpleNamespace(
        datadir=str(datadir),
        output=None,
        assay_outdir=str(tmp_path / "assays"),
        per_assay_csv=True,
        types=None,
        json=False,
        quiet=True,
    )
    code = data_cli.cmd_data_merge(args)

    assert code == 0
    assert captured["data_dir"] == datadir
    assert captured["assay_output_dir"] == tmp_path / "assays"
    assert captured["output_path"] == datadir / "merged_deduped.tsv"


def test_cmd_data_merge_default_disables_assay_csv(tmp_path, monkeypatch):
    datadir = tmp_path / "data"
    datadir.mkdir()
    captured = {}

    def fake_deduplicate_all(**kwargs):
        captured.update(kwargs)
        return [], {"total_input": 0, "total_output": 0}

    monkeypatch.setattr(data_cli, "deduplicate_all", fake_deduplicate_all)
    args = SimpleNamespace(
        datadir=str(datadir),
        output=None,
        assay_outdir=str(tmp_path / "assays"),
        per_assay_csv=False,
        types=None,
        json=False,
        quiet=True,
    )
    code = data_cli.cmd_data_merge(args)

    assert code == 0
    assert captured["assay_output_dir"] is None


def test_cmd_data_mhc_index_build_handles_errors(monkeypatch, capsys):
    args = SimpleNamespace(
        imgt_fasta="imgt.fasta",
        ipd_mhc_dir=None,
        out_csv="mhc_index.csv",
        out_fasta=None,
        quiet=False,
    )

    def fake_build(**kwargs):
        raise MHCIndexError("boom")

    monkeypatch.setattr(data_cli, "build_mhc_index", fake_build)
    code = data_cli.cmd_data_mhc_index_build(args)

    assert code == 1
    err = capsys.readouterr().err
    assert "Error building MHC index: boom" in err


def test_cmd_data_mhc_index_augment_handles_errors(monkeypatch, capsys):
    args = SimpleNamespace(
        index_csv="mhc_index.csv",
        out_csv="mhc_index_augmented.csv",
        quiet=False,
    )

    def fake_augment(**kwargs):
        raise MHCIndexError("boom")

    monkeypatch.setattr(data_cli, "augment_mhc_index", fake_augment)
    code = data_cli.cmd_data_mhc_index_augment(args)

    assert code == 1
    err = capsys.readouterr().err
    assert "Error augmenting MHC index: boom" in err


def test_cmd_data_mhc_index_resolve_requires_input(capsys):
    args = SimpleNamespace(
        index_csv="mhc_index.csv",
        alleles=None,
        allele_file=None,
        column=None,
        no_seq=False,
        format="json",
        output=None,
    )
    code = data_cli.cmd_data_mhc_index_resolve(args)
    assert code == 1
    err = capsys.readouterr().err
    assert "No alleles provided" in err


def test_cmd_data_mhc_index_resolve_reads_file_and_writes_csv(
    tmp_path,
    monkeypatch,
):
    allele_file = tmp_path / "alleles.csv"
    out_csv = tmp_path / "resolved.csv"
    allele_file.write_text("allele\nA0201\n", encoding="utf-8")

    captured = {}

    def fake_resolve_alleles(index_csv, alleles, include_sequence):
        captured["index_csv"] = index_csv
        captured["alleles"] = alleles
        captured["include_sequence"] = include_sequence
        return [
            {
                "input": "A0201",
                "normalized": "HLA-A*02:01",
                "resolved": "HLA-A*02:01:01",
                "found": True,
                "seq_len": 365,
            }
        ]

    monkeypatch.setattr(data_cli, "resolve_alleles", fake_resolve_alleles)

    args = SimpleNamespace(
        index_csv="mhc_index.csv",
        alleles=None,
        allele_file=str(allele_file),
        column=None,
        no_seq=True,
        format="csv",
        output=str(out_csv),
    )
    code = data_cli.cmd_data_mhc_index_resolve(args)

    assert code == 0
    assert captured["index_csv"] == "mhc_index.csv"
    assert captured["alleles"] == ["A0201"]
    assert captured["include_sequence"] is False

    text = out_csv.read_text(encoding="utf-8")
    assert "input,normalized,resolved,found,seq_len" in text
    assert "A0201,HLA-A*02:01,HLA-A*02:01:01,True,365" in text


def test_cmd_data_mhc_index_report_writes_json(tmp_path, monkeypatch):
    output_json = tmp_path / "report.json"

    def fake_report(index_csv):
        assert index_csv == "mhc_index.csv"
        return {"total_records": 2, "by_source": {"imgt": 2}}

    monkeypatch.setattr(data_cli, "summarize_mhc_index", fake_report)
    args = SimpleNamespace(
        index_csv="mhc_index.csv",
        output=str(output_json),
        format="json",
    )
    code = data_cli.cmd_data_mhc_index_report(args)
    assert code == 0
    text = output_json.read_text(encoding="utf-8")
    assert '"total_records": 2' in text


def test_cmd_data_mhc_index_validate_exit_code(monkeypatch, capsys):
    monkeypatch.setattr(
        data_cli,
        "validate_mhc_index",
        lambda index_csv: {
            "valid": False,
            "error_count": 1,
            "warning_count": 0,
            "errors": [{"row": 2, "code": "duplicate_normalized"}],
            "warnings": [],
        },
    )
    args = SimpleNamespace(index_csv="mhc_index.csv", output=None, format="json")
    code = data_cli.cmd_data_mhc_index_validate(args)
    assert code == 1
    out = capsys.readouterr().out
    assert '"error_count": 1' in out


def test_cmd_data_mhc_index_refresh_uses_existing_paths(tmp_path, monkeypatch):
    datadir = tmp_path / "data"
    datadir.mkdir()
    out_csv = tmp_path / "mhc_index.csv"
    out_fasta = tmp_path / "mhc_index.fasta"
    imgt = datadir / "imgt" / "hla_prot.fasta"
    ipd = datadir / "ipd_mhc" / "ipd_mhc_prot.fasta"
    imgt.parent.mkdir(parents=True)
    ipd.parent.mkdir(parents=True)
    imgt.write_text(">HLA-A*02:01\nAAAA\n", encoding="utf-8")
    ipd.write_text(">Mamu-A*01:01\nCCCC\n", encoding="utf-8")

    def fake_get_dataset_path(dataset_name, data_dir):
        assert data_dir == datadir
        if dataset_name == "imgt_hla":
            return imgt
        if dataset_name == "ipd_mhc_nhp":
            return ipd
        return None

    build_calls = {}

    def fake_build(imgt_fasta, ipd_mhc_dir, out_csv, out_fasta):
        build_calls["imgt_fasta"] = imgt_fasta
        build_calls["ipd_mhc_dir"] = ipd_mhc_dir
        build_calls["out_csv"] = out_csv
        build_calls["out_fasta"] = out_fasta
        return {"total": 2, "parsed": 2, "skipped": 0, "duplicates": 0, "replaced": 0}

    monkeypatch.setattr(data_cli, "get_dataset_path", fake_get_dataset_path)
    monkeypatch.setattr(data_cli, "build_mhc_index", fake_build)

    args = SimpleNamespace(
        datadir=str(datadir),
        imgt_fasta=None,
        ipd_mhc_dir=None,
        out_csv=str(out_csv),
        out_fasta=str(out_fasta),
        download_missing=False,
        quiet=True,
    )
    code = data_cli.cmd_data_mhc_index_refresh(args)
    assert code == 0
    assert build_calls["imgt_fasta"] == str(imgt)
    assert build_calls["ipd_mhc_dir"] == str(datadir / "ipd_mhc")
    assert build_calls["out_csv"] == str(out_csv)
    assert build_calls["out_fasta"] == str(out_fasta)


def test_cmd_data_mhc_index_refresh_fails_when_missing(tmp_path, monkeypatch, capsys):
    datadir = tmp_path / "data"
    datadir.mkdir()
    monkeypatch.setattr(data_cli, "get_dataset_path", lambda dataset_name, data_dir: None)

    args = SimpleNamespace(
        datadir=str(datadir),
        imgt_fasta=None,
        ipd_mhc_dir=None,
        out_csv=str(tmp_path / "mhc_index.csv"),
        out_fasta=None,
        download_missing=False,
        quiet=True,
    )
    code = data_cli.cmd_data_mhc_index_refresh(args)
    assert code == 1
    err = capsys.readouterr().err
    assert "Unable to locate IMGT or IPD-MHC inputs" in err


def test_cmd_data_mhc_index_mouse_overlay_uses_defaults(tmp_path, monkeypatch):
    datadir = tmp_path / "data"
    datadir.mkdir()
    captured = {}

    def fake_build_mouse_mhc_overlay(**kwargs):
        captured.update(kwargs)
        return {
            "imgt_genes": 7,
            "catalog_rows": 11,
            "selected_alleles": 5,
            "fasta_records": 5,
        }

    monkeypatch.setattr(data_cli, "build_mouse_mhc_overlay", fake_build_mouse_mhc_overlay)

    args = SimpleNamespace(
        datadir=str(datadir),
        out_csv=None,
        out_fasta=None,
        imgt_url=None,
        include_unreviewed=False,
        max_genes=0,
        quiet=True,
    )
    code = data_cli.cmd_data_mhc_index_mouse_overlay(args)
    assert code == 0
    assert captured["out_csv"] == str(datadir / "ipd_mhc" / "mouse_uniprot_overlay.csv")
    assert captured["out_fasta"] == str(datadir / "ipd_mhc" / "mouse_uniprot_overlay.fasta")
    assert captured["reviewed_only"] is True


def test_cmd_data_mhc_index_mouse_overlay_handles_errors(tmp_path, monkeypatch, capsys):
    datadir = tmp_path / "data"
    datadir.mkdir()

    def fake_build_mouse_mhc_overlay(**kwargs):
        raise RuntimeError("network down")

    monkeypatch.setattr(data_cli, "build_mouse_mhc_overlay", fake_build_mouse_mhc_overlay)

    args = SimpleNamespace(
        datadir=str(datadir),
        out_csv=None,
        out_fasta=None,
        imgt_url=None,
        include_unreviewed=False,
        max_genes=0,
        quiet=True,
    )
    code = data_cli.cmd_data_mhc_index_mouse_overlay(args)
    assert code == 1
    err = capsys.readouterr().err
    assert "Error building mouse MHC overlay: network down" in err
