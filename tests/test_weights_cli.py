"""Tests for weights CLI commands."""

import json
from pathlib import Path
from types import SimpleNamespace

from presto.cli import weights as weights_cli
from presto.cli.main import create_parser


def test_parser_wires_weights_list_command():
    parser = create_parser()
    args = parser.parse_args(["weights", "list"])
    assert args.func is weights_cli.cmd_weights_list


def test_parser_wires_weights_download_command():
    parser = create_parser()
    args = parser.parse_args(
        [
            "weights",
            "download",
            "--name",
            "foundation-v1",
            "--output",
            "presto.pt",
        ]
    )
    assert args.func is weights_cli.cmd_weights_download
    assert args.name == "foundation-v1"
    assert args.output == "presto.pt"


def test_cmd_weights_list_json(monkeypatch, capsys):
    monkeypatch.setattr(
        weights_cli,
        "load_weight_registry",
        lambda source=None: {
            "models": {
                "foundation-v1": {
                    "url": "https://example.org/foundation-v1.pt",
                    "sha256": "abc",
                    "description": "Test model",
                }
            }
        },
    )
    args = SimpleNamespace(registry=None, json=True)
    code = weights_cli.cmd_weights_list(args)
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert "models" in payload
    assert "foundation-v1" in payload["models"]


def test_cmd_weights_download_from_registry(tmp_path, monkeypatch):
    out = tmp_path / "model.pt"

    monkeypatch.setattr(
        weights_cli,
        "load_weight_registry",
        lambda source=None: {
            "models": {
                "foundation-v1": {
                    "url": "https://example.org/foundation-v1.pt",
                    "sha256": "abc",
                }
            }
        },
    )

    calls = {}

    def fake_download(url: str, output_path: Path, expected_sha256=None):
        calls["url"] = url
        calls["output_path"] = output_path
        calls["expected_sha256"] = expected_sha256
        output_path.write_bytes(b"weights")
        return {"bytes": 7, "sha256": "abc"}

    monkeypatch.setattr(weights_cli, "download_file", fake_download)

    args = SimpleNamespace(
        name="foundation-v1",
        url=None,
        registry=None,
        output=str(out),
        cache_dir=None,
        force=False,
    )
    code = weights_cli.cmd_weights_download(args)
    assert code == 0
    assert out.exists()
    assert calls["url"] == "https://example.org/foundation-v1.pt"
    assert calls["output_path"] == out
    assert calls["expected_sha256"] == "abc"


def test_cmd_weights_download_requires_name_or_url(capsys):
    args = SimpleNamespace(
        name=None,
        url=None,
        registry=None,
        output=None,
        cache_dir=None,
        force=False,
    )
    code = weights_cli.cmd_weights_download(args)
    assert code == 1
    err = capsys.readouterr().err
    assert "Provide --name or --url" in err
