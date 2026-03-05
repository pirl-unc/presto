"""Tests for predict CLI parser + command wiring."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from presto.cli import predict as predict_cli
from presto.cli.main import create_parser
from presto.inference.predictor import TiledPresentationHit, TiledPresentationResult


def test_parser_wires_predict_tile():
    parser = create_parser()
    args = parser.parse_args(
        [
            "predict",
            "tile",
            "--checkpoint",
            "model.pt",
            "--protein-sequence",
            "MPEPTIDESEQ",
            "--allele",
            "HLA-A*02:01",
            "--min-length",
            "8",
            "--max-length",
            "11",
            "--top-k",
            "20",
            "--sort-by",
            "binding",
        ]
    )
    assert args.func is predict_cli.cmd_predict_tile
    assert args.min_length == 8
    assert args.max_length == 11
    assert args.top_k == 20
    assert args.sort_by == "binding"


def test_parser_predict_presentation_accepts_species_and_index():
    parser = create_parser()
    args = parser.parse_args(
        [
            "predict",
            "presentation",
            "--checkpoint",
            "model.pt",
            "--peptide",
            "SLLQHLIGL",
            "--mhc-sequence",
            "MAVMAPRTLLLLLSGALALTQTWAG",
            "--species",
            "human",
            "--mhc-species",
            "murine",
            "--immune-species",
            "human",
            "--species-of-origin",
            "viruses",
            "--index-csv",
            "mhc_index.csv",
        ]
    )
    assert args.species == "human"
    assert args.mhc_species == "murine"
    assert args.immune_species == "human"
    assert args.species_of_origin == "viruses"
    assert args.index_csv == "mhc_index.csv"


def test_cmd_predict_tile_requires_one_sequence_source():
    args = SimpleNamespace(
        checkpoint="model.pt",
        protein_sequence="AAAA",
        protein_file="protein.fa",
    )
    with pytest.raises(ValueError):
        predict_cli.cmd_predict_tile(args)


def test_cmd_predict_tile_reads_fasta_file(tmp_path, monkeypatch):
    fasta = tmp_path / "protein.fa"
    fasta.write_text(">PRAME\nMPEPTIDESEQ\n", encoding="utf-8")

    captured = {}

    class _DummyPredictor:
        def predict_tiled_presentation(self, **kwargs):
            captured.update(kwargs)
            return TiledPresentationResult(
                protein_length=len(kwargs["protein_sequence"]),
                total_candidates=1,
                min_length=8,
                max_length=8,
                flank_size=0,
                mhc_class="I",
                sort_by="presentation",
                hits=[
                    TiledPresentationHit(
                        peptide="MPEPTIDE",
                        start=0,
                        end=8,
                        flank_n="",
                        flank_c="Q",
                        processing_prob=0.2,
                        binding_prob=0.3,
                        presentation_prob=0.1,
                        assays={},
                    )
                ],
            )

    monkeypatch.setattr(predict_cli, "_build_predictor", lambda args: _DummyPredictor())

    args = SimpleNamespace(
        checkpoint="model.pt",
        protein_sequence=None,
        protein_file=str(fasta),
        allele="HLA-A*02:01",
        mhc_sequence=None,
        mhc_b_sequence=None,
        mhc_class="I",
        species="human",
        mhc_species="murine",
        immune_species="human",
        species_of_origin="viruses",
        min_length=8,
        max_length=8,
        flank_size=0,
        batch_size=4,
        top_k=10,
        sort_by="presentation",
        json=False,
        output=str(tmp_path / "out.json"),
    )
    code = predict_cli.cmd_predict_tile(args)
    assert code == 0
    assert captured["protein_sequence"] == "MPEPTIDESEQ"
    assert captured["mhc_species"] == "murine"
    assert captured["immune_species"] == "human"
    assert captured["species_of_origin"] == "viruses"
    assert Path(args.output).exists()
