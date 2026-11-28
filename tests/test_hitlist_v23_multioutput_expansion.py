"""Unit tests for `data/hitlist_v23_multioutput_expansion.py`."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pandas as pd
import pytest

from presto.data.hitlist_binding_expansion import FastGrooveResolver, ProtectedIdentityIndex
from presto.data.hitlist_v23_multioutput_expansion import (
    allocate_v23_elution_strata,
    build_expanded_bekker_jensen_bulk,
    build_hla_ligand_atlas_expansion,
    build_nonclassical_mhc_expansion,
    normalize_human_hla_token,
    resolve_human_mhc_bag,
)
from presto.data.modular import Observation

MHC_CSV = (
    Path(__file__).resolve().parents[1] / "data/shared-presentation-20261117/mhc-full-seqs.csv"
)


def _write_synthetic_human_fasta(path: Path) -> Path:
    seq1 = "MAAAAGILGFVFTLPPPPEEEEKRRRRWYYYYQQQQVLLLLSSSSDDDDNNNNGGGGTTTTAAAA"
    seq2 = "MGGGGNLVPMVATVKKKKQQQQLLLLPPPPAAAAFFFFWWWWEEEEYYYYTTTTDDDDSSSSRRRR"
    seq3 = "MDDDDPKYVKQNTLKLATGGGGLLLLPPPPAAAAFFFFWWWWEEEEYYYYTTTTDDDDSSSSRRRR"
    path.write_text(
        f">sp|P00001|GENE1_HUMAN GN=GENE1\n{seq1}\n"
        f">sp|P00002|GENE2_HUMAN GN=GENE2\n{seq2}\n"
        f">sp|P00003|GENE3_HUMAN GN=GENE3\n{seq3}\n"
    )
    return path


def test_normalize_and_resolve_human_mhc_bag():
    assert normalize_human_hla_token("A*11:01") == "HLA-A*11:01"
    assert normalize_human_hla_token("DPA1*01:03-DPB1*04:01") == "HLA-DPA1*01:03/HLA-DPB1*04:01"
    assert normalize_human_hla_token("HLA-DRB1*04:01") == "HLA-DRB1*04:01"

    if not MHC_CSV.exists():
        pytest.skip("mhc-full-seqs.csv absent locally")
    resolver = FastGrooveResolver(MHC_CSV)
    bag_i = resolve_human_mhc_bag(["A*02:01", "B*07:02", "A*02:01"], "I", resolver)
    assert len(bag_i) == 2
    assert all(m.mhc_class == "I" for m in bag_i)
    bag_ii = resolve_human_mhc_bag(
        ["HLA-DQA1*05:01/DQB1*03:01;HLA-DQA1*05:03/DQB1*03:01"], "II", resolver
    )
    assert len(bag_ii) >= 1
    assert all(m.mhc_class == "II" for m in bag_ii)


def test_hla_ligand_atlas_builder_embargo_and_context(tmp_path: Path):
    if not MHC_CSV.exists():
        pytest.skip("mhc-full-seqs.csv absent locally")
    atlas_dir = tmp_path / "atlas"
    atlas_dir.mkdir()
    pd.DataFrame(
        [
            {"donor": "DN1", "hla_allele": "A*02:01"},
            {"donor": "DN1", "hla_allele": "B*07:02"},
            {"donor": "DN1", "hla_allele": "DRB1*01:01"},
        ]
    ).to_csv(atlas_dir / "donors.tsv.gz", sep="\t", index=False)
    pd.DataFrame(
        [
            {"peptide_sequence_id": 1, "peptide_sequence": "GILGFVFTL"},
            {"peptide_sequence_id": 2, "peptide_sequence": "NLVPMVATV"},  # embargoed!
            {"peptide_sequence_id": 3, "peptide_sequence": "PKYVKQNTLKLAT"},
        ]
    ).to_csv(atlas_dir / "peptides.tsv.gz", sep="\t", index=False)
    pd.DataFrame(
        [
            {"peptide_sequence_id": 1, "donor": "DN1", "tissue": "Lung", "hla_class": "HLA-I"},
            {"peptide_sequence_id": 1, "donor": "DN1", "tissue": "Spleen", "hla_class": "HLA-I"},
            {"peptide_sequence_id": 2, "donor": "DN1", "tissue": "Lung", "hla_class": "HLA-I"},
            {"peptide_sequence_id": 3, "donor": "DN1", "tissue": "Thymus", "hla_class": "HLA-II"},
        ]
    ).to_csv(atlas_dir / "sample_hits.tsv.gz", sep="\t", index=False)

    fasta = _write_synthetic_human_fasta(tmp_path / "human.fasta")
    protected = ProtectedIdentityIndex(["NLVPMVATV"])
    out_p = tmp_path / "atlas.jsonl.gz"

    stats = build_hla_ligand_atlas_expansion(
        atlas_dir=atlas_dir,
        human_fasta=fasta,
        mhc_full_seqs_csv=MHC_CSV,
        protected=protected,
        global_observed=set(),
        out_path=out_p,
        seed=42,
    )
    assert stats["positives_admitted"] == 2
    assert stats["total_rows_written"] == 4
    assert stats["pre_filter_counts"]["dropped_embargo"] == 1

    with gzip.open(out_p, "rt") as fh:
        rows = [Observation.from_dict(json.loads(line)) for line in fh]
    peps = {r.peptide for r in rows}
    assert "NLVPMVATV" not in peps
    assert "GILGFVFTL" in peps
    assert "PKYVKQNTLKLAT" in peps
    assert all(r.context.cell_type == "healthy_tissue" for r in rows)
    assert all(protected.reason(r.peptide) is None for r in rows)


def test_nonclassical_mhc_excludes_cd1_mr1_and_admits_hla_egf(tmp_path: Path):
    if not MHC_CSV.exists():
        pytest.skip("mhc-full-seqs.csv absent locally")
    fasta = _write_synthetic_human_fasta(tmp_path / "human.fasta")
    obs_p = tmp_path / "obs.parquet"
    bind_p = tmp_path / "bind.parquet"
    pd.DataFrame(
        [
            {
                "peptide": "GILGFVFTL",
                "mhc_class": "non-classical",
                "mhc_restriction": "HLA-E*01:03",
                "has_ptm": False,
                "is_binding_assay": False,
                "pmid": 12345,
            },
            {
                "peptide": "NLVPMVATV",
                "mhc_class": "non-classical",
                "mhc_restriction": "human-CD1d",
                "has_ptm": False,
                "is_binding_assay": False,
                "pmid": 12345,
            },
            {
                "peptide": "NLVPMVATV",
                "mhc_class": "non-classical",
                "mhc_restriction": "human-MR1",
                "has_ptm": False,
                "is_binding_assay": False,
                "pmid": 12345,
            },
        ]
    ).to_parquet(obs_p)
    pd.DataFrame(
        [
            {
                "peptide": "GILGFVFTL",
                "mhc_class": "non-classical",
                "mhc_restriction": "HLA-G*01:01",
                "has_ptm": False,
                "response_measured": "qualitative binding",
                "qualitative_measurement": "Positive",
                "pmid": 99999,
            },
            {
                "peptide": "NLVPMVATV",
                "mhc_class": "non-classical",
                "mhc_restriction": "human-CD1b",
                "has_ptm": False,
                "response_measured": "qualitative binding",
                "qualitative_measurement": "Positive",
                "pmid": 99999,
            },
        ]
    ).to_parquet(bind_p)

    stats = build_nonclassical_mhc_expansion(
        observations_parquet=obs_p,
        binding_parquet=bind_p,
        human_fasta=fasta,
        mhc_full_seqs_csv=MHC_CSV,
        protected=ProtectedIdentityIndex([]),
        global_observed=set(),
        out_elution_path=tmp_path / "nc_elut.jsonl.gz",
        out_binding_path=tmp_path / "nc_bind.jsonl.gz",
        seed=42,
    )
    assert stats["elution"]["positives_admitted"] == 1
    assert stats["elution"]["pre_filter_counts"]["dropped_non_peptide_mhc"] == 2
    assert stats["binding"]["admitted"] == 1
    assert stats["binding"]["counts"]["dropped_non_peptide_mhc"] == 1


def test_expanded_bekker_jensen_bulk_motif_and_embargo(tmp_path: Path):
    fasta = tmp_path / "bulk.fasta"
    # Two clear tryptic cleavage fragments of length 10+
    fasta.write_text(
        ">sp|P10001|TEST_HUMAN GN=TEST\nMAAAAAAAAAKBBBBBBBBBBRCCCCCCCCCCKDDDDDDDDDDK\n".replace(
            "B", "A"
        )
    )
    # Use canonical amino acids only:
    seq = "MAAAAAAAAGKVVVVVVVVVVRLLLLLLLLLLKGGGGGGGGGGK"
    fasta.write_text(f">sp|P10001|TEST_HUMAN GN=TEST\n{seq}\n")
    bulk_p = tmp_path / "bulk.parquet"
    pd.DataFrame(
        [
            {
                "source": "Bekker-Jensen_2017",
                "granularity": "peptide",
                "digestion_enzyme": "Trypsin/P (cleaves K/R except before P)",
                "peptide": "VVVVVVVVVVR",
            },
            {
                "source": "Bekker-Jensen_2017",
                "granularity": "peptide",
                "digestion_enzyme": "Trypsin/P (cleaves K/R except before P)",
                "peptide": "LLLLLLLLLLK",
            },
        ]
    ).to_parquet(bulk_p)

    protected = ProtectedIdentityIndex(["LLLLLLLLLLK"])
    out_p = tmp_path / "bulk_out.jsonl.gz"
    stats = build_expanded_bekker_jensen_bulk(
        bulk_proteomics_parquet=bulk_p,
        human_fasta=fasta,
        protected=protected,
        out_path=out_p,
        seed=42,
    )
    assert stats["pairs_by_enzyme"]["trypsin_p"] == 1
    assert stats["total_rows_written"] == 2
    with gzip.open(out_p, "rt") as fh:
        rows = [Observation.from_dict(json.loads(line)) for line in fh]
    assert all(r.endpoint == "bulk_observation" for r in rows)
    assert all(r.context.enzyme == "trypsin_p" for r in rows)
    assert all(protected.reason(r.peptide) is None for r in rows)


def test_allocate_v23_elution_strata_preserves_non_elution_mass():
    base_w = {
        ("ic50", "I"): 0.30,
        ("bulk_observation", "chymotrypsin"): 0.08,
        ("half_life", "I"): 0.06,
        ("elution_observation", "human_study_1"): 0.28,
        ("elution_observation", "human_study_2"): 0.28,
    }
    counts = {
        ("ic50", "I"): 50000,
        ("bulk_observation", "chymotrypsin"): 40000,
        ("half_life", "I"): 18000,
        ("elution_observation", "human_study_1"): 100000,
        ("elution_observation", "human_study_2"): 100000,
        ("elution_observation", "v23_atlas:I"): 80000,
        ("elution_observation", "v23_atlas:II"): 80000,
        ("elution_observation", "v23_dq:II"): 10000,
        ("elution_observation", "v23_dodm:II"): 12000,
        ("elution_observation", "v23_nonclassical:I"): 5000,
    }
    out = allocate_v23_elution_strata(
        base_w,
        counts,
        budgets={
            "v23_atlas:": 0.08,
            "v23_dq:": 0.025,
            "v23_dodm:": 0.02,
            "v23_nonclassical:": 0.015,
        },
        max_total_fraction_of_elution=0.30,
    )
    # Non-elution budgets MUST be bit-for-bit identical
    assert out[("ic50", "I")] == pytest.approx(0.30, abs=1e-12)
    assert out[("bulk_observation", "chymotrypsin")] == pytest.approx(0.08, abs=1e-12)
    assert out[("half_life", "I")] == pytest.approx(0.06, abs=1e-12)
    # Total elution mass MUST be preserved identically
    elut_sum = sum(v for k, v in out.items() if k[0] == "elution_observation")
    assert elut_sum == pytest.approx(0.56, abs=1e-12)
    assert out[("elution_observation", "v23_atlas:I")] == pytest.approx(0.04, abs=1e-12)
    assert out[("elution_observation", "v23_dq:II")] == pytest.approx(0.025, abs=1e-12)
