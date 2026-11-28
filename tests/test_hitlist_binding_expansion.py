"""Tests for leak-free hitlist purified binding expansion builder."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
import pandas as pd

from presto.data.assay_measurement import observation_track
from presto.data.hitlist_binding_expansion import (
    ProtectedIdentityIndex,
    build_hitlist_binding_expansion,
    load_protected_identity_index,
)
from presto.data.modular import Observation


def test_protected_identity_index_blocks_exact_8mer_and_hamming1(tmp_path: Path):
    emb = tmp_path / "embargo.txt.gz"
    with gzip.open(emb, "wt", encoding="utf-8") as f:
        f.write("SIINFEKL\n")
        f.write("PKYVKQNTLKLAT\n")
    val_f = tmp_path / "val.jsonl"
    val_f.write_text(json.dumps({"peptide": "GILGFVFTL"}) + "\n", encoding="utf-8")

    idx, summary = load_protected_identity_index(emb, [val_f])
    assert summary["test_accessed"] is False
    assert idx.is_protected("SIINFEKL")  # exact
    assert idx.is_protected("AASIINFEKLGG")  # 8-mer substring
    assert idx.is_protected("SIINFEKA")  # Hamming-1
    assert idx.is_protected("GILGFVFTL")  # from val.jsonl
    assert not idx.is_protected("NLVPMVATV")


def test_build_hitlist_binding_expansion_slices_and_isolation(tmp_path: Path, monkeypatch):
    import presto.data.hitlist_binding_expansion as mod

    class _FakeResolver:
        def __init__(self, mhc_full_seqs_csv=None, **kwargs):
            pass

        def resolve(self, restriction: str, mhc_class: str):
            g1 = (
                "GSHSMRYFFTSVSRPGRGEPRFIAVGYVDDTQFVRFDSDAASQRME"
                "PRAPWIEQEGPEYWDGETRKVKAHSQTHRVDLGTLRGYYNQSEA"
            )
            g2 = (
                "GSHTVQRMYGCDVGSDWRFLRGYHQYAYDGKDYIALKEDLRSWTAA"
                "DMAAQTTKHKWEAAHVAEQLRAYLEGTCVEWLRRYLENGKETLQ"
            )
            return (g1, g2) if mhc_class in ("I", "II") else None

    monkeypatch.setattr(mod, "FastGrooveResolver", _FakeResolver)

    rows = [
        # Slice 1: Class II HT binding (1 pos + 5 neg -> 1:2 balanced keeps 1 pos + 2 neg)
        {
            "peptide": "AAYLLPRRGPRLGVR",
            "has_ptm": False,
            "mhc_restriction": "HLA-DRB1*04:04",
            "mhc_species": "Homo sapiens",
            "mhc_class": "II",
            "assay_iri": "a1",
            "reference_iri": "r1",
            "pmid": 35486722,
            "assay_comments": "",
            "qualitative_measurement": "Positive",
            "assay_method": "High throughput multiplexed assay",
            "response_measured": "MHC binding",
            "measurement_units": None,
            "measurement_inequality": None,
            "quantitative_measurement": None,
            "quantitative_value": None,
        },
        *[
            {
                "peptide": f"GGYLLPRRGPRLGV{aa}",
                "has_ptm": False,
                "mhc_restriction": "HLA-DRB1*04:04",
                "mhc_species": "Homo sapiens",
                "mhc_class": "II",
                "assay_iri": f"an{i}",
                "reference_iri": "r1",
                "pmid": 35486722,
                "assay_comments": "",
                "qualitative_measurement": "Negative",
                "assay_method": "High throughput multiplexed assay",
                "response_measured": "MHC binding",
                "measurement_units": None,
                "measurement_inequality": None,
                "quantitative_measurement": None,
                "quantitative_value": None,
            }
            for i, aa in enumerate("ACDEF")
        ],
        # Embargoed peptide must be dropped
        {
            "peptide": "PKYVKQNTLKLATGM",
            "has_ptm": False,
            "mhc_restriction": "HLA-DRB1*04:04",
            "mhc_species": "Homo sapiens",
            "mhc_class": "II",
            "assay_iri": "a_emb",
            "reference_iri": "r1",
            "pmid": 35486722,
            "assay_comments": "",
            "qualitative_measurement": "Positive",
            "assay_method": "High throughput multiplexed assay",
            "response_measured": "MHC binding",
            "measurement_units": None,
            "measurement_inequality": None,
            "quantitative_measurement": None,
            "quantitative_value": None,
        },
        # Slice 2: Non-human purified quant affinity
        {
            "peptide": "NLVPMVATV",
            "has_ptm": False,
            "mhc_restriction": "H2-K*b",
            "mhc_species": "Mus musculus",
            "mhc_class": "I",
            "assay_iri": "nh1",
            "reference_iri": "r2",
            "pmid": 123456,
            "assay_comments": "",
            "qualitative_measurement": "Positive",
            "assay_method": "purified MHC/competitive/radioactivity",
            "response_measured": "half maximal inhibitory concentration (IC50)",
            "measurement_units": "nM",
            "measurement_inequality": "=",
            "quantitative_measurement": "42.0",
            "quantitative_value": 42.0,
        },
        # Slice 3: PMID 12135423 Class I KD -> must land on track=None (protocol_citation_conflict)
        {
            "peptide": "YMLDLQPET",
            "has_ptm": False,
            "mhc_restriction": "HLA-A*02:01",
            "mhc_species": "Homo sapiens",
            "mhc_class": "I",
            "assay_iri": "p12_1",
            "reference_iri": "r3",
            "pmid": 12135423,
            "assay_comments": "Reference 12135423 protocol",
            "qualitative_measurement": "Negative",
            "assay_method": "purified MHC/competitive/fluorescence",
            "response_measured": "dissociation constant KD",
            "measurement_units": "nM",
            "measurement_inequality": ">",
            "quantitative_measurement": ">20000",
            "quantitative_value": 20000.0,
        },
    ]
    pq_path = tmp_path / "binding.parquet"
    pd.DataFrame(rows).to_parquet(pq_path)

    out_path = tmp_path / "expansion.jsonl"
    prot = ProtectedIdentityIndex({"PKYVKQNTLKLAT"})
    allowed_tracks = {("ic50", "purified", "competitive", "radioactivity")}
    stats = build_hitlist_binding_expansion(
        pq_path,
        prot,
        allowed_tracks=allowed_tracks,
        out_path=out_path,
        ht_neg_per_pos=2,
    )
    assert stats["test_accessed"] is False
    assert stats["slice1_ht_class2"]["admitted"] == 3  # 1 pos + 2 balanced neg
    assert stats["slice1_ht_class2"]["drop_embargo"] == 1
    assert stats["slice2_nonhuman_quant"]["admitted"] == 1
    assert stats["slice3_pmid12135423_kd"]["admitted"] == 1

    loaded = [Observation.from_dict(json.loads(line)) for line in out_path.read_text().splitlines()]
    assert len(loaded) == 5
    by_src = {o.source: [] for o in loaded}
    for o in loaded:
        by_src[o.source].append(o)
    # PMID 12135423 isolated on track=None
    assert observation_track(by_src["hitlist_pmid12135423_kd"][0]) is None
    assert by_src["hitlist_pmid12135423_kd"][0].qualifier == ">"
    # Non-human quant routed to exact purified track
    nh_track = observation_track(by_src["hitlist_nonhuman_purified_quant"][0])
    assert nh_track is not None and nh_track.key == "ic50/purified/competitive/radioactivity"
    # Class II HT direct binding routed as qualitative_binding on track=None
    assert all(
        o.endpoint == "qualitative_binding" and observation_track(o) is None
        for o in by_src["hitlist_ht_class2_binding"]
    )
