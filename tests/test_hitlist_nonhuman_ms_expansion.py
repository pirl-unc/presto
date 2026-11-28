"""Tests for leak-free non-human MS elution expansion builder."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pandas as pd
import pytest

from presto.data.hitlist_binding_expansion import FastGrooveResolver, ProtectedIdentityIndex
from presto.data.hitlist_nonhuman_ms_expansion import (
    allocate_nonhuman_ms_strata,
    build_hitlist_nonhuman_ms_expansion,
    read_proteome_fasta,
    read_proteome_fasta_with_reviewed,
    resolve_nonhuman_mhc_candidates,
    resolve_source_junction,
)
from presto.data.modular import Observation
from presto.models.presto import Presto, PrestoConfig

G1_A = "GSHSMRYFFTSVSRPGRGEPRFIAVGYVDDTQFVRFDSDAASQRMEPRAPWIEQEGPEYWDGETRKVKAHSQTHRVDLGTLRGYYNQSEA"
G2_A = "GSHTVQRMYGCDVGSDWRFLRGYHQYAYDGKDYIALKEDLRSWTAADMAAQTTKHKWEAAHVAEQLRAYLEGTCVEWLRRYLENGKETLQ"
G1_B = "GPHSMRYFETSVSRPGLGEPRFIIVGYVDDTQFVRFDSDAENPRYEPRARWMEQEGPEYWERETQKAKGNEQSFRVDLRTLLGYYNQSKG"
G2_B = "GSHTIQVISGCEVGSDGRLLRGYQQYAYDGCDYIALNEDLKTWTAADMAALITKHKWEQAGEAERLRAYLEGTCVEWLRRYLKNGNATLL"
G1_C2 = "EDDIEADHVGTYGISVYQSPGDIGQYTFEFDGDELFYVDLDKKETVWMLPEFGQLASFDPQGGLQNIAVVKHNLGVLTKRSNSTPATNE"
G2_C2 = (
    "GDSERHFVYQFMGECYFTNGTQRIRYVTRYIYNREEYVRYDSDVGEHRAVTELGRPDAEYWNSQPEILERTRAELDTVCRHNYEGPETHTS"
)


class _FakeResolver:
    _MAP = {
        ("H2-K*b", "I"): (G1_A, G2_A),
        ("H2-D*b", "I"): (G1_B, G2_B),
        ("H2-AA*b/AB*b", "II"): (G1_C2, G2_C2),
        ("Calu-88*501:01", "I"): (G1_A, G2_B),
        ("Mamu-B*008:01", "I"): (G1_B, G2_A),
    }

    def __init__(self, mhc_full_seqs_csv=None, **kwargs):
        pass

    def resolve(self, restriction: str, mhc_class: str):
        return self._MAP.get((restriction.strip(), mhc_class.strip()))


def test_protected_identity_index_reason_and_long_hamming1():
    mid_pep = "ACDEFGHIKLMNPQR"  # 15-mer: mutating center index 7 leaves max 7-mer match
    mid_ham1 = "ACDEFGHAKLMNPQR"
    idx = ProtectedIdentityIndex({"SIINFEKL", mid_pep})
    assert idx.reason("SIINFEKL") == "exact"
    assert idx.reason("AASIINFEKLGG") == "shared8mer"
    assert idx.reason("SIINFEKA") == "hamming1"
    assert idx.reason(mid_ham1) == "hamming1"
    assert idx.is_protected(mid_ham1)
    assert idx.reason("NLVPMVATV") is None


def test_fast_groove_resolver_canis_and_synonymous_three_field(tmp_path: Path):
    import mhcseqs

    real_csv = Path(mhcseqs.find_csv("mhc-full-seqs.csv"))
    if not real_csv.exists():
        pytest.skip("local mhc-full-seqs.csv unavailable")
    resolver = FastGrooveResolver(mhc_full_seqs_csv=real_csv)

    # Dog Calu-88*501:01 -> CanisSp-88*501:01
    calu_501 = resolver.resolve("Calu-88*501:01", "I")
    dla_501 = resolver.resolve("CanisSp-88*501:01", "I")
    assert calu_501 is not None
    assert calu_501 == dla_501
    assert resolver.resolve("Calu-88*508:01", "I") is not None

    # Pig SLA-1*04:01 -> synonymous 3-field SusSp-1*04:01:01/:02
    sla = resolver.resolve("SLA-1*04:01", "I")
    assert sla is not None
    assert len(sla[0]) > 50 and len(sla[1]) > 50

    # Mouse H2-b alleles resolve cleanly
    assert resolver.resolve("H2-K*b", "I") is not None
    assert resolver.resolve("H2-D*b", "I") is not None
    assert resolver.resolve("H2-AA*b/AB*b", "II") is not None


def test_read_proteome_fasta_ensembl_and_uniprot(tmp_path: Path):
    ens_gz = tmp_path / "mouse.fa.gz"
    with gzip.open(ens_gz, "wt", encoding="utf-8") as f:
        f.write(">ENSMUSP00001.1 pep chromosome:GRCm39 gene:ENSMUSG00001.1\n")
        f.write("MAAAAKLLLLVVVV\nFFFFYYYY*\n")
    prots_ens = read_proteome_fasta(ens_gz)
    assert "ENSMUSP00001.1" in prots_ens
    assert prots_ens["ENSMUSP00001.1"].gene == "ENSMUSG00001.1"
    assert prots_ens["ENSMUSP00001.1"].sequence == "MAAAAKLLLLVVVVFFFFYYYY"

    uni_fa = tmp_path / "dog.fasta"
    uni_fa.write_text(
        ">tr|A0A8C0LRN9|A0A8C0LRN9_CANLF Phospholipase OS=Canis lupus GN=PLA2G4A PE=4\n"
        "MGGGGWWWWYYYYPPPP\n",
        encoding="utf-8",
    )
    prots_uni = read_proteome_fasta(uni_fa)
    assert "A0A8C0LRN9" in prots_uni
    assert prots_uni["A0A8C0LRN9"].gene == "PLA2G4A"
    assert prots_uni["A0A8C0LRN9"].sequence == "MGGGGWWWWYYYYPPPP"


def test_resolve_nonhuman_mhc_candidates_exact_haplotypes_and_rejections():
    resolver = _FakeResolver()

    # Exact H2-b haplotypes
    h2b_i = resolve_nonhuman_mhc_candidates(
        "H2-b class I", "mus_musculus", "Mus musculus", "I", resolver, allow_exact_haplotypes=True
    )
    assert h2b_i is not None
    assert len(h2b_i[0]) == 2
    assert h2b_i[2] == "exact_haplotype"

    h2b_ii = resolve_nonhuman_mhc_candidates(
        "H2-b class II", "mus_musculus", "Mus musculus", "II", resolver, allow_exact_haplotypes=True
    )
    assert h2b_ii is not None
    assert len(h2b_ii[0]) == 1
    assert h2b_ii[2] == "exact_haplotype"

    # Disabled when allow_exact_haplotypes=False
    assert (
        resolve_nonhuman_mhc_candidates(
            "H2-b class I",
            "mus_musculus",
            "Mus musculus",
            "I",
            resolver,
            allow_exact_haplotypes=False,
        )
        is None
    )

    # Strictly rejected: class-only, incomplete H2-d class II, Saha, mutants, partial multiallelic
    for raw, sp, cls in (
        ("Saha class I", "sarcophilus_harrisii", "I"),
        ("H2 class I", "mus_musculus", "I"),
        ("H2 class II", "mus_musculus", "II"),
        ("H2-d class II", "mus_musculus", "II"),
        ("H2-K*b Y22F mutant", "mus_musculus", "I"),
        ("H2-K*b;H2-UNKNOWN", "mus_musculus", "I"),
    ):
        assert resolve_nonhuman_mhc_candidates(raw, sp, sp, cls, resolver) is None


def test_build_nonhuman_ms_expansion_end_to_end_and_presto_forward(tmp_path: Path, monkeypatch):
    import presto.data.hitlist_nonhuman_ms_expansion as mod

    monkeypatch.setattr(mod, "FastGrooveResolver", _FakeResolver)

    # Build synthetic mouse + dog proteomes with unique target motifs:
    # - ENSMUSP001: unique occurrence of WWYFFCCMM (unanimous flanks)
    # - ENSMUSP002 & ENSMUSP003: same peptide VVVWWWYYY with DIFFERENT flanks (ambiguous)
    # - ENSMUSP004: unique 15-mer Class II peptide HHNNKKQQDDYYMMR
    # - DOG001: unique dog 9-mer FFGGHHIIK
    mouse_fa = tmp_path / "mouse.fa.gz"
    with gzip.open(mouse_fa, "wt", encoding="utf-8") as f:
        f.write(">ENSMUSP001 gene:ENSMUSG001\n")
        f.write(
            "ACDEFGHIKLMNPQR"  # 15-AA N-flank
            "WWYFFCCMM"  # target 9-mer (unanimous)
            "VWYACDEFGHIKLMN"  # 15-AA C-flank
            "PQRSTVWYACDEFGHIKLMNPQRSTVWYACDEFGH\n"
        )
        f.write(">ENSMUSP002 gene:ENSMUSG002\n")
        f.write(
            "AAAAAAAAAAAAAAA"
            "VVVWWWYYY"  # ambiguous flanks (A vs D)
            "CCCCCCCCCCCCCCC"
            "MNPQRSTVWYDEFGHIKLMNPQRSTVWYDEFGHIK\n"
        )
        f.write(">ENSMUSP003 gene:ENSMUSG003\n")
        f.write("DDDDDDDDDDDDDDDVVVWWWYYYEEEEEEEEEEEEEEEQRVWYACDEFGHIKLMNPQRSTVWYACDEFGHIKL\n")
        f.write(">ENSMUSP004 gene:ENSMUSG004\n")
        f.write(
            "GGGGGGGGGGGGGGG"
            "HHNNKKQQDDYYMMR"  # 15-mer Class II (unanimous)
            "HHHHHHHHHHHHHHH"
            "CDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRST\n"
        )

    dog_fa = tmp_path / "dog.fasta"
    dog_fa.write_text(
        ">tr|DOG001|DOG001_CANLF Test OS=Canis lupus GN=DOGG1\n"
        "TTTTTTTTTTTTTTT"
        "FFGGHHIIK"
        "SSSSSSSSSSSSSSS"
        "WYRSTVWYACDEFGHIKLMNPQRSTVWYACDEFGH\n",
        encoding="utf-8",
    )

    def _row(**overrides):
        base = {
            "peptide": "WWYFFCCMM",
            "mhc_species": "Mus musculus",
            "mhc_class": "I",
            "mhc_restriction": "H2-K*b",
            "mhc_allele_set": "",
            "has_ptm": False,
            "is_non_peptide_ligand": False,
            "is_potential_contaminant": None,
            "restriction_evidence": "unknown",
            "assay_method": "cellular MHC/mass spectrometry",
            "response_measured": "ligand presentation",
            "qualitative_measurement": "Positive",
            "pmid": 12345678,
            "assay_iri": "iri:1",
            "reference_iri": "ref:1",
        }
        base.update(overrides)
        return base

    # Include a human MS peptide inside ENSMUSP001 to verify global observed exclusion
    human_obs_pep = "PQRSTVWYA"
    obs_rows = [
        _row(peptide=human_obs_pep, mhc_species="Homo sapiens", mhc_restriction="HLA-A*02:01"),
        # Admitted: mouse Class I unanimous flank
        _row(peptide="WWYFFCCMM", assay_iri="iri:pos1"),
        # Duplicate (peptide, mhc, pmid) -> must be deduplicated
        _row(peptide="WWYFFCCMM", assay_iri="iri:pos1_dup"),
        # Admitted: mouse Class I H2-b haplotype with AMBIGUOUS flanks
        _row(peptide="VVVWWWYYY", mhc_restriction="H2-b class I", assay_iri="iri:pos2"),
        # Admitted: mouse Class II H2-b haplotype
        _row(
            peptide="HHNNKKQQDDYYMMR",
            mhc_class="II",
            mhc_restriction="H2-b class II",
            assay_iri="iri:pos3",
        ),
        # Admitted: dog Calu-88*501:01
        _row(
            peptide="FFGGHHIIK",
            mhc_species="Canis lupus",
            mhc_restriction="Calu-88*501:01",
            assay_iri="iri:pos4",
        ),
        # Dropped: embargoed peptide (exact)
        _row(peptide="EMBARGOAA", assay_iri="iri:drop_emb"),
        # Dropped: predicted restriction
        _row(
            peptide="WWYFFCCMM",
            restriction_evidence="predicted",
            pmid=999,
            assay_iri="iri:drop_pred",
        ),
        # Dropped: PTM
        _row(peptide="WWYFFCCMM", has_ptm=True, pmid=998, assay_iri="iri:drop_ptm"),
        # Dropped: Saha class I
        _row(
            peptide="WWYFFCCMM",
            mhc_species="Sarcophilus harrisii",
            mhc_restriction="Saha class I",
            assay_iri="iri:drop_saha",
        ),
        # Dropped: viral peptide not in mouse proteome
        _row(peptide="QQQQQQQQQ", assay_iri="iri:drop_unmapped"),
    ]
    obs_parquet = tmp_path / "observations.parquet"
    pd.DataFrame(obs_rows).to_parquet(obs_parquet, index=False)

    # Protect EMBARGOAA plus a substring inside ENSMUSP001 to ensure negatives also avoid protected
    protected = ProtectedIdentityIndex({"EMBARGOAA", "VWYACDEF"})
    out_jsonl = tmp_path / "nonhuman-ms.jsonl.gz"

    stats = build_hitlist_nonhuman_ms_expansion(
        obs_parquet,
        protected,
        proteome_paths={"mus_musculus": mouse_fa, "canis_lupus": dog_fa},
        out_path=out_jsonl,
        neg_per_pos=2,
        seed=43,
        allow_exact_haplotypes=True,
    )

    assert stats["test_accessed"] is False
    assert stats["positives_admitted"] == 4
    assert stats["negatives_admitted"] == 8
    assert stats["total_rows_written"] == 12
    assert stats["counts"]["drop_duplicate_pep_mhc_pmid"] == 1
    assert stats["counts"]["drop_predicted_restriction"] == 1
    assert stats["counts"]["drop_ptm"] == 1
    assert stats["counts"]["drop_unmapped_to_proteome"] == 1
    assert stats["positives_by_junction_policy"]["unanimous_reference_junction"] == 3
    assert stats["positives_by_junction_policy"]["ambiguous_cross_gene_junction"] == 1

    with gzip.open(out_jsonl, "rt", encoding="utf-8") as f:
        parsed = [Observation.from_dict(json.loads(line)) for line in f]
    assert len(parsed) == 12

    by_pair: dict[str, list[Observation]] = {}
    for obs in parsed:
        obs.validate()
        # Strict embargo on ALL rows (positives AND negatives)
        assert protected.reason(obs.peptide) is None
        # Zero human APC/TPM/intervention leakage
        assert obs.context.cell_type is None
        assert obs.context.components == {}
        assert obs.context.cytokines == {}
        assert obs.context.interventions == {}
        assert obs.endpoint == "elution_observation"
        assert obs.split == "train"
        assert obs.provenance["study_id"].startswith("nh_ms:")
        if obs.value == 0.0:
            assert obs.peptide != human_obs_pep
        by_pair.setdefault(obs.provenance["pair_id"], []).append(obs)

    assert len(by_pair) == 4
    for pair_rows in by_pair.values():
        pos_rows = [r for r in pair_rows if r.value == 1.0]
        neg_rows = [r for r in pair_rows if r.value == 0.0]
        assert len(pos_rows) == 1 and len(neg_rows) == 2
        pos = pos_rows[0]
        # Distinct negatives per replicate & exact length match
        assert len({r.peptide for r in neg_rows}) == 2
        for neg in neg_rows:
            assert len(neg.peptide) == len(pos.peptide)
            assert neg.mhc == pos.mhc
        # Symmetric missing-junction check
        if pos.peptide == "VVVWWWYYY":
            assert pos.context.source_n is None and pos.context.source_c is None
            for neg in neg_rows:
                assert neg.context.source_n is None and neg.context.source_c is None
        elif pos.peptide == "WWYFFCCMM":
            assert pos.context.source_n == "ACDEFGHIKLMNPQR"
            assert pos.context.source_c == "VWYACDEFGHIKLMN"
            for neg in neg_rows:
                assert neg.context.source_n is not None

    # Determinism check
    out_jsonl_2 = tmp_path / "nonhuman-ms-2.jsonl.gz"
    build_hitlist_nonhuman_ms_expansion(
        obs_parquet,
        protected,
        proteome_paths={"mus_musculus": mouse_fa, "canis_lupus": dog_fa},
        out_path=out_jsonl_2,
        neg_per_pos=2,
        seed=43,
        allow_exact_haplotypes=True,
    )
    with gzip.open(out_jsonl, "rt") as f1, gzip.open(out_jsonl_2, "rt") as f2:
        assert f1.read() == f2.read()

    # Forward + backward compatibility with Presto-Omni v2.2 model
    import torch
    from presto.training.modular import train_step

    cfg = PrestoConfig(
        endpoints=("ic50", "elution_observation"),
        d_model=32,
        n_layers=1,
        n_heads=2,
        d_ff=64,
        source_context=True,
        elution_staging="affinity_downstream",
        elution_affinity_endpoint="ic50",
        elution_shared_processing=True,
        shared_processing_v2=True,
        elution_pmhc_residual=True,
        interaction_mode="anchor_v3",
        groove_segment_embedding=True,
        groove_cross_chain=True,
        query_anchor_embedding=True,
        groove_pocket_pool=True,
        register_pooling_mode="thermodynamic",
    )
    model = Presto(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    metrics = train_step(model, opt, parsed, {})
    assert metrics["loss"] > 0.0 and torch.isfinite(torch.tensor(metrics["loss"])).item()


def test_allocate_nonhuman_ms_strata_zero_regression_guard():
    wmap = {
        ("ic50", "I", "hu_ic50"): 0.20,
        ("ic50", "II", "hu_ic50_2"): 0.22,
        ("bulk_observation", "none", "28561026:trypsin_p"): 0.08,
        ("tcell_response", "I", "tcell_s1"): 0.10,
        ("elution_observation", "I", "hu_ms_1"): 0.25,
        ("elution_observation", "II", "hu_ms_2"): 0.15,
        ("elution_observation", "I", "nh_ms:mus_musculus:h2_k_b"): 1e-4,
        ("elution_observation", "I", "nh_ms:macaca_mulatta:mamu_b_008_01"): 1e-4,
    }
    counts = {
        ("elution_observation", "I", "nh_ms:mus_musculus:h2_k_b"): 25000,
        ("elution_observation", "I", "nh_ms:macaca_mulatta:mamu_b_008_01"): 900,
    }
    res = allocate_nonhuman_ms_strata(
        wmap, counts, nh_ms_draw_mass=0.06, max_fraction_of_elution=0.25
    )

    # Non-elution strata are 100% untouched (exact bit-for-bit parity!)
    for k in (
        ("ic50", "I", "hu_ic50"),
        ("ic50", "II", "hu_ic50_2"),
        ("bulk_observation", "none", "28561026:trypsin_p"),
        ("tcell_response", "I", "tcell_s1"),
    ):
        assert res[k] == wmap[k]

    # Total elution mass preserved & human elution >= 75% of elution pool
    orig_elut = sum(v for k, v in wmap.items() if k[0] == "elution_observation")
    new_elut = sum(v for k, v in res.items() if k[0] == "elution_observation")
    assert new_elut == pytest.approx(orig_elut, rel=1e-9)

    nh_sum = sum(v for k, v in res.items() if str(k[2]).startswith("nh_ms:"))
    assert nh_sum == pytest.approx(0.06, rel=1e-9)
    # Sqrt-balancing: 25000 vs 900 -> sqrt ratio is 158.11 / 30 = 5.27 (not 27.8x!)
    ratio = (
        res[("elution_observation", "I", "nh_ms:mus_musculus:h2_k_b")]
        / res[("elution_observation", "I", "nh_ms:macaca_mulatta:mamu_b_008_01")]
    )
    assert 5.0 < ratio < 5.6


def test_nullable_pandas_na_pmid_and_fields(tmp_path: Path, monkeypatch):
    import presto.data.hitlist_nonhuman_ms_expansion as mod

    monkeypatch.setattr(mod, "FastGrooveResolver", _FakeResolver)
    mouse_fa = tmp_path / "mouse.fa"
    mouse_fa.write_text(
        ">ENSMUSP099 gene:ENSMUSG099\n"
        "ACDEFGHIKLMNPQRWWYFFCCMMVWYACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRST\n",
        encoding="utf-8",
    )
    df = pd.DataFrame(
        {
            "peptide": ["WWYFFCCMM"],
            "mhc_species": ["Mus musculus"],
            "mhc_class": ["I"],
            "mhc_restriction": ["H2-K*b"],
            "mhc_allele_set": pd.Series([pd.NA], dtype="string"),
            "has_ptm": pd.Series([False], dtype="boolean"),
            "is_non_peptide_ligand": pd.Series([False], dtype="boolean"),
            "is_potential_contaminant": pd.Series([pd.NA], dtype="boolean"),
            "restriction_evidence": pd.Series([pd.NA], dtype="string"),
            "assay_method": ["cellular MHC/mass spectrometry"],
            "response_measured": ["ligand presentation"],
            "qualitative_measurement": ["Positive"],
            "pmid": pd.Series([pd.NA], dtype="Int64"),
            "assay_iri": pd.Series([pd.NA], dtype="string"),
            "reference_iri": pd.Series([pd.NA], dtype="string"),
        }
    )
    obs_parquet = tmp_path / "obs_na.parquet"
    df.to_parquet(obs_parquet, index=False)
    out_jsonl = tmp_path / "out_na.jsonl"
    stats = build_hitlist_nonhuman_ms_expansion(
        obs_parquet,
        ProtectedIdentityIndex(set()),
        proteome_paths={"mus_musculus": mouse_fa},
        out_path=out_jsonl,
        neg_per_pos=1,
    )
    assert stats["positives_admitted"] == 1
    assert stats["negatives_admitted"] == 1
    rows = [json.loads(line) for line in out_jsonl.read_text().splitlines()]
    assert rows[0]["provenance"]["pmid"] == "unknown"


def test_resolve_source_junction_within_gene_canonical_and_reviewed(tmp_path: Path):
    from presto.data.presentation import Protein

    # Case 1: Single gene (ENSMUSG010) with canonical isoform + differing splice variant
    prots = {
        "ENSMUSP010.1": Protein(
            "ENSMUSP010.1", "ENSMUSG010.1", "AAAAAAAAAAAAAAAWWYFFCCMMCCCCCCCCCCCCCCC"
        ),
        "ENSMUSP011.1": Protein(
            "ENSMUSP011.1", "ENSMUSG010.1", "DDDDDDDDDDDDDDDWWYFFCCMMGGGGGGGGGGGGGGG"
        ),
        # Case 2: Internal repeat inside a canonical transcript
        "ENSMUSP020.1": Protein(
            "ENSMUSP020.1",
            "ENSMUSG020.1",
            "AAAAAKLMNPQRSTCCCCCCKLMNPQRSTDDDDDD",
        ),
        # Case 3: Cross-gene paralogs (ENSMUSG030 vs ENSMUSG031)
        "ENSMUSP030.1": Protein(
            "ENSMUSP030.1", "ENSMUSG030.1", "EEEEEEEEEEEEEEEPPQQRRSSTHHHHHHHHHHHHHHH"
        ),
        "ENSMUSP031.1": Protein(
            "ENSMUSP031.1", "ENSMUSG031.1", "FFFFFFFFFFFFFFFPPQQRRSSTIIIIIIIIIIIIIII"
        ),
    }
    canon_ids = {"ENSMUSP010", "ENSMUSP020.1", "ENSMUSP030.1"}

    # Single-gene canonical resolves flanks & narrows background preference to canonical isoform
    ctx1, pol1, pref1 = resolve_source_junction(
        "WWYFFCCMM",
        [("ENSMUSP010.1", 15), ("ENSMUSP011.1", 15)],
        prots,
        canonical_ids=canon_ids,
    )
    assert pol1 == "within_gene_canonical_junction"
    assert ctx1["source_n"] == "AAAAAAAAAAAAAAA"
    assert ctx1["source_c"] == "CCCCCCCCCCCCCCC"
    assert pref1 == [("ENSMUSP010.1", 15)]

    # Internal repeat inside canonical isoform remains conservatively masked
    ctx2, pol2, _ = resolve_source_junction(
        "KLMNPQRST",
        [("ENSMUSP020.1", 5), ("ENSMUSP020.1", 20)],
        prots,
        canonical_ids=canon_ids,
    )
    assert pol2 == "ambiguous_within_gene_junction"
    assert ctx2["source_n"] is None and ctx2["source_c"] is None

    # Cross-gene paralogs NEVER guessed even when one paralog is in canonical_ids
    ctx3, pol3, _ = resolve_source_junction(
        "PPQQRRSST",
        [("ENSMUSP030.1", 15), ("ENSMUSP031.1", 15)],
        prots,
        canonical_ids=canon_ids,
    )
    assert pol3 == "ambiguous_cross_gene_junction"
    assert ctx3["source_n"] is None and ctx3["source_c"] is None

    # Case 4: Single-gene UniProt Swiss-Prot (sp|) resolves over TrEMBL (tr|)
    uni_fa = tmp_path / "uni.fasta"
    uni_fa.write_text(
        ">sp|P12345|TEST_CANLF Reviewed OS=Canis lupus GN=GENE1\n"
        "MMMMMMMMMMMMMMMFFGGHHIIKNNNNNNNNNNNNNNN\n"
        ">tr|A0A999|A0A999_CANLF Unreviewed OS=Canis lupus GN=GENE1\n"
        "PPPPPPPPPPPPPPPFFGGHHIIKQQQQQQQQQQQQQQQ\n",
        encoding="utf-8",
    )
    uni_prots, rev_ids = read_proteome_fasta_with_reviewed(uni_fa)
    assert rev_ids == {"P12345"}
    ctx4, pol4, pref4 = resolve_source_junction(
        "FFGGHHIIK",
        [("P12345", 15), ("A0A999", 15)],
        uni_prots,
        reviewed_ids=rev_ids,
    )
    assert pol4 == "within_gene_reviewed_junction"
    assert ctx4["source_n"] == "MMMMMMMMMMMMMMM"
    assert ctx4["source_c"] == "NNNNNNNNNNNNNNN"
    assert pref4 == [("P12345", 15)]
