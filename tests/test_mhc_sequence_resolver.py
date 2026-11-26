"""Tests for mhcseqs-first exact sequence resolution."""

import types

import pytest

from presto.data import mhc_sequence_resolver as resolver
from presto.data.mhc_sequence_resolver import ExactMHCInput


def test_resolve_exact_mhc_sequences_prefers_mhcseqs(monkeypatch):
    monkeypatch.setattr(
        resolver,
        "_load_mhcseqs_input_lookup",
        lambda search_dir=None: {
            "HLA-A*02:01": ExactMHCInput(
                allele="HLA-A*02:01",
                sequence="A" * 80,
                source="mhcseqs",
            ),
            "HLA-A*02:01".upper(): ExactMHCInput(
                allele="HLA-A*02:01",
                sequence="A" * 80,
                source="mhcseqs",
            ),
        },
    )
    monkeypatch.setattr(
        resolver,
        "_require_mhcseqs",
        lambda: types.SimpleNamespace(normalize_allele_name=lambda allele: allele),
    )

    mapping, stats = resolver.resolve_exact_mhc_sequences(
        ["HLA-A*02:01", "HLA-B*07:02"],
        index_csv=None,
        prefer_mhcseqs=True,
    )

    assert mapping == {"HLA-A*02:01": "A" * 80}
    assert stats["resolved"] == 1
    assert stats["resolved_mhcseqs"] == 1
    assert stats["resolved_index"] == 0
    assert stats["missing"] == 1


def test_resolve_exact_mhc_sequences_falls_back_to_index(monkeypatch):
    monkeypatch.setattr(
        resolver,
        "_load_mhcseqs_input_lookup",
        lambda search_dir=None: {},
    )
    monkeypatch.setattr(
        resolver,
        "_require_mhcseqs",
        lambda: types.SimpleNamespace(normalize_allele_name=lambda allele: allele),
    )
    monkeypatch.setattr(
        resolver,
        "resolve_alleles",
        lambda index_csv, alleles, include_sequence: [
            {
                "input": allele,
                "found": allele == "HLA-B*07:02",
                "sequence": "B" * 80 if allele == "HLA-B*07:02" else "",
            }
            for allele in alleles
        ],
    )

    mapping, stats = resolver.resolve_exact_mhc_sequences(
        ["HLA-A*02:01", "HLA-B*07:02"],
        index_csv="dummy.csv",
        prefer_mhcseqs=True,
    )

    assert mapping == {"HLA-B*07:02": "B" * 80}
    assert stats["resolved"] == 1
    assert stats["resolved_mhcseqs"] == 0
    assert stats["resolved_index"] == 1
    assert stats["missing"] == 1


def test_resolve_exact_mhc_inputs_prefers_mhcseqs_grooves(monkeypatch):
    monkeypatch.setattr(
        resolver,
        "_load_mhcseqs_input_lookup",
        lambda search_dir=None: {
            "HLA-A*02:01": ExactMHCInput(
                allele="HLA-A*02:01",
                sequence="A" * 181,
                groove1="A" * 90,
                groove2="B" * 93,
                mhc_class="I",
                source="mhcseqs",
            ),
            "HLA-A*02:01".upper(): ExactMHCInput(
                allele="HLA-A*02:01",
                sequence="A" * 181,
                groove1="A" * 90,
                groove2="B" * 93,
                mhc_class="I",
                source="mhcseqs",
            ),
        },
    )
    monkeypatch.setattr(
        resolver,
        "_require_mhcseqs",
        lambda: types.SimpleNamespace(normalize_allele_name=lambda allele: allele),
    )

    mapping, stats = resolver.resolve_exact_mhc_inputs(
        ["HLA-A*02:01"],
        index_csv=None,
        prefer_mhcseqs=True,
    )

    assert mapping["HLA-A*02:01"].groove1 == "A" * 90
    assert mapping["HLA-A*02:01"].groove2 == "B" * 93
    assert mapping["HLA-A*02:01"].source == "mhcseqs"
    assert stats["resolved"] == 1
    assert stats["resolved_mhcseqs"] == 1


def test_lookup_exact_mhc_input_normalizes_query(monkeypatch):
    record = ExactMHCInput(
        allele="HLA-A*02:01",
        sequence="A" * 181,
        groove1="A" * 90,
        groove2="C" * 93,
        mhc_class="I",
        source="mhcseqs",
    )
    monkeypatch.setattr(
        resolver,
        "_load_mhcseqs_input_lookup",
        lambda search_dir=None: {
            "HLA-A*02:01": record,
            "HLA-A*02:01".upper(): record,
        },
    )
    monkeypatch.setattr(
        resolver,
        "_require_mhcseqs",
        lambda: types.SimpleNamespace(normalize_allele_name=lambda allele: "HLA-A*02:01"),
    )

    resolved = resolver.lookup_exact_mhc_input("A0201")

    assert resolved == record


def test_load_mhcseqs_catalog_inputs_deduplicates_lookup_aliases(monkeypatch):
    record = ExactMHCInput(
        allele="HLA-A*02:01",
        sequence="A" * 181,
        groove1="A" * 90,
        groove2="C" * 93,
        mhc_class="I",
        source="mhcseqs",
    )
    monkeypatch.setattr(
        resolver,
        "_load_mhcseqs_input_lookup",
        lambda search_dir=None: {
            "HLA-A*02:01": record,
            "hla-a*02:01": record,
        },
    )

    catalog = resolver.load_mhcseqs_catalog_inputs()

    assert catalog == {"HLA-A*02:01": record}


def test_load_mhcseqs_catalog_inputs_fails_loudly_when_catalog_is_empty(monkeypatch):
    monkeypatch.setattr(
        resolver,
        "_load_mhcseqs_input_lookup",
        lambda search_dir=None: {},
    )

    with pytest.raises(RuntimeError, match="no usable exact MHC inputs"):
        resolver.load_mhcseqs_catalog_inputs()


def test_resolve_class_i_groove_halves_prefers_exact_mhcseqs(monkeypatch):
    record = ExactMHCInput(
        allele="HLA-A*02:01",
        sequence="A" * 181,
        groove1="G" * 90,
        groove2="H" * 93,
        mhc_class="I",
        source="mhcseqs",
    )
    monkeypatch.setattr(
        resolver,
        "lookup_exact_mhc_input",
        lambda allele, mhcseqs_search_dir=None: record,
    )

    resolved = resolver.resolve_class_i_groove_halves(
        allele="HLA-A*02:01",
        allele_sequences={"HLA-A*02:01": "S" * 181},
    )

    assert resolved == ("G" * 90, "H" * 93)


def test_resolve_class_i_groove_halves_falls_back_to_sequence_map(monkeypatch):
    monkeypatch.setattr(
        resolver,
        "lookup_exact_mhc_input",
        lambda allele, mhcseqs_search_dir=None: None,
    )
    monkeypatch.setattr(
        resolver,
        "prepare_mhc_input",
        lambda *, mhc_a, mhc_class: types.SimpleNamespace(
            groove_half_1="A" * 90,
            groove_half_2="B" * 93,
        ),
    )

    resolved = resolver.resolve_class_i_groove_halves(
        allele="HLA-A*02:01",
        allele_sequences={"A*02:01": "S" * 181},
    )

    assert resolved == ("A" * 90, "B" * 93)


_HLA_B2705_SEQ = (
    "MRVTAPRTLLLLLWGAVALTETWAGSHSMRYFHTSVSRPGRGEPRFITVGYVDDTLFVRFDSDAASPREEPRAPWIEQEGPEYWDRETQICKA"
    "KAQTDREDLRTLLRYYNQSEAGSHTLQNMYGCDVGPDGRLLRGYHQDAYDGKDYIALNEDLSSWTAADTAAQITQRKWEAARVAEQLRAYLEG"
    "ECVEWLRRYLENGKETLQRADPPKTHVTHHPISDHEATLRCWALGFYPAEITLTWQRDGEDQTQDTELVETRPAGDRTFQKWAAVVVPSGEEQ"
    "RYTCHVQHEGLPKPLTLRWEPSSQSTVPIVGIVAGLAVLAVVVIGAVVAAVMCRRKSSGGKGGSYSQAACSDSAQGSDVSLTA"
)
_HLA_B2705_G1 = (
    "GSHSMRYFHTSVSRPGRGEPRFITVGYVDDTLFVRFDSDAASPREEPRAPWIEQEGPEYWDRETQICKAKAQTDREDLRTLLRYYNQSEA"
)
_HLA_B2705_G2 = (
    "GSHTLQNMYGCDVGPDGRLLRGYHQDAYDGKDYIALNEDLSSWTAADTAAQITQRKWEAARVAEQLRAYLEGECVEWLRRYLENGKETLQRAD"
)

_H2_KB_SEQ = (
    "MVPCTLLLLLAAALAPTQTRAGPHSLRYFVTAVSRPGLGEPRYMEVGYVDDTEFVRFDSDAENPRYEPRARWMEQEGPEYWERETQKAKGNEQ"
    "SFRVDLRTLLGYYNQSKGGSHTIQVISGCEVGSDGRLLRGYQQYAYDGCDYIALNEDLKTWTAADMAALITKHKWEQAGEAERLRAYLEGTCV"
    "EWLRRYLKNGNATLLRTDSPKAHVTHHSRPEDKVTLRCWALGFYPADITLTWQLNGEELIQDMELVETRPAGDGTFQKWASVVVPLGKEQYYT"
    "CHVYHQGLPEPLTLRWEPPPSTVSNMATVAVLVVLGAAIVTGAVVAFVMKMRRRNTGGKGGDYALAPGSQTSDLSLPDCKVMVHDPHSLA"
)
_H2_KB_G1 = (
    "GPHSLRYFVTAVSRPGLGEPRYMEVGYVDDTEFVRFDSDAENPRYEPRARWMEQEGPEYWERETQKAKGNEQSFRVDLRTLLGYYNQSKG"
)
_H2_KB_G2 = (
    "GSHTIQVISGCEVGSDGRLLRGYQQYAYDGCDYIALNEDLKTWTAADMAALITKHKWEQAGEAERLRAYLEGTCVEWLRRYLKNGNATLLRTD"
)

_HLA_DRA0101_SEQ = (
    "MAISGVPVLGFFIIAVLMSAQESWAIKEEHVIIQAEFYLNPDQSGEFMFDFDGDEIFHVDMAKKETVWRLEEFGRFASFEAQGALANIAVDKA"
    "NLEIMTKRSNYTPITNVPPEVTVLTNSPVELREPNVLICFIDKFTPPVVNVTWLRNGKPVTTGVSETVFLPREDHLFRKFHYLPFLPSTEDVY"
    "DCRVEHWGLDEPLLKHWEFDAPSPLPETTENVVCALGLTVGLVGIIIGTIFIIKGVRKSNAAERRGPL"
)
_HLA_DRA0101_G1 = (
    "IKEEHVIIQAEFYLNPDQSGEFMFDFDGDEIFHVDMAKKETVWRLEEFGRFASFEAQGALANIAVDKANLEIMTKRSNYTPIT"
)

_HLA_DRB1_0101_SEQ = (
    "MVCLKLPGGSCMTALTVTLMVLSSPLALAGDTRPRFLWQLKFECHFFNGTERVRLLERCIYNQEESVRFDSDVGEYRAVTELGRPDAEYWNSQ"
    "KDLLEQRRAAVDTYCRHNYGVGESFTVQRRVEPKVTVYPSKTQPLQHHNLLVCSVSGFYPGSIEVRWFRNGQEEKAGVVSTGLIQNGDWTFQT"
    "LVMLETVPRSGEVYTCQVEHPSVTSPLTVEWRARSESAQSKMLSGVGGFVLGLLFLGAGLFIYFRNQKGHSGLQPTGFLS"
)
_HLA_DRB1_0101_G2 = (
    "GDTRPRFLWQLKFECHFFNGTERVRLLERCIYNQEESVRFDSDVGEYRAVTELGRPDAEYWNSQKDLLEQRRAAVDTYCRHNYGVGESFTVQR"
)


def _mutant_fixture_lookup():
    entries = [
        ExactMHCInput(
            allele="HomoSapiens-B*27:05",
            sequence=_HLA_B2705_SEQ,
            groove1=_HLA_B2705_G1,
            groove2=_HLA_B2705_G2,
            mhc_class="I",
            chain="alpha",
            groove_status="ok",
            source="mhcseqs",
        ),
        ExactMHCInput(
            allele="MusMusculus-K*b",
            sequence=_H2_KB_SEQ,
            groove1=_H2_KB_G1,
            groove2=_H2_KB_G2,
            mhc_class="I",
            chain="alpha",
            groove_status="ok",
            source="mhcseqs",
        ),
        ExactMHCInput(
            allele="HomoSapiens-DRA*01:01",
            sequence=_HLA_DRA0101_SEQ,
            groove1=_HLA_DRA0101_G1,
            groove2="",
            mhc_class="II",
            chain="alpha",
            groove_status="ok",
            source="mhcseqs",
        ),
        ExactMHCInput(
            allele="HomoSapiens-DRB1*01:01",
            sequence=_HLA_DRB1_0101_SEQ,
            groove1="",
            groove2=_HLA_DRB1_0101_G2,
            mhc_class="II",
            chain="beta",
            groove_status="ok",
            source="mhcseqs",
        ),
    ]
    lookup = {}
    for rec in entries:
        lookup[rec.allele] = rec
        lookup[rec.allele.upper()] = rec
    return lookup


def test_lookup_exact_mhc_input_resolves_class_i_point_mutant(monkeypatch):
    monkeypatch.setattr(
        resolver,
        "_load_mhcseqs_input_lookup",
        lambda search_dir=None: _mutant_fixture_lookup(),
    )

    resolved = resolver.lookup_exact_mhc_input("HLA-B*27:05 C67S mutant")

    assert resolved is not None
    assert resolved.allele == "HLA-B*27:05 C67S mutant"
    assert resolved.source == "mhcseqs_mutant"
    assert resolved.mhc_class == "I"
    assert len(resolved.groove1) == len(_HLA_B2705_G1)
    assert resolved.groove1[66] == "S"
    assert _HLA_B2705_G1[66] == "C"
    assert sum(a != b for a, b in zip(_HLA_B2705_G1, resolved.groove1)) == 1
    assert resolved.groove2 == _HLA_B2705_G2

    halves = resolver.resolve_class_i_groove_halves(allele="HLA-B*27:05 C67S mutant")
    assert halves == (resolved.groove1, resolved.groove2)


def test_lookup_exact_mhc_input_resolves_multi_mutant_mouse_class_i(monkeypatch):
    monkeypatch.setattr(
        resolver,
        "_load_mhcseqs_input_lookup",
        lambda search_dir=None: _mutant_fixture_lookup(),
    )

    resolved = resolver.lookup_exact_mhc_input("H2-Kb Y22F M23I E24S D30N mutant")

    assert resolved is not None
    assert resolved.source == "mhcseqs_mutant"
    assert resolved.groove1[21] == "F"
    assert resolved.groove1[22] == "I"
    assert resolved.groove1[23] == "S"
    assert resolved.groove1[29] == "N"
    assert sum(a != b for a, b in zip(_H2_KB_G1, resolved.groove1)) == 4
    assert resolved.groove2 == _H2_KB_G2


def test_lookup_exact_mhc_input_resolves_class_ii_chain_and_pair_mutants(monkeypatch):
    monkeypatch.setattr(
        resolver,
        "_load_mhcseqs_input_lookup",
        lambda search_dir=None: _mutant_fixture_lookup(),
    )

    beta_mut = resolver.lookup_exact_mhc_input("HLA-DRB1*01:01 C30S mutant")
    assert beta_mut is not None
    assert beta_mut.source == "mhcseqs_mutant"
    assert beta_mut.groove1 == ""
    assert beta_mut.groove2[29] == "S"
    assert _HLA_DRB1_0101_G2[29] == "C"
    assert sum(a != b for a, b in zip(_HLA_DRB1_0101_G2, beta_mut.groove2)) == 1

    pair_beta_mut = resolver.lookup_exact_mhc_input("HLA-DRA*01:01/DRB1*01:01 C30S mutant")
    assert pair_beta_mut is not None
    assert pair_beta_mut.source == "mhcseqs_mutant"
    assert pair_beta_mut.mhc_class == "II"
    assert pair_beta_mut.chain == "pair"
    assert pair_beta_mut.groove1 == _HLA_DRA0101_G1
    assert pair_beta_mut.groove2 == beta_mut.groove2

    pair_alpha_mut = resolver.lookup_exact_mhc_input("HLA-DRA*01:01 F54C mutant/DRB1*01:01")
    assert pair_alpha_mut is not None
    assert pair_alpha_mut.source == "mhcseqs_mutant"
    assert pair_alpha_mut.groove1[53] == "C"
    assert _HLA_DRA0101_G1[53] == "F"
    assert sum(a != b for a, b in zip(_HLA_DRA0101_G1, pair_alpha_mut.groove1)) == 1
    assert pair_alpha_mut.groove2 == _HLA_DRB1_0101_G2

    pair_wt = resolver.lookup_exact_mhc_input("HLA-DRA*01:01/DRB1*01:01")
    assert pair_wt is not None
    assert pair_wt.source == "mhcseqs_pair"
    assert pair_wt.groove1 == _HLA_DRA0101_G1
    assert pair_wt.groove2 == _HLA_DRB1_0101_G2


def test_lookup_exact_mhc_input_rejects_mismatched_or_out_of_range_mutant(monkeypatch):
    monkeypatch.setattr(
        resolver,
        "_load_mhcseqs_input_lookup",
        lambda search_dir=None: _mutant_fixture_lookup(),
    )

    # Mature pos 67 in HLA-B*27:05 is C, not A — must fail closed rather than mis-mutate.
    assert resolver.lookup_exact_mhc_input("HLA-B*27:05 A67S mutant") is None
    assert resolver.lookup_exact_mhc_input("HLA-B*27:05 C999S mutant") is None
    assert resolver.lookup_exact_mhc_input("HLA-B*99:99 C67S mutant") is None
