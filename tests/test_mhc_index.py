"""Tests for MHC index build + resolve utilities."""

import csv
from pathlib import Path

from presto.data import mhc_index


def _write_fasta(path: Path, records: list[tuple[str, str]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for header, sequence in records:
            f.write(f">{header}\n")
            f.write(f"{sequence}\n")


LONG_SEQ_A = "A" * 80
LONG_SEQ_C = "C" * 80
LONG_SEQ_D = "D" * 80


def test_build_mhc_index_writes_outputs_and_deduplicates(tmp_path, monkeypatch):
    imgt_fasta = tmp_path / "imgt.fasta"
    ipd_fasta = tmp_path / "ipd.fasta"
    out_csv = tmp_path / "mhc_index.csv"
    out_fasta = tmp_path / "mhc_index.fasta"

    _write_fasta(
        imgt_fasta,
        [
            ("HLA-A*02:01 desc", LONG_SEQ_A),
        ],
    )
    _write_fasta(
        ipd_fasta,
        [
            ("HLA-A*02:01 alt", LONG_SEQ_D),
            ("Mamu-A*01:01 alt", LONG_SEQ_C),
        ],
    )

    monkeypatch.setattr(mhc_index, "_require_mhcgnomes", lambda: object())

    def fake_resolve(header: str):
        token = header.split()[0]
        if token == "HLA-A*02:01":
            return "HLA-A*02:01", "A", "I", "human", token
        if token == "Mamu-A*01:01":
            return "Mamu-A*01:01", "A", "I", "macaque", token
        raise AssertionError(f"Unexpected header: {header}")

    monkeypatch.setattr(mhc_index, "_resolve_header_allele", fake_resolve)

    stats = mhc_index.build_mhc_index(
        imgt_fasta=str(imgt_fasta),
        ipd_mhc_dir=str(ipd_fasta),
        out_csv=str(out_csv),
        out_fasta=str(out_fasta),
    )

    assert stats["total"] == 3
    assert stats["parsed"] == 2
    assert stats["duplicates"] == 1
    assert stats["replaced"] == 0
    assert stats["skipped"] == 0

    rows = list(csv.DictReader(open(out_csv, "r", encoding="utf-8")))
    assert len(rows) == 2
    by_norm = {row["normalized"]: row for row in rows}
    assert by_norm["HLA-A*02:01"]["source"] == "imgt"
    assert by_norm["HLA-A*02:01"]["sequence"] == LONG_SEQ_A
    assert by_norm["Mamu-A*01:01"]["source"] == "ipd_mhc"

    fasta_text = out_fasta.read_text(encoding="utf-8")
    assert ">HLA-A*02:01 source=imgt" in fasta_text
    assert LONG_SEQ_A in fasta_text


def test_build_mhc_index_prefers_protein_fasta_and_skips_nucleotide_entries(tmp_path, monkeypatch):
    imgt_fasta = tmp_path / "imgt.fasta"
    ipd_dir = tmp_path / "ipd_mhc"
    ipd_dir.mkdir()
    ipd_nuc = ipd_dir / "ipd_mhc_nuc.fasta"
    ipd_prot = ipd_dir / "ipd_mhc_prot.fasta"
    out_csv = tmp_path / "mhc_index.csv"

    _write_fasta(imgt_fasta, [])
    _write_fasta(
        ipd_nuc,
        [("Mamu-A*01:01 nuc", "ACG" * 120)],
    )
    _write_fasta(
        ipd_prot,
        [("Mamu-A*01:01 prot", "M" + ("A" * 180))],
    )

    monkeypatch.setattr(mhc_index, "_require_mhcgnomes", lambda: object())

    def fake_resolve(header: str):
        token = header.split()[0]
        if token == "Mamu-A*01:01":
            return "Mamu-A*01:01", "A", "I", "macaque", token
        raise AssertionError(f"Unexpected header: {header}")

    monkeypatch.setattr(mhc_index, "_resolve_header_allele", fake_resolve)

    stats = mhc_index.build_mhc_index(
        imgt_fasta=str(imgt_fasta),
        ipd_mhc_dir=str(ipd_dir),
        out_csv=str(out_csv),
    )

    assert stats["total"] == 2
    assert stats["parsed"] == 1
    assert stats["skipped_nucleotide"] == 1

    rows = list(csv.DictReader(open(out_csv, "r", encoding="utf-8")))
    assert len(rows) == 1
    assert rows[0]["normalized"] == "Mamu-A*01:01"
    assert rows[0]["sequence"] == "M" + ("A" * 180)


def test_build_mhc_index_skips_trivially_short_fragments(tmp_path, monkeypatch):
    imgt_fasta = tmp_path / "imgt.fasta"
    out_csv = tmp_path / "mhc_index.csv"

    _write_fasta(
        imgt_fasta,
        [
            ("HLA-DRB1*03:03 short", "A" * 60),
            ("HLA-DRB1*03:03 okay", "A" * 89),
        ],
    )

    monkeypatch.setattr(mhc_index, "_require_mhcgnomes", lambda: object())
    monkeypatch.setattr(
        mhc_index,
        "_resolve_header_allele",
        lambda header: ("HLA-DRB1*03:03", "DRB1", "II", "human", "HLA-DRB1*03:03"),
    )

    stats = mhc_index.build_mhc_index(
        imgt_fasta=str(imgt_fasta),
        ipd_mhc_dir=None,
        out_csv=str(out_csv),
    )

    assert stats["total"] == 2
    assert stats["parsed"] == 1
    assert stats["skipped_short"] == 1

    rows = list(csv.DictReader(open(out_csv, "r", encoding="utf-8")))
    assert len(rows) == 1
    assert rows[0]["normalized"] == "HLA-DRB1*03:03"
    assert rows[0]["seq_len"] == "89"


def test_resolve_alleles_uses_aliases_from_index(tmp_path, monkeypatch):
    index_csv = tmp_path / "mhc_index.csv"
    with open(index_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=mhc_index.INDEX_FIELDS)
        writer.writeheader()
        writer.writerow(
            {
                "allele_raw": "HLA-A*02:01:01",
                "normalized": "HLA-A*02:01:01",
                "gene": "A",
                "mhc_class": "I",
                "species": "human",
                "source": "imgt",
                "seq_len": "80",
                "sequence": LONG_SEQ_A,
            }
        )

    monkeypatch.setattr(mhc_index, "_require_mhcgnomes", lambda: object())

    def fake_normalize(allele: str):
        if allele == "A0201":
            return "HLA-A*02:01", None, None, None
        if allele == "HLA-B*07:02":
            return "HLA-B*07:02", None, None, None
        raise ValueError("bad allele")

    monkeypatch.setattr(mhc_index, "_normalize_with_mhcgnomes", fake_normalize)

    resolved = mhc_index.resolve_alleles(
        index_csv=str(index_csv),
        alleles=["A0201", "HLA-B*07:02"],
        include_sequence=False,
    )

    assert len(resolved) == 2
    assert resolved[0]["input"] == "A0201"
    assert resolved[0]["normalized"] == "HLA-A*02:01"
    assert resolved[0]["resolved"] == "HLA-A*02:01:01"
    assert resolved[0]["found"] is True
    assert "sequence" not in resolved[0]

    assert resolved[1]["input"] == "HLA-B*07:02"
    assert resolved[1]["found"] is False


def test_resolve_alleles_supports_h2_alias_forms(tmp_path, monkeypatch):
    index_csv = tmp_path / "mhc_index.csv"
    with open(index_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=mhc_index.INDEX_FIELDS)
        writer.writeheader()
        writer.writerow(
            {
                "allele_raw": "H-2-Kd",
                "normalized": "H2-K*d",
                "gene": "K",
                "mhc_class": "I",
                "species": "mouse",
                "source": "ipd_mhc",
                "seq_len": "80",
                "sequence": LONG_SEQ_A,
            }
        )

    monkeypatch.setattr(mhc_index, "_require_mhcgnomes", lambda: object())

    def fake_normalize(allele: str):
        value = allele.strip()
        if value in {"H-2Kd", "H-2-Kd", "H2-Kd", "H2Kd"}:
            return "H2-K*d", None, None, None
        raise ValueError("bad allele")

    monkeypatch.setattr(mhc_index, "_normalize_with_mhcgnomes", fake_normalize)

    resolved = mhc_index.resolve_alleles(
        index_csv=str(index_csv),
        alleles=["H-2Kd", "H-2-Kd", "H2-Kd", "H2Kd"],
        include_sequence=False,
    )
    assert len(resolved) == 4
    assert all(row["found"] for row in resolved)
    assert all(row["resolved"] == "H2-K*d" for row in resolved)


def test_resolve_alleles_falls_back_when_mhcgnomes_query_parse_fails(tmp_path, monkeypatch):
    index_csv = tmp_path / "mhc_index.csv"
    with open(index_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=mhc_index.INDEX_FIELDS)
        writer.writeheader()
        writer.writerow(
            {
                "allele_raw": "Mamu-A*01:01",
                "normalized": "Mamu-A*01:01",
                "gene": "A",
                "mhc_class": "I",
                "species": "macaque",
                "source": "ipd_mhc",
                "seq_len": "80",
                "sequence": LONG_SEQ_C,
            }
        )

    monkeypatch.setattr(mhc_index, "_require_mhcgnomes", lambda: object())
    monkeypatch.setattr(
        mhc_index,
        "_normalize_with_mhcgnomes",
        lambda allele: (_ for _ in ()).throw(ValueError("bad allele")),
    )

    resolved = mhc_index.resolve_alleles(
        index_csv=str(index_csv),
        alleles=["Mamu-A*01:01:01"],
        include_sequence=False,
    )
    assert len(resolved) == 1
    assert resolved[0]["found"] is True
    assert resolved[0]["resolved"] == "Mamu-A*01:01"
    assert resolved[0]["normalized"] == "Mamu-A*01:01:01"


def test_resolve_header_allele_has_non_mhcgnomes_fallback(monkeypatch):
    monkeypatch.setattr(
        mhc_index,
        "_normalize_with_mhcgnomes",
        lambda token: (_ for _ in ()).throw(ValueError("no parse")),
    )
    normalized, gene, mhc_class, species, token = mhc_index._resolve_header_allele(
        "IPD-MHC:TEST H-2Kd 365 bp"
    )
    assert token == "H-2Kd"
    assert normalized == "H2-K*d"
    assert gene == "K"
    assert mhc_class == "I"
    assert species == "murine"


def test_validate_mhc_index_rejects_nucleotide_like_sequences(tmp_path, monkeypatch):
    index_csv = tmp_path / "mhc_index.csv"
    with open(index_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=mhc_index.INDEX_FIELDS)
        writer.writeheader()
        writer.writerow(
            {
                "allele_raw": "Mamu-A*01:01",
                "normalized": "Mamu-A*01:01",
                "gene": "A",
                "mhc_class": "I",
                "species": "macaque",
                "source": "ipd_mhc",
                "seq_len": "180",
                "sequence": "ACG" * 60,
            }
        )

    monkeypatch.setattr(mhc_index, "_require_mhcgnomes", lambda: object())
    monkeypatch.setattr(
        mhc_index,
        "_normalize_with_mhcgnomes",
        lambda allele: (allele, None, None, None),
    )

    report = mhc_index.validate_mhc_index(str(index_csv))
    assert report["valid"] is False
    assert any(error["code"] == "nucleotide_like_sequence" for error in report["errors"])


def test_validate_mhc_index_rejects_trivially_short_sequences(tmp_path, monkeypatch):
    index_csv = tmp_path / "mhc_index.csv"
    with open(index_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=mhc_index.INDEX_FIELDS)
        writer.writeheader()
        writer.writerow(
            {
                "allele_raw": "HLA-DRB1*03:03",
                "normalized": "HLA-DRB1*03:03",
                "gene": "DRB1",
                "mhc_class": "II",
                "species": "human",
                "source": "imgt",
                "seq_len": "60",
                "sequence": "A" * 60,
            }
        )

    monkeypatch.setattr(mhc_index, "_require_mhcgnomes", lambda: object())
    monkeypatch.setattr(
        mhc_index,
        "_normalize_with_mhcgnomes",
        lambda allele: (allele, None, None, None),
    )

    report = mhc_index.validate_mhc_index(str(index_csv))
    assert report["valid"] is False
    assert any(error["code"] == "sequence_too_short" for error in report["errors"])


def test_classify_unresolved_allele_uses_mhcgnomes_parse_kinds(monkeypatch):
    class _Species:
        def __init__(self, name: str):
            self.name = name

    class _Gene:
        def __init__(self, name: str):
            self.name = name

    class Pair:
        species = _Species("Mus musculus")
        gene = None
        mhc_class = "IIa"

        def to_string(self):
            return "H2-AA*b/AB*b"

    class Haplotype:
        species = _Species("Mus musculus")
        gene = None
        mhc_class = None

        def to_string(self):
            return "H2-b class I"

    class Allele:
        species = _Species("Mus musculus")
        gene = _Gene("K")
        mhc_class = "Ia"

        def to_string(self):
            return "H2-K*b"

    class Serotype:
        species = _Species("Homo sapiens")
        gene = _Gene("A")
        mhc_class = None

        def to_string(self):
            return "HLA-A68"

    class Class2Locus:
        species = _Species("Homo sapiens")
        gene = None
        mhc_class = "II"

        def to_string(self):
            return "HLA-DR"

    class Gene:
        species = _Species("Homo sapiens")
        gene = _Gene("BTN3A1")
        mhc_class = "Ic"

        def to_string(self):
            return "HLA-BTN3A1"

    mapping = {
        "H2-AA*b/AB*b": Pair(),
        "H2-b class I": Haplotype(),
        "H2-K*b": Allele(),
        "HLA-A68": Serotype(),
        "HLA-DR": Class2Locus(),
        "HLA-BTN3A1": Gene(),
    }

    class _FakeMHCGnomes:
        @staticmethod
        def parse(token: str):
            return mapping[token]

    monkeypatch.setattr(mhc_index, "_require_mhcgnomes", lambda: _FakeMHCGnomes())

    checks = [
        ("H2-AA*b/AB*b", "murine_pair_shorthand", "Pair"),
        ("H2-b class I", "murine_haplotype", "Haplotype"),
        ("H2-K*b", "murine_allele_missing_sequence", "Allele"),
        ("HLA-A68", "human_serotype", "Serotype"),
        ("HLA-DR", "human_locus", "Class2Locus"),
        ("HLA-BTN3A1", "human_nonclassical_gene", "Gene"),
    ]
    for token, expected_category, expected_type in checks:
        info = mhc_index.classify_unresolved_allele(token)
        assert info["category"] == expected_category
        assert info["parsed_type"] == expected_type
        assert info["parse_error"] == ""


def test_classify_unresolved_allele_falls_back_when_parser_unavailable(monkeypatch):
    monkeypatch.setattr(
        mhc_index,
        "_require_mhcgnomes",
        lambda: (_ for _ in ()).throw(RuntimeError("mhcgnomes unavailable")),
    )
    info = mhc_index.classify_unresolved_allele("HLA-DR")
    assert info["category"] == "human_locus"
    assert info["parsed_type"] == ""
    assert "mhcgnomes unavailable" in info["parse_error"]


def test_summarize_mhc_index_counts(tmp_path):
    index_csv = tmp_path / "mhc_index.csv"
    with open(index_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=mhc_index.INDEX_FIELDS)
        writer.writeheader()
        writer.writerow(
            {
                "allele_raw": "HLA-A*02:01",
                "normalized": "HLA-A*02:01",
                "gene": "A",
                "mhc_class": "I",
                "species": "human",
                "source": "imgt",
                "seq_len": "80",
                "sequence": LONG_SEQ_A,
            }
        )
        writer.writerow(
            {
                "allele_raw": "Mamu-A*01:01",
                "normalized": "Mamu-A*01:01",
                "gene": "A",
                "mhc_class": "I",
                "species": "macaque",
                "source": "ipd_mhc",
                "seq_len": "80",
                "sequence": LONG_SEQ_C,
            }
        )
        writer.writerow(
            {
                "allele_raw": "HLA-DRB1*01:01",
                "normalized": "HLA-DRB1*01:01",
                "gene": "DRB1",
                "mhc_class": "II",
                "species": "human",
                "source": "imgt",
                "seq_len": "4",
                "sequence": "DDDD",
            }
        )

    report = mhc_index.summarize_mhc_index(str(index_csv))
    assert report["total_records"] == 3
    assert report["by_source"] == {"imgt": 2, "ipd_mhc": 1}
    assert report["by_species"] == {"human": 2, "macaque": 1}
    assert report["by_mhc_class"] == {"I": 2, "II": 1}


def test_validate_mhc_index_detects_duplicates_and_seq_len_mismatch(tmp_path, monkeypatch):
    index_csv = tmp_path / "mhc_index.csv"
    with open(index_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=mhc_index.INDEX_FIELDS)
        writer.writeheader()
        writer.writerow(
            {
                "allele_raw": "HLA-A*02:01",
                "normalized": "HLA-A*02:01",
                "gene": "A",
                "mhc_class": "I",
                "species": "human",
                "source": "imgt",
                "seq_len": "80",
                "sequence": LONG_SEQ_A,
            }
        )
        writer.writerow(
            {
                "allele_raw": "HLA-A*02:01 dup",
                "normalized": "HLA-A*02:01",
                "gene": "A",
                "mhc_class": "I",
                "species": "human",
                "source": "imgt",
                "seq_len": "9",
                "sequence": LONG_SEQ_A,
            }
        )

    monkeypatch.setattr(mhc_index, "_require_mhcgnomes", lambda: object())
    monkeypatch.setattr(
        mhc_index,
        "_normalize_with_mhcgnomes",
        lambda allele: (allele, None, None, None),
    )

    report = mhc_index.validate_mhc_index(str(index_csv))
    assert report["total_rows"] == 2
    assert report["error_count"] == 1
    assert report["warning_count"] == 1
    assert report["valid"] is False


def test_validate_mhc_index_detects_non_canonical_alleles(tmp_path, monkeypatch):
    index_csv = tmp_path / "mhc_index.csv"
    with open(index_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=mhc_index.INDEX_FIELDS)
        writer.writeheader()
        writer.writerow(
            {
                "allele_raw": "HLA-A2",
                "normalized": "HLA-A2",
                "gene": "A",
                "mhc_class": "I",
                "species": "human",
                "source": "imgt",
                "seq_len": "80",
                "sequence": LONG_SEQ_A,
            }
        )

    monkeypatch.setattr(mhc_index, "_require_mhcgnomes", lambda: object())
    monkeypatch.setattr(
        mhc_index,
        "_normalize_with_mhcgnomes",
        lambda allele: ("HLA-A*02", None, None, None),
    )

    report = mhc_index.validate_mhc_index(str(index_csv))
    assert report["error_count"] == 0
    assert report["warning_count"] == 1
    assert report["valid"] is True
