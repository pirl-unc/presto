"""Tests for mouse MHC overlay builder."""

from pathlib import Path

from presto.data import mouse_mhc_overlay


def test_parse_imgt_mouse_mhc_genes_extracts_gene_like_symbols():
    html = """
    <html><body>
      <table>
        <tr><td>H2-A</td><td>MH2-AA</td></tr>
        <tr><td>H2-AB</td><td>H2-Q3</td></tr>
        <tr><td>H2-K</td><td>MH1-K1</td></tr>
      </table>
    </body></html>
    """
    genes = mouse_mhc_overlay.parse_imgt_mouse_mhc_genes(html)
    assert "H2-AA" in genes
    assert "H2-AB" in genes
    assert "H2-Q3" in genes
    assert "H2-K1" in genes
    assert "H2-A" not in genes
    assert "H2-K" not in genes


def test_derive_alleles_from_uniprot_row_uses_haplotype_suffix():
    alleles = mouse_mhc_overlay._derive_alleles_from_uniprot_row(
        imgt_gene_symbol="H2-K1",
        uniprot_gene_query="H2-K1",
        protein_name="H-2 class I histocompatibility antigen, K-B alpha chain",
        uniprot_gene_names=["H2-K1", "H2-K"],
    )
    assert ("H2-K*b", "protein_name_haplotype") in alleles


def test_build_mouse_mhc_overlay_writes_catalog_and_fasta(tmp_path, monkeypatch):
    out_csv = tmp_path / "mouse_overlay.csv"
    out_fasta = tmp_path / "mouse_overlay.fasta"

    monkeypatch.setattr(
        mouse_mhc_overlay,
        "fetch_imgt_mouse_mhc_genes",
        lambda imgt_url, timeout: ["H2-K1"],
    )

    def fake_search(query: str, timeout: int = 30, size: int = 100):
        return [
            {
                "primaryAccession": "P01901",
                "uniProtkbId": "HA1K_MOUSE",
                "entryType": "UniProtKB reviewed (Swiss-Prot)",
                "proteinDescription": {
                    "recommendedName": {
                        "fullName": {"value": "H-2 class I histocompatibility antigen, K-B alpha chain"}
                    }
                },
                "genes": [{"geneName": {"value": "H2-K1"}}],
                "sequence": {"value": "A" * 298},
            },
            {
                "primaryAccession": "P01901X",
                "uniProtkbId": "HA1K_MOUSE_X",
                "entryType": "UniProtKB reviewed (Swiss-Prot)",
                "proteinDescription": {
                    "recommendedName": {
                        "fullName": {"value": "H-2 class I histocompatibility antigen, K-B alpha chain"}
                    }
                },
                "genes": [{"geneName": {"value": "H2-K1"}}],
                "sequence": {"value": "A" * 320},
            },
        ]

    monkeypatch.setattr(mouse_mhc_overlay, "_uniprot_search", fake_search)

    stats = mouse_mhc_overlay.build_mouse_mhc_overlay(
        out_csv=str(out_csv),
        out_fasta=str(out_fasta),
        reviewed_only=True,
    )

    assert stats["imgt_genes"] == 1
    assert stats["catalog_rows"] >= 2
    assert stats["selected_alleles"] == 1
    assert out_csv.exists()
    assert out_fasta.exists()

    csv_text = out_csv.read_text(encoding="utf-8")
    assert "imgt_source_url" in csv_text
    assert "uniprot_record_url" in csv_text
    assert "allele_derivation_rule" in csv_text
    assert "H2-K*b" in csv_text
    assert "P01901X" in csv_text

    fasta_text = out_fasta.read_text(encoding="utf-8")
    assert ">H2-K*b source=uniprot_mouse_overlay accession=P01901X gene=H2-K1" in fasta_text

