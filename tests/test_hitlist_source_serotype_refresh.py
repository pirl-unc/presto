"""Loader-level contracts required by Presto's Hitlist 2.1.12 floor."""

import pandas as pd
import pytest


@pytest.mark.parametrize("loader_name", ["load_observations", "load_binding"])
def test_hitlist_source_refreshes_serotypes_without_changing_evidence_roles(
    tmp_path, monkeypatch, loader_name
):
    import hitlist
    import hitlist.observations as observations
    from presto.data.hitlist_source import require_supported_hitlist

    require_supported_hitlist(hitlist.__version__)
    path = tmp_path / "existing-index.parquet"
    restrictions = [
        "HLA-A*02:01",
        "HLA-B*15:112",
        "HLA-A2",
        "HLA-A*02:01;HLA-B*15:112",
    ]
    pd.DataFrame(
        {
            "peptide": ["ALLELE_A", "ALLELE_B", "REPORTED", "DONOR"],
            "mhc_restriction": restrictions,
            "serotype": ["HLA-A2", "HLA-B75", "HLA-A2", "HLA-B75"],
            "serotypes": ["HLA-A2", "HLA-B75;HLA-B15", "HLA-A2", "HLA-A2;HLA-B75"],
            "serotype_source": ["derived", "derived", "reported", "donor_set"],
            "assay_method": [
                "binding assay" if loader_name == "load_binding" else "mass spectrometry"
            ]
            * 4,
            "qualitative_measurement": ["Positive"] * 4,
        }
    ).to_parquet(path, index=False)
    original_bytes = path.read_bytes()
    monkeypatch.setattr(observations, "observations_path", lambda: path)
    monkeypatch.setattr(observations, "binding_path", lambda: path)
    loader = getattr(observations, loader_name)
    frame = loader()
    assert frame.mhc_restriction.tolist() == restrictions
    assert frame.serotype_source.tolist() == ["derived", "derived", "reported", "donor_set"]
    assert "HLA-A2.1" in frame.iloc[0].serotypes.split(";")
    assert "HLA-B75" not in frame.iloc[1].serotypes.split(";")
    assert loader(serotype="HLA-A2.1", columns=["peptide"]).peptide.tolist() == [
        "ALLELE_A",
        "DONOR",
    ]
    assert loader(serotype="HLA-B75", columns=["peptide"]).empty
    for empty in ([], "", [""], " , "):
        result = loader(serotype=empty, columns=["peptide", "serotype_source"])
        assert result.empty
        assert result.columns.tolist() == ["peptide", "serotype_source"]
    assert path.read_bytes() == original_bytes, "Refreshing annotations must reuse the built index"
