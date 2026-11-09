"""Declare all source decisions through the shared recipe; no workbook parsing."""

from collections import Counter
import math
import re
import sys

from common import ROOT, digest, read, request, write


def derive(row):
    value, dose = row["target_value"], row["amount.nominal_amount_fmol"]
    if row["decision"] != "accepted" or value is None or dose is None:
        raise ValueError("Only admitted measured values can be transformed")
    if not all(math.isfinite(v) and v > 0 for v in (value, dose)):
        raise ValueError("Dose and signal must be finite and positive")
    if (
        row["target_unit"] != "source_arbitrary_summed_intensity"
        or row["amount.measurement_unit"] != "fmol"
    ):
        raise ValueError("Unsupported quantity units")
    return {
        **row,
        "sequence": row["standard.peptide_sequence"],
        "gene": row["standard.gene_symbol_raw"],
        "dose_fmol": dose,
        "x": math.log10(dose),
        "y": math.log10(value),
    }


def main():
    req = request()
    cfg = read(ROOT / "reproduce/launch.json")
    out = ROOT / "results" / req["output_subdir"]
    out.mkdir(exist_ok=False)
    sys.path.insert(0, cfg["hitlist_checkout"])
    import pyarrow.parquet as pq
    from hitlist.evidence_store import EvidenceStore
    from hitlist.nontryptic_calibration_schema import SCHEMAS

    store = EvidenceStore.open(cfg["package"])
    assert digest(store.directory / "manifest.json") == req["package_manifest_sha256"]
    queries, metadata = {}, {}
    for name in (
        "standards",
        "nominal_amounts",
        "source_slots",
        "reported_forms",
        "isotope_assertions",
        "protocol_assertions",
        "material_contexts",
        "material_links",
        "assets",
    ):
        queries[name] = {
            "evidence_type": "nontryptic_" + name,
            "select": SCHEMAS[name].names,
            "limit": None,
        }
        metadata[name] = store.query(**queries[name]).table.to_pylist()
    standards = metadata["standards"]
    assert len(standards) == len({r["peptide_sequence"] for r in standards}) == 18
    assert all(re.fullmatch(r"[A-Za-z0-9_.-]+", r["gene_symbol_raw"] or "") for r in standards)
    genes = sorted({r["gene_symbol_raw"] for r in standards})
    assert len(genes) >= 4
    folds = [
        {
            "held_gene": gene,
            "held_sequences": sorted(
                r["peptide_sequence"] for r in standards if r["gene_symbol_raw"] == gene
            ),
            "training_sequences": sorted(
                r["peptide_sequence"] for r in standards if r["gene_symbol_raw"] != gene
            ),
        }
        for gene in genes
    ]
    # This artifact is written from sequence/gene metadata before target extraction.
    write(out / "folds.json", folds)
    write(out / "metadata.json", metadata)
    write(out / "queries.json", queries)
    recipe = read(ROOT / "reproduce/recipe.json")
    manifest = store.materialize(recipe, cfg["recipe_destination"])
    decisions = [
        r
        for disposition in ("accepted", "excluded", "ambiguous")
        for r in pq.read_table(
            manifest.parent / "sources/nontryptic_measurements" / (disposition + ".parquet")
        ).to_pylist()
    ]
    decisions.sort(key=lambda r: r["measurement_id"])
    assert len(decisions) == len({r["measurement_id"] for r in decisions}) == 348
    accepted = []
    for row in decisions:
        assert row["slot.acquisition_id"] is None
        assert row["amount.nominal_status"] == "source_reported_not_independently_certified"
        if row["amount.isotopologue_label"] == "1H":
            assert row["decision"] == "accepted" and "DatasetS3A!" in row["source_locator"]
            accepted.append(derive(row))
        else:
            assert (
                row["amount.isotopologue_label"] == "2H" and "DatasetS3B!" in row["source_locator"]
            )
            assert (
                row["decision"] == "excluded" and "context_precision_unmet" in row["reason_codes"]
            )
    assert len(accepted) == 213
    assert len({r["sequence"] for r in accepted}) == 18
    assert len({r["nominal_amount_id"] for r in accepted}) == 71
    assert Counter(r["decision"] for r in decisions) == {"accepted": 213, "excluded": 135}
    for fold in folds:
        train = [r for r in accepted if r["gene"] != fold["held_gene"]]
        assert len({r["gene"] for r in train}) >= 3
        assert not set(fold["held_sequences"]) & {r["sequence"] for r in train}
    write(out / "decisions.json", decisions)
    write(out / "admitted.json", accepted)
    write(
        out / "summary.json",
        {
            "status": "admitted_development_response_only",
            "n_rows": len(accepted),
            "n_precursor_groups": len(genes),
            "n_dose_rows": 71,
            "decisions": dict(Counter(r["decision"] for r in decisions)),
            "recipe_manifest": str(manifest),
            "recipe_manifest_sha256": digest(manifest),
            "package_manifest_sha256": digest(store.directory / "manifest.json"),
            "independent_biological_test": False,
            "old_biological_heldouts_accessed": False,
            "model_fits": 0,
        },
    )
    print((out / "summary.json").read_text(), flush=True)


if __name__ == "__main__":
    main()
