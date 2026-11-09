"""Cross-check membership receipts with pandas and already-verified array cells."""

import gzip
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

FAMILY = Path(__file__).resolve().parents[1]


def main():
    started = time.monotonic()
    paths = json.loads((FAMILY / "results/input_paths.json").read_text())
    source = pd.read_csv(paths["archive_members"]["txt/peptides.txt"], sep="\t", dtype=str, keep_default_na=False)
    source = source.set_index("id", drop=False)
    with np.load(FAMILY / "results/filter_comparisons.npz", allow_pickle=False) as arrays:
        ids, experiments, measured = arrays["peptide_ids"], arrays["experiments"], arrays["measured"]
    source = source.loc[ids]
    receipt = json.loads((FAMILY / "results/membership-summary.json").read_text())
    with gzip.open(FAMILY / "results/original_workbook_cells.json.gz", "rt") as stream:
        publisher = json.load(stream)
    paper_rows = publisher["suppdata2.xlsx"]["Sheet1"][4:]
    ledger = pd.read_csv(FAMILY / "results/publisher_membership.csv.gz", dtype=str, keep_default_na=False)
    for condition, column in (("UT", 2), ("TNF+IFN", 3)):
        indices = [i for i, e in enumerate(experiments) if f"_{condition}_" in e]
        eligible = source[(source["Reverse"] != "+") & (source["Potential contaminant"] != "+")
            & ((measured[:, indices] > 0).sum(axis=1) >= 2)]
        published = {r[0] for r in paper_rows if r[column] is not None}
        assert set(eligible["Sequence"]) == published
        r = receipt["conditions"][condition]
        assert len(eligible) == r["source_eligible_without_unique_group_restriction"] == r["publisher_present"]
        assert eligible["Leading razor protein"].nunique() == r["distinct_nonempty_leading_razor_labels"]
        assert eligible["Protein group IDs"].str.contains(";", regex=False).sum() == r["multiple_group_id_rows"]
        assert eligible["Proteins"].str.contains(";", regex=False).sum() == r["multiple_protein_id_rows"]
        subset = ledger[ledger["condition"] == condition].set_index("source_peptide_id")
        assert subset.index.is_unique and set(subset.index) == set(eligible.index)
        for target, original in (("sequence", "Sequence"), ("source_leading_razor_protein", "Leading razor protein"),
            ("source_protein_ids", "Proteins"), ("source_group_ids", "Protein group IDs"),
            ("source_unique_groups", "Unique (Groups)"), ("source_unique_proteins", "Unique (Proteins)")):
            assert (subset.loc[eligible.index, target].to_numpy() == eligible[original].to_numpy()).all()
    original3 = pd.read_excel(paths["publisher"]["suppdata3.xlsx"], header=None).to_numpy()
    assert original3.shape == (133, 5)
    assert np.isfinite(original3[2:, 1:].astype(float)).all()
    assert (original3[2:, 1:].astype(float) > 0).sum() == receipt["data3_positive_numeric_cells"] == 524
    assert len(set(original3[2:, 0])) == receipt["data3_unique_gene_labels"] == 131
    result = {"status": "passed", "membership_rows_verified": len(ledger),
        "mapping_fields_verified_per_row": 6, "data3_quantity_cells_verified": 524,
        "seconds": time.monotonic()-started}
    (FAMILY / "results/membership-verification.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
