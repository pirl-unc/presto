"""Independent pandas replay of source structure and quantity/support audit."""

import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd


FAMILY = Path(__file__).resolve().parents[2 if Path(__file__).parent.name == "source" else 1]
REPO = FAMILY.parents[1]


def main():
    start = time.perf_counter()
    result = FAMILY / "results"
    paths = json.loads((result / "input_paths.json").read_text())
    manifest = json.loads((FAMILY / "reproduce/input_assets.json").read_text())
    members = json.loads((FAMILY / "reproduce/archive_members.json").read_text())
    launch = json.loads((FAMILY / "reproduce/launch.json").read_text())
    for filename, key in [("input_assets.json", "input_manifest_sha256"),
                          ("archive_members.json", "member_manifest_sha256"),
                          ("source/audit.py", "source_sha256")]:
        assert hashlib.sha256((FAMILY / "reproduce" / filename).read_bytes()).hexdigest() == launch[key]
    for group, expected in [("original", manifest["assets"]), ("archive_members", members)]:
        for name, receipt in expected.items():
            p = REPO / paths[group][name]
            assert hashlib.sha256(p.read_bytes()).hexdigest() == receipt["sha256"]
            assert p.stat().st_size == receipt.get("size", receipt.get("size_bytes"))
    hashes = json.loads((result / "output_hashes.json").read_text())
    for name, expected in hashes.items():
        assert hashlib.sha256((result / name).read_bytes()).hexdigest() == expected

    def read(name, **kwargs):
        return pd.read_csv(REPO / paths["archive_members"]["txt/" + name + ".txt"],
                           sep="\t", dtype=str, keep_default_na=False, **kwargs)

    inventory = json.loads((result / "table_inventory.json").read_text())
    for item in inventory:
        table = read(item["member"].removeprefix("txt/").removesuffix(".txt"))
        assert list(table.columns) == item["columns"] and len(table) == item["n_rows"]
        del table
    p = read("peptides").set_index("id", verify_integrity=True)
    g = read("proteinGroups")
    good = g[g["id"].str.fullmatch(r"\d+")].set_index("id", verify_integrity=True)
    quarantine = json.loads((result / "quarantined_group_rows.json").read_text())
    invalid = g[~g["id"].str.fullmatch(r"\d+")]
    assert invalid.index.to_list() == [row["source_row"] - 2 for row in quarantine]
    for (_, source), row in zip(invalid.iterrows(), quarantine):
        assert hashlib.sha256(json.dumps(source.to_dict(), sort_keys=True).encode()).hexdigest() == row["parsed_row_sha256"]
    run_source = read("summary")
    run_source = run_source[run_source["Experiment"].ne("")].set_index("Raw file", verify_integrity=True)
    runs = json.loads((result / "runs.json").read_text())["acquisitions"]
    assert len(run_source) == len(runs)
    for run in runs:
        assert run_source.loc[run["raw_file"], "Experiment"] == run["experiment"]
        assert run_source.loc[run["raw_file"], "Enzyme mode"] == run["enzyme_mode"]

    sums, counts, row_count, types = None, None, 0, None
    columns = ["Peptide ID", "Sequence", "Experiment", "Raw file", "Intensity", "MS/MS Count", "Type"]
    for chunk in read("evidence", usecols=columns, chunksize=20000):
        assert np.array_equal(chunk["Sequence"].to_numpy(), p.loc[chunk["Peptide ID"], "Sequence"].to_numpy())
        assert np.array_equal(chunk["Experiment"].to_numpy(), run_source.loc[chunk["Raw file"], "Experiment"].to_numpy())
        chunk["value"] = pd.to_numeric(chunk["Intensity"].replace("", "0"))
        chunk["direct"] = pd.to_numeric(chunk["MS/MS Count"].replace("", "0")).gt(0).astype(int)
        block_sums = chunk.groupby(["Peptide ID", "Experiment"])["value"].sum()
        block_counts = chunk.groupby(["Peptide ID", "Experiment"])["direct"].sum()
        block_types = chunk["Type"].value_counts()
        sums = block_sums if sums is None else sums.add(block_sums, fill_value=0)
        counts = block_counts if counts is None else counts.add(block_counts, fill_value=0)
        types = block_types if types is None else types.add(block_types, fill_value=0)
        row_count += len(chunk)
    comparison = pd.read_csv(result / "quantity_comparisons.csv.gz", dtype=str, keep_default_na=False)
    index = pd.MultiIndex.from_frame(comparison[["peptide_id", "experiment"]])
    assert not index.duplicated().any()
    expected_sum = sums.reindex(index, fill_value=0).to_numpy()
    expected_direct = counts.reindex(index, fill_value=0).to_numpy()
    np.testing.assert_allclose(comparison["linked_evidence_intensity_sum"].astype(float), expected_sum, rtol=1e-12, atol=1e-5)
    np.testing.assert_array_equal(comparison["direct_msms_evidence_rows"].astype(int), expected_direct)
    quantities = {column: column[len("Intensity "):].replace("-MAPPbeadFilt", "")
                  for column in p.columns if column.startswith("Intensity ")}
    assert len(comparison) == len(quantities) * len(p)
    for column, experiment in quantities.items():
        block = comparison[comparison["experiment"].eq(experiment)]
        assert block["quantity_column"].eq(column).all()
        np.testing.assert_array_equal(block["source_intensity_text"], p.loc[block["peptide_id"], column])
    reported = comparison["source_intensity_text"].astype(float).to_numpy()
    state = np.full(len(comparison), "both_positive_disagree", dtype=object)
    state[(reported > 0) & (expected_sum > 0) & np.isclose(reported, expected_sum, rtol=1e-4, atol=1)] = "both_positive_agree_within_rounding_tolerance"
    state[(reported == 0) & (expected_sum == 0)] = "both_zero"
    state[(reported == 0) & (expected_sum > 0)] = "reported_zero_evidence_positive"
    state[(reported > 0) & (expected_sum == 0)] = "reported_positive_evidence_zero"
    np.testing.assert_array_equal(comparison["state"], state)
    ledger = pd.read_csv(result / "peptide_support_ledger.csv.gz", dtype=str, keep_default_na=False).set_index("peptide_id", verify_integrity=True)
    assert set(ledger.index) == set(p.index)
    ledger = ledger.loc[p.index]
    np.testing.assert_array_equal(ledger["sequence"], p["Sequence"])
    np.testing.assert_array_equal(ledger["protein_ids"], p["Proteins"])
    np.testing.assert_array_equal(ledger["protein_group_ids"], p["Protein group IDs"])
    expected = {
        "canonical_sequence": p["Sequence"].str.fullmatch("[ACDEFGHIKLMNPQRSTVWY]+"),
        "reverse": p["Reverse"].eq("+"), "contaminant": p["Potential contaminant"].eq("+"),
        "single_group": p["Protein group IDs"].str.fullmatch(r"\d+"),
        "single_protein": p["Proteins"].ne("") & ~p["Proteins"].str.contains(";"),
    }
    for key, expected_value in expected.items():
        np.testing.assert_array_equal(ledger[key].eq("True"), expected_value)
    for key, token in [("ut", "_UT_"), ("cytokine", "_TNF+IFN_")]:
        cols = [c for c, e in quantities.items() if token in e]
        assert len(cols) == 3
        expected[key + "_positive"] = p[cols].astype(float).gt(0).sum(axis=1)
        direct_matrix = counts.unstack(fill_value=0).reindex(p.index, fill_value=0)
        expected[key + "_direct_msms_runs"] = direct_matrix[[quantities[c] for c in cols]].gt(0).sum(axis=1)
        for name in [key + "_positive", key + "_direct_msms_runs"]:
            np.testing.assert_array_equal(ledger[name].astype(int), expected[name])
    support = expected["canonical_sequence"] & ~expected["reverse"] & ~expected["contaminant"] & expected["single_group"]
    support &= expected["ut_positive"].ge(2) & expected["cytokine_positive"].ge(2)
    support &= expected["ut_direct_msms_runs"].ge(1) & expected["cytokine_direct_msms_runs"].ge(1)
    np.testing.assert_array_equal(ledger["structural_contrast_support"].eq("True"), support)
    assert ledger["admission"].eq("deferred").all()
    summary = json.loads((result / "summary.json").read_text())
    assert summary["n_peptide_rows"] == len(p) and summary["n_indexable_protein_groups"] == len(good)
    assert summary["n_evidence_rows"] == row_count and summary["n_quantity_comparisons"] == len(comparison)
    assert summary["quantity_states"] == pd.Series(state).value_counts().to_dict()
    assert summary["evidence_types"] == types.astype(int).to_dict()
    assert summary["peptide_counts"]["structural_contrast_support"] == int(support.sum())
    sizes = p.loc[support, "Protein group IDs"].value_counts()
    assert summary["structural_contrast_groups"] == len(sizes)
    assert summary["structural_contrast_multipeptide_groups"] == int(sizes.gt(1).sum())
    assert summary["structural_contrast_multipeptide_rows"] == int(sizes[sizes.gt(1)].sum())
    receipt = {"passed": True, "n_original_asset_hashes": len(manifest["assets"]), "n_member_hashes": len(members),
               "n_output_hashes": len(hashes), "n_source_evidence_rows": row_count,
               "n_quantity_comparisons": len(comparison), "n_peptide_ledger_rows": len(ledger),
               "n_quarantined_group_rows": len(invalid), "seconds": time.perf_counter() - start,
               "pandas": pd.__version__, "numpy": np.__version__,
               "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "scope": "Independent source/quantity/support verification; no cohort admitted or model evaluated"}
    output = FAMILY / "verification"
    output.mkdir(exist_ok=True)
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
