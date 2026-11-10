"""Verify source qualification from native SQL rows without using the extractor."""

import argparse
import gzip
import json
import time
from collections import Counter, defaultdict
from pathlib import Path


def load_rows(path):
    with gzip.open(path, "rt") as stream:
        for line in stream:
            yield json.loads(line)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--native-results", type=Path, required=True)
    p.add_argument("--extraction", type=Path, required=True)
    p.add_argument("--samples", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    started = time.perf_counter()
    if args.output.exists():
        raise FileExistsError(args.output)
    sources = {}
    for file in args.native_results.glob("*.msf.json.gz"):
        r = json.loads(gzip.decompress(file.read_bytes()))
        q = r["queries"]
        expected = {
            row[0]: dict(zip(q["target_psms"]["columns"], row))
            for row in q["target_psms"]["rows"]
            if row[2] == 3
        }
        high_sql = sum(
            row[3] for row in q["independent_high_counts"]["rows"] if row[0] == 3
        )
        assert len(expected) == high_sql
        tables = {}
        for name in (
            "native_modifications",
            "native_terminal_modifications",
            "native_protein_mappings",
            "native_protein_annotations",
        ):
            index = defaultdict(list)
            for values in q[name]["rows"]:
                index[values[0]].append(dict(zip(q[name]["columns"], values)))
            tables[name] = index
        fdr = {row[0]: row[1] for row in q["confidence_parameters"]["rows"]}
        assert float(fdr["TargetFPRHigh"]) == 0.01
        tops = Counter(
            row["SpectrumID"]
            for row in expected.values()
            if row["SearchEngineRank"] == 1 and row["Hidden"] == 0
        )
        sources[r["receipt"]["source"]["name"]] = {
            "receipt": r["receipt"],
            "queries": q,
            "high": expected,
            "tables": tables,
            "tops": tops,
        }
    samples = {
        r["file"].replace(".mzid.gz", ".msf"): r
        for r in json.loads(args.samples.read_text())
    }
    n_psms = 0
    seen = defaultdict(set)
    candidate_support = defaultdict(set)
    protein_names = defaultdict(set)
    run_names = defaultdict(set)
    for row in load_rows(args.extraction / "native-high-confidence-psms.jsonl.gz"):
        name = row["source_file"]
        source = sources[name]
        q = source["queries"]
        pid = row["native_psm"]["PeptideID"]
        assert pid not in seen[name]
        seen[name].add(pid)
        assert row["native_psm"] == source["high"][pid]
        assert row["source_sha256"] == source["receipt"]["sha256"]
        assert row["sample"] == samples[name]
        mods = source["tables"]["native_modifications"][pid]
        terms = source["tables"]["native_terminal_modifications"][pid]
        assert row["modifications"] == mods and row["terminal_modifications"] == terms
        pro_ids = sorted(
            {r["ProteinID"] for r in source["tables"]["native_protein_mappings"][pid]}
        )
        proteins = {
            str(pro): source["tables"]["native_protein_annotations"][pro]
            for pro in pro_ids
        }
        assert proteins == row["protein_mappings"]
        native = row["native_psm"]
        sequence = native["Sequence"]
        single_top = source["tops"][native["SpectrumID"]] == 1
        candidate = (
            not mods
            and not terms
            and 10 <= len(sequence) <= 512
            and all(c in "ACDEFGHIKLMNPQRSTVWY" for c in sequence)
            and native["SearchEngineRank"] == 1
            and native["Hidden"] == 0
            and single_top
            and bool(pro_ids)
            and all(proteins.values())
        )
        assert row["protein_mapping_complete"] == (
            bool(pro_ids) and all(proteins.values())
        )
        assert candidate == row["source_quality_candidate"]
        assert row["confidence"] == "native_high" and row["target_fdr_high"] == 0.01
        assert row["training_eligible"] is False and row["split"] == "unpartitioned"
        if candidate:
            key = (row["sample"]["biological_sample"], sequence)
            candidate_support[key].add(
                (name, row["source_sha256"], pid, native["SpectrumID"])
            )
            run_names[key].add(row["sample"]["run"])
            protein_names[key].update(
                a["Description"] for v in proteins.values() for a in v
            )
        n_psms += 1
    assert set(seen) == set(sources) and all(
        seen[name] == set(r["high"]) for name, r in sources.items()
    )
    control_peptides = {
        seq
        for (sample, seq) in candidate_support
        if sample.startswith("No_DP_control_")
    }
    qualified = 0
    distinct = set()
    bio_counts = defaultdict(Counter)
    keys = set()
    for row in load_rows(args.extraction / "native-peptide-samples.jsonl.gz"):
        key = (row["sample"]["biological_sample"], row["peptide"])
        assert key not in keys
        keys.add(key)
        actual = {
            (r["source_file"], r["source_sha256"], r["peptide_id"], r["spectrum_id"])
            for r in row["supporting_psms"]
        }
        assert (
            len(actual) == len(row["supporting_psms"])
            and actual == candidate_support[key]
        )
        assert set(row["runs"]) == run_names[key]
        assert {r["Description"] for r in row["protein_annotations"]} == protein_names[
            key
        ]
        assert row["has_ptm"] is False and row["peptide_modifications"] == ""
        control_overlap = key[1] in control_peptides
        assert row["observed_in_no_dp_control"] == control_overlap
        admitted = not key[0].startswith("No_DP_control_") and not control_overlap
        assert row["source_qualified_candidate"] == admitted
        assert row["training_eligible"] is False and row["split"] == "unpartitioned"
        assert not {"file", "run", "technical_replicate"} & set(row["sample"])
        qualified += admitted
        if admitted:
            distinct.add(key[1])
        bio_counts[key[0]].update(
            {
                "n_peptide_sample_rows": 1,
                "n_control_overlap_rows": int(control_overlap),
                "n_source_qualified_candidate_rows": int(admitted),
                "n_multi_acquisition_peptides": int(len(run_names[key]) >= 2),
            }
        )
    assert keys == set(candidate_support)
    summary = json.loads((args.extraction / "summary.json").read_text())
    assert n_psms == summary["counts"]["n_native_high_psms"]
    assert (
        len(keys) == summary["n_peptide_sample_rows"]
        and qualified == summary["n_source_qualified_candidate_rows"]
    )
    assert len(distinct) == summary["n_unique_candidate_peptides"]
    for row in summary["biological_samples"]:
        for k, v in bio_counts[row["biological_sample"]].items():
            assert row[k] == v
    result = {
        "status": "passed",
        "n_native_files": len(sources),
        "n_original_high_psms_replayed": n_psms,
        "n_peptide_sample_rows_replayed": len(keys),
        "n_source_qualified_candidate_rows": qualified,
        "n_distinct_candidate_peptides": len(distinct),
        "n_training_eligible_rows": 0,
        "seconds": time.perf_counter() - started,
        "verification_basis": "original hash-verified read-only SQLite query outputs and independent SQL histograms; no extractor imports",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
