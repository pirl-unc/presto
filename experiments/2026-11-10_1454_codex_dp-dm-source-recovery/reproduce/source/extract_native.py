"""Recover high-confidence native source records, retaining acquisition lineage."""

import argparse
import csv
import gzip
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path


def keyed(query):
    return [dict(zip(query["columns"], row)) for row in query["rows"]]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--native-results", type=Path, required=True)
    p.add_argument("--samples", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    started = time.perf_counter()
    paths = sorted(args.native_results.glob("*.msf.json.gz"))
    if len(paths) != 32:
        raise ValueError(f"Need all 32 original results; found {len(paths)}")
    args.output.mkdir(parents=True, exist_ok=False)
    samples = {
        s["file"].replace(".mzid.gz", ".msf"): s
        for s in json.loads(args.samples.read_text())
    }
    total = Counter()
    file_counts = []
    groups = {}
    psm_path = args.output / "native-high-confidence-psms.jsonl.gz"
    with gzip.open(psm_path, "wt") as output:
        for path in paths:
            result = json.loads(gzip.decompress(path.read_bytes()))
            q = result["queries"]
            source = result["receipt"]
            name = source["source"]["name"]
            sample = samples[name]
            assert source["sha1"] == source["source"]["expected_sha1"]
            assert source["n_bytes"] == source["source"]["max_bytes"]
            params = {
                r["ParameterName"]: r["ParameterValue"]
                for r in keyed(q["confidence_parameters"])
            }
            assert float(params["TargetFPRHigh"]) == 0.01
            assert float(params["TargetFPRMiddle"]) == 0.05
            assert any(
                r["NodeName"] == "PeptideValidatorNode" for r in keyed(q["nodes"])
            )
            assert any(
                r["FileName"].replace("\\", "/").split("/")[-1]
                == name.replace(".msf", ".raw")
                for r in keyed(q["file_identities"])
            )
            targets = keyed(q["target_psms"])
            ids = {r["PeptideID"] for r in targets}
            assert len(ids) == len(targets)
            original_counts = Counter()
            for row in q["independent_high_counts"]["rows"]:
                original_counts[(row[0], row[1], row[2])] += row[3]
            assert original_counts == Counter(
                (r["ConfidenceLevel"], r["Hidden"], r["SearchEngineRank"])
                for r in targets
            )
            modifications = defaultdict(list)
            terminals = defaultdict(list)
            mappings = defaultdict(set)
            annotations = defaultdict(list)
            for row in keyed(q["native_modifications"]):
                assert row["PeptideID"] in ids and row["ModificationName"] is not None
                modifications[row["PeptideID"]].append(row)
            for row in keyed(q["native_terminal_modifications"]):
                assert row["PeptideID"] in ids and row["ModificationName"] is not None
                terminals[row["PeptideID"]].append(row)
            for row in keyed(q["native_protein_mappings"]):
                mappings[row["PeptideID"]].add(row["ProteinID"])
            for row in keyed(q["native_protein_annotations"]):
                annotations[row["ProteinID"]].append(row)
            high = [r for r in targets if r["ConfidenceLevel"] == 3]
            tops = Counter(
                r["SpectrumID"]
                for r in targets
                if r["ConfidenceLevel"] == 3
                and r["SearchEngineRank"] == 1
                and r["Hidden"] == 0
            )
            counts = Counter(
                n_native_target_psms=len(targets), n_native_high_psms=len(high)
            )
            for row in high:
                pid = row["PeptideID"]
                sequence = row["Sequence"]
                protein_mapping_complete = bool(mappings[pid]) and all(
                    annotations[prot] for prot in mappings[pid]
                )
                modified = bool(modifications[pid] or terminals[pid])
                canonical = 10 <= len(sequence) <= 512 and not set(sequence) - set(
                    "ACDEFGHIKLMNPQRSTVWY"
                )
                unique_top = (
                    row["SearchEngineRank"] == 1
                    and row["Hidden"] == 0
                    and tops[row["SpectrumID"]] == 1
                )
                candidate = (
                    canonical
                    and not modified
                    and unique_top
                    and protein_mapping_complete
                )
                counts["n_high_modified_psms"] += modified
                counts[
                    "n_high_incomplete_protein_mapping_psms"
                ] += not protein_mapping_complete
                counts["n_high_noncanonical_or_short_psms"] += not canonical
                counts["n_high_not_unique_visible_top_psms"] += not unique_top
                counts["n_qualified_candidate_psms"] += candidate
                record = {
                    "source_file": name,
                    "source_sha256": source["sha256"],
                    "source_table": "Peptides",
                    "native_psm": row,
                    "sample": sample,
                    "modifications": modifications[pid],
                    "terminal_modifications": terminals[pid],
                    "protein_mappings": {
                        str(prot): annotations[prot] for prot in sorted(mappings[pid])
                    },
                    "confidence": "native_high",
                    "protein_mapping_complete": protein_mapping_complete,
                    "target_fdr_high": 0.01,
                    "has_ptm": modified,
                    "source_quality_candidate": candidate,
                    "training_eligible": False,
                    "split": "unpartitioned",
                }
                output.write(
                    json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
                )
                if candidate:
                    key = (sample["biological_sample"], sequence)
                    if key not in groups:
                        groups[key] = {
                            "peptide": sequence,
                            "sample": {
                                k: v
                                for k, v in sample.items()
                                if k
                                not in {
                                    "file",
                                    "sha256",
                                    "run",
                                    "source_sample_name",
                                    "technical_replicate",
                                }
                            },
                            "has_ptm": False,
                            "peptide_modifications": "",
                            "supporting_psms": [],
                            "protein_annotations": {},
                            "runs": set(),
                        }
                    group = groups[key]
                    group["runs"].add(sample["run"])
                    group["supporting_psms"].append(
                        {
                            "source_file": name,
                            "source_sha256": source["sha256"],
                            "peptide_id": pid,
                            "spectrum_id": row["SpectrumID"],
                        }
                    )
                    for prot in sorted(mappings[pid]):
                        for annotation in annotations[prot]:
                            group["protein_annotations"][annotation["Description"]] = (
                                annotation
                            )
            file_counts.append(
                {
                    "file": name,
                    "source_sha256": source["sha256"],
                    "sample": sample,
                    **dict(counts),
                }
            )
            total.update(counts)
    controls = {
        seq
        for (bio, seq), r in groups.items()
        if r["sample"]["capture_role"] == "no_dp_capture_control"
    }
    group_path = args.output / "native-peptide-samples.jsonl.gz"
    bio_counts = defaultdict(Counter)
    with gzip.open(group_path, "wt") as out:
        for key in sorted(groups):
            group = groups[key]
            group["runs"] = sorted(group["runs"])
            group["protein_annotations"] = list(group["protein_annotations"].values())
            group["observed_in_no_dp_control"] = group["peptide"] in controls
            group["training_eligible"] = False
            group["split"] = "unpartitioned"
            group["source_qualified_candidate"] = (
                group["sample"]["capture_role"] == "engineered_dp_capture"
                and not group["observed_in_no_dp_control"]
            )
            out.write(json.dumps(group, sort_keys=True, separators=(",", ":")) + "\n")
            counts = bio_counts[key[0]]
            counts["n_peptide_sample_rows"] += 1
            counts["n_control_overlap_rows"] += group["observed_in_no_dp_control"]
            counts["n_source_qualified_candidate_rows"] += group[
                "source_qualified_candidate"
            ]
            counts["n_multi_acquisition_peptides"] += len(group["runs"]) >= 2
    summary = {
        "status": "source_extraction_complete_not_training_admission",
        "n_files": len(paths),
        "counts": dict(total),
        "n_peptide_sample_rows": len(groups),
        "n_control_peptides": len(controls),
        "n_unique_candidate_peptides": len(
            {seq for (bio, seq), r in groups.items() if r["source_qualified_candidate"]}
        ),
        "n_source_qualified_candidate_rows": sum(
            c["n_source_qualified_candidate_rows"] for c in bio_counts.values()
        ),
        "biological_samples": [
            {
                **next(s for s in samples.values() if s["biological_sample"] == bio),
                **dict(counts),
            }
            for bio, counts in sorted(bio_counts.items())
        ],
        "n_training_eligible_rows": 0,
        "source_filter": "native ConfidenceLevel=3 and TargetFPRHigh=.01; only unique visible rank-1, unmodified canonical 10-512 aa candidates; complete original protein mappings; no-DP controls kept separate; control overlap flagged",
        "pending": [
            "source integration in shared Hitlist",
            "lineage and prior-cohort peptide overlap",
            "prospective partitions and comparison contract",
        ],
        "native_high_psms_path": str(psm_path),
        "native_high_psms_sha256": hashlib.sha256(psm_path.read_bytes()).hexdigest(),
        "peptide_samples_path": str(group_path),
        "peptide_samples_sha256": hashlib.sha256(group_path.read_bytes()).hexdigest(),
        "seconds": time.perf_counter() - started,
    }
    for name, value in [("summary.json", summary), ("source-files.json", file_counts)]:
        (args.output / name).write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n"
        )
    with (args.output / "sample-counts.csv").open("w") as out:
        fields = [
            "biological_sample",
            "mhc_alpha",
            "mhc_beta",
            "dm_state",
            "capture_role",
            "n_peptide_sample_rows",
            "n_control_overlap_rows",
            "n_source_qualified_candidate_rows",
            "n_multi_acquisition_peptides",
        ]
        writer = csv.DictWriter(out, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary["biological_samples"])
    print(
        json.dumps(
            {k: v for k, v in summary.items() if k not in ("biological_samples",)},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
