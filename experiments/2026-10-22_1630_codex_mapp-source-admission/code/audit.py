"""Source admission evidence, not a production curator or training adapter."""

from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import platform
import subprocess
import tempfile
import time

import datacache


FAMILY = Path(__file__).resolve().parents[2 if Path(__file__).parent.name == "source" else 1]
REPO = FAMILY.parents[1]
AA = set("ACDEFGHIKLMNPQRSTVWY")


def save(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def source_paths():
    frozen = json.loads((FAMILY / "reproduce/input_assets.json").read_text())
    assets = frozen["assets"]
    seeds = {k: v for k, v in frozen["initial_seed_paths"].items() if Path(v).is_file()}
    # Checked-in snapshots also work when original absolute seed paths moved.
    for name in assets:
        accession, kind = name.split("/")
        candidate = REPO / "data/availability_sources/snapshots/2026-10-22" / name
        if not candidate.is_file():
            candidate = REPO / "data/availability_sources/snapshots/2026-10-22-mapp-series" / f"{accession}.{kind}"
        if candidate.is_file():
            seeds[name] = str(candidate)
    identity = hashlib.sha256(json.dumps(assets, sort_keys=True).encode()).hexdigest()
    paths = datacache.install_bundle(REPO / "data/availability_sources" / identity, assets,
                                     source_paths=seeds, download_options={"show_progress": False})
    archive = paths["PXD009092/SEARCH.7z"]
    members = json.loads((FAMILY / "reproduce/archive_members.json").read_text())
    assert all(name.startswith("txt/") and ".." not in Path(name).parts for name in members)
    # Original member bytes are immutable derived assets, backed by this archive.
    with tempfile.TemporaryDirectory(prefix="presto-mapp-members-") as tmp:
        local, derived = {}, {}
        for name, receipt in members.items():
            destination = Path(tmp) / Path(name).name
            with destination.open("wb") as handle:
                subprocess.run(["tar", "-xOf", str(archive), name], stdout=handle, check=True)
            assert destination.stat().st_size == receipt["size_bytes"]
            assert hashlib.sha256(destination.read_bytes()).hexdigest() == receipt["sha256"]
            local[name] = str(destination)
            derived[name] = {"url": assets["PXD009092/SEARCH.7z"]["url"] + "#" + name,
                             "size": receipt["size_bytes"], "sha256": receipt["sha256"], "decompress": False}
        member_id = hashlib.sha256(json.dumps(members, sort_keys=True).encode()).hexdigest()
        member_paths = datacache.install_bundle(REPO / "data/availability_sources" / member_id,
                                               derived, source_paths=local)
    return paths, member_paths


def records(path):
    with open(path, newline="") as handle:
        yield from csv.DictReader(handle, delimiter="\t")


def tokens(value):
    return value.split(";") if value else []


def dump_csv(path, rows):
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as handle:
                writer = None
                for row in rows:
                    if writer is None:
                        writer = csv.DictWriter(handle, fieldnames=list(row), lineterminator="\n")
                        writer.writeheader()
                    writer.writerow(row)


def main():
    started = time.perf_counter()
    paths, member_paths = source_paths()
    result = FAMILY / "results"
    result.mkdir(exist_ok=True)
    inventory = []
    for name, path in member_paths.items():
        with open(path, newline="") as handle:
            reader = csv.reader(handle, delimiter="\t")
            header = next(reader)
            n_rows = 0
            for row in reader:
                assert len(row) == len(header), (name, reader.line_num, len(row))
                n_rows += 1
        assert len(header) == len(set(header))
        inventory.append({"member": name, "n_rows": n_rows, "columns": header,
                          "size_bytes": Path(path).stat().st_size})
    save(result / "table_inventory.json", inventory)
    save(result / "input_paths.json", {"original": {k: str(Path(v).relative_to(REPO)) for k, v in paths.items()},
         "archive_members": {k: str(Path(v).relative_to(REPO)) for k, v in member_paths.items()}})
    series = []
    for name in sorted(paths):
        if not name.endswith("/project.json"):
            continue
        accession = name.split("/")[0]
        project = json.loads(Path(paths[name]).read_text())
        files = json.loads(Path(paths[f"{accession}/files.json"]).read_text())
        assert len(files) < 100, "Must paginate rather than claim a truncated file inventory"
        series.append({"accession": accession, "title": project["title"],
                       "description": project.get("projectDescription"),
                       "sample_processing": project.get("sampleProcessingProtocol"),
                       "data_processing": project.get("dataProcessingProtocol"),
                       "n_files": len(files), "files": [{"name": f["fileName"], "bytes": f["fileSizeBytes"],
                           "category": f["fileCategory"].get("value")} for f in files]})
    save(result / "deposited_series.json", series)
    parameters = list(records(member_paths["txt/parameters.txt"]))
    save(result / "parameters.json", [dict(row, source_row=i) for i, row in enumerate(parameters, 2)
         if row["Parameter"] not in {"User name", "Machine name"}])
    summary = list(records(member_paths["txt/summary.txt"]))
    runs = {r["Raw file"]: dict(r, source_row=i) for i, r in enumerate(summary, 2) if r["Experiment"]}
    assert len(runs) == 18
    experiments = {r["Experiment"] for r in runs.values()}
    files = json.loads(Path(paths["PXD009092/files.json"]).read_text())
    raw_names = {f["fileName"].rsplit(".", 1)[0] for f in files if f["fileName"].lower().endswith(".raw")}
    assert set(runs) == raw_names
    peptides = {r["id"]: dict(r, source_row=i) for i, r in enumerate(records(member_paths["txt/peptides.txt"]), 2)}
    group_records = list(records(member_paths["txt/proteinGroups.txt"]))
    groups = {r["id"]: r for r in group_records if r["id"].isdigit()}
    quarantined = [{"source_row": i, "source_id": r["id"],
                    "reason": "missing_or_nonnumeric_source_group_id",
                    "nonempty_fields": {k: {"length": len(v), "prefix": v[:80]} for k, v in r.items() if v},
                    "parsed_row_sha256": hashlib.sha256(json.dumps(r, sort_keys=True).encode()).hexdigest()}
                   for i, r in enumerate(group_records, 2) if not r["id"].isdigit()]
    save(result / "quarantined_group_rows.json", quarantined)
    assert len(peptides) == next(r["n_rows"] for r in inventory if r["member"] == "txt/peptides.txt")
    assert len(group_records) == next(r["n_rows"] for r in inventory if r["member"] == "txt/proteinGroups.txt")
    assert len(groups) + len(quarantined) == len(group_records)
    totals = defaultdict(float)
    direct = Counter()
    evidence_types, per_run_types, mods = Counter(), defaultdict(Counter), Counter()
    link_issues = Counter()
    seen_ids, evidence_links = set(), defaultdict(set)
    for row in records(member_paths["txt/evidence.txt"]):
        assert row["id"] not in seen_ids
        seen_ids.add(row["id"])
        pid, experiment = row["Peptide ID"], row["Experiment"]
        assert experiment in experiments
        assert runs[row["Raw file"]]["Experiment"] == experiment
        if pid not in peptides:
            link_issues["evidence_to_missing_peptide"] += 1
        elif row["Sequence"] != peptides[pid]["Sequence"]:
            link_issues["evidence_peptide_sequence_mismatch"] += 1
        evidence_links[pid].add(row["id"])
        value = float(row["Intensity"]) if row["Intensity"] else 0.0
        assert math.isfinite(value) and value >= 0
        totals[(pid, experiment)] += value
        if int(row["MS/MS Count"] or 0) > 0:
            direct[(pid, experiment)] += 1
        evidence_types[row["Type"]] += 1
        per_run_types[experiment][row["Type"]] += 1
        mods[row["Modifications"]] += 1
        for gid in tokens(row["Protein group IDs"]):
            if gid not in groups:
                link_issues["evidence_to_missing_group"] += 1
    for pid, p in peptides.items():
        if set(tokens(p["Evidence IDs"])) != evidence_links[pid]:
            link_issues["peptide_reciprocal_evidence_mismatch"] += 1
        for gid in tokens(p["Protein group IDs"]):
            if gid not in groups:
                link_issues["peptide_to_missing_group"] += 1
            elif pid not in tokens(groups[gid]["Peptide IDs"]):
                link_issues["peptide_group_reciprocal_mismatch"] += 1
    group_orphans = sum(pid not in peptides for g in groups.values() for pid in tokens(g["Peptide IDs"]))
    link_issues["group_to_missing_peptide"] += group_orphans
    first = next(iter(peptides.values()))
    quantity_columns = {c: c.removeprefix("Intensity ").replace("-MAPPbeadFilt", "")
                        for c in first if c.startswith("Intensity ")}
    assert len(quantity_columns) == len(set(quantity_columns.values())) == 15
    assert set(quantity_columns.values()) < experiments
    for experiment in experiments:
        assert "Identification type " + experiment in first
    comparisons, states, per_run_quantity = [], Counter(), defaultdict(Counter)
    for pid, p in peptides.items():
        for column, experiment in quantity_columns.items():
            text = p[column]
            assert text != ""
            source = float(text)
            assert math.isfinite(source) and source >= 0
            evidence = totals[(pid, experiment)]
            if source == 0 and evidence > 0:
                state = "reported_zero_evidence_positive"
            elif source > 0 and evidence == 0:
                state = "reported_positive_evidence_zero"
            elif source == evidence == 0:
                state = "both_zero"
            elif math.isclose(source, evidence, rel_tol=1e-4, abs_tol=1):
                state = "both_positive_agree_within_rounding_tolerance"
            else:
                state = "both_positive_disagree"
            states[state] += 1
            per_run_quantity[experiment][state] += 1
            comparisons.append({"peptide_id": pid, "source_row": p["source_row"],
                "experiment": experiment, "quantity_column": column,
                "source_intensity_text": text, "linked_evidence_intensity_sum": evidence,
                "state": state, "direct_msms_evidence_rows": direct[(pid, experiment)]})
    dump_csv(result / "quantity_comparisons.csv.gz", comparisons)
    ut = sorted(e for e in experiments if "_UT_" in e)
    cytokine = sorted(e for e in experiments if "_TNF+IFN_" in e)
    assert len(ut) == len(cytokine) == 3
    by_experiment = {e: c for c, e in quantity_columns.items()}
    ledger, counts, lengths, group_support = [], Counter(), Counter(), Counter()
    for pid, p in peptides.items():
        flags = {
            "canonical_sequence": bool(p["Sequence"]) and set(p["Sequence"]) <= AA,
            "reverse": p["Reverse"] == "+", "contaminant": p["Potential contaminant"] == "+",
            "single_group": len(tokens(p["Protein group IDs"])) == 1,
            "single_protein": len(tokens(p["Proteins"])) == 1,
            "ut_positive": sum(float(p[by_experiment[e]]) > 0 for e in ut),
            "cytokine_positive": sum(float(p[by_experiment[e]]) > 0 for e in cytokine),
            "ut_direct_msms_runs": sum(direct[(pid, e)] > 0 for e in ut),
            "cytokine_direct_msms_runs": sum(direct[(pid, e)] > 0 for e in cytokine),
        }
        structural = flags["canonical_sequence"] and not flags["reverse"] and not flags["contaminant"] and flags["single_group"]
        structural = structural and flags["ut_positive"] >= 2 and flags["cytokine_positive"] >= 2
        structural = structural and flags["ut_direct_msms_runs"] >= 1 and flags["cytokine_direct_msms_runs"] >= 1
        lengths[len(p["Sequence"]) ] += 1
        for key in ["reverse", "contaminant", "canonical_sequence", "single_group", "single_protein"]:
            counts[key] += int(flags[key])
        counts["structural_contrast_support"] += int(structural)
        if structural:
            group_support[p["Protein group IDs"]] += 1
        ledger.append({"source_row": p["source_row"], "peptide_id": pid, "sequence": p["Sequence"],
            "protein_ids": p["Proteins"], "protein_group_ids": p["Protein group IDs"], "source_pep": p["PEP"],
            **flags, "structural_contrast_support": structural, "admission": "deferred",
            "reason": "source_confidence_and_postprocessing_resolution;shared_metadata_export;reference_and_material_links"})
    dump_csv(result / "peptide_support_ledger.csv.gz", ledger)
    run_rows = [{"raw_file": raw, "experiment": r["Experiment"], "summary_source_row": r["source_row"],
        "enzyme_mode": r["Enzyme mode"], "evidence_types": per_run_types[r["Experiment"]],
        "peptide_quantity_column": by_experiment.get(r["Experiment"]),
        "quantity_comparison_states": per_run_quantity[r["Experiment"]]} for raw, r in runs.items()]
    save(result / "runs.json", {"acquisitions": run_rows, "other_summary_rows": [r["Raw file"] for r in summary if not r["Experiment"]]})
    report = {"n_peptide_rows": len(peptides), "n_unique_sequences": len({p["Sequence"] for p in peptides.values()}),
        "n_indexable_protein_groups": len(groups), "n_quarantined_group_records": len(quarantined),
        "n_evidence_rows": len(seen_ids), "n_acquisitions": len(runs),
        "n_quantity_comparisons": len(comparisons), "quantity_states": states,
        "peptide_counts": counts, "peptide_lengths": lengths, "evidence_types": evidence_types,
        "modifications": mods, "link_issues": link_issues,
        "structural_contrast_groups": len(group_support),
        "structural_contrast_multipeptide_groups": sum(n > 1 for n in group_support.values()),
        "structural_contrast_multipeptide_rows": sum(n for n in group_support.values() if n > 1),
        "n_training_admitted": 0}
    save(result / "summary.json", report)
    save(result / "runtime.json", {"seconds": time.perf_counter() - started, "python": platform.python_version(),
         "platform": platform.platform(), "model_fitting": False, "gpu": None})
    output_names = ["table_inventory.json", "input_paths.json", "deposited_series.json", "parameters.json",
                    "quantity_comparisons.csv.gz", "peptide_support_ledger.csv.gz", "runs.json", "summary.json", "runtime.json",
                    "quarantined_group_rows.json"]
    save(result / "output_hashes.json", {n: hashlib.sha256((result / n).read_bytes()).hexdigest() for n in output_names})
    print(json.dumps(report))


if __name__ == "__main__":
    main()
