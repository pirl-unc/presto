#!/usr/bin/env python
"""Frozen, read-only source inventory for the registered coverage family."""

import argparse
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import sqlite3
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = Path(__file__).resolve().parents[1]
RAW = ROOT / "artifacts" / EXPERIMENT.name
sys.path.insert(0, str(ROOT))


def json_write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def stat_key(path):
    value = path.stat()
    return value.st_size, value.st_mtime_ns, value.st_ino


def hash_file(path):
    digest = hashlib.sha256()
    before = stat_key(path)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if stat_key(path) != before:
        raise RuntimeError(f"Source changed while hashing: {path}")
    return digest.hexdigest()


def freeze_invocation(args, output):
    """Freeze only an explicit environment allowlist; never dump credentials."""
    receipt = {
        "schema_version": 1,
        "agent_model": "Codex / GPT-6",
        "phase": args.phase,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "argv": [sys.executable, *sys.argv],
        "cwd": str(ROOT),
        "environment": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "HITLIST_DATA_DIR")
        },
        "git": {
            "commit": git("rev-parse", "HEAD"),
            "branch": git("branch", "--show-current"),
            "status": git("status", "--porcelain"),
        },
        "packages": {
            distribution.metadata["Name"]: distribution.version
            for distribution in importlib.metadata.distributions()
        },
        "output_dir": str(output),
        "launcher_sha256": hash_file(Path(__file__)),
        "package_source_files": {},
    }
    for package in ("hitlist", "mhcseqs", "mhcgnomes"):
        package_root = Path(importlib.util.find_spec(package).origin).parent
        receipt["package_source_files"][package] = {
            str(path.relative_to(package_root)): hash_file(path)
            for path in sorted(package_root.rglob("*"))
            if path.is_file() and path.suffix in {".py", ".yaml", ".json", ".csv", ".tsv"}
        }
    receipt["git"]["dirty"] = bool(receipt["git"]["status"])
    json_write(output / "invocation.json", receipt)
    bundle = EXPERIMENT / "reproduce"
    bundle.mkdir(exist_ok=True)
    json_write(bundle / f"{output.name}_invocation.json", receipt)
    snapshot = bundle / "source" / output.name
    snapshot.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, snapshot / "launch.py")
    (bundle / "environment.txt").write_text(
        "\n".join(
            f"{name}=={version}"
            for name, version in sorted(receipt["packages"].items())
            if name.lower() != "presto"
        )
        + "\n"
    )
    registry = ROOT / "experiments" / "experiment_log.md"
    if f"### {EXPERIMENT.name}" not in registry.read_text():
        with registry.open("a") as handle:
            handle.write(
                f"\n### {EXPERIMENT.name}\n\n"
                "- **Date / agent**: 2026-09-09; Codex / GPT-6.\n"
                f"- **Experiment**: [{EXPERIMENT.name}]({EXPERIMENT.name}/).\n"
                "- **Status**: source inventory running; uncapped census/update phases pending.\n"
                "- **Contract**: current merged TSV and exclusive Hitlist cache, read-only; "
                "prospective 80/10/10 peptide splits, data seed 17, split/model seed 42.\n"
                "- **Training / synthetic data**: none in inventory; later measured/augmented "
                "conditions and bounded optimizer diagnostics are specified in the README.\n"
                "- **Validation/test metrics**: not applicable to metadata inventory; "
                "no prediction-quality claim.\n"
                "- **Hardware**: local CPU; no Modal GPU requested.\n"
                "- **Reproduction**: experiment `reproduce/launch.sh`, phase invocation and "
                "source snapshot; pinned isolated environment.\n"
            )
    return receipt


def insert_excluded(db, name, rows):
    db.executemany(
        "INSERT INTO excluded VALUES (?,?,?,?)",
        ((name, str(pmid), str(peptide or ""), str(kind or "")) for pmid, peptide, kind in rows),
    )


def scan_parquet(path, excluded, db):
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    file = pq.ParquetFile(path)
    result = {
        "rows": file.metadata.num_rows,
        "row_groups": file.metadata.num_row_groups,
        "schema": {field.name: str(field.type) for field in file.schema_arrow},
    }
    if path.name not in {"observations.parquet", "binding.parquet"}:
        return result
    names = set(file.schema_arrow.names)
    if not {"pmid", "peptide"} <= names:
        result["exclusion_check"] = "missing pmid/peptide columns"
        return result
    total = matched = 0
    wanted = pa.array(sorted(excluded), type=pa.string())
    for batch in file.iter_batches(batch_size=131072, columns=["pmid", "peptide"]):
        total += batch.num_rows
        pmids = pc.cast(batch.column("pmid"), pa.string())
        mask = pc.is_in(pmids, value_set=wanted)
        selected = batch.filter(mask).to_pydict()
        matched += len(selected["pmid"])
        insert_excluded(
            db,
            path.name,
            (
                (pmid, peptide, "ms" if path.name == "observations.parquet" else "binding")
                for pmid, peptide in zip(selected["pmid"], selected["peptide"])
            ),
        )
    result.update(scanned_rows=total, rows_from_excluded_ms_studies=matched)
    if total != file.metadata.num_rows:
        raise RuntimeError(f"Parquet row-count reconciliation failed: {path}")
    return result


def scan_merged(path, excluded, db, output):
    counts, sources, total, matched = Counter(), Counter(), 0, 0
    subset = output / "excluded_studies.tsv"
    with path.open(newline="") as handle, subset.open("w", newline="") as sink:
        reader = csv.DictReader(handle, delimiter="\t")
        columns = reader.fieldnames
        if not {"peptide", "pmid", "record_type", "source"} <= set(columns or []):
            raise ValueError("Merged input lacks the required source-inventory fields")
        writer = csv.DictWriter(sink, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        for row in reader:
            total += 1
            counts[row["record_type"]] += 1
            sources[row["source"]] += 1
            # This schema contains one PMID, rather than a bibliography string.
            pmid = row["pmid"].strip()
            if pmid in excluded:
                matched += 1
                writer.writerow(row)
                insert_excluded(db, path.name, [(pmid, row["peptide"], row["record_type"])])
    return {
        "rows": total,
        "columns": columns,
        "record_types": dict(counts),
        "sources": dict(sources),
        "rows_from_excluded_ms_studies": matched,
        "excluded_subset": str(subset),
        "subset_sha256": hash_file(subset),
    }


def excluded_studies():
    from hitlist.curation import load_pmid_overrides

    return {
        str(key): value
        for key, value in load_pmid_overrides().items()
        if value.get("exclude_from_ms") is True
    }


def inventory(args, output):

    hitlist_root = Path(importlib.util.find_spec("hitlist").origin).parent
    overrides = hitlist_root / "data" / "pmid_overrides.yaml"
    excluded = excluded_studies()
    if not excluded:
        raise RuntimeError("No excluded studies found; inspect the curation schema/version")
    curation = {
        "path": str(overrides),
        "sha256": hash_file(overrides),
        "exclusions": {
            key: {"study_label": value.get("study_label", "")} for key, value in excluded.items()
        },
    }
    json_write(output / "curation.json", curation)
    sources = [
        ROOT / "data" / "merged_deduped.tsv",
        ROOT / "data" / "merged_deduped_funnel.tsv",
        *[
            args.hitlist_dir / name
            for name in (
                "observations.parquet",
                "binding.parquet",
                "bulk_proteomics.parquet",
                "peptide_mappings.parquet",
                "observations_meta.json",
                "peptide_mappings_meta.json",
                "manifest.json",
            )
        ],
    ]
    records = {}
    for path in sources:
        if not path.exists():
            records[str(path)] = {"exists": False}
            continue
        print(f"Hashing {path}", flush=True)
        records[str(path)] = {"exists": True, "stat": stat_key(path), "sha256": hash_file(path)}
    json_write(output / "source_files.json", records)
    raw_phase = RAW / output.name
    raw_phase.mkdir(parents=True, exist_ok=True)
    db_path = raw_phase / "excluded.sqlite"
    if db_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing evidence database: {db_path}")
    with sqlite3.connect(db_path) as db:
        db.execute("CREATE TABLE excluded (source_file TEXT, pmid TEXT, peptide TEXT, kind TEXT)")
        for path in sources:
            if not path.exists():
                continue
            print(f"Inspecting {path}", flush=True)
            if path.suffix == ".parquet":
                records[str(path)]["contents"] = scan_parquet(path, excluded, db)
            elif path.name == "merged_deduped.tsv":
                records[str(path)]["contents"] = scan_merged(path, excluded, db, raw_phase)
            if tuple(records[str(path)]["stat"]) != stat_key(path):
                raise RuntimeError(f"Source changed during inventory: {path}")
            db.commit()
        summary = [
            {
                "source_file": name,
                "pmid": pmid,
                "kind": kind,
                "rows": rows,
                "distinct_peptides": peptides,
                "study_label": curation["exclusions"][pmid]["study_label"],
            }
            for name, pmid, kind, rows, peptides in db.execute(
                "SELECT source_file,pmid,kind,COUNT(*),COUNT(DISTINCT peptide) "
                "FROM excluded GROUP BY source_file,pmid,kind ORDER BY source_file,pmid,kind"
            )
        ]
    # Verify real merged source routing on the complete flagged-study subset.
    from presto.scripts.train_iedb import load_records_from_merged_tsv

    subset = raw_phase / "excluded_studies.tsv"
    lists = load_records_from_merged_tsv(
        subset,
        max_binding=0,
        max_kinetics=0,
        max_stability=0,
        max_processing=0,
        max_elution=0,
        max_tcell=0,
        max_vdjdb=0,
    )
    routed = {}
    for name, values in zip(
        ("binding", "kinetics", "stability", "processing", "elution", "tcell", "tcr_evidence"),
        lists[:-1],
    ):
        routed[name] = {
            "records": len(values),
            "pmids": dict(Counter(str(getattr(record, "pmid", "")) for record in values)),
        }
    for path in sources:
        if path.exists() and tuple(records[str(path)]["stat"]) != stat_key(path):
            raise RuntimeError(f"Source changed before inventory closure: {path}")
    result = {
        "schema_version": 1,
        "phase": "inventory",
        "source_files": records,
        "curation": curation,
        "excluded_study_counts": summary,
        "merged_flagged_subset_loader": {"modalities": routed, "stats": lists[-1]},
        "raw_sqlite": str(db_path),
        "interpretation": "MS exclusions apply to MS evidence, not all binding/T-cell rows. "
        "Loader counts precede MHC resolution/filtering; this is not prediction evidence.",
    }
    json_write(output / "result.json", result)
    with (output / "excluded_study_counts.csv").open("w", newline="") as handle:
        fields = ("source_file", "pmid", "kind", "rows", "distinct_peptides", "study_label")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)
    print(
        json.dumps(
            {
                "excluded_studies": len(excluded),
                "flagged_counts": summary,
                "merged_routing": routed,
            },
            indent=2,
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["inventory"])
    parser.add_argument(
        "--hitlist-dir",
        type=Path,
        default=Path(os.environ.get("HITLIST_DATA_DIR", "/Users/iskander/.hitlist")),
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output = args.output_dir or EXPERIMENT / "results" / args.phase
    if output.exists():
        raise FileExistsError(f"Choose a new --output-dir; existing phase artifacts: {output}")
    receipt = freeze_invocation(args, output)
    started = time.perf_counter()
    try:
        inventory(args, output)
    except BaseException as exc:
        json_write(
            output / "status.json",
            {
                "status": "failed",
                "error": repr(exc),
                "elapsed_seconds": time.perf_counter() - started,
            },
        )
        raise
    json_write(
        output / "status.json",
        {
            "status": "completed",
            "elapsed_seconds": time.perf_counter() - started,
            "git": receipt["git"],
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )


if __name__ == "__main__":
    main()
