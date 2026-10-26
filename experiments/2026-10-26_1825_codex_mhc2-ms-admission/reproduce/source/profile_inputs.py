"""Profile actual class-II MS source contracts before row admission or fitting."""

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import sys
import time

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

FAMILY = Path(__file__).resolve().parents[1]
RAW = Path("/private/tmp/presto-mhc2-20261026/ms-source")
CONTRIBUTORS = Path(
    "/Users/iskander/.hitlist/corrected-corpus/public-downloads/v4/"
    "bundles/2026-10-16T07-20-43Z/observation_contributors.parquet"
)
MOLECULAR = (
    "pmid",
    "mhc_restriction",
    "mhc_allele_set",
    "sample_mhc",
    "restriction_evidence",
    "mhc_allele_provenance",
    "sample_attribution",
)
CONTEXT = (
    "pmid",
    "condition_id",
    "condition_status",
    "condition_evidence",
    "condition_reference",
    "condition_control",
    "condition_control_for",
    "condition_combination",
    "condition_knockout_genes",
    "condition_knockdown_genes",
    "condition_overexpression_genes",
    "condition_transfection",
    "condition_transduction",
    "condition_cytokines",
    "condition_drugs",
    "condition_stimulation",
    "condition_antigen_exposure",
    "condition_background",
    "condition_mhc_context",
    "condition_culture",
    "condition_material",
    "cell_type",
    "cell_line_name",
    "sample_attribution",
    "sample_label",
    "attributed_sample_label",
    "sample_mhc_origin",
)
CATEGORICAL = (
    "ms_outcome_status",
    "assay_modality",
    "has_ptm",
    "peptide_modifications",
    "is_non_peptide_ligand",
    "restriction_evidence",
    "mhc_allele_provenance",
    "sample_attribution",
    "condition_status",
    "condition_evidence",
    "provenance_status",
    "mhc_species",
    "source_species",
    "species",
    "is_monoallelic",
    "is_potential_contaminant",
    "evidence_kind",
)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def rss():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform == "darwin" else value * 1024


def count_groups(table, names, counts):
    subset = table.select(names)
    subset = pa.table({name: pc.cast(subset[name], pa.string()) for name in names})
    subset = subset.append_column("_rows", pa.array([1] * len(subset), type=pa.int64()))
    for row in (
        subset.group_by(list(names), use_threads=False).aggregate([("_rows", "sum")]).to_pylist()
    ):
        counts[tuple(row[name] for name in names)] += row["_rows_sum"]


def save_groups(path, names, counts):
    rows = [{**dict(zip(names, key)), "rows": count} for key, count in counts.items()]
    rows.sort(key=lambda row: (-row["rows"], json.dumps(row, sort_keys=True)))
    path.write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")


def main():
    started = time.monotonic()
    signal.alarm(300)
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    output = FAMILY / "results" / "input-profile"
    output.mkdir(parents=True, exist_ok=False)
    source_sha = "f902341e90c1c57defc93f4bc540ffcbea1b41061afb4492ddd8aaa7642c65bd"
    manifest_sha = "b180eba804532f633a6d61bb350584dbe24f8df4b4b96f02c6ec492c60144372"
    assert digest(RAW / "manifest.json") == manifest_sha
    assert digest(RAW / "training.parquet") == source_sha
    manifest = json.loads((RAW / "manifest.json").read_text())
    expected = manifest["inputs"]["build_metadata"]["provenance"]["contributors"]
    assert CONTRIBUTORS.stat().st_size == expected["size_bytes"]
    assert digest(CONTRIBUTORS) == expected["sha256"]
    launch = {
        "agent_model": "Codex / GPT-6",
        "utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "dirty_status": subprocess.check_output(["git", "status", "--short"], text=True),
        "argv": sys.argv,
        "python": sys.executable,
        "pyarrow": pa.__version__,
        "script_sha256": digest(Path(__file__)),
        "training": {"path": str(RAW / "training.parquet"), "sha256": source_sha},
        "manifest_sha256": manifest_sha,
        "contributors": {"path": str(CONTRIBUTORS), **expected},
        "limits": {"threads": 1, "seconds": 300, "rss_bytes": 3 * 1024**3},
        "environment_overrides": {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        },
    }
    (FAMILY / "reproduce" / "profile-launch.json").write_text(json.dumps(launch, indent=2) + "\n")
    shutil.copyfile(__file__, FAMILY / "reproduce" / "source" / "profile_inputs.py")
    columns = sorted(set(MOLECULAR + CONTEXT + CATEGORICAL + ("mhc_class", "is_ms_observation")))
    molecular, contexts = Counter(), Counter()
    categories = {name: Counter() for name in CATEGORICAL}
    source_rows = selected_rows = 0
    for batch in pq.ParquetFile(RAW / "training.parquet").iter_batches(
        batch_size=8192, columns=columns, use_threads=False
    ):
        source_rows += len(batch)
        selected = pa.Table.from_batches([batch]).filter(
            pc.and_(pc.equal(batch.column("mhc_class"), "II"), batch.column("is_ms_observation"))
        )
        selected_rows += len(selected)
        if len(selected):
            count_groups(selected, MOLECULAR, molecular)
            count_groups(selected, CONTEXT, contexts)
            for name in CATEGORICAL:
                for item in pc.value_counts(selected[name]).to_pylist():
                    categories[name][item["values"]] += item["counts"]
        if rss() > 3 * 1024**3:
            raise RuntimeError("profile exceeded 3-GiB RSS limit")
    assert source_rows == manifest["n_rows"]
    assert sum(molecular.values()) == sum(contexts.values()) == selected_rows
    for counts in categories.values():
        assert sum(counts.values()) == selected_rows
    save_groups(output / "molecular-groups.json", MOLECULAR, molecular)
    save_groups(output / "context-groups.json", CONTEXT, contexts)
    fields = {
        name: [{"value": value, "rows": count} for value, count in counts.most_common()]
        for name, counts in categories.items()
    }
    (output / "categorical-fields.json").write_text(json.dumps(fields, indent=2) + "\n")
    summary = {
        "status": "profiled; no observations admitted",
        "source_rows": source_rows,
        "class_ii_ms_rows": selected_rows,
        "molecular_groups": len(molecular),
        "context_groups": len(contexts),
        "elapsed_seconds": time.monotonic() - started,
        "peak_rss_bytes": rss(),
        "contributor_source_hash_verified": True,
        "training_rows_admitted": 0,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    signal.alarm(0)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
