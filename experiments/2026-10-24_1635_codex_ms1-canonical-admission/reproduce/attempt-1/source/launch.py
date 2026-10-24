"""Build canonical source/recipe and independently verify original quantity cells."""

import csv
import hashlib
import io
import json
import math
import os
import resource
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
RAW = REPO / "data/measured_ms1_source_qualification/2026-10-24_1546_v1"
OUTPUT = REPO / "data/measured_ms1_source_admission/2026-10-24_1635_v1"
HITLIST = Path("/private/tmp/hitlist-proteometools-ms1")
SAMPLES = [
    ("HLA1-first-DDA.zip", "TUM_HLA_1_01_01_DDA-1h-R1-unspecific.zip", "TUM_HLA_1", 677),
    ("HLA2-first-DDA.zip", "TUM_HLA2_1_01_01_DDA-1h-R1-unspecific.zip", "TUM_HLA2_1", 470),
]


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def git(repo, *args):
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    from hitlist import proteometools_ms1 as source
    from hitlist import proteometools_response as response
    from hitlist.evidence_recipe import materialize_recipe, verify_recipe_package
    from hitlist.evidence_store import EvidenceStore
    from hitlist.provenance import file_digest

    assert Path(source.__file__).resolve().parent.parent == HITLIST
    assert not OUTPUT.exists()
    OUTPUT.mkdir(parents=True)
    started = time.perf_counter()
    metadata = {
        "agent_model": "Codex / GPT-6",
        "python": sys.executable,
        "python_version": sys.version,
        "argv": sys.argv,
        "environment": {
            k: os.environ[k]
            for k in ("PYTHONPATH", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
            if k in os.environ
        },
        "presto_commit": git(REPO, "rev-parse", "HEAD"),
        "presto_tracked_status": git(REPO, "status", "--short", "--untracked-files=no"),
        "hitlist_commit": git(HITLIST, "rev-parse", "HEAD"),
        "hitlist_status": git(HITLIST, "status", "--short"),
        "source_directory": str(RAW),
        "output_directory": str(OUTPUT),
        "phase": "source_admission_no_predictive_fit",
    }
    write(ROOT / "reproduce/launch.json", metadata)
    frozen = ROOT / "reproduce/source/hitlist"
    changed = (
        git(HITLIST, "diff", "--name-only").splitlines()
        + git(HITLIST, "ls-files", "--others", "--exclude-standard").splitlines()
    )
    for name in changed:
        original = HITLIST / name
        if original.is_file():
            destination = frozen / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, destination)
    (ROOT / "reproduce/hitlist.patch").write_text(git(HITLIST, "diff") + "\n")
    shutil.copy2(__file__, ROOT / "reproduce/source/launch.py")
    archives = [
        {"filename": filename, "source_name": name, **file_digest(RAW / filename)}
        for filename, name, _, _ in SAMPLES
    ]
    write(ROOT / "reproduce/archive-receipts.json", archives)
    source.write_proteometools_ms1_package(RAW, OUTPUT / "source", archives=archives)
    source_seconds = time.perf_counter() - started
    store = EvidenceStore.open(OUTPUT / "source")
    recipe = {
        "schema_version": 1,
        "name": "audited_two_pool_relative_MS1",
        "registered_recipe": response.RECIPE_ID,
    }
    materialize_recipe(store, recipe, OUTPUT / "recipe")
    verify_recipe_package(OUTPUT / "recipe", source_package=OUTPUT / "source")
    build_seconds = time.perf_counter() - started
    print("Canonical package and recipe built; independent original-cell checks next", flush=True)

    typed = {
        name: pq.read_table(
            OUTPUT / "source" / source.ROOT / "typed" / f"{name}.parquet"
        ).to_pylist()
        for name in ("runs", "peptides", "forms", "source_records")
    }
    runs = {row["source_file"]: row for row in typed["runs"]}
    records = {row["source_record_id"]: row for row in typed["source_records"]}
    examples = json.loads((OUTPUT / "recipe/examples.json").read_text())
    with (
        REPO
        / "experiments/2026-10-24_1546_codex_measured-ms1-source/results/planned-product-dispositions.csv"
    ).open() as stream:
        membership = list(csv.DictReader(stream))
    pool_results = []
    n_cells = 0
    for filename, _, pool, n_expected in SAMPLES:
        run_id = runs[filename]["run_id"]
        originals = {}
        with zipfile.ZipFile(RAW / filename) as archive:
            for member, table in [
                ("peptides.txt", "peptides"),
                ("modificationSpecificPeptides.txt", "forms"),
            ]:
                frame = pd.read_csv(
                    io.BytesIO(archive.read(member)), sep="\t", dtype=str, keep_default_na=False
                )
                originals[table] = frame
                rows = {
                    row["source_id_raw"]: row for row in typed[table] if row["run_id"] == run_id
                }
                assert len(rows) == len(frame) == frame["id"].nunique()
                for original in frame.to_dict(orient="records"):
                    projected = rows[original["id"]]
                    record = json.loads(records[projected["source_record_id"]]["source_row_json"])
                    assert dict(zip(record["headers"], record["cells"], strict=True)) == original
                    n_cells += len(original)
                    for raw_key, typed_key in [
                        ("Sequence", "peptide_sequence"),
                        ("Intensity", "quantitative_raw"),
                        ("PEP", "pep_raw"),
                        ("Reverse", "reverse_raw"),
                        ("Potential contaminant", "contaminant_raw"),
                    ]:
                        assert original[raw_key] == projected[typed_key]
                    assert float(original["Intensity"]) == projected["quantitative_value"]
        candidate_sequences = {
            row["sequence"]
            for row in membership
            if row["archive"] == filename and row["planned_positive_quality"] == "True"
        }
        frame = originals["forms"]
        eligible = frame[
            frame.Sequence.isin(candidate_sequences)
            & (frame.Modifications == "Unmodified")
            & (frame.Intensity.astype(float) > 0)
            & (frame.PEP.astype(float) < 0.01)
            & (frame.PEP.astype(float) >= 0)
            & (frame.Reverse == "")
            & (frame["Potential contaminant"] == "")
        ]
        assert len(eligible) == n_expected
        admitted = [row for row in examples if row["run_id"] == run_id]
        assert len(admitted) == len(eligible)
        observed = {
            row["Sequence"]: float(row["Intensity"]) for row in eligible.to_dict(orient="records")
        }
        assert set(observed) == {row["sequence"] for row in admitted}
        center = math.fsum(math.log10(value) for value in observed.values()) / len(observed)
        for row in admitted:
            assert row["pool_name"] == pool
            assert abs(row["log10_signal"] - math.log10(observed[row["sequence"]])) < 1e-12
            assert (
                abs(
                    row["target_centered_log10_signal"]
                    - (math.log10(observed[row["sequence"]]) - center)
                )
                < 1e-12
            )
        pool_results.append(
            {
                "pool": pool,
                "run_id": run_id,
                "n_original_peptides": len(originals["peptides"]),
                "n_original_forms": len(originals["forms"]),
                "n_admitted_products": len(admitted),
                "run_center_audit_only": center,
            }
        )
    manifest = json.loads((OUTPUT / "source/manifest.json").read_text())
    assert manifest["reconciliation"]["n_planned_products_rows"] == 305331
    for name in (
        "summary",
        "examples",
        "dispositions",
        "planned_dispositions",
        "dependence_groups",
        "contract",
    ):
        shutil.copy2(OUTPUT / "recipe" / f"{name}.json", ROOT / "results" / f"{name}.json")
    for kind in ("source", "recipe"):
        shutil.copy2(OUTPUT / kind / "manifest.json", ROOT / "results" / f"{kind}-manifest.json")
    elapsed = time.perf_counter() - started
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3
    result = {
        "status": "passed",
        "n_original_literal_cells_verified": n_cells,
        "pools": pool_results,
        "source_build_verify_seconds": source_seconds,
        "canonical_build_verify_seconds": build_seconds,
        "total_seconds": elapsed,
        "peak_process_rss_gib": rss,
        "source_manifest_sha256": digest(OUTPUT / "source/manifest.json"),
        "recipe_manifest_sha256": digest(OUTPUT / "recipe/manifest.json"),
        "n_source_bytes": sum(
            item["size_bytes"]
            for item in json.loads((OUTPUT / "source" / source.ROOT / "assets.json").read_text())[
                "assets"
            ]
        ),
    }
    write(ROOT / "results/independent-verification.json", result)
    assert rss <= 3, "Exceeded prospective 3 GiB process cap"
    write(
        ROOT / "results/frozen-code-hashes.json",
        {
            str(p.relative_to(ROOT / "reproduce")): digest(p)
            for p in sorted((ROOT / "reproduce/source").rglob("*"))
            if p.is_file()
        },
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
