"""Replay frozen IC50 labels through the canonical census, without a model."""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import presto
import torch
from datacache import install_bundle, materialize
from presto.data.collate import PrestoCollator, PrestoSample
from presto.training.output_contract import OutputConfiguration, build_output_contract
from presto.training.output_coverage import OutputCoverageCensus
from presto.training.supervision import LOSS_TASK_SPECS

FAMILY = Path(__file__).resolve().parents[1]
REPO = FAMILY.parents[1]
PINS = {
    "ic50-raw-effective-targets.tsv": {
        "url": "https://github.com/pirl-unc/presto/releases/download/corrected-human-target-policy-audit-v1/ic50-raw-effective-targets.tsv",
        "sha256": "59c73feeec2ba00041fcc3f8a91b6701ef31bd3f5d3ce6db01de188c45f44583",
        "size": 3468131,
    },
    "archived-supported-outputs.json": {
        "url": "https://github.com/pirl-unc/presto/releases/download/corrected-human-preflight-v1/prospective-supported-outputs.json",
        "sha256": "13fe8bd4dab158bd06ae31fc7996fba98f73b3944fa68c3f57463beec34e7cb0",
        "size": 3346,
    },
}
QUALIFIERS = {"": 0, "=": 0, "<": -1, "<=": -1, "≤": -1, ">": 1, ">=": 1, "≥": 1}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def pin(path):
    data = Path(path).read_bytes()
    return {"sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}


def git(*args):
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def make_sample(row):
    fields = (
        "sample_id",
        "evidence_row_id",
        "sample_source",
        "source_value",
        "source_units",
        "source_endpoint",
        "source_inequality",
        "pmid",
        "doi",
        "reference_iri",
        "assay_iri",
        "source_bundle_id",
        "source_partition",
        "provenance_id",
    )
    kwargs = {field: row[field] for field in fields}
    for field in ("source_mhc_alleles", "resolved_mhc_alleles"):
        kwargs[field] = tuple(json.loads(row[field]))
    kwargs["target_provenance"] = json.loads(row["target_provenance"])
    return PrestoSample(
        **kwargs,
        peptide=row["peptide"],
        mhc_class=row["native_mhc_class"],
        bind_value=float(row["source_value"]),
        bind_qual=QUALIFIERS[row["source_inequality"]],
        bind_measurement_type="IC50",
        binding_assay_type="IC50",
        binding_assay_method=row["native_assay_method"] or None,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-root", type=Path, default=Path.home() / ".hitlist/presto-affinity-target-policy"
    )
    args = parser.parse_args()
    assert presto.__version__ == "0.1.16"
    assert Path(presto.__file__).resolve().parent == REPO
    assert not git("status", "--porcelain"), "Run the frozen recipe from a clean checkout"
    producer = {
        "source_commit": git("rev-parse", "HEAD"),
        "dirty": False,
        "presto_version": presto.__version__,
        "source_checkout": str(REPO),
        "imported_module": str(Path(presto.__file__).resolve()),
        "dependencies": {
            p: importlib.metadata.version(p)
            for p in ("torch", "numpy", "pandas", "pyarrow", "mhcseqs", "datacache")
        },
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "gpu": None,
        "device": "cpu",
        "torch_threads": 1,
        "n_workers": 0,
        "batch_size": 128,
        "timeout_seconds": 600,
    }

    def timeout(*_):
        raise TimeoutError("600-second audit limit")

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(600)
    torch.set_num_threads(1)
    start = time.monotonic()
    inputs = install_bundle(
        args.cache_root / "inputs/v1", PINS, download_options={"show_progress": False}
    )
    names = (
        "affinity_target_audit.json",
        "affinity_target_changes.tsv",
        "counts.tsv",
        "clipped-source-contexts.tsv",
        "summary.json",
    )

    def builder(source_paths, output_paths):
        with Path(source_paths["ic50-raw-effective-targets.tsv"]).open() as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        assert len(rows) == 2614 and len({r["sample_id"] for r in rows}) == len(rows)
        assert {r["source_units"] for r in rows} == {"nM"}
        assert {r["source_endpoint"] for r in rows} == {
            "half maximal inhibitory concentration (IC50)"
        }
        manifest = json.loads(Path(source_paths["archived-supported-outputs.json"]).read_text())
        contract = build_output_contract(OutputConfiguration(**manifest["model_contract"]))
        spec = next(s for s in LOSS_TASK_SPECS if s.name == "binding_ic50")
        samples = [make_sample(r) for r in rows]
        effective = spec.target_transform(
            torch.tensor([s.bind_value for s in samples], dtype=torch.float32)
        )
        for row, sample, value in zip(rows, samples, effective.tolist()):
            assert math.isclose(
                value, float(row["effective_target_log10_nM_float32"]), abs_tol=1e-7
            )
            assert (
                sample.bind_qual
                == {"exact": 0, "left_censored": -1, "right_censored": 1}[row["label_kind"]]
            )
        with (
            OutputCoverageCensus(contract) as census,
            tempfile.TemporaryDirectory(prefix="affinity-audit-") as tmp,
        ):
            for split in ("train", "val", "test"):
                selected = [s for s in samples if s.source_partition == split]
                for offset in range(0, len(selected), 128):
                    batch = selected[offset : offset + 128]
                    census.add_batch(split, batch, PrestoCollator()(batch))
            paths = census.affinity_target_audit.write_artifacts(tmp, census.splits)
            for key, path in paths.items():
                shutil.copyfile(
                    path, output_paths[key + (".json" if key.endswith("audit") else ".tsv")]
                )
            audit = census.affinity_target_audit.report(census.splits)
        counts = [
            r
            for r in audit["counts"]
            if r["objective_id"] == "row:binding_ic50" and not r["column"]
        ]
        assert [
            (
                r["n_observations"],
                r["n_clipped_observations"],
                r["n_clipped_exact_observations"],
                r["n_clipped_right_censored_observations"],
            )
            for r in counts
        ] == [(2074, 157, 39, 118), (265, 31, 8, 23), (275, 23, 7, 16)]
        with Path(output_paths["affinity_target_changes.tsv"]).open() as handle:
            changed = [
                r
                for r in csv.DictReader(handle, delimiter="\t")
                if r["objective_id"] == "row:binding_ic50"
            ]
        expected = {r["sample_id"]: r for r in rows if r["physical_value_clipped"] == "True"}
        assert {r["sample_id"] for r in changed} == set(expected) and len(changed) == 211
        for row in changed:
            source = expected[row["sample_id"]]
            for field in (
                "evidence_row_id",
                "source_value",
                "source_units",
                "source_inequality",
                "pmid",
                "doi",
                "reference_iri",
                "assay_iri",
            ):
                assert row[field] == source[field]
            assert row["split"] == source["source_partition"]
            assert int(row["qualifier"]) == QUALIFIERS[source["source_inequality"]]
            assert float(row["effective_target_log10_nM"]) == float(
                source["effective_target_log10_nM_float32"]
            )
        with Path(output_paths["counts.tsv"]).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(audit["counts"][0]), delimiter="\t")
            writer.writeheader()
            writer.writerows(audit["counts"])
        with Path(output_paths["clipped-source-contexts.tsv"]).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
            writer.writeheader()
            writer.writerows(
                sorted(expected.values(), key=lambda r: (r["source_partition"], r["sample_id"]))
            )
        summary = {
            "status": "canonical label audit matches the independent published target audit",
            "producer": producer,
            "input_pins": PINS,
            "direct_ic50_counts": counts,
            "n_verified_observations": len(rows),
            "n_clipped_observations": len(changed),
            "n_clipped_exact_observations": sum(r["n_clipped_exact_observations"] for r in counts),
            "n_predictions": 0,
            "n_checkpoints": 0,
            "n_optimization_steps": 0,
            "wall_seconds_including_input_acquisition": time.monotonic() - start,
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * (1 if sys.platform == "darwin" else 1024),
            "limitations": [
                "Label-only samples omit MHC sequences and do not requalify complete model inputs or auxiliary-label coverage.",
                "The archived supported-output manifest supplies model configuration only; it is not rebound or qualified.",
                "Objective and panel exposures are not independent source observations.",
                "Unknown donor/specimen/acquisition lineage and shared studies remain unchanged.",
                "No model, predictive evaluation or training-readiness claim is produced.",
            ],
        }
        write_json(output_paths["summary.json"], summary)

    output_paths = materialize(
        args.cache_root / "audits" / producer["source_commit"],
        {
            name: {"path": str(path), "sha256": PINS[name]["sha256"]}
            for name, path in inputs.items()
        },
        transform={
            "version": "canonical-affinity-target-policy-audit-v1",
            "options": {"producer": producer, "code": pin(__file__)},
        },
        outputs={name: {} for name in names},
        builder=builder,
    )
    for name, path in output_paths.items():
        shutil.copyfile(path, FAMILY / "results" / name)
    receipt = {
        "producer": producer,
        "inputs": PINS,
        "artifacts": {
            name: {"path": str(path), **pin(path)} for name, path in output_paths.items()
        },
    }
    write_json(FAMILY / "results/receipt.json", receipt)
    write_json(
        FAMILY / "reproduce/launch.json",
        {
            "argv": sys.argv,
            "environment": {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
            "producer": producer,
        },
    )
    signal.alarm(0)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
