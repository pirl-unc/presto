"""Add canonical lineage and complete Stage B development embargoes to BA TRAIN."""

from __future__ import annotations

from collections import Counter
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import resource
import time

import modal


FID = "2026-11-08_1740_codex_netmhc-ba-parity-projection"
PRESENTATION_FID = "2026-11-07_1915_codex_unified-presentation"
V1 = {
    "netmhcpan42": {
        "path": f"{FID}/prepared/netmhcpan42-ba-train.jsonl",
        "bytes": 114_297_607,
        "sha256": "4491918763c1cf1c64825af0228357f43516293d1b6ea73ef8a6a7554d4635df",
    },
    "netmhciipan43": {
        "path": f"{FID}/prepared/netmhciipan43-ba-train.jsonl",
        "bytes": 91_225_030,
        "sha256": "153a51f8997093c0b8cf724dc8ff649a676ee3ee2e516c52dff810bb18d00ee2",
    },
    "summary": {
        "path": f"{FID}/prepared/summary.json",
        "bytes": 5_622,
        "sha256": "9411e528a8b5ce7a4aa527835d6a2750d9cfc1b36198e711ef3241f064002576",
    },
}
PRESENTATION_DEVELOPMENT = {
    "path": f"{PRESENTATION_FID}/prepared-v2/presentation-val.jsonl",
    "bytes": 218_542_980,
    "sha256": "0313b2d6b394320da7dcf6aa9144cbc44b73abe48b16380e03c1b27a9559fc52",
}
ARCHIVE_HASHES = {
    "netmhcpan42": "243a3075887fef3c975da232a87e8d1e17b1aafbf7dba4e52180361d657b5d14",
    "netmhciipan43": "b3602edd48fd34a2d4dcd574bea7781bb75c50f9f4dbc3a2d7178301b735c521",
}
REQUIREMENTS = (
    "torch==2.14.1",
    "numpy==2.5.4",
    "pyyaml==6.0.3",
    "mhcgnomes==3.64.6",
    "mhcseqs==2.6.13",
    "tqdm==4.70.1",
)

app = modal.App("presto-netmhc-ba-projection-v2-20261108")
volume = modal.Volume.from_name("presto-checkpoints")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(*REQUIREMENTS)
    .env({"PYTHONPATH": "/runtime:/bundle", "PYTHONDONTWRITEBYTECODE": "1"})
)
if modal.is_local():
    FAMILY = Path(
        os.environ.get("PRESTO_EXPERIMENT_DIR", Path(__file__).resolve().parents[1])
    ).resolve()
    ROOT = FAMILY.parents[1]
    image = (
        image.add_local_file(FAMILY / "reproduce/v2/launch.json", "/bundle/launch.json")
        .add_local_file(FAMILY / "code/reproject_v2.py", "/bundle/reproject_v2.py")
        .add_local_file(ROOT / "data/netmhc_training.py", "/bundle/netmhc_training.py")
        .add_local_file(ROOT / "__init__.py", "/runtime/presto/__init__.py")
        .add_local_dir(ROOT / "data", "/runtime/presto/data")
    )


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def write_jsonl(path: Path, rows) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False))
            stream.write("\n")
    temporary.replace(path)


def write_gzip_jsonl(path: Path, rows) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8") as stream:
                for row in rows:
                    stream.write(
                        json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False)
                    )
                    stream.write("\n")
    temporary.replace(path)


def artifact(path: Path) -> dict:
    return {
        "path": str(path.relative_to("/volume")),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def presentation_embargo(path: Path):
    from netmhc_training import peptide_cores, peptide_deletion_masks

    peptides, cores8, masks = set(), set(), set()
    rows = 0
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            peptide = str(row["peptide"])
            peptides.add(peptide)
            cores8.update(peptide_cores(peptide, 8))
            masks.update(peptide_deletion_masks(peptide))
            rows += 1
    return peptides, cores8, masks, rows


def project_row(row: dict, peptides: set, cores8: set, masks: set):
    from netmhc_training import archive_measurement_id, peptide_cores, peptide_deletion_masks

    peptide = str(row["peptide"])
    if peptide in peptides:
        return None, "presentation_development_exact_peptide_embargo"
    if any(core in cores8 for core in peptide_cores(peptide, 8)):
        return None, "presentation_development_core8_embargo"
    if any(mask in masks for mask in peptide_deletion_masks(peptide)):
        return None, "presentation_development_hamming1_embargo"
    result = json.loads(json.dumps(row))
    provenance = result["provenance"]
    archive = str(provenance["archive"])
    if provenance["archive_sha256"] != ARCHIVE_HASHES[archive]:
        raise ValueError("v1 row archive hash changed")
    provenance["measurement_lineage"] = {
        "namespace": provenance["archive_sha256"],
        "measurement_id": archive_measurement_id(
            archive,
            peptide,
            str(provenance["archive_identity"]),
            str(provenance["archive_context"]),
        ),
        "replicate_id": "",
    }
    provenance["projection_continuation"] = {
        "parent": "prepared-v1",
        "policy": "canonical_measurement_lineage_plus_stage_b_presentation_development_embargo",
    }
    return result, "retained_v2_train"


@app.function(
    image=image,
    cpu=4,
    memory=16384,
    timeout=3600,
    max_containers=1,
    retries=0,
    single_use_containers=True,
    block_network=True,
    volumes={"/volume": volume},
)
def reproject_v2() -> dict:
    from presto.data.modular import Observation
    from presto.data.modular_lineage import measurement_key

    started = time.monotonic()
    contract = json.loads(Path("/bundle/launch.json").read_text())
    for name, path in (
        ("code/reproject_v2.py", "/bundle/reproject_v2.py"),
        ("data/netmhc_training.py", "/bundle/netmhc_training.py"),
    ):
        if sha256(path) != contract["source_sha256"][name]:
            raise ValueError(f"source differs from frozen contract: {name}")
    volume.reload()
    for spec in (*V1.values(), PRESENTATION_DEVELOPMENT):
        path = Path("/volume") / spec["path"]
        if path.stat().st_size != spec["bytes"] or sha256(path) != spec["sha256"]:
            raise ValueError(f"input changed: {spec['path']}")
    parent = json.loads((Path("/volume") / V1["summary"]["path"]).read_text())
    if parent["status"] != "completed" or parent["output_rows"] != 143_307:
        raise ValueError("v1 parent projection contract changed")
    peptides, cores8, masks, development_rows = presentation_embargo(
        Path("/volume") / PRESENTATION_DEVELOPMENT["path"]
    )

    output = Path("/volume") / FID / "prepared-v2"
    output.mkdir(parents=True, exist_ok=False)
    dispositions, outputs = [], {}
    disposition_counts, class_counts, qualifier_counts = Counter(), Counter(), Counter()
    for archive in ("netmhcpan42", "netmhciipan43"):
        admitted = []
        with (Path("/volume") / V1[archive]["path"]).open() as stream:
            for line in stream:
                original = json.loads(line)
                row, disposition = project_row(original, peptides, cores8, masks)
                disposition_counts[disposition] += 1
                dispositions.append(
                    {
                        "archive": archive,
                        "observation_id": original["observation_id"],
                        "disposition": disposition,
                    }
                )
                if row is None:
                    continue
                observation = Observation.from_dict(row)
                if measurement_key(observation) is None:
                    raise ValueError("v2 row lacks canonical measurement lineage")
                admitted.append(row)
                class_counts[row["mhc"][0]["mhc_class"]] += 1
                qualifier_counts[row["qualifier"]] += 1
        path = output / f"{archive}-ba-train.jsonl"
        write_jsonl(path, admitted)
        outputs[archive] = artifact(path)
    disposition_path = output / "v1-admitted-row-dispositions.jsonl.gz"
    write_gzip_jsonl(disposition_path, dispositions)
    outputs["v1-admitted-row-dispositions"] = artifact(disposition_path)

    summary = {
        "schema": "presto.netmhc_ba_projection_continuation.v2",
        "status": "completed",
        "experiment": FID,
        "parent": V1,
        "presentation_development": PRESENTATION_DEVELOPMENT,
        "presentation_development_rows": development_rows,
        "presentation_development_unique_peptides": len(peptides),
        "input_v1_admitted_rows": parent["output_rows"],
        "dispositions": dict(sorted(disposition_counts.items())),
        "output_rows": sum(class_counts.values()),
        "output_rows_by_class": dict(sorted(class_counts.items())),
        "output_rows_by_qualifier": dict(sorted(qualifier_counts.items())),
        "measurement_lineage": {
            "namespace": "source archive SHA-256",
            "measurement_id": "SHA-256(archive, peptide, identity, context)",
            "replicate_id": "empty; unavailable in archive",
        },
        "development_embargo": {
            "affinity": "inherited v1 exact peptide plus shared core9",
            "presentation": "v2 exact peptide plus shared core8 plus same-length Hamming1",
        },
        "split": {
            "train": sum(class_counts.values()),
            "validation": 0,
            "test": 0,
            "test_absence_reason": "This continuation only removes or annotates v1 TRAIN rows.",
        },
        "artifacts": outputs,
        "evaluation": {
            "model_loaded": False,
            "predictions": False,
            "test_accessed": False,
            "locked_confirmation_accessed": False,
        },
        "resources": {
            "requested_gpu": None,
            "observed_gpu": None,
            "cpu": 4,
            "memory_mib": 16384,
            "elapsed_seconds": time.monotonic() - started,
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        },
    }
    if sum(disposition_counts.values()) != parent["output_rows"]:
        raise ValueError("v2 dispositions do not close over v1 admitted rows")
    summary_path = output / "summary.json"
    write_json(summary_path, summary)
    volume.commit()
    return {
        "status": "completed",
        "summary": artifact(summary_path),
        "artifacts": outputs,
        "output_rows": summary["output_rows"],
        "dispositions": summary["dispositions"],
        "resources": summary["resources"],
    }


@app.function(
    image=image,
    cpu=4,
    memory=16384,
    timeout=3600,
    max_containers=1,
    retries=0,
    single_use_containers=True,
    block_network=True,
    volumes={"/volume": volume},
)
def verify_v2() -> dict:
    from netmhc_training import peptide_cores, peptide_deletion_masks
    from presto.data.modular import Observation
    from presto.data.modular_lineage import measurement_key

    volume.reload()
    summary_path = Path("/volume") / FID / "prepared-v2/summary.json"
    summary = json.loads(summary_path.read_text())
    for spec in summary["artifacts"].values():
        path = Path("/volume") / spec["path"]
        if path.stat().st_size != spec["bytes"] or sha256(path) != spec["sha256"]:
            raise ValueError(f"v2 output changed: {spec['path']}")
    peptides, cores8, masks, _ = presentation_embargo(
        Path("/volume") / PRESENTATION_DEVELOPMENT["path"]
    )
    counts, qualifiers, observation_ids, measurement_ids = Counter(), Counter(), set(), set()
    for archive in ("netmhcpan42", "netmhciipan43"):
        path = Path("/volume") / summary["artifacts"][archive]["path"]
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                observation = Observation.from_dict(row)
                measurement = measurement_key(observation)
                if measurement is None or measurement in measurement_ids:
                    raise ValueError("missing or duplicate v2 measurement lineage")
                measurement_ids.add(measurement)
                peptide = row["peptide"]
                if (
                    peptide in peptides
                    or any(core in cores8 for core in peptide_cores(peptide, 8))
                    or any(mask in masks for mask in peptide_deletion_masks(peptide))
                ):
                    raise ValueError("v2 row violates presentation development embargo")
                if row["observation_id"] in observation_ids:
                    raise ValueError("duplicate v2 observation ID")
                observation_ids.add(row["observation_id"])
                counts[row["mhc"][0]["mhc_class"]] += 1
                qualifiers[row["qualifier"]] += 1
    if len(observation_ids) != summary["output_rows"]:
        raise ValueError("v2 output row count mismatch")
    if dict(sorted(counts.items())) != summary["output_rows_by_class"]:
        raise ValueError("v2 class count mismatch")
    if dict(sorted(qualifiers.items())) != summary["output_rows_by_qualifier"]:
        raise ValueError("v2 qualifier count mismatch")
    receipt = {
        "status": "passed",
        "summary_path": str(summary_path.relative_to("/volume")),
        "summary_sha256": sha256(summary_path),
        "output_rows": len(observation_ids),
        "unique_observation_ids": len(observation_ids),
        "unique_measurement_ids": len(measurement_ids),
        "output_rows_by_class": dict(sorted(counts.items())),
        "output_rows_by_qualifier": dict(sorted(qualifiers.items())),
        "presentation_development_embargo_verified": True,
        "model_prediction_test_or_locked_access": False,
    }
    path = Path("/volume") / FID / "prepared-v2/verification.json"
    write_json(path, receipt)
    volume.commit()
    return {**receipt, "verification": artifact(path)}


@app.local_entrypoint()
def main(verify: bool = False) -> None:
    contract = json.loads((FAMILY / "reproduce/v2/launch.json").read_text())
    if sha256(FAMILY / "code/reproject_v2.py") != contract["source_sha256"][
        "code/reproject_v2.py"
    ]:
        raise ValueError("local v2 source differs from frozen contract")
    name = "verify-call.json" if verify else "call.json"
    call_path = FAMILY / "reproduce/v2" / name
    if call_path.exists():
        raise ValueError(f"{name} already exists")
    call = (verify_v2 if verify else reproject_v2).spawn()
    write_json(call_path, {"call_id": call.object_id, "submitted_unix": time.time()})
    print(call.object_id)
