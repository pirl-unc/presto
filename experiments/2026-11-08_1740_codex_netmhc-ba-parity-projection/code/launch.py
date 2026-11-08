"""Project pinned public NetMHC BA archives into source-faithful TRAIN rows."""

from __future__ import annotations

from collections import Counter
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import tarfile
import time

import modal


FID = "2026-11-08_1740_codex_netmhc-ba-parity-projection"
AUDIT_FID = "2026-11-08_0542_codex_baseline-training-admission-audit"
ARCHIVES = {
    "netmhcpan42": {
        "path": "2026-11-07_1645_codex_unified-data-audit/audit/sources/netmhcpan42.tar.gz",
        "bytes": 269075958,
        "sha256": "243a3075887fef3c975da232a87e8d1e17b1aafbf7dba4e52180361d657b5d14",
        "pseudo_member": "NetMHCpan_train/pseudoseqs",
        "source": "NetMHCpan-4.2-training/BA",
    },
    "netmhciipan43": {
        "path": "2026-11-07_1645_codex_unified-data-audit/audit/sources/netmhciipan43.tar.gz",
        "bytes": 139440263,
        "sha256": "b3602edd48fd34a2d4dcd574bea7781bb75c50f9f4dbc3a2d7178301b735c521",
        "pseudo_member": "NetMHCIIpan_train/pseudosequence.2023.dat",
        "source": "NetMHCIIpan-4.3-training/BA",
    },
}
CURRENT = {
    "train": {
        "path": "2026-11-07_1720_codex_affinity-development/prepared/concentrations-s42-train.jsonl",
        "sha256": "acc22d1481ce8e0b45a6df86ca9a62e70721752955070b7eb26b85e48361635d",
    },
    "development": {
        "path": "2026-11-07_1720_codex_affinity-development/prepared/concentrations-s42-validation.jsonl",
        "sha256": "08e41a6c006274b5929142fc6bd4687a289a08df9cc5212edd92399f907f2f8a",
    },
}
AUDIT_INPUTS = {
    "summary": {
        "path": f"{AUDIT_FID}/audit-phase2-continuation/summary.json",
        "sha256": "b4f86522dd5fb0a22a90261d979613dec03d17c229f4a3a146410f563b9cd91f",
    },
    "admission": {
        "path": f"{AUDIT_FID}/audit-phase2-continuation/admission-decisions.json",
        "sha256": "84fb6dfeec2e6c64f227721d7d99236e2079038f8fc9efc56d6b3e451c499d0a",
    },
}
CATALOG = {
    "path": f"{FID}/inputs/mhc-full-seqs.csv",
    "bytes": 61469362,
    "sha256": "0e26d38ed0c572692a3486e100272c56bd1134ce74d2b6122fa4be2f4c6b0a82",
}
FOLD = re.compile(r"c00([0-4])_ba$")

app = modal.App("presto-netmhc-ba-parity-projection-20261108")
volume = modal.Volume.from_name("presto-checkpoints")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("mhcseqs==2.6.13")
    .env({"PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": "/bundle"})
)
if modal.is_local():
    FAMILY = Path(
        os.environ.get("PRESTO_EXPERIMENT_DIR", Path(__file__).resolve().parents[1])
    ).resolve()
    ROOT = FAMILY.parents[1]
    image = (
        image.add_local_file(FAMILY / "reproduce/launch.json", "/bundle/launch.json")
        .add_local_file(ROOT / "data/netmhc_training.py", "/bundle/netmhc_training.py")
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


def load_jsonl_identity(path: Path):
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            # Outcome/value/qualifier fields are intentionally not accessed.
            yield {"peptide": row["peptide"], "mhc": row["mhc"]}


def parse_identity_asset(raw: bytes) -> set[str]:
    result = set()
    for line in raw.decode("utf-8").splitlines():
        fields = line.split()
        if len(fields) != 2 or fields[0] in result:
            raise ValueError("malformed or duplicate archive pseudosequence asset")
        result.add(fields[0])
    if not result:
        raise ValueError("empty archive pseudosequence asset")
    return result


def human_resolver():
    import mhcseqs

    lookup = {}
    for row in mhcseqs.load_sequences_dict():
        record = {
            "allele": str(row.get("two_field_allele") or row.get("representative_allele") or ""),
            "mhc_class": str(row.get("mhc_class") or row.get("class") or "").upper(),
            "chain": str(row.get("chain") or "").lower(),
            "groove1": str(row.get("groove1") or "").upper(),
            "groove2": str(row.get("groove2") or "").upper(),
        }
        for token in (row.get("two_field_allele"), row.get("representative_allele")):
            if token:
                lookup[str(token)] = record
                lookup[str(token).upper()] = record

    def one(token):
        candidates = [token, token.upper()]
        try:
            normalized = mhcseqs.normalize_allele_name(token)
        except Exception:  # noqa: BLE001 - upstream forwards parser-specific errors
            normalized = ""
        if normalized:
            candidates.extend((normalized, normalized.upper()))
        return next((lookup[value] for value in candidates if value in lookup), None)

    dra = one("DRA1_0101")
    if dra is None or not dra["groove1"]:
        raise ValueError("frozen catalog lacks canonical human DRA")

    def resolve(token):
        if token.startswith("HLA-D") and "-D" in token[4:]:
            alpha_token, beta_token = token[4:].split("-", 1)
            chains = [one(alpha_token), one(beta_token)]
        else:
            record = one(token)
            if record and record["mhc_class"] == "I":
                chains = [record]
            elif record and record["mhc_class"] == "II" and record["chain"] == "beta":
                chains = [dra, record] if "DRB" in record["allele"] else [None, record]
            else:
                chains = [record]
        if any(chain is None for chain in chains):
            return None
        if len(chains) == 1:
            record = chains[0]
            if (
                record["mhc_class"] != "I"
                or not record["groove1"]
                or not record["groove2"]
                or not record["allele"].startswith("HomoSapiens-")
            ):
                return None
            return {
                "catalog_alleles": [record["allele"]],
                "groove1": record["groove1"],
                "groove2": record["groove2"],
                "mhc_class": "I",
            }
        alpha, beta = chains
        if (
            alpha["mhc_class"] != "II"
            or beta["mhc_class"] != "II"
            or alpha["chain"] != "alpha"
            or beta["chain"] != "beta"
            or not alpha["groove1"]
            or not beta["groove2"]
            or not alpha["allele"].startswith("HomoSapiens-")
            or not beta["allele"].startswith("HomoSapiens-")
        ):
            return None
        return {
            "catalog_alleles": [alpha["allele"], beta["allele"]],
            "groove1": alpha["groove1"],
            "groove2": beta["groove2"],
            "mhc_class": "II",
        }

    return resolve


def artifact(path: Path) -> dict:
    return {
        "bytes": path.stat().st_size,
        "path": str(path.relative_to("/volume")),
        "sha256": sha256(path),
    }


@app.function(
    image=image,
    cpu=8,
    memory=32768,
    ephemeral_disk=131072,
    timeout=3600,
    max_containers=1,
    retries=0,
    single_use_containers=True,
    block_network=True,
    volumes={"/volume": volume},
)
def project() -> dict:
    import resource
    from netmhc_training import (
        ArchiveBARow,
        current_identity_sets,
        parse_ba_line,
        project_ba_rows,
    )

    started = time.monotonic()
    contract = json.loads(Path("/bundle/launch.json").read_text())
    assert sha256("/bundle/netmhc_training.py") == contract["source_sha256"][
        "data/netmhc_training.py"
    ]
    volume.reload()
    input_specs = [*ARCHIVES.values(), *CURRENT.values(), *AUDIT_INPUTS.values(), CATALOG]
    for spec in input_specs:
        path = Path("/volume") / spec["path"]
        if "bytes" in spec:
            assert path.stat().st_size == spec["bytes"]
        assert sha256(path) == spec["sha256"]
    audit_summary = json.loads((Path("/volume") / AUDIT_INPUTS["summary"]["path"]).read_text())
    decisions = json.loads((Path("/volume") / AUDIT_INPUTS["admission"]["path"]).read_text())
    assert audit_summary["status"] == "completed"
    assert {
        (row["archive"], row["decision"])
        for row in decisions
        if row["modality"] == "ba"
    } == {
        ("netmhcpan42", "source-faithful measured output"),
        ("netmhciipan43", "source-faithful measured output"),
    }

    cache = Path("/root/.cache/mhcseqs/mhc-full-seqs.csv")
    cache.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Path("/volume") / CATALOG["path"], cache)
    assert sha256(cache) == CATALOG["sha256"]
    resolve = human_resolver()

    train_peptides, train_cores, train_pmhc = current_identity_sets(
        load_jsonl_identity(Path("/volume") / CURRENT["train"]["path"])
    )
    development_peptides, development_cores, development_pmhc = current_identity_sets(
        load_jsonl_identity(Path("/volume") / CURRENT["development"]["path"])
    )
    if train_peptides & development_peptides or train_cores & development_cores:
        raise ValueError("frozen current TRAIN/development peptide embargo changed")

    source_rows = []
    pseudosequence_counts = {}
    scratch = Path("/scratch") / FID
    scratch.mkdir(parents=True, exist_ok=False)
    for archive_name, spec in ARCHIVES.items():
        local = scratch / f"{archive_name}.tar.gz"
        shutil.copyfile(Path("/volume") / spec["path"], local)
        assert local.stat().st_size == spec["bytes"] and sha256(local) == spec["sha256"]
        with tarfile.open(local) as archive:
            pseudosequences = parse_identity_asset(archive.extractfile(spec["pseudo_member"]).read())
            pseudosequence_counts[archive_name] = len(pseudosequences)
            for member in sorted(archive.getmembers(), key=lambda value: value.name):
                match = FOLD.match(Path(member.name).name)
                if not member.isfile() or match is None:
                    continue
                stream = archive.extractfile(member)
                if stream is None:
                    raise ValueError(f"cannot read {member.name}")
                for line_number, raw in enumerate(stream, start=1):
                    if not raw.strip():
                        continue
                    row = parse_ba_line(
                        raw,
                        archive=archive_name,
                        member=member.name,
                        fold=int(match.group(1)),
                        line_number=line_number,
                    )
                    if isinstance(row, ArchiveBARow) and row.identity not in pseudosequences:
                        raise ValueError("BA row identity absent from archive pseudosequences")
                    source_rows.append(row)

    projected = project_ba_rows(
        source_rows,
        resolve_identity=resolve,
        archive_hashes={name: spec["sha256"] for name, spec in ARCHIVES.items()},
        source_names={name: spec["source"] for name, spec in ARCHIVES.items()},
        development_peptides=development_peptides,
        development_cores=development_cores,
        train_pmhc=train_pmhc,
    )
    output = Path("/volume") / FID / "prepared"
    output.mkdir(parents=True, exist_ok=False)
    by_archive = {
        name: [row for row in projected.observations if row["provenance"]["archive"] == name]
        for name in ARCHIVES
    }
    paths = {
        "netmhcpan42-ba-train": output / "netmhcpan42-ba-train.jsonl",
        "netmhciipan43-ba-train": output / "netmhciipan43-ba-train.jsonl",
        "source-row-dispositions": output / "source-row-dispositions.jsonl.gz",
        "source-target-conflicts": output / "source-target-conflicts.jsonl.gz",
        "source-exact-duplicates": output / "source-exact-duplicates.jsonl.gz",
    }
    for name, rows in by_archive.items():
        write_jsonl(paths[f"{name}-ba-train"], rows)
    write_gzip_jsonl(paths["source-row-dispositions"], projected.dispositions)
    write_gzip_jsonl(paths["source-target-conflicts"], projected.conflicts)
    write_gzip_jsonl(paths["source-exact-duplicates"], projected.duplicates)

    dispositions = Counter(row["disposition"] for row in projected.dispositions)
    qualifiers = Counter(row["qualifier"] for row in projected.observations)
    classes = Counter(row["mhc"][0]["mhc_class"] for row in projected.observations)
    archive_outputs = Counter(row["provenance"]["archive"] for row in projected.observations)
    summary = {
        "status": "completed",
        "experiment": FID,
        "policy": (
            "TRAIN-only comparator-parity projection; source semantics and censoring preserved; "
            "current outcomes, model, predictions, TEST and locked confirmation never accessed"
        ),
        "inputs": {
            "archives": ARCHIVES,
            "audit": AUDIT_INPUTS,
            "catalog": CATALOG,
            "current_identity_only": CURRENT,
        },
        "source_rows": len(source_rows),
        "dispositions": dict(sorted(dispositions.items())),
        "output_rows": len(projected.observations),
        "output_rows_by_archive": dict(sorted(archive_outputs.items())),
        "output_rows_by_class": dict(sorted(classes.items())),
        "output_rows_by_qualifier": dict(sorted(qualifiers.items())),
        "output_peptide_lengths": dict(
            sorted(Counter(len(row["peptide"]) for row in projected.observations).items())
        ),
        "output_archive_identities": {
            name: len({row["provenance"]["archive_identity"] for row in rows})
            for name, rows in by_archive.items()
        },
        "source_target_conflict_keys": len(projected.conflicts),
        "source_exact_duplicate_keys": len(projected.duplicates),
        "archive_pseudosequences": pseudosequence_counts,
        "current_identity": {
            "train": {
                "peptides": len(train_peptides),
                "cores9": len(train_cores),
                "pmhc": len(train_pmhc),
            },
            "development": {
                "peptides": len(development_peptides),
                "cores9": len(development_cores),
                "pmhc": len(development_pmhc),
            },
        },
        "assay_label_output_mapping": {
            "archive_ba": "ic50/reference with unknown preparation, mechanism and detector",
            "target": "IC50_nM=50000**(1-y); y=0 => >=50000; y=1 => <=1",
        },
        "split": {
            "train": len(projected.observations),
            "validation": 0,
            "test": 0,
            "test_absence_reason": (
                "Comparator training archives overlap known data and expose no source-study IDs; "
                "they cannot define an independent test."
            ),
        },
        "synthetic_data": None,
        "loss_terms": None,
        "artifacts": {},
        "resources": {
            "requested_gpu": None,
            "observed_gpu": None,
            "cpu": 8,
            "memory_mib": 32768,
            "elapsed_seconds": time.monotonic() - started,
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        },
        "evaluation": {
            "model_loaded": False,
            "predictions": False,
            "current_outcomes_read": False,
            "test_accessed": False,
            "locked_confirmation_accessed": False,
        },
    }
    for name, path in paths.items():
        summary["artifacts"][name] = artifact(path)
    summary_path = output / "summary.json"
    write_json(summary_path, summary)
    volume.commit()
    return {
        "status": "completed",
        "summary_path": str(summary_path.relative_to("/volume")),
        "summary_bytes": summary_path.stat().st_size,
        "summary_sha256": sha256(summary_path),
        "artifacts": summary["artifacts"],
        "resources": summary["resources"],
    }


@app.function(
    image=image,
    cpu=4,
    memory=8192,
    timeout=1800,
    max_containers=1,
    retries=0,
    single_use_containers=True,
    block_network=True,
    volumes={"/volume": volume},
)
def verify_projection() -> dict:
    volume.reload()
    summary_path = Path("/volume") / FID / "prepared/summary.json"
    summary = json.loads(summary_path.read_text())
    assert summary["status"] == "completed"
    for spec in summary["artifacts"].values():
        path = Path("/volume") / spec["path"]
        assert path.stat().st_size == spec["bytes"] and sha256(path) == spec["sha256"]
    rows = Counter()
    qualifiers = Counter()
    observation_ids = set()
    for archive in ARCHIVES:
        path = Path("/volume") / summary["artifacts"][f"{archive}-ba-train"]["path"]
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                assert row["schema"] == "presto.modular.observation.v1"
                assert row["split"] == "train" and row["endpoint"] == "ic50"
                assert row["assay_method"] == {
                    "detector": "unknown",
                    "mechanism": "unknown",
                    "preparation": "unknown",
                    "review_status": "unknown",
                }
                assert row["provenance"]["assay_output"] == "unspecified_assay/IC50"
                assert row["qualifier"] in ("=", "<=", ">=")
                if row["qualifier"] == "<=":
                    assert row["value"] == 1
                elif row["qualifier"] == ">=":
                    assert row["value"] == 50000
                assert row["observation_id"] not in observation_ids
                observation_ids.add(row["observation_id"])
                rows[archive] += 1
                qualifiers[row["qualifier"]] += 1
    assert sum(rows.values()) == summary["output_rows"]
    assert dict(sorted(rows.items())) == summary["output_rows_by_archive"]
    assert dict(sorted(qualifiers.items())) == summary["output_rows_by_qualifier"]
    assert sum(summary["dispositions"].values()) == summary["source_rows"]
    assert summary["evaluation"] == {
        "current_outcomes_read": False,
        "locked_confirmation_accessed": False,
        "model_loaded": False,
        "predictions": False,
        "test_accessed": False,
    }
    receipt = {
        "status": "passed",
        "summary_path": str(summary_path.relative_to("/volume")),
        "summary_sha256": sha256(summary_path),
        "output_rows": sum(rows.values()),
        "output_rows_by_archive": dict(sorted(rows.items())),
        "output_rows_by_qualifier": dict(sorted(qualifiers.items())),
        "unique_observation_ids": len(observation_ids),
        "model_prediction_test_or_locked_access": False,
    }
    path = Path("/volume") / FID / "prepared/verification.json"
    write_json(path, receipt)
    volume.commit()
    return {**receipt, "verification": artifact(path)}


@app.local_entrypoint()
def main(verify: bool = False) -> None:
    contract = json.loads((FAMILY / "reproduce/launch.json").read_text())
    assert sha256(FAMILY / "code/launch.py") == contract["source_sha256"]["code/launch.py"]
    name = "verify-call.json" if verify else "call.json"
    call_path = FAMILY / "reproduce" / name
    if call_path.exists():
        raise ValueError(f"{name} already exists")
    call = (verify_projection if verify else project).spawn()
    write_json(call_path, {"call_id": call.object_id, "submitted_unix": time.time()})
    print(call.object_id, flush=True)
