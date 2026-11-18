"""Freeze peptide-only ancestry/validation protection without opening TEST outcomes."""

from collections import defaultdict
import gzip
import json
from pathlib import Path

from .modular import AA
from .modular_sources import file_sha256


def freeze_identities(manifest_path, output_dir):
    path = Path(manifest_path).resolve()
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != "presto.presentation-identities.v1":
        raise ValueError("unsupported identity source manifest")
    sources = manifest.get("sources", [])
    if not sources or any(
        source.get("role") not in {"ancestry", "fixed_validation"}
        or source.get("split") not in {"train", "validation"}
        for source in sources
    ):
        raise ValueError("declare TRAIN ancestry or VALIDATION identities; never TEST outcomes")
    identities = defaultdict(set)
    receipts = []
    for source in sources:
        expected = "train" if source["role"] == "ancestry" else "validation"
        if source["split"] != expected:
            raise ValueError("identity role disagrees with declared split")
        filename = (path.parent / source["path"]).resolve()
        if not source.get("sha256") or file_sha256(filename) != source["sha256"]:
            raise ValueError("identity source hash mismatch")
        kind = source.get("kind", "observations")
        if kind not in {"observations", "candidate_options"}:
            raise ValueError("unsupported identity source kind")
        opener = gzip.open if filename.suffix == ".gz" else open
        count = 0
        with opener(filename, "rt") as stream:
            for line in stream:
                row = json.loads(line)
                if kind == "observations":
                    if row["split"] != expected:
                        raise ValueError("identity source contains an undeclared partition")
                    peptides = [row["peptide"]]
                else:
                    peptides = [option["peptide"] for option in row["options"]]
                for peptide in peptides:
                    if not peptide or set(peptide) - set(AA):
                        raise ValueError("noncanonical protected peptide")
                    identities[source["role"]].add(peptide)
                count += 1
        receipts.append({**source, "rows": count})
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    files = []
    for role, values in sorted(identities.items()):
        filename = output / f"{role}.txt.gz"
        with gzip.open(filename, "wt") as stream:
            stream.write("\n".join(sorted(values)) + "\n")
        files.append(
            {
                "role": role,
                "path": str(filename.resolve()),
                "sha256": file_sha256(filename),
                "peptides": len(values),
            }
        )
    receipt = {
        "manifest_sha256": file_sha256(path),
        "sources": receipts,
        "identities": files,
        "test_outcomes_accessed": False,
    }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt
