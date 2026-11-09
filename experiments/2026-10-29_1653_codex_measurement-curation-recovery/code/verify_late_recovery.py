"""Verify the collected late source batch independently of model admission."""

from pathlib import Path
import csv
import hashlib
import json
import shutil
import sys
import time

F = Path(__file__).resolve().parents[1]
A = Path("/Users/iskander/code/presto/artifacts") / F.name


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    result = json.loads((F / "results/recovery-late.json").read_text())
    assert sha(A / "recovered-late.tar.gz") == result["archive"]["sha256"]
    root = A / "recovery-batch-late/recovered-late"
    for name, field in (
        ("strazar_2023_recovered_unmodified.csv", "output"),
        ("source-evidence.jsonl", "evidence"),
    ):
        assert sha(root / name) == result[field]["sha256"]
    rows = {}
    with (root / "strazar_2023_recovered_unmodified.csv").open() as stream:
        for row in csv.DictReader(stream):
            assert row["has_ptm"] == "False" and row["peptide_modifications"] == "Unmodified"
            assert row["source_evidence_id"] not in rows
            rows[row["source_evidence_id"]] = row
    assert len(rows) == result["positive_pmhcs"] == 33683
    sources = {(Path(s["path"]).name, s["sha256"]) for s in result["sources"]}
    verified, support_rows = set(), 0
    with (root / "source-evidence.jsonl").open() as stream:
        for line in stream:
            entry = json.loads(line)
            if entry["state"] != "unmodified":
                continue
            key = entry["evidence_id"]
            assert key not in verified
            verified.add(key)
            row = rows[key]
            assert entry["peptide"] == row["peptide"]
            assert entry["mhc_restriction"] == row["mhc_restriction"]
            identity = json.dumps([entry["peptide"], entry["mhc_restriction"], "unmodified"])
            assert hashlib.sha256(identity.encode()).hexdigest() == key
            assert entry["support"]
            for support in entry["support"]:
                assert (support["source_file"], support["source_sha256"]) in sources
                assert support["source_row"] >= 2
                assert support["nterm"] == "Hydrogen" and support["cterm"] == "Free Acid"
                assert all(not support[k] for k in ("modifications", "variableSites", "cys"))
                support_rows += 1
    assert verified == rows.keys()
    output = dict(
        status="passed",
        source_verified_positive_pmhcs=len(rows),
        unmodified_support_rows=support_rows,
        source_files=len(sources),
        archive_sha256=result["archive"]["sha256"],
        original_positive_retention_verified_remotely=result["verification"][
            "all_previous_positive_identities_retained"
        ],
        model_admission="not performed for this later batch",
        active_training_unchanged=True,
        verifier_sha256=sha(Path(__file__)),
        argv=[sys.executable, str(Path(__file__).resolve())],
        verified_unix=time.time(),
    )
    target = F / "results/recovery-late-verification.json"
    assert not target.exists()
    shutil.copyfile(__file__, F / "reproduce/source/verify_late_recovery.py")
    target.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print("Verified", len(rows), "source-positive pMHCs and", support_rows, "support rows")


if __name__ == "__main__":
    main()
