"""Input-only integration replay; never fit, predict, or modify frozen evidence."""

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess

from presto.data.modular_hitlist import AdmissionError, HumanMHCCatalog, MHCReferenceAssumption


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("family", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    closure_path = args.family / "results/closure.json"
    closure = json.loads(closure_path.read_text())
    for relative, expected in closure["files"].items():
        if sha(args.family / relative) != expected:
            raise ValueError(f"frozen evidence changed: {relative}")
    source = args.family / "results/molecular-assumption.json"
    record = json.loads(source.read_text())
    if sha(record["catalog_path"]) != record["catalog_sha256"]:
        raise ValueError("catalog differs from the frozen experiment")
    with Path(record["catalog_path"]).open(newline="") as stream:
        catalog = HumanMHCCatalog(csv.DictReader(stream))
    provenance = record["provenance"]
    assert provenance["source_reported_alpha_allele"] is None
    reported = provenance["source_reported_beta_allele"]
    try:
        catalog.resolve(reported, "II")
    except AdmissionError as error:
        if error.reason != "missing_explicit_mhc_partner":
            raise ValueError("strict resolution failed for an unexpected reason") from error
    else:
        raise AssertionError("strict resolution admitted unknown source alpha")
    selected = "/".join((provenance["modeled_alpha_allele"], reported))
    pair = catalog.resolve(
        reported,
        "II",
        reference_assumption=MHCReferenceAssumption(
            selected, provenance["reason"], f"sha256:{sha(source)}"
        ),
    )
    assert pair.groove1 == record["alpha"]["groove1"]
    assert pair.groove2 == record["beta"]["groove2"]
    assert pair.provenance["reported_restriction"] == reported
    assert pair.provenance["assumed_alleles"] == [provenance["modeled_alpha_allele"]]
    root = Path(__file__).resolve().parents[1]
    result = {
        "status": "passed",
        "scope": "input-only integration; no biological fit, evaluation or qualification",
        "base_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "resolver_sha256": sha(root / "data/modular_hitlist.py"),
        "replay_script_sha256": sha(__file__),
        "family": str(args.family.resolve()),
        "closure_sha256": sha(closure_path),
        "frozen_files_verified": len(closure["files"]),
        "molecular_assumption_sha256": sha(source),
        "catalog_sha256": record["catalog_sha256"],
        "catalog_dispositions": dict(catalog.dispositions),
        "exact_grooves_preserved": True,
        "strict_missing_alpha_rejected": True,
        "source_alpha_identity": None,
        "resolved_mhc": asdict(pair),
        "eligible_alias_counts": {name: len(rows) for name, rows in catalog.aliases().items()},
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "frozen_files_verified", "scope")}))


if __name__ == "__main__":
    main()
