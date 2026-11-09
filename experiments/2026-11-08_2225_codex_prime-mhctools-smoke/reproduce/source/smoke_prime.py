#!/usr/bin/env python3
"""Run a hash-pinned PRIME/mhctools software smoke test."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import time

import mhctools
from mhctools import PRIME


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prime", required=True, type=Path)
    parser.add_argument("--mixmhcpred", required=True, type=Path)
    parser.add_argument("--backend-python", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    for path in (args.prime, args.mixmhcpred, args.backend_python):
        if not path.is_file():
            raise ValueError(f"missing executable: {path}")

    began = time.monotonic()
    predictor = PRIME(
        alleles=["HLA-A*02:01"],
        program_name=str(args.prime),
        mixmhcpred_path=str(args.mixmhcpred),
        mixmhcpred_python=str(args.backend_python),
        timeout=300,
    )
    peptides = ["NLVPMVATV", "GILGFVFTL"]
    results = predictor.predict(peptides)
    if [result.peptide for result in results] != peptides:
        raise ValueError("PRIME output order or coverage differs from the input")
    if any(len(result.preds) != 1 for result in results):
        raise ValueError("PRIME did not return exactly one allele prediction per peptide")
    checkout_root = Path(mhctools.__file__).resolve().parents[1]
    checkout = subprocess.check_output(
        ["git", "-C", str(checkout_root), "describe", "--tags", "--always"],
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()
    receipt = {
        "schema": "presto.prime_mhctools_smoke.v1",
        "status": "passed",
        "peptides": peptides,
        "alleles": ["HLA-A*02:01"],
        "predictions": [result.to_dict() for result in results],
        "runtime_info": predictor.runtime_info,
        "mhctools_distribution_version": importlib.metadata.version("mhctools"),
        "mhctools_checkout_describe": checkout,
        "executables": {
            "prime": {"path": str(args.prime), "sha256": sha256(args.prime)},
            "mixmhcpred": {
                "path": str(args.mixmhcpred),
                "sha256": sha256(args.mixmhcpred),
            },
            "backend_python": {"path": str(args.backend_python)},
        },
        "license_scope": "academic/non-commercial; see local PRIME and MixMHCpred licenses",
        "requested_gpu": None,
        "observed_gpu": None,
        "training": None,
        "validation_or_test": None,
        "elapsed_seconds": time.monotonic() - began,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
