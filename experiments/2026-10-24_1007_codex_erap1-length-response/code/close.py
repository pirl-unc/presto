"""Seal independently verified experiment artifacts without hiding failed attempts."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    verification = json.loads((ROOT / "results/independent-verification.json").read_text())
    summary = json.loads((ROOT / "results/fits/summary.json").read_text())
    manifest = json.loads((ROOT / "manifest.json").read_text())
    assert verification["status"] == "passed"
    assert len(summary["arms"]) == 4
    assert all(len(v["fits"]) == 6 for v in summary["arms"].values())
    assert not summary["outer_validation_test_scored"]
    prior = ROOT / "reproduce/manifest-before-closure.json"
    if not prior.exists():
        prior.write_text(json.dumps(manifest, indent=2) + "\n")
    manifest["source_sha256"] = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((ROOT / "reproduce/source").glob("*.py"))
    }
    manifest["executed_phase_receipts"] = [
        "reproduce/qualification/execution.json",
        "reproduce/training/execution.json",
        "reproduce/verification/execution.json",
        "reproduce/verification-v2/execution.json",
    ]
    manifest.update(
        status="complete_independently_verified",
        biological_gate=summary["development_gate_pass"],
        n_qualified_fits=24,
        preserved_unqualified_attempt="results/attempt-global-prefix",
        broader_availability_goal="unachieved",
    )
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    destination = ROOT / "results/closure.json"
    hashes = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(ROOT.rglob("*"))
        if p.is_file()
        and p != destination
        and "__pycache__" not in p.parts
        and p.name != ".DS_Store"
    }
    output = {
        "status": "complete_independently_verified",
        "n_files": len(hashes),
        "artifact_sha256": hashes,
        "notes": "All qualified and preserved failed-attempt artifacts sealed. Outer validation/test intentionally unscored. Optimizer/RNG trajectories retained, not independently replayed. This does not complete the APC availability/concentration goal.",
    }
    destination.write_text(json.dumps(output, indent=2) + "\n")
    for name, expected in hashes.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected
    print(json.dumps({k: v for k, v in output.items() if k != "artifact_sha256"}))


if __name__ == "__main__":
    main()
