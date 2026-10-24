"""Independent full-path zero-length equality and registered gradient reductions."""

import hashlib
import json
import resource
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    start = time.perf_counter()
    cfg = json.loads((ROOT / "reproduce/launch.json").read_text())
    coverage = json.loads((ROOT / "results/input-coverage.json").read_text())
    artifact = ROOT / "results/path-inputs.npz"
    assert hashlib.sha256(artifact.read_bytes()).hexdigest() == coverage["files"][artifact.name]
    with np.load(artifact) as archive:
        p = {name: archive[name] for name in archive.files}
    direction = np.random.default_rng(42).normal(size=23)
    probes = [np.zeros(23), 0.2 * direction, -0.2 * direction]
    checks = []
    for index, probe in enumerate(probes):
        for label, scale in [("wt", 0.0), ("ko", -np.logaddexp(0, -probe[-1]))]:
            rates = probe[:20] + scale
            survival = -np.logaddexp(0, -rates)
            hazard = np.logaddexp(0, rates[p["target_residue"]])
            hazard[p["target_monomer"]] = 0
            old_scores = p["log_birth"] + p["counts"] @ survival - hazard[p["owner"]]
            state_survival = survival[p["state_residue"]]
            before = np.concatenate(([0.0], np.cumsum(state_survival)[:-1]))
            state_hazard = np.logaddexp(0, rates[p["state_residue"]])
            state_hazard[p["state_monomer"]] = 0
            new_scores = (
                p["log_birth"]
                + (before[p["terminal_index"]] - before)
                - state_hazard[p["terminal_index"]]
            )
            error = float(np.max(abs(old_scores - new_scores)))
            checks.append(
                {
                    "probe": index,
                    "condition": label,
                    "n_paths": len(old_scores),
                    "max_absolute_error": error,
                }
            )
            assert error <= cfg["old_model_absolute_tolerance"]
    result = {
        "status": "passed",
        "checks": checks,
        "seconds": time.perf_counter() - start,
        "peak_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "Independent NumPy reconstruction of every zero-length-modifier path; supplements the pre-fit Torch response/objective equality checks. Ran after biological launch, with no training changes or gate relaxation.",
    }
    (ROOT / "results/independent-path-equivalence.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
