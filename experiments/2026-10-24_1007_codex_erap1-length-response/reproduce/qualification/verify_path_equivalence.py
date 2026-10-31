"""Independent full-path zero-length equality and registered gradient reductions."""

import hashlib
import json
import resource
import time
from pathlib import Path

import numpy as np
import torch
from presto.models.n_terminal_trimming import (
    log_state_trimming_paths,
    trimming_origin_layout,
)

ROOT = Path(__file__).resolve().parents[1]


def main():
    start = time.perf_counter()
    torch.set_num_threads(1)
    cfg = json.loads((ROOT / "reproduce/launch.json").read_text())
    coverage = json.loads((ROOT / "results/input-coverage.json").read_text())
    artifact = ROOT / "results/path-inputs.npz"
    assert hashlib.sha256(artifact.read_bytes()).hexdigest() == coverage["files"][artifact.name]
    with np.load(artifact) as archive:
        p = {name: archive[name] for name in archive.files}
    direction = np.random.default_rng(42).normal(size=23)
    probes = [np.zeros(23), 0.2 * direction, -0.2 * direction]
    checks = []
    terminal = torch.tensor(p["terminal_index"])
    monomer = torch.tensor(p["state_monomer"])
    layout = trimming_origin_layout(terminal)
    for index, probe in enumerate(probes):
        for label, scale in [("wt", 0.0), ("ko", -np.logaddexp(0, -probe[-1]))]:
            rates = probe[:20] + scale
            survival = -np.logaddexp(0, -rates)
            hazard = np.logaddexp(0, rates[p["target_residue"]])
            hazard[p["target_monomer"]] = 0
            old_scores = p["log_birth"] + p["counts"] @ survival - hazard[p["owner"]]
            new_scores = log_state_trimming_paths(
                torch.tensor(p["log_birth"]),
                torch.tensor(rates[p["state_residue"]]),
                terminal,
                monomer,
                origin_layout=layout,
            ).numpy()
            error = float(np.max(abs(old_scores - new_scores)))
            checks.append(
                {
                    "probe": index,
                    "condition": label,
                    "n_paths": len(old_scores),
                    "max_absolute_error": error,
                }
            )
    result = {
        "status": "passed"
        if all(c["max_absolute_error"] <= cfg["old_model_absolute_tolerance"] for c in checks)
        else "failed",
        "checks": checks,
        "seconds": time.perf_counter() - start,
        "peak_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "Actual Torch origin-local path scores versus independent NumPy residue-count reference at zero length modifiers, all paths and three WT/KO probes; required before biological training.",
        "padded_states": sum(indices.numel() for indices, _, _, _ in layout),
        "model_source_sha256": hashlib.sha256(
            (ROOT.parents[1] / "models/n_terminal_trimming.py").read_bytes()
        ).hexdigest(),
    }
    (ROOT / "results/independent-path-equivalence.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result), flush=True)
    assert result["status"] == "passed"


if __name__ == "__main__":
    main()
