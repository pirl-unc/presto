"""Record complete training diagnostic artifacts and explicitly absent held-outs."""

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for stage in ("diagnose", "replay"):
    assert json.loads((ROOT / f"results/{stage}-resource.json").read_text())["passed"]
assert json.loads((ROOT / "results/independent-replay.json").read_text())["passed"]
summary = json.loads((ROOT / "results/summary.json").read_text())
columns = ["condition", "success", "iterations", "function_evaluations", "seconds", "objective", "mse",
           "original_adam_mse", "mse_improvement_percent", "tau_minutes"]
with (ROOT / "results/comparison.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows({k: row[k] for k in columns} for row in summary["fits"])
files = {}
for path in sorted(ROOT.rglob("*")):
    if not path.is_file() or "__pycache__" in path.parts:
        continue
    name = str(path.relative_to(ROOT))
    if name in {"artifact-manifest.json", "results/closure.json", "results/closure-resource.json",
                "results/closure-started.json", "launch_logs/closure.log"}:
        continue
    files[name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "size_bytes": path.stat().st_size}
(ROOT / "artifact-manifest.json").write_text(json.dumps({"files": files}, indent=2) + "\n")
result = {"passed": True, "n_conditions": len(summary["fits"]), "n_hashed_files": len(files),
          "logical_bytes": sum(f["size_bytes"] for f in files.values()),
          "heldout_metrics": None,
          "heldout_absence_reason": "training-only diagnosis after original 0046 outer outcomes were inspected",
          "no_model_promotion": True, "decisions": summary["decisions"]}
(ROOT / "results/closure.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result), flush=True)
