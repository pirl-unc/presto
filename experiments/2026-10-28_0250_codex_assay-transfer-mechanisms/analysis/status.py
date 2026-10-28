"""Print only live validation evidence; never open test predictions."""

from pathlib import Path
import json
import sys

F = Path(__file__).resolve().parents[1]
live = json.loads((F / "results/live-status.json").read_text())
controller = json.loads((F / "reproduce/controller.json").read_text())
print(
    f"{controller['status']}: {len(live['completed'])}/12 collected; failures={live['failed']}"
)
for name, state in live["conditions"].items():
    status = state["status"]["status"]
    if status == "completed" and "--all" not in sys.argv:
        continue
    p = state.get("progress", {})
    h = state.get("history", [])
    validation = state.get("metrics", {}).get("validation", {})
    selected = bool(validation)
    if not validation and h:
        validation = h[-1]["validation"]
    metrics = validation.get("ic50", {}).get("by_class", {})
    scores = {
        c: {m: round(metrics[c][m], 4) for m in ("spearman", "rmse")} for c in metrics
    }
    print(
        name,
        status,
        p.get("steps", 0),
        "selected validation"
        if selected
        else f"validation@{h[-1]['steps']}"
        if h
        else "no validation yet",
        scores,
    )
