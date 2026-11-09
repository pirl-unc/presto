"""Posthoc validation-history diagnostics; never select or evaluate a model."""

from pathlib import Path
import json
import math

F = Path(__file__).resolve().parents[1]


def summarize(history, selected_steps):
    rows = []
    for event in history:
        metrics = event["validation"]["ic50"]
        classes = metrics["by_class"]
        count = sum(classes[c]["n"] for c in ("I", "II"))
        pooled = math.sqrt(
            sum(classes[c]["n"] * classes[c]["rmse"] ** 2 for c in ("I", "II")) / count
        )
        assert math.isclose(pooled, metrics["exact"]["rmse"], rel_tol=1e-12)
        rows.append(
            dict(
                steps=event["steps"],
                pooled_rmse=pooled,
                equal_class_mean_rmse=sum(classes[c]["rmse"] for c in ("I", "II")) / 2,
                class_I_fraction=classes["I"]["n"] / count,
                by_class={
                    c: {k: classes[c][k] for k in ("n", "rmse", "spearman")}
                    for c in ("I", "II")
                },
            )
        )
    actual = min(rows, key=lambda r: r["pooled_rmse"])
    assert actual["steps"] == selected_steps
    alternatives = {
        "equal_class_mean_rmse": min(rows, key=lambda r: r["equal_class_mean_rmse"])
    }
    for cls in ("I", "II"):
        alternatives[f"{cls}_minimum_rmse"] = min(
            rows, key=lambda r: r["by_class"][cls]["rmse"]
        )
        alternatives[f"{cls}_maximum_spearman"] = max(
            rows, key=lambda r: r["by_class"][cls]["spearman"]
        )
    return dict(actual_selected=actual, descriptive_alternatives=alternatives)


def main():
    pins = json.loads((F / "reproduce/source-pins.json").read_text())
    results = {}
    for name in pins["conditions"]:
        directory = F / "results" / name
        history = json.loads((directory / "history.json").read_text())
        report = json.loads((directory / "report.json").read_text())
        results[name] = summarize(history, report["selected_steps"])
    output = dict(
        scope="Posthoc validation-history description added during execution. Alternatives are not selected, refitted, promoted or evaluated on test. Original selected checkpoints and decision remain unchanged.",
        conditions=results,
    )
    (F / "results/checkpoint-selection-diagnostics.json").write_text(
        json.dumps(output, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
