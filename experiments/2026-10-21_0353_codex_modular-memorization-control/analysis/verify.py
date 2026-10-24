"""Fresh-process replay and plots from the completed fixed diagnostic; no fitting."""

import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from presto.data.modular_sources import ObservationIndex, file_sha256
from presto.inference.modular import ModularPredictor
from presto.training.modular import load_checkpoint
from presto.training.modular_evaluation import evaluate


def main():
    torch.set_num_threads(1)
    family = Path(__file__).resolve().parents[1]
    results = family / "results"
    summary = json.loads((results / "summary.json").read_text())
    curation = json.loads((results / "curation.json").read_text())
    manifest = Path(curation["manifest_path"])
    assert file_sha256(manifest) == summary["source_manifest_sha256"]
    rows = list(ObservationIndex(manifest))
    path = Path(summary["raw_artifacts"]) / "final.pt"
    assert file_sha256(path) == summary["checkpoint_sha256"]
    model, scales, _ = load_checkpoint(path)
    metrics, records = evaluate(model, rows, range(32), scales)
    saved = [
        json.loads(line) for line in (results / "final-predictions.jsonl").read_text().splitlines()
    ]
    assert records == saved and metrics == summary["metrics"]
    served = ModularPredictor(path).predict(rows)
    assert all(
        value["transformed_value"] == record["prediction"]
        for value, record in zip(served, saved, strict=True)
    )
    assert all(
        record["target"] == math.log10(row.value) for record, row in zip(records, rows, strict=True)
    )
    diagnostic_path = results / "register-diagnostics.jsonl"
    with diagnostic_path.open("w") as stream:
        for row, output in zip(rows, model.predict(rows), strict=True):
            stream.write(
                json.dumps(
                    {
                        "observation_id": row.observation_id,
                        "mhc_class": row.mhc[0].mhc_class,
                        "reported_restriction": row.mhc[0].provenance["reported_restriction"],
                        "diagnostics": output.diagnostics["candidates"],
                    },
                    sort_keys=True,
                    default=lambda value: (
                        value.detach().cpu().tolist() if isinstance(value, torch.Tensor) else value
                    ),
                )
                + "\n"
            )
    coverage = json.loads((results / "coverage.json").read_text())
    register = coverage["parameters"]["register_score.weight"]
    assert register["gradient_and_update_steps"] > 0
    report = {
        "model_commit": json.loads((family / "reproduce/launch.json").read_text())["git_commit"],
        "fresh_process_checkpoint_predictions": "identical for all 32 observations",
        "public_predictor": "identical transformed values; all support guards passed",
        "raw_target_replay": "all 32 equal log10(raw nM)",
        "coverage_parameter_count": len(coverage["parameters"]),
        "register_gradient_update_steps": register["gradient_and_update_steps"],
        "register_max_gradient_norm": register["max_gradient_norm"],
        "register_diagnostics_sha256": hashlib.sha256(diagnostic_path.read_bytes()).hexdigest(),
        "training_only": True,
        "validation_test": "not used by design; no generalization estimate",
        "independent_rmse": math.sqrt(
            sum((row["target"] - row["prediction"]) ** 2 for row in records) / 32
        ),
    }
    (results / "fresh-process-verification.json").write_text(json.dumps(report, indent=2) + "\n")
    history = [json.loads(line) for line in (results / "history.jsonl").read_text().splitlines()]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), layout="constrained")
    colors = {"I": "#1f78b4", "II": "#df6c20"}
    for cls, color in colors.items():
        selected = [row for row in records if row["mhc_classes"] == [cls]]
        axes[0].scatter(
            [row["target"] for row in selected],
            [row["prediction"] for row in selected],
            label=f"MHC {cls} (16 fit targets)",
            color=color,
            s=34,
        )
        axes[1].plot(
            [row["step"] for row in history],
            [row["metrics"]["ic50"]["by_class"][cls]["rmse"] for row in history],
            marker="o",
            label=f"MHC {cls}",
            color=color,
        )
    axes[0].plot([-0.2, 6.1], [-0.2, 6.1], linestyle="--", color="0.5", linewidth=1)
    axes[0].set(
        xlabel="Measured log10(nM)", ylabel="Fitted log10(nM)", title="Final training predictions"
    )
    axes[0].legend(loc="upper left", fontsize=8)
    axes[1].axhline(0.1, linestyle="--", color="0.5", label="Fixed RMSE gate")
    axes[1].set(
        xlabel="Full-batch update",
        ylabel="Training RMSE, log10(nM)",
        title="Both classes pass at update 100",
        yscale="log",
    )
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.15)
    fig.suptitle("Assay-only memorization • no pretraining • no held-out evaluation", fontsize=12)
    fig.savefig(results / "training-diagnostic.png", dpi=180)
    plt.close(fig)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
