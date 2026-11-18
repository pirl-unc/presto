"""Exercise installed train/evaluate/predict commands on artificial both-class data."""

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys

from ..data.modular import MHCInput, Observation
from ..data.modular_sources import MANIFEST_VERSION, file_sha256, write_observations


def run_smoke(directory):
    root = Path(directory).resolve()
    root.mkdir(parents=True, exist_ok=False)
    peptides = (
        "ACDEFGHIK",
        "LMNPQRSTV",
        "WYACDEFGHIKLM",
        "NPQRSTVWYACD",
        "CDEFGHIKL",
        "MNPQRSTVW",
        "YACDEFGHIKLMN",
        "PQRSTVWYACDE",
    )
    rows = [
        Observation(
            f"fixture-{i}",
            "software-workflow-fixture",
            "ic50",
            peptide,
            (MHCInput("ACDEFGHIKLM", "NPQRSTVWYAC", "I" if i % 4 < 2 else "II"),),
            value=10.0 if i % 2 else 1000.0,
            unit="nM",
            split="train" if i < 4 else "validation",
            group_id=f"group-{i}",
            provenance={"original_measurement_id": f"measurement-{i}"},
        )
        for i, peptide in enumerate(peptides)
    ]
    sources = []
    for split in ("train", "validation"):
        receipt = write_observations(
            root / f"{split}.jsonl", [row for row in rows if row.split == split]
        )
        sources.append(
            {
                "path": f"{split}.jsonl",
                "sha256": receipt["sha256"],
                "source": "software-workflow-fixture",
                "split": split,
                "population": "artificial software check",
                "label_policy": "artificial numeric targets; no biological claims",
                "split_policy": "distinct artificial peptides and groups",
            }
        )
    manifest = {
        "schema": MANIFEST_VERSION,
        "sources": sources,
        "model": {
            "d_model": 16,
            "n_layers": 1,
            "n_heads": 2,
            "d_ff": 32,
            "endpoints": ["ic50"],
            "identity_tasks": [],
            "dropout": 0,
        },
        "training": {
            "max_epochs": 1,
            "max_steps": 1,
            "batch_size": 4,
            "test_policy": "no_test_training_diagnostic",
            "no_test_reason": "Artificial integration fixture; no biological TEST outcomes",
        },
    }
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest))
    run = root / "experiments/2000-01-01_0000_codex_workflow-smoke"
    env = {**os.environ, "PRESTO_WORKSPACE": str(root), "OMP_NUM_THREADS": "2"}
    env.pop("PYTHONPATH", None)

    def command(*args):
        with (root / f"{args[0]}.log").open("w") as log:
            subprocess.run(
                [sys.executable, "-I", "-m", "presto", *map(str, args)],
                cwd=root,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )

    command(
        "train",
        "--manifest",
        path,
        "--run-dir",
        run,
        "--device",
        "cpu",
        "--agent",
        "codex",
    )
    checkpoint = run / "selected.pt"
    assert checkpoint.is_file()
    assert (run / "reproduce/source/models/modular.py").is_file()
    assert "Closure:" in (root / "experiments/experiment_log.md").read_text()
    # Evaluation must select declared validation sources before accessing TEST.
    manifest["sources"].append({"path": "SEALED-NOT-PRESENT.jsonl", "split": "test"})
    path.write_text(json.dumps(manifest))
    command(
        "evaluate",
        "observations",
        "--checkpoint",
        checkpoint,
        "--manifest",
        path,
        "--output-dir",
        root / "evaluation",
    )
    evaluation = json.loads((root / "evaluation/receipt.json").read_text())
    assert evaluation["rows"] == 4 and evaluation["test_accessed"] is False
    queries = root / "queries.jsonl"
    queries.write_text(
        "".join(
            json.dumps(replace(row, value=None, split="predict").to_dict()) + "\n"
            for row in rows[-4:]
        )
    )
    command(
        "predict",
        "observations",
        "--checkpoint",
        checkpoint,
        "--checkpoint-sha256",
        file_sha256(checkpoint),
        "--inputs",
        queries,
        "--output",
        root / "predictions.json",
    )
    predicted = json.loads((root / "predictions.json").read_text())
    assert len(predicted["predictions"]) == 4
    receipt = {
        "status": "passed",
        "kind": "artificial software fixture",
        "classes": ["I", "II"],
        "optimizer_steps": 1,
        "validation_rows": 4,
        "label_free_prediction_rows": 4,
        "checkpoint_sha256": file_sha256(checkpoint),
        "test_outcomes_accessed": False,
    }
    (root / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    print(json.dumps(run_smoke(parser.parse_args().output_dir), indent=2))
