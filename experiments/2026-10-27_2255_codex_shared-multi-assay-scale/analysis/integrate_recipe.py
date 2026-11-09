"""Materialize the validation-selected recipe and explicitly experimental wider-output recipes."""

from pathlib import Path
import copy
import hashlib
import json
import sys

F = Path(__file__).resolve().parents[1]
ROOT = F.parents[1]


def main():
    decision = json.loads((F / "results/validation-decision.json").read_text())
    verification = json.loads((F / "results/verification-summary.json").read_text())
    replay = json.loads((F / "results/gpu-training-replay.json").read_text())
    assert decision["test_used_for_decision"] is False
    assert verification["status"] == "passed"
    assert len(replay["results"]) == 8 and all(
        r["status"] == "passed" for r in replay["results"]
    )
    pins = json.loads((F / "reproduce/source-pins.json").read_text())
    specifications = {
        "ic50_joint_development": decision["preferred_variant"],
        "multi_assay_development": "multi-assay",
        "multi_assay_wide_development": "multi-assay-wide",
    }
    sys.path.insert(0, str(ROOT.parent))
    from presto.data.modular_sources import ObservationIndex
    from presto.training.modular_run import preflight

    receipts = {}
    for label, variant in specifications.items():
        document = copy.deepcopy(pins["manifests"][f"{variant}-s42"])
        document["training"]["selection_endpoint"] = "ic50"
        for source in document["sources"]:
            source["path"] = f"../artifacts/{F.name}/upload/inputs/{source['path']}"
        destination = ROOT / "configs" / f"{label}.json"
        destination.write_text(json.dumps(document, indent=2) + "\n")
        preflight(ObservationIndex(destination))
        receipts[label] = dict(
            variant=variant,
            path=str(destination.relative_to(ROOT)),
            sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
            scope="validation-selected IC50 development recipe"
            if label.startswith("ic50")
            else "experimental multi-output recipe; not globally promoted",
            preflight="passed",
            fitting_performed=False,
        )
    (F / "results/integrated-recipes.json").write_text(
        json.dumps(receipts, indent=2) + "\n"
    )
    print(json.dumps(receipts, indent=2), flush=True)


if __name__ == "__main__":
    main()
