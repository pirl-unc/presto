"""Admit the frozen shared F3 endpoint; do not fit, predict or select a model."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.requests.read_text())
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "reproduce/launch.json").read_text())
    assert request["action"] == "shared_organ_endpoint_admission"
    assert json.loads((root / "results/metadata-support.json").read_text())[
        "metadata_gate_passed"
    ]
    checkout = Path(request["hitlist_checkout"]).resolve()
    assert (
        subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=checkout, text=True
        ).strip()
        == request["hitlist_commit"]
    )
    assert not subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=checkout,
        text=True,
    ).strip()
    # This stage explicitly uses its own frozen feature commit; the metadata
    # stage's release checkout and inherited environment remain unchanged.
    sys.path.insert(0, str(checkout))
    import hitlist
    from hitlist.evidence_recipe import verify_recipe_package
    from hitlist.evidence_store import EvidenceStore

    assert Path(hitlist.__file__).resolve().is_relative_to(checkout)
    destination = Path(request["recipe_destination"]).resolve()
    assert destination.is_relative_to(root.parents[1] / "data/intact_organ_recipes")
    store = EvidenceStore.open(config["source_package"])
    store.materialize(request["recipe"], destination)
    manifest = verify_recipe_package(destination, source_package=store.directory)
    result = {
        "action": request["action"],
        "recipe_directory": str(destination),
        "recipe_manifest_sha256": hashlib.sha256(
            (destination / "manifest.json").read_bytes()
        ).hexdigest(),
        "source_manifest_sha256": config["source_manifest_sha256"],
        "hitlist_checkout": str(checkout),
        "hitlist_import_file": str(Path(hitlist.__file__).resolve()),
        "hitlist_commit": request["hitlist_commit"],
        "hitlist_version": hitlist.__version__,
        "dependency_selection": "explicit stage-local sys.path entry pinned in request; no environment installation",
        "coverage": manifest["coverage"],
        "models_fitted_or_scored": False,
        "next_action": "independent numeric replay before frozen predictions"
        if manifest["coverage"]["support_passed"]
        else "stop scoring; preserve failed support and close family",
    }
    with (root / "results/admission.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
