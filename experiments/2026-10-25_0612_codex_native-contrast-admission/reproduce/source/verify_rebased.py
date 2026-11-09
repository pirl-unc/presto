"""Verify the admitted package through the current-main public recipe API."""

import hashlib
import json
from pathlib import Path

from hitlist.evidence_recipe import verify_recipe_package

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "reproduce/rebased-launch.json").read_text())
manifest = verify_recipe_package(CFG["package"], source_package=CFG["source_package"])
assert manifest == json.loads((ROOT / "results/package-manifest.json").read_text())
report = dict(
    passed=True,
    hitlist_commit=CFG["hitlist_base"],
    recipe_manifest_unchanged=True,
    manifest_sha256=hashlib.sha256(
        (Path(CFG["package"]) / "manifest.json").read_bytes()
    ).hexdigest(),
    qualified_for_screen=manifest["coverage"]["qualified_for_screen"],
    predictor_fitted=False,
)
(ROOT / "results/rebased-public-verification.json").write_text(
    json.dumps(report, indent=2) + "\n"
)
print(json.dumps(report), flush=True)
