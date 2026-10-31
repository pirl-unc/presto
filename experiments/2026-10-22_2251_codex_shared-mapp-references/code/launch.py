"""Materialize a shared MAPP source generation, preserving original input bytes."""

import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import datacache
from hitlist.external_evidence import write_external_evidence_package

FAMILY = Path(__file__).resolve().parents[1]
HITLIST = Path("/private/tmp/hitlist-mapp-source")
OUTPUT = Path("/private/tmp/hitlist-v2-mapp-reference-second")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main():
    assert not (FAMILY / "results/execution.json").exists()
    tick = time.monotonic()
    receipt_path = HITLIST / "hitlist/data/evidence_revisions/mapp_assets.json"
    receipt = json.loads(receipt_path.read_text())
    assets = {
        k: {f: v[f] for f in ("url", "size", "sha256", "decompress")}
        for k, v in receipt["assets"].items()
    }
    seeds = json.loads((FAMILY / "reproduce/input_locations.json").read_text())
    parent = Path("/private/tmp/hitlist-v2-supporting-18221-audited")
    dump(
        FAMILY / "reproduce/launch.json",
        {
            "agent_model": "Codex / GPT-6",
            "argv": sys.argv,
            "python": sys.version,
            "executable": sys.executable,
            "hitlist_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=HITLIST, text=True
            ).strip(),
            "hitlist_dirty": subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=HITLIST, text=True
            ),
            "presto_dirty": subprocess.check_output(
                ["git", "status", "--porcelain"], text=True
            ).strip(),
            "presto_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "environment_overrides": {"PYTHONPATH": str(HITLIST)},
            "source_receipt_sha256": sha(receipt_path),
            "parent": str(parent),
            "parent_manifest_sha256": sha(parent / "manifest.json"),
            "output": str(OUTPUT),
            "training": None,
            "pretraining": None,
            "model": None,
            "device": "local CPU",
            "no_predictive_metrics_reason": "source importer verification; no model fit or split",
        },
    )
    for name in (
        "mapp_evidence.py",
        "external_evidence.py",
        "sequence_reference_evidence.py",
        "evidence_catalog.py",
    ):
        shutil.copy2(HITLIST / "hitlist" / name, FAMILY / "reproduce/source" / name)
    for name in ("mapp_assets.json", "mapp_contract.json"):
        shutil.copy2(
            HITLIST / "hitlist/data/evidence_revisions" / name,
            FAMILY / "reproduce/source" / name,
        )
    shutil.copy2(__file__, FAMILY / "reproduce/source/launch.py")
    cache = FAMILY.parents[1] / "data/availability_sources" / sha(receipt_path)
    inputs = datacache.install_bundle(cache, assets, source_paths=seeds)
    root = Path(inputs["txt/evidence.txt"]).parents[1]
    assert all(root / k == Path(v) for k, v in inputs.items())
    dump(FAMILY / "results/input_paths.json", {k: str(v) for k, v in inputs.items()})
    write_external_evidence_package(
        parent,
        OUTPUT,
        source_roots={
            "pxd009092": root,
            "pxd057588": Path(
                "/private/tmp/hitlist-v2-external-18222-audited/external/pxd057588/assets"
            ),
        },
        source_ids=("pxd009092", "pxd057588"),
    )
    manifest = json.loads((OUTPUT / "manifest.json").read_text())
    dump(
        FAMILY / "results/execution.json",
        {
            "seconds": time.monotonic() - tick,
            "manifest_sha256": sha(OUTPUT / "manifest.json"),
            "output": str(OUTPUT),
            "sources": manifest["sources"],
            "n_package_files": len(manifest["files"]),
        },
    )
    shutil.copy2(OUTPUT / "manifest.json", FAMILY / "results/shared_manifest.json")
    print(
        json.dumps(
            {
                "seconds": time.monotonic() - tick,
                "source": manifest["sources"]["pxd009092"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
