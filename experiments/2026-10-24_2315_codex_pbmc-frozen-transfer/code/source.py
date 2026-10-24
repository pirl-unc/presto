"""Append the pinned PBMC source through Hitlist; no model outcome inspection."""

import hashlib
import json
import shutil
import signal
from pathlib import Path

from hitlist.evidence_store import EvidenceStore
from hitlist.external_successor import write_external_successor

ROOT = Path(__file__).resolve().parents[1]


def digest(path, kind="sha256"):
    value = hashlib.new(kind)
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            value.update(block)
    return value.hexdigest()


def terminate(*_):
    raise RuntimeError("Resource supervisor requested termination")


def main():
    signal.signal(signal.SIGTERM, terminate)
    config = json.loads((ROOT / "reproduce/launch.json").read_text())
    hl = Path(config["hitlist_checkout"])
    receipt = hl / "hitlist/data/evidence_revisions/mapp_pbmc_assets.json"
    assert digest(receipt) == config["source_receipt_sha256"]
    assets = json.loads(receipt.read_text())
    for relative, expected in assets["assets"].items():
        path = Path(config["source_root"]) / relative
        assert path.stat().st_size == expected["size"] and digest(path) == expected["sha256"]
    archive = Path(config["archive"])
    expected = assets["archive_provenance"]["PXD009438"]
    assert archive.stat().st_size == expected["size"]
    assert (
        digest(archive) == expected["sha256"] and digest(archive, "sha1") == expected["pride_sha1"]
    )
    assert (
        digest(Path(config["predecessor"]) / "manifest.json")
        == config["predecessor_manifest_sha256"]
    )
    print("Original asset, archive and predecessor fingerprints pass", flush=True)
    output = write_external_successor(
        config["predecessor"], config["package"], source_roots={"pxd009438": config["source_root"]}
    )
    manifest = json.loads(output.read_text())
    store = EvidenceStore.open(config["package"])
    names = [
        name
        for name in store.bindings
        if name.startswith("pbmc_mapp") or name == "external_pbmc_proteasome_fragment_quantity"
    ]
    result = {
        "source_manifest_sha256": digest(output),
        "reconciliation": manifest["appended_sources"]["pxd009438"]["reconciliation"],
        "source_registry": manifest["appended_sources"]["pxd009438"]["registry"],
        "query_relations": names,
        "predecessor_snapshot": "existing3.8GiBreference generation; independently snapshotted with CoW, not used as PBMCtraining data",
        "source_schema_version": manifest["schema_version"],
        "source_catalog_schema_version": manifest["catalog_schema_version"],
    }
    shutil.copy2(output, ROOT / "results/source-manifest.json")
    (ROOT / "results/source-qualification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
