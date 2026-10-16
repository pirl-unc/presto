"""Exact source and public catalog pins shared with the verified preflight."""

import hashlib
import importlib.metadata
import json
import shutil
import tempfile
from pathlib import Path

ROOT = Path("/opt/qualification")
SOURCE = Path(
    "/data/hitlist-qualification/runs/a6933c31825caf35475ba38ae243551bb815d65ba03ee93d9ddb768869e4b5a0/human-cohort/c923e5f8f05d1df7156625740b84f5a58f25108f703a01146a2b62336dc54443/artifacts/bundles/2026-10-16T07-29-43Z"
)
CACHE = Path("/data/presto-qualification/effective-target-fit-v1")
HITLIST_CACHE = Path("/data/presto-qualification/corrected-human-v1/hitlist")
MHC_CACHE = Path("/data/presto-qualification/human-baseline-v1")


def digest(path):
    path = Path(path)
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            sha.update(block)
    return {"sha256": sha.hexdigest(), "size_bytes": path.stat().st_size}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def build_mhc_catalog():
    import datacache
    import mhcseqs

    sources = json.loads((ROOT / "mhc-sources.json").read_text())
    files = datacache.install_bundle(
        MHC_CACHE / "mhc-sources",
        {name: {k: row[k] for k in ("url", "sha256", "size")} for name, row in sources.items()},
        download_options={"show_progress": False},
    )
    names = (
        "mhc-seqs-raw.csv",
        "mhc-full-seqs.csv",
        "mhc-merge-report.txt",
        "mhc-validation-report.txt",
    )

    def builder(inputs, outputs):
        # The existing verified FASTAs prevent mhcseqs.build from fetching mutable Latest URLs.
        with tempfile.TemporaryDirectory(prefix="mhc-catalog-") as tmp:
            work = Path(tmp)
            fasta = work / "fasta"
            fasta.mkdir()
            for name, path in inputs.items():
                shutil.copyfile(path, fasta / name)
            mhcseqs.build(output_dir=str(work), data_dir=str(fasta), force_download=False)
            for name in names:
                shutil.copyfile(work / name, outputs[name])

    outputs = datacache.materialize(
        MHC_CACHE / "mhc-catalog",
        {
            name: {"path": str(path), "sha256": sources[name]["sha256"]}
            for name, path in files.items()
        },
        transform={
            "version": "mhcseqs-2.6.13-canonical-build-v1",
            "options": {"mhcseqs": importlib.metadata.version("mhcseqs")},
        },
        outputs={name: {} for name in names},
        builder=builder,
    )
    expected = {
        "mhc-seqs-raw.csv": "4c61d2683c07f8aeee7845fb75038636fbb8d867adf5d55535d371ece16a94eb",
        "mhc-full-seqs.csv": "1d64f5af13f49497ea5c7219f8ec21edd1f973aecd3a6627a065fc554ef022c5",
        "mhc-merge-report.txt": "090f90ca4713cb55b22f43cfefb99e0d2246ebe2cbf2b5ac450393ad701964cb",
        "mhc-validation-report.txt": "349f7aeafdbaa1edfe2de53f8bd674810a1e4fecaa2c7356feeeabc15016ce39",
    }
    for name, path in outputs.items():
        datacache.validate_file(path, expected[name])
    return {
        "shared_catalog_root": str(MHC_CACHE),
        "directory": str(Path(outputs[names[0]]).parent),
        "artifacts": {name: digest(path) for name, path in outputs.items()},
        "sources": sources,
    }
