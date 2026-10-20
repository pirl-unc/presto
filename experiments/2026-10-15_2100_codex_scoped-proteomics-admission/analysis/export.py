"""Publish a portable, path-redacted admission audit through datacache."""

import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path

from datacache import materialize
from hitlist.downloads import data_dir

EXPERIMENT = Path(__file__).resolve().parents[1]
DATASET = "presto-pxd013455-admission-v1"
ARCHIVE = DATASET + ".tar.gz"
SOURCE = "https://github.com/pirl-unc/hitlist/releases/tag/pxd013455-hela-tryp46-v1"


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            value.update(block)
    return value.hexdigest()


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def run():
    records = json.loads((EXPERIMENT / "results" / "artifacts.json").read_text())
    sources = {
        name: {"path": item["path"], "sha256": item["sha256"], "size": item["size_bytes"]}
        for name, item in records.items()
    }
    receipt = json.loads(Path(records["source_receipt.json"]["path"]).read_text())
    for name, item in receipt["projection"]["artifacts"].items():
        sources["projection_" + name] = {
            "path": str(Path(receipt["projection"]["directory"]) / name),
            "sha256": item["sha256"],
            "size": item["size_bytes"],
        }

    def build(inputs, outputs):
        summary = json.loads(Path(inputs["summary.json"]).read_text())
        summary.pop("imports")
        source = json.loads(Path(inputs["source_receipt.json"]).read_text())
        source["frozen_bundle"].pop("directory")
        source["projection"].pop("directory")
        source["source_download"] = SOURCE
        readme = (
            "# Presto PXD013455 admission audit v1\n\n"
            "This contains a complete one-vote projection of 1,656,863 candidate "
            "occurrences into 1,451,158 peptide records, plus a fixed seed-42 sample "
            "of 10,000 records and its canonical collation/support audit.\n\n"
            "Full labels: 143,216 MS/MS, 20,778 matching, 1,287,164 scoped absences. "
            "Admitted labels: 960 MS/MS, 154 matching, 8,886 scoped absences. "
            "All are proxy support for observation within this specific search. "
            "No excision or depth labels; no invented biological replicate counts.\n\n"
            "| File | Meaning |\n| --- | --- |\n"
            "| projection_records.parquet | Full one-vote projection, with occurrence lookup |\n"
            "| projection_inspection.tsv | Up to 20 example peptides per evidence category |\n"
            "| projection_coverage.json | Full source/occurrence/peptide census |\n"
            "| projection_label_policy.json | Original and consumer label policies |\n"
            "| admitted.tsv / admitted.parquet | Exact 10,000-record reservoir membership |\n"
            "| source_receipt.json | Source hashes, citations, cell/search/assay contracts |\n"
            "| split_support.* / row_split_support.csv / mil_split_support.csv | "
            "Canonical support audit; admitted is not a training partition |\n"
            "| output_coverage.* | Declared output support, including unsupported outputs |\n"
            "| summary.json | Runtime, hardware, software and limitations |\n"
            "| manifest.json | Payload hashes and derivation |\n\n"
            f"Original frozen evidence: {SOURCE}\n\n"
            "Multiple candidate occurrences mask parent and flank inputs. Their complete "
            "source occurrences remain recoverable by frozen bundle + peptide. Source "
            "observation IDs exist only for positives; absence has candidate linkage. "
            "The legacy census's untraceable_rows means no measured observation ID.\n\n"
            "No train/validation/test assignments, fitting, prediction exports or held-out "
            "performance claim. One cultured HeLa search does not establish independent "
            "experiment/donor/specimen generalization. MBR is not direct MS/MS; leading-razor "
            "support is not proof of a unique parent. Training readiness is not established.\n"
        ).encode()
        replacements = {
            "summary.json": encoded(summary),
            "source_receipt.json": encoded(source),
            "README.md": readme,
        }
        inventory = {}
        for name, path in inputs.items():
            payload = replacements.get(name)
            inventory[name] = {
                "sha256": hashlib.sha256(payload).hexdigest()
                if payload is not None
                else digest(path),
                "size_bytes": len(payload) if payload is not None else Path(path).stat().st_size,
            }
        manifest = {
            "schema_version": 1,
            "dataset_id": DATASET,
            "consumer_source_commit": summary["git_commit"],
            "input_manifest_sha256": source["frozen_bundle"]["manifest_sha256"],
            "source_download": SOURCE,
            "artifacts": inventory,
            "original_audit_hashes": {name: item["sha256"] for name, item in records.items()},
            "path_redaction": "Local import and cache paths removed from portable summary/receipt",
            "qualification": "admission_only; no partitions or held-out evaluation",
        }
        manifest_bytes = encoded(manifest)
        with open(outputs[ARCHIVE], "wb") as raw:
            with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as compressed:
                with tarfile.open(fileobj=compressed, mode="w|") as archive:
                    for name in sorted([*inputs, "manifest.json"]):
                        payload = (
                            manifest_bytes if name == "manifest.json" else replacements.get(name)
                        )
                        info = tarfile.TarInfo(name)
                        info.mode = 0o644
                        info.size = (
                            len(payload)
                            if payload is not None
                            else Path(inputs[name]).stat().st_size
                        )
                        if payload is not None:
                            archive.addfile(info, io.BytesIO(payload))
                        else:
                            with open(inputs[name], "rb") as stream:
                                archive.addfile(info, stream)
        for name, payload in (
            ("manifest.json", manifest_bytes),
            ("summary.json", encoded(summary)),
            ("README.md", readme),
        ):
            Path(outputs[name]).write_bytes(payload)
        Path(outputs["SHA256SUMS"]).write_text(
            "".join(
                f"{digest(outputs[name])}  {name}\n"
                for name in (ARCHIVE, "manifest.json", "summary.json", "README.md")
            )
        )

    recipe_id = hashlib.sha256(
        encoded(
            {
                "sources": {name: item["sha256"] for name, item in sources.items()},
                "exporter_sha256": digest(__file__),
            }
        )
    ).hexdigest()
    paths = materialize(
        data_dir() / "presto-scoped-proteomics" / "publications" / DATASET / recipe_id,
        sources,
        transform={
            "version": "presto-admission-portable-v1",
            "options": {"exporter_sha256": digest(__file__)},
        },
        outputs={
            name: {}
            for name in (ARCHIVE, "manifest.json", "summary.json", "README.md", "SHA256SUMS")
        },
        builder=build,
        download_options={"show_progress": False},
    )
    result = {
        name: {"path": str(path), "sha256": digest(path), "size_bytes": Path(path).stat().st_size}
        for name, path in paths.items()
    }
    (EXPERIMENT / "results" / "publication.json").write_bytes(encoded(result))
    print(encoded(result).decode())


if __name__ == "__main__":
    run()
