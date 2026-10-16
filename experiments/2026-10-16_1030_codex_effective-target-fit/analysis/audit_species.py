"""Expose native species axes for the unchanged frozen membership (#115)."""

import csv
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from datacache import materialize

FAMILY = Path(__file__).resolve().parents[1]
SOURCE = {
    "url": "https://github.com/pirl-unc/hitlist/releases/download/human-cohort-corrected-v1/training.parquet",
    "sha256": "f902341e90c1c57defc93f4bc540ffcbea1b41061afb4492ddd8aaa7642c65bd",
    "size": 188175209,
}
NATIVE_MANIFEST_SHA256 = (
    "b180eba804532f633a6d61bb350584dbe24f8df4b4b96f02c6ec492c60144372"
)


def pin(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"sha256": digest.hexdigest(), "size_bytes": path.stat().st_size}


def main():
    frozen = json.loads((FAMILY / "reproduce/frozen-inputs.json").read_text())
    columns = [
        "evidence_row_id",
        "peptide",
        "pmid",
        "reference_iri",
        "reference_title",
        "assay_iri",
        "response_measured",
        "mhc_restriction",
        "mhc_species",
        "mhc_species_source",
        "mhc_species_context_disagrees",
        "host",
        "source_organism",
        "cell_line_name",
        "cell_type",
        "source_tissue",
        "species_context_id",
        "species_context_status",
        "species_context_scope",
        "species_context_kind",
        "species_context_reference",
        "presenting_species",
        "in_vivo_host_species",
        "lineage_host_species",
        "introduced_mhc_species",
        "supported_foreign_species",
        "reviewed_candidate_species",
        "species_context_note",
    ]

    def build(inputs, outputs):
        assert pin(inputs["training.parquet"]) == {
            "sha256": SOURCE["sha256"],
            "size_bytes": SOURCE["size"],
        }
        with Path(inputs["membership.tsv"]).open() as handle:
            membership = list(csv.DictReader(handle, delimiter="\t"))
        selected = {row["evidence_row_id"]: row for row in membership}
        assert len(selected) == len(membership) == 9053
        rows = {}
        for batch in pq.ParquetFile(inputs["training.parquet"]).iter_batches(
            batch_size=20000, columns=columns
        ):
            data = batch.to_pydict()
            for index, evidence_id in enumerate(data["evidence_row_id"]):
                if evidence_id not in selected:
                    continue
                assert evidence_id not in rows
                record = {name: values[index] for name, values in data.items()}
                original = selected[evidence_id]
                assert record["peptide"] == original["peptide"]
                rows[evidence_id] = {
                    "sample_id": original["sample_id"],
                    "source_partition": original["source_partition"],
                    **record,
                }
        assert rows.keys() == selected.keys()
        ordered = [rows[row["evidence_row_id"]] for row in membership]
        with Path(outputs["source-species.tsv"]).open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["sample_id", "source_partition", *columns],
                delimiter="\t",
            )
            writer.writeheader()
            writer.writerows(ordered)
        nonhuman_hosts = [
            row for row in ordered if row["host"] != "Homo sapiens (human)"
        ]
        assert len(nonhuman_hosts) == 8
        assert all(
            row["host"] == "Rattus norvegicus (brown rat)" for row in nonhuman_hosts
        )
        assert all(row["mhc_species"] == "Homo sapiens" for row in ordered)
        assert all(
            row["response_measured"] == "ligand presentation" for row in nonhuman_hosts
        )
        report = {
            "native_manifest_sha256": NATIVE_MANIFEST_SHA256,
            "training_source": SOURCE,
            "membership": frozen["membership.tsv"],
            "n_verified_observations": len(ordered),
            "n_observations_by_axis": {
                name: dict(Counter(row[name] or "unknown" for row in ordered))
                for name in (
                    "host",
                    "mhc_species",
                    "source_organism",
                    "species_context_status",
                )
            },
            "n_observations_with_nonhuman_recorded_host_by_partition": dict(
                Counter(row["source_partition"] for row in nonhuman_hosts)
            ),
            "nonhuman_host_rows": nonhuman_hosts,
            "issue": "https://github.com/pirl-unc/presto/issues/115",
            "limitations": [
                "Recorded source organism, host and introduced MHC are distinct axes; partial native attribution is not independently resolved here.",
                "The released consumer uses host for processing_species and MHC-species supervision; this historical run is unchanged.",
                "All eight affected observations are elution rows, not direct IC50. Their effect through shared objectives or representations is unquantified.",
                "Compact historical context omitted these species axes. This additive table preserves them without rewriting any frozen input.",
            ],
        }
        Path(outputs["species-audit.json"]).write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )

    paths = materialize(
        Path.home()
        / ".hitlist/presto-effective-target-fit/species-audit"
        / pin(__file__)["sha256"],
        {"training.parquet": SOURCE, "membership.tsv": frozen["membership.tsv"]},
        transform={
            "version": "native-species-inspection-v1",
            "options": {"script_sha256": pin(__file__)["sha256"]},
        },
        outputs={"source-species.tsv": {}, "species-audit.json": {}},
        builder=build,
    )
    receipt = {name: {"path": str(path), **pin(path)} for name, path in paths.items()}
    for name, path in paths.items():
        shutil.copyfile(path, FAMILY / "results" / name)
    (FAMILY / "results/species-audit-receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
