"""Join the small PRIDE/SDRF inventories; preserve evidence qualifications."""

from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

FAMILY = Path(__file__).resolve().parents[1]
RAW = Path("/private/tmp/presto-mhc2-20261026/human-cell-contrast")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    result = FAMILY / "results/qualification.json"
    assert not result.exists()
    inventory_path = RAW / "pride/PXD024417-files.json"
    sdrf_path = RAW / "sdrf/PXD024417_community_annotated.sdrf.tsv"
    inventory = json.loads(inventory_path.read_text())
    with sdrf_path.open() as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    files = {row["fileName"]: row for row in inventory}
    assert len(files) == len(inventory) == 81
    assert len(rows) == 40
    raw_files = {name for name in files if name.endswith(".raw")}
    assert {row["comment[data file]"] for row in rows} == raw_files
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["source name"]].append(row)
    assert len(grouped) == 8
    samples = []
    for name, group in sorted(grouped.items()):
        assert len(group) == 5
        assert {r["comment[technical replicate]"] for r in group} == set("12345")
        samples.append(
            {
                "source_name": name,
                "biological_replicate_labels": sorted(
                    {r["characteristics[biological replicate]"] for r in group}
                ),
                "technical_replicates": 5,
                "raw_files": [r["comment[data file]"] for r in group],
                "capture_annotations": sorted(
                    {r["characteristics[antibody enrichment]"] for r in group}
                ),
                "mhc_typing_annotations_unverified": sorted(
                    {r["characteristics[mhc typing]"] for r in group}
                ),
                "mhc_typing_method": sorted(
                    {r["characteristics[mhc typing method]"] for r in group}
                ),
            }
        )
    assert {r["characteristics[mhc typing method]"] for r in rows} == {"not available"}
    current = json.loads((FAMILY / "reproduce/current-curation-operation11.json").read_text())[
        "structuredContent"
    ]
    local_lines = (
        (FAMILY / "reproduce/source/pmid_overrides.yaml").read_text().splitlines(keepends=True)
    )
    assert current["content"].splitlines() == [line.rstrip("\n") for line in local_lines[8650:8815]]
    audit = json.loads((FAMILY / "results/audit.json").read_text())
    assert audit["independent_count_and_position_replay"]
    source = {
        "status": "source_audit_complete_not_admitted_for_paired_condition_fitting",
        "agent_model": "Codex / GPT-6",
        "source_export": {"path": audit["source_path"], "sha256": audit["source_sha256"]},
        "code_sha256": sha(__file__),
        "primary_xml_sha256": sha(RAW / "primary/PMC9724678.xml"),
        "pride_inventory_sha256": sha(inventory_path),
        "sdrf_sha256": sha(sdrf_path),
        "current_curation_github_blob": current["sha"],
        "current_curation_line_content_equal": True,
        "class_ii_rows": 11875,
        "single_condition_rows": 5980,
        "shared_condition_rows": 5895,
        "all_class_ii_restrictions": "class_only; no candidate alpha/beta allele set",
        "quantitative_class_ii_values": 0,
        "samples": samples,
        "file_categories": dict(Counter(x["fileCategory"]["value"] for x in inventory)),
        "paper_evidence": {
            "reference": "PMC9724678, immunopeptidome Results/Methods and Data availability",
            "capture": "pan class II; expressed DRB1*01:02 does not uniquely assign each ligand",
            "binding_estimates": "computational estimates; not measured affinity labels",
            "quantities": "core-epitope relative intensities aggregate peptide MS signals",
            "replicates": (
                "five technical runs; paper reports two siRNA biological experiments; "
                "SDRF maps one source per siRNA arm and two Mock sources"
            ),
            "search": "human proteome; II length12-25; methionine oxidation allowed; FDR5%",
        },
        "unresolved": [
            "Merged export lacks per-run quantities, units and original measurement IDs.",
            "Community SDRF extended MHC typing lacks established primary provenance.",
            "Complete MHC candidates, biological replicates and precursor amounts are unresolved.",
            "Shared memberships are not independent measurements; missing is not negative.",
        ],
        "next": "Qualify primary measurements, complete MHC candidates and replicate metadata.",
        "prediction_metrics": None,
        "test_split_reason": "source qualification only; no model or predictive partition",
    }
    result.write_text(json.dumps(source, indent=2) + "\n")
    (FAMILY / "results/pride-file-inventory.json").write_text(
        json.dumps(inventory, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                key: source[key]
                for key in (
                    "status",
                    "class_ii_rows",
                    "single_condition_rows",
                    "shared_condition_rows",
                    "current_curation_line_content_equal",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
