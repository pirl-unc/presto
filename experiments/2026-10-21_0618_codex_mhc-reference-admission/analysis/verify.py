"""Independent source replay and split-concentration report; no predictions."""

from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path

FAMILY = Path(__file__).resolve().parents[1]
report = json.loads((FAMILY / "results/admission.json").read_text())
receipt = json.loads((FAMILY / "reproduce/executed-launch.json").read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


source = Path(receipt["source"]["path"])
require(sha(source) == receipt["source"]["sha256"], "source hash changed")
with source.open(newline="") as stream:
    raw = list(csv.DictReader(stream))
with gzip.open(FAMILY / "results/source-dispositions.jsonl.gz", "rt") as stream:
    dispositions = [json.loads(line) for line in stream]
with Path(report["normalized_observations"]["path"]).open() as stream:
    observations = [json.loads(line) for line in stream]
require(
    sha(report["normalized_observations"]["path"]) == report["normalized_observations"]["sha256"],
    "observation hash changed",
)
require(sha(report["manifest"]["path"]) == report["manifest"]["sha256"], "manifest hash changed")
require(len(raw) == len(dispositions) == report["raw_rows"], "row census differs")
source_annotations, parent_groups, sequence_groups = (
    defaultdict(set),
    defaultdict(set),
    defaultdict(set),
)
groups, source_facts = defaultdict(list), 0
for row in observations:
    require(row["endpoint"] == "identity" and row["value"] is None, "unexpected assay target")
    require(not row["peptide"] and not row["tcr_a"] and not row["tcr_b"], "unexpected input")
    require(len(row["mhc"]) == len(row["identity_labels"]) == 1, "not a single-domain annotation")
    mhc = row["mhc"][0]
    require(bool(mhc["groove1"]) != bool(mhc["groove2"]), "partner was invented")
    slot = "groove1" if mhc["groove1"] else "groove2"
    label = "mhc1" if slot == "groove1" else "mhc2"
    sequence, species = mhc[slot], row["identity_labels"][label]
    require(
        mhc["mhc_class"] is None and mhc["class_override"] is None, "class target/input invented"
    )
    require(len(sequence) >= 60 and len(sequence) <= 130, "out-of-scope sequence length")
    require(set(sequence) <= set("ACDEFGHIKLMNPQRSTVWY"), "noncanonical sequence")
    groups[row["group_id"]].append(row)
    sequence_groups[sequence].add((row["group_id"], row["split"]))
    for alias in row["provenance"]["source_aliases"]:
        i = alias["source_row"]
        original = raw[i]
        require(original[slot] == sequence, "wrong source domain")
        require(receipt["taxon_mapping"][original["species"]] == species, "wrong source species")
        require(
            original["groove_status"] == "ok" and not original["groove_flags"],
            "unsupported source flag",
        )
        require(
            original["is_functional"] == "True"
            and all(
                original[key] == "False" for key in ("is_null", "is_questionable", "is_pseudogene")
            ),
            "bad source functional status",
        )
        require(
            original["chain"] in ("alpha", "beta") and original["mhc_class"] in ("I", "II"),
            "bad class/chain",
        )
        require(
            all(
                original[key] == alias[key]
                for key in (
                    "source",
                    "source_id",
                    "two_field_allele",
                    "gene",
                    "species",
                    "mhc_class",
                    "chain",
                )
            ),
            "source alias differs",
        )
        position = alias["protein_position"]
        require(
            original["sequence"][position : position + len(sequence)] == sequence,
            "bad protein coordinate",
        )
        source_annotations[i].add(row["observation_id"])
        parent_groups[i].add((row["group_id"], row["split"]))
        source_facts += 1
for i, disposition in enumerate(dispositions):
    require(disposition["source_row"] == i, "raw row order changed")
    require(
        all(
            disposition[k] == raw[i][k]
            for k in ("source", "source_id", "species", "mhc_class", "chain")
        ),
        "disposition source mismatch",
    )
    require(
        set(disposition["annotations"]) == source_annotations[i], "annotation conservation failed"
    )
    if disposition["status"] in ("admitted_source", "duplicate_alias"):
        expected_domains = 2 if raw[i]["mhc_class"] == "I" else 1
        require(
            len(source_annotations[i]) == expected_domains, "source domains omitted or invented"
        )
    else:
        require(not source_annotations[i], "excluded/deferred row admitted")
require(
    all(len(value) == 1 for value in parent_groups.values()), "original chain spans groups/splits"
)
require(
    all(len(value) == 1 for value in sequence_groups.values()), "exact domain spans groups/splits"
)
for group_id, rows in groups.items():
    keys = sorted(
        (
            "groove1" if r["mhc"][0]["groove1"] else "groove2",
            r["mhc"][0]["groove1"] or r["mhc"][0]["groove2"],
        )
        for r in rows
    )
    component = hashlib.sha256(json.dumps(keys, sort_keys=True).encode()).hexdigest()
    require(group_id.endswith(component), "component digest mismatch")
    bucket = int(hashlib.sha256(f"{receipt['salt']}:{component}".encode()).hexdigest(), 16) % 100
    expected = "train" if bucket < 70 else "validation" if bucket < 85 else "test"
    require(all(row["split"] == expected for row in rows), "frozen split assignment mismatch")
concentration = []
for split in ("train", "validation", "test"):
    for species in receipt["taxon_mapping"].values():
        counts = Counter(
            row["group_id"]
            for row in observations
            if row["split"] == split and species in row["identity_labels"].values()
        )
        n = sum(counts.values())
        concentration.append(
            {
                "split": split,
                "species": species,
                "n": n,
                "components": len(counts),
                "largest_component": max(counts.values(), default=0),
                "largest_component_fraction": max(counts.values(), default=0) / n if n else None,
                "effective_components_by_size": n
                * n
                / sum(value * value for value in counts.values())
                if n
                else None,
            }
        )
with (FAMILY / "results/split-concentration.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(concentration[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(concentration)
output = {
    "status": "passed",
    "raw_rows": len(raw),
    "annotations": len(observations),
    "source_domain_aliases_replayed": source_facts,
    "components": len(groups),
    "split_concentration": concentration,
    "largest_components": [
        {
            "group_id": group,
            "n": len(rows),
            "split": rows[0]["split"],
            "species": sorted(
                {species for row in rows for species in row["identity_labels"].values()}
            ),
            "source_genes": sorted(
                {alias["gene"] for row in rows for alias in row["provenance"]["source_aliases"]}
            ),
        }
        for group, rows in sorted(groups.items(), key=lambda item: -len(item[1]))[:10]
    ],
}
(FAMILY / "results/independent-replay.json").write_text(
    json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n"
)
print(json.dumps(output, indent=2))
