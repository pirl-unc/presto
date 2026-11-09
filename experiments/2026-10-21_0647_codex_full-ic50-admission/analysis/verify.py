"""Independent replay of the frozen actual source; imports no Presto adapter."""

from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path

import pyarrow.parquet as pq


FAMILY = Path(__file__).resolve().parents[1]
RECEIPT = json.loads((FAMILY / "reproduce/executed-launch.json").read_text())
REPORT = json.loads((FAMILY / "results/admission.json").read_text())
AA = set("ACDEFGHIKLMNPQRSTVWY")
UNITS = {"pM": 0.001, "nM": 1, "uM": 1000, "mM": 1e6, "M": 1e9}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def catalog_assignment(row):
    if row["source"] != "imgt" or row["species"] != "Homo sapiens":
        return None
    if not row["source_id"] or not row["two_field_allele"].startswith("HomoSapiens-"):
        return None
    if row["is_functional"] != "True" or any(
        row[key] != "False" for key in ("is_null", "is_questionable", "is_pseudogene")
    ):
        return None
    if row["groove_status"] != "ok" or row["groove_flags"]:
        return None
    cls, chain = row["mhc_class"], row["chain"]
    if (cls, chain) not in (("I", "alpha"), ("II", "alpha"), ("II", "beta")):
        return None
    slots = (
        ("groove1", "groove2") if cls == "I" else ("groove1",) if chain == "alpha" else ("groove2",)
    )
    end = 0
    for slot in ("groove1", "groove2"):
        sequence = row[slot]
        if slot not in slots:
            if sequence:
                return None
            continue
        if (
            not 60 <= len(sequence) <= 130
            or set(sequence) - AA
            or row[slot + "_len"] != str(len(sequence))
        ):
            return None
        start = row["sequence"].find(sequence, end)
        if start < 0:
            return None
        end = start + len(sequence)
    return cls, chain, row["groove1"], row["groove2"]


def names(raw):
    return [
        part if part.startswith("HLA-") else "HLA-" + part
        for part in raw["mhc_restriction"].split("/")
    ]


for source in RECEIPT["sources"].values():
    require(sha(source["path"]) == source["sha256"], "original source hash changed")
with Path(RECEIPT["sources"]["catalog"]["path"]).open() as stream:
    catalog_rows = list(csv.DictReader(stream))
catalog = defaultdict(lambda: defaultdict(list))
for i, row in enumerate(catalog_rows):
    assignment = catalog_assignment(row)
    if assignment:
        catalog[row["two_field_allele"].replace("HomoSapiens-", "HLA-", 1)][assignment].append(i)
used = json.loads((FAMILY / "results/used-catalog-aliases.json").read_text())
for name, aliases in used.items():
    require(len(catalog[name]) == 1, "used catalog name is unresolved or ambiguous")
    require(
        {alias["source_row"] for alias in aliases} == set(next(iter(catalog[name].values()))),
        "catalog aliases omitted or invented",
    )
    for alias in aliases:
        require(
            {key: value for key, value in alias.items() if key != "source_row"}
            == catalog_rows[alias["source_row"]],
            "catalog alias differs from original source",
        )

manifests = {}
for condition, descriptor in REPORT["manifests"].items():
    require(sha(descriptor["path"]) == descriptor["sha256"], "manifest changed")
    manifests[condition] = json.loads(Path(descriptor["path"]).read_text())
exact_sources = {item["path"]: item for item in manifests["exact"]["sources"]}
full_sources = {item["path"]: item for item in manifests["with-censor"]["sources"]}
require(set(exact_sources) < set(full_sources), "censored condition has no added training source")
require(
    all(value == full_sources[key] for key, value in exact_sources.items()),
    "shared source descriptor differs between conditions",
)
rows, by_ordinal, original_ids = [], {}, set()
for name, descriptor in full_sources.items():
    path = Path(REPORT["manifests"]["with-censor"]["path"]).parent / name
    require(sha(path) == descriptor["sha256"], "observation source changed")
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            ordinal = row["provenance"]["source_ordinal"]
            require(ordinal not in by_ordinal, "source row receives multiple votes")
            require(row["source"] == descriptor["source"], "source label differs")
            if name not in exact_sources:
                require(
                    row["split"] == "train" and row["qualifier"] != "=",
                    "conditions differ beyond training censor inclusion",
                )
            elif row["split"] == "train":
                require(row["qualifier"] == "=", "exact-only condition trains on a bound")
            original = row["provenance"]["original_measurement_id"]
            require(original not in original_ids, "duplicate original assay admitted")
            original_ids.add(original)
            rows.append(row)
            by_ordinal[ordinal] = row

# The actual frozen source has no eligible duplicate assay IRIs. Verify that
# fact below; the separate adapter tests cover conflicting/re-exported assays.
counts, reasons, raw_candidates = Counter(), Counter(), set()
aliases_used, target_inputs = set(), defaultdict(set)
unit_counts, above_50k = Counter(), Counter()
with gzip.open(FAMILY / "results/source-dispositions.jsonl.gz", "rt") as ledger:
    ordinal = 0
    for batch in pq.ParquetFile(RECEIPT["sources"]["binding"]["path"]).iter_batches(
        batch_size=16384
    ):
        for raw in batch.to_pylist():
            ordinal += 1
            disposition = json.loads(next(ledger))
            require(
                disposition["ordinal"] == ordinal and disposition["assay_iri"] == raw["assay_iri"],
                "disposition/source order differs",
            )
            require(
                disposition["source"] == raw["source"] and disposition["class"] == raw["mhc_class"],
                "raw source/class changed",
            )
            counts[disposition["disposition"]] += 1
            reason = disposition["reason"]
            if disposition["disposition"] != "admitted":
                reasons[reason] += 1
                valid_reason = {
                    "outside_direct_iedb_cedar_ic50": lambda raw=raw: (
                        raw["source"] not in ("iedb", "cedar")
                        or raw["response_measured"]
                        != "half maximal inhibitory concentration (IC50)"
                    ),
                    "not_explicit_purified_mhc": lambda raw=raw: (
                        not isinstance(raw["assay_method"], str)
                        or not raw["assay_method"].startswith("purified MHC/")
                    ),
                    "modified_or_unspecified_peptidoform": lambda raw=raw: (
                        raw["has_ptm"] is not False or bool(raw["peptide_modifications"])
                    ),
                    "non_peptide_ligand": lambda raw=raw: bool(raw["is_non_peptide_ligand"]),
                    "unsupported_peptide_or_class": lambda raw=raw: (
                        not isinstance(raw["peptide"], str)
                        or bool(set(raw["peptide"]) - AA)
                        or raw["mhc_class"] not in ("I", "II")
                        or not (9 if raw["mhc_class"] == "II" else 1) <= len(raw["peptide"]) <= 512
                    ),
                    "nonpositive_or_nonfinite_measurement": lambda raw=raw: (
                        raw["quantitative_value"] is None
                        or not math.isfinite(raw["quantitative_value"])
                        or raw["quantitative_value"] <= 0
                    ),
                    "not_explicit_human_restriction": lambda raw=raw: (
                        not isinstance(raw["mhc_restriction"], str)
                        or not raw["mhc_restriction"].startswith("HLA-")
                    ),
                    "missing_explicit_mhc_partner": lambda raw=raw: (
                        len(names(raw)) != (1 if raw["mhc_class"] == "I" else 2)
                    ),
                    "unresolved_catalog_name": lambda raw=raw: any(
                        name not in catalog for name in names(raw)
                    ),
                }
                require(
                    reason in valid_reason and valid_reason[reason](),
                    f"unsupported exclusion/deferral at source row {ordinal}",
                )
                continue
            require(ordinal in by_ordinal, "admitted source row has no canonical observation")
            row = by_ordinal[ordinal]
            require(row["provenance"]["raw_record"] == raw, "raw source columns changed")
            require(
                raw["source"] in ("iedb", "cedar")
                and raw["response_measured"] == "half maximal inhibitory concentration (IC50)",
                "wrong assay",
            )
            require(
                raw["assay_method"].startswith("purified MHC/")
                and raw["has_ptm"] is False
                and not raw["peptide_modifications"]
                and not raw["is_non_peptide_ligand"],
                "unsupported assay/peptidoform",
            )
            peptide, cls = raw["peptide"], raw["mhc_class"]
            require(
                cls in ("I", "II")
                and not set(peptide) - AA
                and (9 if cls == "II" else 1) <= len(peptide) <= 512,
                "unsupported canonical input",
            )
            require(
                row["endpoint"] == "ic50"
                and row["peptide"] == peptide
                and not row["identity_labels"]
                and not row["tcr_a"]
                and not row["tcr_b"],
                "measurement was relabelled",
            )
            value, unit = raw["quantitative_value"], raw["measurement_units"]
            require(
                unit in UNITS and math.isfinite(value) and value > 0, "invalid physical measurement"
            )
            qualifier = (
                "="
                if raw["measurement_inequality"] in (None, "")
                else raw["measurement_inequality"]
            )
            text = raw["quantitative_measurement"].strip()
            prefix = (
                text[:2]
                if text[:2] in ("<=", ">=")
                else text[:1]
                if text[:1] in ("=", "<", ">")
                else ""
            )
            require(not prefix or prefix == qualifier, "raw text bound disagrees")
            require(
                float(text[len(prefix) :]) == value and qualifier in ("=", "<", "<=", ">", ">="),
                "raw numeric target disagrees",
            )
            require(
                (row["value"], row["unit"], row["qualifier"]) == (value, unit, qualifier),
                "original value/unit/bound changed",
            )
            actual_names = names(raw)
            require(len(actual_names) == (1 if cls == "I" else 2), "missing explicit partner")
            assignments = []
            for name in actual_names:
                require(name in used and len(catalog[name]) == 1, "unqualified catalog resolution")
                assignments.append(next(iter(catalog[name])))
                aliases_used.add(name)
            require(all(item[0] == cls for item in assignments), "catalog/source class disagrees")
            by_chain = {item[1]: item for item in assignments}
            require(
                set(by_chain) == ({"alpha"} if cls == "I" else {"alpha", "beta"}),
                "incorrect chain roles",
            )
            first, second = by_chain["alpha"][2], by_chain["alpha" if cls == "I" else "beta"][3]
            require(len(row["mhc"]) == 1, "candidate pair invented")
            mhc = row["mhc"][0]
            require(
                (mhc["groove1"], mhc["groove2"], mhc["mhc_class"], mhc["species1"], mhc["species2"])
                == (first, second, cls, "human", "human"),
                "resolved grooves/species changed",
            )
            require(
                mhc["provenance"]["catalog_alleles"] == actual_names
                and mhc["provenance"]["partner_imputation"] is False,
                "partner provenance differs",
            )
            assay = raw["assay_iri"]
            require(
                assay and assay not in raw_candidates,
                "actual eligible original assays are not unique",
            )
            raw_candidates.add(assay)
            lineage = ["assay_iri", assay, "", "ic50"]
            require(disposition["measurement_lineage"] == lineage, "source lineage changed")
            expected_id = (
                "measurement:"
                + hashlib.sha256(json.dumps(lineage, sort_keys=True).encode()).hexdigest()
            )
            require(
                row["observation_id"] == expected_id,
                "canonical original-measurement identity differs",
            )
            require(
                row["provenance"]["export_aliases"]
                == [
                    {
                        "source": row["source"],
                        "export_id": f"hitlist-v4-row:{ordinal}",
                        "group_id": row["group_id"],
                    }
                ],
                "unexpected source aliases",
            )
            target = math.log10(value * UNITS[unit])
            target_inputs[(peptide, cls, first, second)].add((target, qualifier))
            unit_counts[unit] += 1
            if value * UNITS[unit] > 50000:
                above_50k[row["split"]] += 1
    require(next(ledger, None) is None, "extra raw disposition rows")
require(
    counts == REPORT["dispositions"] and ordinal == REPORT["raw_rows"], "raw conservation differs"
)
require(reasons == REPORT["initial_reasons"], "source reason totals differ")
require(
    len(rows) == len(raw_candidates) == REPORT["admitted_rows"] == REPORT["candidate_rows"],
    "admitted candidate conservation failed",
)
require(aliases_used == set(used), "catalog usage receipt differs")
contradictions = sum(len(values) > 1 for values in target_inputs.values())
require(
    contradictions == REPORT["independent_input_contradictions"],
    "independent disagreement count differs",
)

historical = defaultdict(set)
with Path(RECEIPT["sources"]["historical_membership"]["path"]).open() as stream:
    for item in csv.DictReader(stream, delimiter="\t"):
        historical[item["peptide"]].add(
            "validation" if item["source_partition"] == "val" else item["source_partition"]
        )
with Path(RECEIPT["sources"]["memorization"]["path"]).open() as stream:
    memorized = [json.loads(line) for line in stream]
for item in memorized:
    historical[item["peptide"]].add("train")
groups, membership = defaultdict(set), Counter()
for row in rows:
    # Each eligible assay is unique in this actual source, verified above, so
    # each connected component is exactly one peptide's measurements.
    component = hashlib.sha256(json.dumps([row["peptide"]]).encode()).hexdigest()
    require(row["group_id"] == "ic50-component:" + component, "component hash differs")
    old = historical.get(row["peptide"], set())
    require(len(old) <= 1, "conflicting historical assignments admitted")
    bucket = int(hashlib.sha256(f"{RECEIPT['salt']}:{component}".encode()).hexdigest(), 16) % 100
    expected = (
        next(iter(old))
        if old
        else "train"
        if bucket < 80
        else "validation"
        if bucket < 90
        else "test"
    )
    require(row["split"] == expected, "frozen partition/exposure violated")
    require(
        row["provenance"]["partition_basis"] == ("historical" if old else "novel_component_hash"),
        "partition origin differs",
    )
    groups[row["peptide"]].add(row["split"])
    membership[row["split"]] += 1
require(all(len(value) == 1 for value in groups.values()), "peptide overlap")
output = {
    "status": "passed",
    "raw_rows_and_dispositions_replayed": ordinal,
    "admitted_original_measurements_replayed": len(rows),
    "used_catalog_names_replayed": len(used),
    "original_measurements_unique": True,
    "independent_disagreeing_inputs_retained": contradictions,
    "raw_units": dict(unit_counts),
    "unclipped_measurements_above_50000_nM": dict(above_50k),
    "partitions": dict(membership),
    "peptide_groups": len(groups),
    "paired_conditions_differ_only_by_training_censors": True,
    "historical_assignments_and_memorization_exposure_preserved": True,
    "limitations": (
        "source/target/split replay, not homology/study/donor independence or predictive evaluation"
    ),
}
(FAMILY / "results/independent-replay.json").write_text(
    json.dumps(output, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(output, indent=2), flush=True)
