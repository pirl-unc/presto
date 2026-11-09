"""Independent original-value contrast replay; no predictor or effect summaries."""

from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import re

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "reproduce/launch.json").read_text())
PACKAGE = Path(CFG["package"])
SOURCE = Path(CFG["source_package"]) / "native_peptidomics/typed"
SAMPLES = ("Cont1", "Cont2", "Cont3", "KD1", "KD2", "KD3")


def rows(directory, name):
    for batch in pq.ParquetFile(directory / (name + ".parquet")).iter_batches():
        yield from batch.to_pylist()


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def split(group):
    bucket = int(digest("pxd018722-native-split-v1|" + group), 16) % 20
    return ("train", "val", "test")[0 if bucket < 14 else 1 if bucket < 17 else 2]


def close(a, b):
    assert math.isclose(a, b, abs_tol=2e-12, rel_tol=2e-13), (a, b)


def original(name):
    with Path(CFG["original_assets"][name]).open(
        encoding="utf-8-sig", newline=""
    ) as stream:
        yield from csv.DictReader(stream, delimiter="\t")


raw = {r["id"]: r for r in original("peptides.txt")}
manifest = json.loads((PACKAGE / "manifest.json").read_text())
for relative, receipt in manifest["files"].items():
    path = PACKAGE / relative
    assert path.stat().st_size == receipt["size_bytes"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == receipt["sha256"]
peptides = {r["peptide_id"]: r for r in rows(SOURCE, "peptides")}
assert len(peptides) == len(raw)
native_to_id = {r["native_id"]: pid for pid, r in peptides.items()}
assert set(native_to_id) == set(raw)
refs = {}
fasta = Path(CFG["original_assets"]["HUMAN_NS3_Cyclin.fasta"]).read_text()
entries = [entry for entry in re.split(r"(?m)^>", fasta) if entry]
reference_index = {}
for index, (entry, row) in enumerate(
    zip(entries, rows(SOURCE, "references"), strict=True)
):
    header, *parts = entry.splitlines()
    sequence = "".join(p.strip() for p in parts)
    checksum = digest(sequence)
    accession = header.split()[0].split("|")[1]
    taxon = re.search(r"(?:^|\s)OX=(\d+)(?:\s|$)", header)
    assert row["header_raw"] == header.strip() and row["reference_sequence"] == sequence
    assert row["reference_sequence_sha256"] == checksum
    refs[row["reference_record_id"]] = dict(
        sequence=sequence,
        checksum=checksum,
        accession=accession,
        taxon=taxon.group(1) if taxon else None,
        engineered=accession == "P22222",
    )
    reference_index[index] = row["reference_record_id"]
del fasta, entries
origins = defaultdict(list)
with gzip.open(CFG["origin_audit"], "rt") as stream:
    for row in csv.DictReader(stream):
        pid = native_to_id[row["peptide_id"]]
        rid = reference_index[int(row["reference_record_id"])]
        assert refs[rid]["checksum"] == row["parent_sequence_sha256"]
        origins[pid].append((rid, row["match_type"]))
declared = defaultdict(list)
for row in rows(SOURCE, "declared_parents"):
    declared[row["peptide_id"]].append(row)
source_origins = defaultdict(list)
for row in rows(SOURCE, "origins"):
    source_origins[row["peptide_id"]].append(row)
modified, feature_types = defaultdict(set), defaultdict(set)
for row in original("modificationSpecificPeptides.txt"):
    for label in SAMPLES:
        literal = row["Intensity " + label]
        if literal and float(literal) > 0 and row["Modifications"] != "Unmodified":
            modified[row["Peptide ID"], label].add(row["Modifications"] or "unreported")
for row in original("evidence.txt"):
    if row["Intensity"] and float(row["Intensity"]) > 0:
        feature_types[row["Peptide ID"], row["Experiment"]].add(row["Type"])
groups = {r["peptide_id"]: r for r in rows(SOURCE, "dependency_groups")}
assert groups.keys() == peptides.keys()
observed_groups = {r["peptide_id"]: r for r in rows(PACKAGE / "typed", "components")}
assert observed_groups.keys() == groups.keys()
for pid, row in groups.items():
    assert observed_groups[pid] == {
        k: v
        for k, v in {**row, "split": split(row["dependency_group_id"])}.items()
        if k in {"peptide_id", "dependency_group_id", "split", "dependency_policy"}
    }

# Eligibility is evaluated from original quantities/confidence and independent
# origin/reference evidence, not from the recipe's eligible flag or target.
reasons, parents, eligible = {}, {}, defaultdict(list)
for pid, source in peptides.items():
    row = raw[source["native_id"]]
    seq = row["Sequence"]
    assert seq == source["peptide_sequence"]
    why = set()
    if not seq or not 8 <= len(seq) <= 25 or set(seq) - set("ACDEFGHIKLMNPQRSTVWY"):
        why.add("outside_sequence_search_support")
    if row["Reverse"]:
        why.add("source_reverse_flag")
    if row["Potential contaminant"]:
        why.add("source_contaminant_flag")
    try:
        pep = float(row["PEP"])
    except ValueError:
        pep = math.nan
    if not math.isfinite(pep) or not 0 <= pep <= 0.01:
        why.add("invalid_or_above_threshold_PEP")
    alternatives = origins[pid]
    checksums = {refs[rid]["checksum"] for rid, _ in alternatives}
    parent = next(iter(checksums)) if len(checksums) == 1 else None
    parents[pid] = parent
    if not alternatives:
        why.add("no_deposited_reference_origin")
    elif len(checksums) != 1:
        why.add("multiple_complete_parent_sequences")
    if any(kind != "literal" for _, kind in alternatives):
        why.add("nonliteral_origin_alternative")
    if any(refs[rid]["taxon"] != "9606" for rid, _ in alternatives):
        why.add("nonhuman_or_unresolved_taxonomy")
    if any(refs[rid]["engineered"] for rid, _ in alternatives):
        why.add("engineered_origin_alternative")
    if any(r["reference_record_id"] is None for r in declared[pid]):
        why.add("unresolved_declared_parent")
    if parent and any(
        refs[r["reference_record_id"]]["checksum"] != parent
        for r in declared[pid]
        if r["reference_record_id"] is not None
    ):
        why.add("declared_parent_disagrees_with_unique_origin")
    for label in SAMPLES:
        try:
            value = float(row["Intensity " + label])
        except ValueError:
            value = math.nan
        if not math.isfinite(value) or value <= 0:
            why.add("nonpositive_or_missing_source_signal")
    reasons[pid] = why
    if not why:
        eligible[parent].append(pid)
for pid, why in reasons.items():
    if parents[pid] and len(eligible[parents[pid]]) < 2:
        why.add("parent_has_fewer_than_two_eligible_peptides")
reference = {
    parent: min(
        members,
        key=lambda p: digest(
            "pxd018722-native-reference-v1|"
            + parent
            + "|"
            + peptides[p]["peptide_sequence"]
        ),
    )
    for parent, members in eligible.items()
    if len(members) >= 2
}
decisions = {r["peptide_id"]: r for r in rows(PACKAGE / "typed", "decisions")}
assert decisions.keys() == peptides.keys()
for pid, row in decisions.items():
    assert set(json.loads(row["reasons_json"])) == reasons[pid], pid
    assert row["parent_sequence_sha256"] == parents[pid]
    group = groups[pid]["dependency_group_id"]
    assert (row["dependency_group_id"], row["split"]) == (group, split(group))
    expected = (
        "excluded"
        if reasons[pid]
        else "reference"
        if pid == reference[parents[pid]]
        else "contrast"
    )
    assert row["status"] == expected
    assert row["reference_peptide_id"] == (
        None if reasons[pid] else reference[parents[pid]]
    )
    assert json.loads(row["candidate_origins_json"]) == source_origins[pid]
    assert json.loads(row["declared_parents_json"]) == declared[pid]
    native = peptides[pid]["native_id"]
    assert json.loads(row["modified_form_support_json"]) == {
        label: sorted(modified[native, label]) for label in SAMPLES
    }
    assert json.loads(row["feature_types_by_acquisition_json"]) == {
        label: sorted(feature_types[native, label]) for label in SAMPLES
    }
    values = json.loads(row["source_quantities_json"])
    assert [q["sample_label_raw"] for q in values] == list(SAMPLES)
    for q in values:
        literal = raw[peptides[pid]["native_id"]]["Intensity " + q["sample_label_raw"]]
        assert q["quantitative_raw"] == literal
        if literal:
            assert q["quantitative_value"] == float(literal)

parent_rows = {
    r["parent_sequence_sha256"]: r for r in rows(PACKAGE / "typed", "parents")
}
assert parent_rows.keys() == reference.keys()
for parent, row in parent_rows.items():
    assert digest(row["parent_sequence"]) == parent
    assert row["reference_peptide_id"] == reference[parent]

expected_pairs = {
    (pid, reference[parent])
    for parent, members in eligible.items()
    if len(members) >= 2
    for pid in members
    if pid != reference[parent]
}
examples = list(rows(PACKAGE / "typed", "examples"))
assert len(examples) == len(expected_pairs)
assert {
    (r["peptide_id"], r["reference_peptide_id"]) for r in examples
} == expected_pairs
quantities = {
    r["measurement_id"]: r
    for r in rows(SOURCE, "quantities")
    if r["observation_kind"] == "peptide_sequence"
    and r["quantity_kind"] == "source_intensity"
    and r["acquisition_id"] is not None
}
contributors = defaultdict(list)
for row in rows(PACKAGE / "typed", "contributors"):
    contributors[row["example_id"]].append(row)
parents_in_group, groups_in_split = defaultdict(set), defaultdict(set)
contrasts_per_parent = Counter(r["parent_sequence_sha256"] for r in examples)
for row in examples:
    group, part = row["dependency_group_id"], row["split"]
    parents_in_group[group].add(row["parent_sequence_sha256"])
    groups_in_split[part].add(group)
for row in examples:
    pid, ref = row["peptide_id"], row["reference_peptide_id"]
    a, b = (raw[peptides[p]["native_id"]] for p in (pid, ref))
    ratios = [float(a["Intensity " + s]) / float(b["Intensity " + s]) for s in SAMPLES]
    # Independent arithmetic: log of products of ratios, rather than the
    # production sum of log-intensity differences.
    target = math.log10(math.prod(ratios[3:]) / math.prod(ratios[:3])) / 3
    close(row["target_log10_ratio_change"], target)
    for ratio, observed in zip(
        ratios, json.loads(row["acquisition_log_ratios_json"]), strict=True
    ):
        close(math.log10(ratio), observed)
    group, part, parent = (
        row["dependency_group_id"],
        row["split"],
        row["parent_sequence_sha256"],
    )
    assert (
        groups[pid]["dependency_group_id"]
        == groups[ref]["dependency_group_id"]
        == group
    )
    assert part == split(group) and parents[pid] == parents[ref] == parent
    weight = (
        1
        / len(groups_in_split[part])
        / len(parents_in_group[group])
        / contrasts_per_parent[parent]
    )
    close(row["weight"], weight)
    linked = contributors[row["example_id"]]
    assert len(linked) == len({r["measurement_id"] for r in linked}) == 12
    assert [r["measurement_id"] for r in linked] == json.loads(
        row["measurement_ids_json"]
    )
    assert {(r["sample_label_raw"], r["quantity_role"]) for r in linked} == {
        (s, role) for s in SAMPLES for role in ("peptide", "reference")
    }
    for contribution in linked:
        q = quantities[contribution["measurement_id"]]
        assert all(
            contribution[k] == q[k]
            for k in (
                "peptide_id",
                "source_record_id",
                "acquisition_id",
                "quantitative_raw",
                "quantitative_value",
            )
        )
        coefficient = 1 if contribution["sample_label_raw"].startswith("KD") else -1
        coefficient *= 1 if contribution["quantity_role"] == "peptide" else -1
        assert contribution["peptide_id"] == (
            pid if contribution["quantity_role"] == "peptide" else ref
        )
        close(contribution["target_coefficient"], coefficient / 3)
    close(
        target,
        math.fsum(
            r["target_coefficient"] * math.log10(r["quantitative_value"])
            for r in linked
        ),
    )
covariates = list(rows(PACKAGE / "typed", "covariates"))
assert covariates == [
    {
        k: v
        for k, v in r.items()
        if k not in {"target_log10_ratio_change", "acquisition_log_ratios_json"}
    }
    for r in examples
]
summary = json.loads((PACKAGE / "manifest.json").read_text())["coverage"]
assert summary["n_source_peptides"] == len(raw)
assert summary["n_parents"] == len(reference) and summary["n_contrasts"] == len(
    examples
)
assert summary["n_contributors"] == 12 * len(examples)
assert summary["dispositions"] == dict(Counter(r["status"] for r in decisions.values()))
assert summary["exclusion_reasons"] == dict(
    Counter(reason for why in reasons.values() for reason in why)
)
gate = {}
for part, minimum_groups, minimum_contrasts in [
    ("train", 40, 80),
    ("val", 8, 16),
    ("test", 8, 16),
]:
    members = [r for r in examples if r["split"] == part]
    gate[part] = (
        len(groups_in_split[part]) >= minimum_groups
        and len(members) >= minimum_contrasts
    )
    support = summary["split_support"][part]
    assert support["n_components"] == len(groups_in_split[part])
    assert (
        support["n_contrasts"] == len(members) and support["gate_passed"] == gate[part]
    )
    close(math.fsum(r["weight"] for r in members), 1 if members else 0)
assert summary["qualified_for_screen"] == all(gate.values())
report = dict(
    passed=True,
    n_source_peptides=len(raw),
    n_reference_records=len(refs),
    n_parents=len(reference),
    n_contrasts=len(examples),
    n_measurement_contributions=12 * len(examples),
    all_decisions_original_values_verified=True,
    all_source_components_preserved=True,
    exact_reference_and_split_hashes_verified=True,
    targets_lineage_weights_verified=True,
    source_parent_claims_and_graph_reused_from_independently_verified_0519=True,
    qualified_for_screen=all(gate.values()),
    split_support=summary["split_support"],
    numeric_heldout_target_summaries_disclosed=False,
    predictor_fitted=False,
)
(ROOT / "results/independent-verification.json").write_text(
    json.dumps(report, indent=2) + "\n"
)
print(json.dumps(report), flush=True)
