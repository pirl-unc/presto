"""Independent original-source reconciliation through the public EvidenceStore."""

import csv
import gzip
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

from openpyxl.utils.cell import coordinate_from_string, column_index_from_string

from hitlist.evidence_bindings import LINKS
from hitlist.evidence_store import EvidenceStore
from hitlist.native_peptidomics_schema import RELATIONS

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "reproduce/launch.json").read_text())
ASSETS = {k: Path(v) for k, v in CFG["original_assets"].items()}
CONTRACT = CFG["source_contract"]
STORE = EvidenceStore.open(CFG["package"])
REPORT = {"package": CFG["package"], "training_or_holdout_access": False}


def rows(name, **kwargs):
    for result in STORE.iter_query("native_" + name, **kwargs):
        yield from result.table.to_pylist()


def original_table(name):
    with ASSETS[name].open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


primary_keys = {}
relation_counts = {}
for name, (_, _, key) in RELATIONS.items():
    values = [r[key] for r in rows(name, select=[key])]
    assert None not in values and len(values) == len(set(values)), name
    primary_keys[name] = set(values)
    relation_counts[name] = len(values)
link_counts = {}
for link in LINKS:
    if not link.source.startswith("native_"):
        continue
    source_name, target_name = link.source[7:], link.target[7:]
    assert link.target_key == RELATIONS[target_name][2]
    checked = 0
    for row in rows(source_name, select=[link.source_key]):
        value = row[link.source_key]
        assert value in primary_keys[target_name] or (
            value is None and link.cardinality == "optional_one"
        ), (link.source, link.name, value)
        checked += 1
    link_counts[link.source + "." + link.name] = checked
REPORT.update(relation_rows=relation_counts, verified_relationship_rows=link_counts)
del primary_keys
print("All public primary keys and foreign keys verified", flush=True)

# Reparse original CSV independently of Hitlist's TSV reader and verify ordered
# headers, every literal cell, row number and original-source asset identity.
record_identity = {}
original_counts = {}
tsvs = [
    "peptides.txt",
    "modificationSpecificPeptides.txt",
    "evidence.txt",
    "summary.txt",
    "parameters.txt",
    CONTRACT["design"],
]
for filename in tsvs:
    with ASSETS[filename].open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, delimiter="\t")
        headers = next(reader)
        projected = rows("source_records", where={"source_file": filename})
        count = 0
        for number, (original, row) in enumerate(
            zip(reader, projected, strict=True), 2
        ):
            assert row["source_row_index"] == number
            assert json.loads(row["source_row_json"]) == {
                "headers": headers,
                "values": original,
            }
            record_identity[row["source_record_id"]] = (filename, number)
            count += 1
        original_counts[filename] = count
for filename in CONTRACT["metadata_json"]:
    projected = list(rows("source_records", where={"source_file": filename}))
    assert len(projected) == 1
    assert json.loads(projected[0]["source_row_json"]) == json.loads(
        ASSETS[filename].read_text()
    )
for filename in CONTRACT["metadata_xml"]:
    tree = ET.parse(ASSETS[filename])
    projected = rows("source_records", where={"source_file": filename})
    for node, row in zip(tree.iter(), projected, strict=True):
        assert json.loads(row["source_row_json"]) == dict(
            tag=node.tag, text=node.text, tail=node.tail, attributes=node.attrib
        )
REPORT["original_tsv_rows_reconciled"] = original_counts

peptide_original = original_table("peptides.txt")
form_original = original_table("modificationSpecificPeptides.txt")
feature_original = original_table("evidence.txt")
entities = {
    "peptide_sequence": ("peptides", "peptide_id", peptide_original),
    "modification_specific_peptide": ("forms", "form_id", form_original),
    "feature": ("identifications", "identification_id", feature_original),
}
native_ids, original_by_kind = {}, {}
for kind, (name, key, original) in entities.items():
    projected = list(rows(name))
    lookup = {r["id"]: r for r in original}
    original_by_kind[kind] = lookup
    native_ids[kind] = {r[key]: r["native_id"] for r in projected}
    assert len(projected) == len(lookup)
    for row in projected:
        old = lookup[row["native_id"]]
        assert row["peptide_sequence"] == old["Sequence"]
        assert row["parent_label_raw"] == old["Proteins"]
        assert row["confidence_raw"] == old["PEP"]
        assert row["reverse_raw"] == old["Reverse"]
        assert row["contaminant_raw"] == old["Potential contaminant"]
        if kind != "peptide_sequence":
            assert (
                native_ids["peptide_sequence"][row["peptide_id"]] == old["Peptide ID"]
            )
            assert row["peptide_modifications_raw"] == old["Modifications"]
        if kind == "feature":
            assert (
                native_ids["modification_specific_peptide"][row["form_id"]]
                == old["Mod. peptide ID"]
            )
            assert row["identification_type"] == old["Type"]
            assert row["modified_sequence_raw"] == old["Modified sequence"]
            assert row["charge_raw"] == old["Charge"]
        elif kind == "peptide_sequence":
            assert row["gene_names_raw"] == old["Gene names"]
            assert row["protein_names_raw"] == old["Protein names"]

acquisitions = {r["acquisition_id"]: r for r in rows("acquisitions")}
samples = [r["sample_label"] for r in CONTRACT["acquisitions"]]
expected_quantities = {}
for kind, (_, _, original) in entities.items():
    fields = ["Intensity"]
    if kind != "feature":
        fields += ["Intensity " + s for s in samples]
    if kind == "peptide_sequence":
        fields += ["LFQ intensity " + s for s in samples]
    for row in original:
        for field in fields:
            expected_quantities[kind, row["id"], field] = row[field]
quantity_counts = Counter()
for row in rows("quantities"):
    kind = row["observation_kind"]
    native = native_ids[kind][row["observation_id"]]
    literal = expected_quantities.pop((kind, native, row["source_field"]))
    assert row["quantitative_raw"] == literal
    value = float(literal) if literal else None
    assert value is None or math.isfinite(value)
    assert row["quantitative_value"] == value
    assert row["value_status"] == (
        "missing"
        if value is None
        else "positive"
        if value > 0
        else "zero"
        if value == 0
        else "negative"
    )
    assert row["measurement_unit"] == "source_signal_arbitrary_units"
    old = original_by_kind[kind][native]
    expected_peptide = native if kind == "peptide_sequence" else old["Peptide ID"]
    assert native_ids["peptide_sequence"][row["peptide_id"]] == expected_peptide
    if row["acquisition_id"] is not None:
        sample = acquisitions[row["acquisition_id"]]["sample_label_raw"]
        assert row["experiment_raw"] == sample
        assert (
            sample == old["Experiment"]
            if kind == "feature"
            else row["source_field"].endswith(" " + sample)
        )
    else:
        assert row["source_field"] == "Intensity" and kind != "feature"
    assert row["quantity_kind"] == (
        "source_lfq_intensity"
        if row["source_field"].startswith("LFQ")
        else "source_intensity"
    )
    quantity_counts[kind + ":" + row["quantity_kind"]] += 1
assert not expected_quantities
REPORT["quantities_reconciled_by_grain"] = dict(quantity_counts)
print("Original rows, entity identities and every quantity reconciled", flush=True)

# Full exact FASTA contents are compared, then every origin is compared to the
# independently checked family0424 full-reference audit. No preferred parent.
reference_lines = re.split(r"(?m)^>", ASSETS[CONTRACT["reference"]].read_text())
reference_original = []
for entry in reference_lines:
    if not entry:
        continue
    header, *sequence = entry.splitlines()
    reference_original.append((header.strip(), "".join(s.strip() for s in sequence)))
reference_by_id, accession_hashes = {}, defaultdict(set)
for index, ((header, sequence), row) in enumerate(
    zip(reference_original, rows("references"), strict=True)
):
    digest = hashlib.sha256(sequence.encode()).hexdigest()
    assert (row["header_raw"], row["reference_sequence"]) == (header, sequence)
    assert row["source_slot_index"] == index + 1
    assert row["reference_sequence_sha256"] == digest and row[
        "n_reference_residues"
    ] == len(sequence)
    reference_by_id[row["reference_record_id"]] = (index, row["accession_raw"], digest)
    accession_hashes[row["accession_raw"]].add(digest)
    assert (row["reference_role"] == "engineered_construct") == (
        row["accession_raw"] == "P22222"
    )
with gzip.open(
    Path(CFG["audit0424"]) / "results/reference-origins.csv.gz", "rt"
) as stream:
    original_origins = {
        (
            r["peptide_id"],
            int(r["reference_record_id"]),
            r["parent_accession"],
            r["parent_sequence_sha256"],
            int(r["start_1based"]),
            int(r["end_1based"]),
            r["match_type"],
        )
        for r in csv.DictReader(stream)
    }
graph = defaultdict(set)


def edge(peptide, parent):
    p = "p:" + peptide
    graph[p].add(parent)
    graph[parent].add(p)


origin_counts = Counter()
for row in rows("origins"):
    native = native_ids["peptide_sequence"][row["peptide_id"]]
    index, accession, digest = reference_by_id[row["reference_record_id"]]
    original_origins.remove(
        (
            native,
            index,
            accession,
            digest,
            row["candidate_start_1based"],
            row["candidate_end_1based"],
            row["origin_match_type"],
        )
    )
    edge(native, "s:" + digest)
    origin_counts[row["origin_match_type"]] += 1
assert not original_origins
REPORT["reference_records_verified"] = len(reference_original)
REPORT["origins_verified"] = dict(origin_counts)

# Every explicit OOXML cell is compared using a separate streaming XML traversal.
# Cached raw values and styles are preserved; date-formatted genes remain raw.
NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
publisher = ASSETS[CONTRACT["publisher_workbook"]]
cells = rows("source_cells")
publisher_rows, n_cells = defaultdict(dict), 0
with zipfile.ZipFile(publisher) as archive:
    strings = [
        "".join(n.itertext())
        for n in ET.fromstring(archive.read("xl/sharedStrings.xml"))
    ]
    # This pinned workbook has one worksheet named peptides, independently
    # confirmed in family0424. Preserve exact coordinates rather than row order.
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    sheets = workbook.find(NS + "sheets")
    assert len(sheets) == 1 and sheets[0].get("name") == "peptides"
    with archive.open("xl/worksheets/sheet1.xml") as stream:
        for _, node in ET.iterparse(stream, events=("end",)):
            if node.tag != NS + "c":
                continue
            row = next(cells)
            coordinate = node.get("r")
            column, number = coordinate_from_string(coordinate)
            kind, raw = node.get("t", "n"), node.findtext(NS + "v")
            resolved = strings[int(raw)] if kind == "s" else raw
            formula = node.find(NS + "f")
            assert row["source_cell"] == coordinate and row["sheet_name"] == "peptides"
            assert row["source_row_index"] == number
            assert row["source_column_index"] == column_index_from_string(column)
            assert row["cell_type_raw"] == kind and row["cell_style_raw"] == node.get(
                "s"
            )
            assert (row["cell_value_raw"], row["cell_value_resolved"]) == (
                raw,
                resolved,
            )
            assert row["cell_formula_raw"] == (
                None if formula is None else formula.text
            )
            assert (
                None
                if row["formula_attributes_json"] is None
                else json.loads(row["formula_attributes_json"])
            ) == (None if formula is None else dict(formula.attrib))
            publisher_rows[number][column_index_from_string(column)] = resolved
            record_identity[row["source_record_id"]] = (
                CONTRACT["publisher_workbook"],
                number,
            )
            n_cells += 1
            node.clear()
assert next(cells, None) is None
REPORT["original_ooxml_cells_verified"] = n_cells

# Verify every source parent claim by its original row, entity and reference
# alternatives. Even excluded/unmapped contaminants participate in dependencies.
expected_parents = Counter()
expected_observations = {}
public_ids = {
    kind: {native: key for key, native in mapping.items()}
    for kind, mapping in native_ids.items()
}
for kind, (_, _, original) in entities.items():
    filename = {
        "peptide_sequence": "peptides.txt",
        "modification_specific_peptide": "modificationSpecificPeptides.txt",
        "feature": "evidence.txt",
    }[kind]
    for number, row in enumerate(original, 2):
        native = row["id"] if kind == "peptide_sequence" else row["Peptide ID"]
        expected_observations[filename, number] = public_ids[kind][row["id"]]
        for token in filter(None, row["Proteins"].split(";")):
            # Distinct deposited records can share a full sequence hash.
            expected_parents[filename, number, native, token] += 1
headers = publisher_rows[1]
by_sequence = {r["Sequence"]: r["id"] for r in peptide_original}
for number, values in publisher_rows.items():
    if number == 1:
        continue
    row = {field: values.get(column) for column, field in headers.items()}
    native = by_sequence[row["Sequence"]]
    expected_observations[CONTRACT["publisher_workbook"], number] = public_ids[
        "peptide_sequence"
    ][native]
    for token in filter(None, (row["Proteins"] or "").split(";")):
        expected_parents[CONTRACT["publisher_workbook"], number, native, token] += 1
reference_ids_by_accession = defaultdict(set)
for key, (_, accession, _) in reference_by_id.items():
    reference_ids_by_accession[accession].add(key)
expected_parent_links = Counter()
for (filename, number, native, token), count in expected_parents.items():
    for ref in reference_ids_by_accession[token] or {None}:
        expected_parent_links[filename, number, native, token, ref] += count
    for digest in accession_hashes[token]:
        edge(native, "s:" + digest)
    if not accession_hashes[token]:
        edge(native, "u:" + token)
for row in rows("declared_parents"):
    filename, number = record_identity[row["source_record_id"]]
    assert row["observation_id"] == expected_observations[filename, number]
    native = native_ids["peptide_sequence"][row["peptide_id"]]
    key = (
        filename,
        number,
        native,
        row["parent_accession"],
        row["reference_record_id"],
    )
    assert expected_parent_links[key] > 0
    expected_parent_links[key] -= 1
assert not any(expected_parent_links.values())

components = {}
for native in by_sequence.values():
    start = "p:" + native
    if start in components:
        continue
    pending = [start]
    components[start] = start
    while pending:
        for neighbor in graph[pending.pop()]:
            if neighbor not in components:
                components[neighbor] = start
                pending.append(neighbor)
actual_by_component, components_by_actual = defaultdict(set), defaultdict(set)
for row in rows("dependency_groups"):
    native = native_ids["peptide_sequence"][row["peptide_id"]]
    component = components["p:" + native]
    actual_by_component[component].add(row["dependency_group_id"])
    components_by_actual[row["dependency_group_id"]].add(component)
assert all(len(v) == 1 for v in actual_by_component.values())
assert all(len(v) == 1 for v in components_by_actual.values())
old_to_new, new_to_old = defaultdict(set), defaultdict(set)
with (
    Path(CFG["audit0424"]) / "results/peptide-dependency-groups.csv"
).open() as stream:
    for row in csv.DictReader(stream):
        new = components["p:" + row["peptide_id"]]
        old_to_new[row["dependency_group"]].add(new)
        new_to_old[new].add(row["dependency_group"])
assert all(len(v) == 1 for v in old_to_new.values()), (
    "New source views must not split older dependencies"
)
REPORT["all_source_dependency_components"] = len(actual_by_component)
REPORT["older_components_joined_by_additional_source_views"] = sum(
    len(v) > 1 for v in new_to_old.values()
)
REPORT["old_dependency_components"] = len(old_to_new)
REPORT["date_annotation_discrepancies"] = len(
    list(
        rows(
            "reconciliations",
            where={
                "assertion_status": "date_typed_gene_annotation_original_text_retained"
            },
        )
    )
)
assert REPORT["date_annotation_discrepancies"] == 13
REPORT["assertion_scopes"] = dict(
    Counter(r["source_scope"] for r in rows("assertions"))
)
assert all(
    r["specimen_id"] is None and r["aliquot_id"] is None
    for r in rows("material_contexts")
)
REPORT["passed"] = True
(ROOT / "results/independent-verification.json").write_text(
    json.dumps(REPORT, indent=2) + "\n"
)
print(json.dumps(REPORT), flush=True)
