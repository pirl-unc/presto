"""Independent original-reader checks; does not import the source compiler."""

import csv
import datetime
import hashlib
import io
import json
import math
import re
import resource
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import openpyxl
import pyarrow.parquet as pq
from openpyxl.cell.read_only import EmptyCell
from openpyxl.utils.datetime import to_excel

EXPERIMENT = Path(__file__).resolve().parents[1]
REPOSITORY = EXPERIMENT.parents[1]
PACKAGE = REPOSITORY / "data/mapp2025_source/2026-10-24_2210_canonical"
ROOT = PACKAGE / "goldberg_mapp"


def typed(name, columns=None):
    for batch in pq.ParquetFile(ROOT / "typed" / (name + ".parquet")).iter_batches(columns=columns):
        yield from batch.to_pylist()


def number(raw):
    if raw in ("", None):
        return None, "missing"
    try:
        value = float(raw)
    except (ValueError, TypeError):
        return None, "invalid_literal"
    if not math.isfinite(value):
        return None, "nonfinite"
    return value, "positive" if value > 0 else "zero" if value == 0 else "negative"


def original_rows(archive, member):
    with archive.open(member) as handle:
        rows = csv.reader(io.TextIOWrapper(handle), delimiter="\t")
        header = next(rows)
        for index, values in enumerate(rows, 2):
            assert len(values) == len(header)
            yield index, header, values


def main():
    start = time.monotonic()
    csv.field_size_limit(16 * 1024**2)
    contract = json.loads((ROOT / "contract.json").read_text())
    report = {
        "scope": "independent original-row, all search-intensity, literal workbook-cell, pooled-slot and material checks; compiler replay separately covers every typed relation"
    }
    for asset in contract["assets"]:
        path = ROOT / "source_assets" / asset["filename"]
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        assert digest == asset["sha256"] and path.stat().st_size == asset["size_bytes"]
    acquisitions = {r["raw_file"]: r for r in typed("acquisitions")}
    deposition = openpyxl.load_workbook(
        ROOT / "source_assets" / contract["deposition"], read_only=True
    )
    rows = deposition.active.values
    headers = next(rows)
    for row in rows:
        native = dict(zip(headers, row, strict=True))
        target = acquisitions[native["Raw File"]]
        assert target["sample_label_raw"] == native["Sample Name"]
        assert target["assay_name_raw"] == native["Experiment"]
        assert target["cell_line_name"] == native["Cell Type"]
        assert target["specimen_id"] is None and target["aliquot_id"] is None
    deposition.close()
    source_records = (r for r in typed("source_records") if r["source_file"] == contract["archive"])
    search_counts = {}
    experiment_acquisition = {}
    baseline = Counter()
    header_taxa = defaultdict(set)
    peptides = {r["native_id"]: r for r in typed("peptides")}
    quantities = typed("quantities")
    with zipfile.ZipFile(ROOT / "source_assets" / contract["archive"]) as archive:
        for member in [
            "parameters.txt",
            "summary.txt",
            "proteinGroups.txt",
            "peptides.txt",
            "modificationSpecificPeptides.txt",
            "evidence.txt",
        ]:
            n_rows, n_fields = 0, 0
            for index, headers, values in original_rows(archive, member):
                actual = next(source_records)
                assert actual["source_locator"] == f"{member}:row:{index}"
                assert actual["source_row_index"] == index
                assert json.loads(actual["source_row_json"]) == {
                    "headers": headers,
                    "values": values,
                }
                n_rows += 1
                n_fields += len(values)
                row = dict(zip(headers, values, strict=True))
                if member == "proteinGroups.txt":
                    for header in row["Fasta headers"].split(";"):
                        pieces = header.split("|")
                        taxon = re.findall(r"\bOX=(\d+)(?=\s)", header)
                        if len(pieces) >= 3 and len(taxon) == 1:
                            header_taxa[pieces[1]].add(taxon[0])
                if member == "peptides.txt":
                    p = peptides[row["id"]]
                    for field, column in [
                        ("peptide_sequence", "Sequence"),
                        ("parent_label_raw", "Proteins"),
                        ("source_evidence_ids_raw", "Evidence IDs"),
                        ("source_form_ids_raw", "Mod. peptide IDs"),
                        ("reverse_raw", "Reverse"),
                        ("contaminant_raw", "Potential contaminant"),
                        ("source_group_ids_raw", "Protein group IDs"),
                    ]:
                        assert p[field] == row[column]
                    sets = [header_taxa[accession] for accession in row["Proteins"].split(";")]
                    merged = set().union(*sets)
                    status = "mixed_or_unresolved"
                    if all(sets) and merged == {"9606"}:
                        status = "source_human_only"
                    elif all(sets) and merged and "9606" not in merged:
                        status = "source_nonhuman_only"
                    assert p["source_organism_status"] == status
                if member == "summary.txt" and row["Experiment"]:
                    experiment_acquisition[row["Experiment"]] = acquisitions[row["Raw file"]]
                if member not in (
                    "peptides.txt",
                    "modificationSpecificPeptides.txt",
                    "evidence.txt",
                ):
                    continue
                kind = {
                    "peptides.txt": "peptide_intensity",
                    "modificationSpecificPeptides.txt": "form_intensity",
                    "evidence.txt": "identification_intensity",
                }[member]
                for field, raw in row.items():
                    if field != "Intensity" and not field.startswith("Intensity "):
                        continue
                    q = next(quantities)
                    expected_value, expected_status = number(raw)
                    assert q["source_record_id"] == actual["source_record_id"]
                    assert q["source_field"] == field and q["quantitative_raw"] == raw
                    assert (q["quantitative_value"], q["value_status"]) == (
                        expected_value,
                        expected_status,
                    )
                    assert q["quantity_kind"] == kind
                    acq = None
                    if member == "evidence.txt":
                        acq = acquisitions[row["Raw file"]]
                    elif field != "Intensity":
                        acq = experiment_acquisition[field[len("Intensity ") :]]
                    assert q["acquisition_id"] == (acq["acquisition_id"] if acq else None)
                    if (
                        member == "peptides.txt"
                        and acq
                        and acq["condition_status"] == "source_control_label"
                    ):
                        p = peptides[row["id"]]
                        flags = (
                            "flagged"
                            if row["Reverse"] == "+" or row["Potential contaminant"] == "+"
                            else "unflagged"
                        )
                        baseline[
                            "|".join([flags, p["source_organism_status"], expected_status])
                        ] += 1
            search_counts[member] = {"n_rows": n_rows, "n_literal_fields": n_fields}
    assert next(source_records, None) is None
    assert next(quantities, None) is None
    report["original_search_tables"] = search_counts
    report["baseline_cell_dispositions_from_verified_original_values"] = dict(baseline)

    cells = typed("publisher_cells")
    slots = typed("pooled_slots")
    workbook_counts = {}
    n_slots = 0
    with zipfile.ZipFile(ROOT / "source_assets" / contract["publisher"]) as archive:
        for block in contract["publisher_workbooks"]:
            book = openpyxl.load_workbook(
                io.BytesIO(archive.read(block["member"])), read_only=True, data_only=False
            )
            for sheet in book:
                count, date_count = 0, 0
                for row in sheet:
                    for cell in row:
                        if isinstance(cell, EmptyCell):
                            continue
                        q = next(cells)
                        assert q["source_file"] == contract["publisher"] + "!" + block["member"]
                        assert (
                            q["sheet_name"] == sheet.title and q["source_cell"] == cell.coordinate
                        )
                        assert (
                            q["source_row_index"] == cell.row
                            and q["source_column_index"] == cell.column
                        )
                        value = cell.value
                        if isinstance(value, (datetime.datetime, datetime.date, datetime.time)):
                            assert float(q["cell_value_resolved"]) == to_excel(value, book.epoch)
                            date_count += 1
                        elif isinstance(value, bool):
                            assert q["cell_value_resolved"] == ("1" if value else "0")
                        elif isinstance(value, (int, float)):
                            assert float(q["cell_value_resolved"]) == value
                        elif cell.data_type == "f":
                            assert q["cell_formula_raw"] == value[1:]
                        else:
                            assert q["cell_value_resolved"] == (value or "")
                        count += 1
                        if sheet.title == "MAPP database" and cell.column == 6 and cell.row > 1:
                            for index, raw_slot in enumerate(value.split(";"), 1):
                                label, raw_value = raw_slot.split(":", 1)
                                s = next(slots)
                                assert (
                                    s["source_cell"] == cell.coordinate
                                    and s["source_slot_index"] == index
                                )
                                assert (
                                    s["sample_label_raw"] == label
                                    and s["quantitative_raw"] == raw_value
                                )
                                assert (s["quantitative_value"], s["value_status"]) == number(
                                    raw_value
                                )
                                assert (
                                    s["acquisition_id"] is None
                                    and s["specimen_id"] is None
                                    and s["aliquot_id"] is None
                                )
                                n_slots += 1
                workbook_counts[block["member"] + "!" + sheet.title] = {
                    "n_cells": count,
                    "n_date_styled_cells": date_count,
                }
            book.close()
    assert next(cells, None) is None and next(slots, None) is None
    report["publisher_workbooks"] = workbook_counts
    report["n_pooled_slots_checked"] = n_slots
    links = list(typed("material_links"))
    assert not any(
        r["material_link_precision"] in ("exact_specimen", "exact_aliquot") for r in links
    )
    contexts = list(typed("material_contexts"))
    assert all(r["specimen_id"] is None and r["aliquot_id"] is None for r in contexts)
    report["n_material_contexts"] = len(contexts)
    report["n_material_links"] = len(links)
    report["publisher_reconciliation"] = dict(
        Counter(r["link_status"] for r in typed("publisher_links"))
    )
    report["runtime"] = {
        "wall_seconds": time.monotonic() - start,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    assert report["runtime"]["wall_seconds"] <= 1200
    assert report["runtime"]["peak_rss_bytes"] <= 6 * 1024**3
    (EXPERIMENT / "results/independent-source-replay.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
