"""Summarize audited source panels without emitting training observations."""

import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

MAIN = "Source Data_Main Figures.xlsx"
SUPP = "Source Data_Supplementary Figures.xlsx"
GROUPS = {
    MAIN: {
        "structure_or_structure_derived": [
            "Fig 1a",
            "Fig 1b",
            "Fig 1c",
            "Fig 1d",
            "Fig 3f",
            "Fig 3g",
            "Fig 3h",
            "Fig 4c",
            "Fig 4d",
            "Fig 4e",
            "Fig 5e",
            "Fig 5f",
        ],
        "peptide_assay": ["Fig 2h"],
        "reporter_activity_including_some_fits": [
            "Fig 2i",
            "Fig 2j",
            "Fig 4b",
            "Fig 5b",
            "Fig 5c",
            "Fig 5d",
        ],
    },
    SUPP: {
        "structure_or_structure_derived": [
            "Fig S2a",
            "Fig S2b",
            "Fig S2c",
            "Fig S7",
            "Fig S9a",
            "Fig S9b",
            "Fig S9c",
            "Fig S9d",
            "Fig S9e",
        ],
        "peptide_assay": ["Fig S8d"],
        "reporter_activity_including_some_fits": ["Fig S5", "Fig S8c", "Fig S12a"],
        "reporter_fitted_parameters": ["Fig S12b"],
        "image_panel": ["Fig S1b"],
    },
}
REASONS = {
    "structure_or_structure_derived": "Not a peptide turnover or cellular abundance endpoint",
    "peptide_assay": "Narrow biochemical evidence; units/normalization/experiment grouping unresolved; insufficient substrate diversity for the proposed sequence model",
    "reporter_activity_including_some_fits": "Reporter turnover or modulation, not peptide-specific trimming rates; fitted blocks must remain derived",
    "reporter_fitted_parameters": "Derived reporter parameters, not independent raw peptide measurements",
    "image_panel": "No tabular quantitative peptide assay; image interpretation is outside this inventory",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    target = args.out / "qualification.json"
    assert not target.exists()
    meta = json.loads((args.inventory_dir / "inventory.json").read_text())
    with gzip.open(args.inventory_dir / "cells.jsonl.gz", "rt") as stream:
        cells = {(x["workbook"], x["sheet"], x["address"]): x for x in map(json.loads, stream)}
    mapping = {
        (book, sheet): category
        for book, groups in GROUPS.items()
        for category, names in groups.items()
        for sheet in names
    }
    rows = []
    for book in meta["workbooks"]:
        for sheet in book["sheets"]:
            category = mapping.pop((book["name"], sheet["name"]))
            rows.append(
                {
                    "workbook": book["name"],
                    "sheet": sheet["name"],
                    "dimension": sheet["dimension"],
                    "category": category,
                    "n_nonempty_cells": sheet["occupied_cells"],
                    "n_numeric_cells": sheet["cell_types"].get("n", 0),
                    "training_admission": "not_admitted",
                    "reason": REASONS[category],
                }
            )
    assert not mapping
    blocks = []
    for book, sheet, columns, start, end, description in [
        (
            MAIN,
            "Fig 2h",
            "DEFGHI",
            5,
            16,
            "one peptide, three enzyme conditions, duplicate slots across 12 concentrations",
        ),
        (
            SUPP,
            "Fig S8d",
            "DEIJ",
            5,
            13,
            "two peptide probes, duplicate slots across nine time points",
        ),
    ]:
        present, missing, zero, above = [], [], [], []
        for column in columns:
            for row in range(start, end + 1):
                address = f"{column}{row}"
                cell = cells.get((book, sheet, address))
                if cell is None:
                    missing.append(address)
                    continue
                assert cell["type"] == "n"
                value = float(cell["value"])
                present.append(address)
                if value == 0:
                    zero.append(address)
                if value > 100:
                    above.append(address)
        blocks.append(
            {
                "workbook": book,
                "sheet": sheet,
                "description": description,
                "n_expected_response_slots": len(columns) * (end - start + 1),
                "n_numeric_response_cells": len(present),
                "missing_cells": missing,
                "reported_zero_cells": zero,
                "above_100_cells": above,
                "n_training_observations_admitted": 0,
            }
        )
    # A source-layout hazard: the nominal C404S heading precedes a duplicate x column.
    duplicate_axis = all(
        cells[(MAIN, "Fig 2i", f"K{r}")]["value"] == cells[(MAIN, "Fig 2i", f"L{r}")]["value"]
        for r in range(11, 17)
    )
    assert duplicate_axis
    result = {
        "decision": "does_not_qualify_as_sequence_diverse_peptide_processing_training_cohort",
        "n_sheets": len(rows),
        "sheet_categories": dict(Counter(x["category"] for x in rows)),
        "n_nonempty_cells": sum(x["n_nonempty_cells"] for x in rows),
        "n_numeric_cells": sum(x["n_numeric_cells"] for x in rows),
        "candidate_peptide_blocks": blocks,
        "fig2i_duplicate_axis": {
            "range_a": "K11:K16",
            "range_b": "L11:L16",
            "all_equal": duplicate_axis,
            "meaning": "Do not read L as a fourth rate replicate just because L9 labels C404S",
        },
        "n_training_observations_admitted": 0,
        "article_sha256": hashlib.sha256(
            (args.inventory_dir / "article.xml").read_bytes()
        ).hexdigest(),
        "measurement_limits": [
            "Many numbers are structural calculations, uncertainties or fitted summaries; numeric cells are not observation counts",
            "One unmodified peptide and one BPA-modified probe; neither a sequence library nor a product-resolved cellular peptidome",
            "The SIINFEKL condition is a structural measurement, not a measured peptide trimming trajectory",
            "Fig2h rate scale and independent experiment identity are not established by workbook headers",
            "FigS8d response normalization and product identity require qualification; values above 100 remain unchanged",
            "Article describes experiments beyond the two replicate slots shown; do not invent independent repetitions",
            "No APC machinery context, paired precursor abundance, or intracellular concentration labels",
        ],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2) + "\n")
    with (args.out / "sheet-dispositions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
