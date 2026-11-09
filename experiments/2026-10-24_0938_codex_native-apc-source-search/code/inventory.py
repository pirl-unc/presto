"""Audit public metadata and nominate source leads, never training labels."""

import argparse
import hashlib
import json
import re
import resource
import time
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile

import openpyxl

TERMS = re.compile(
    r"macrophag|dendritic|monocyt|THP[- ]?1|RAW\s?264|J774|\bBMDC|\bBMDM|"
    r"\bPBMC|\bRaji\b|\bB[- ]cells?\b|lymphoblast",
    re.IGNORECASE,
)


def read(path):
    with ZipFile(path) as archive:
        assert archive.testzip() is None
    book = openpyxl.load_workbook(path, read_only=True, data_only=False)
    sheets = []
    for sheet in book:
        rows = list(sheet.iter_rows(values_only=True))
        assert not any(isinstance(x, str) and x.startswith("=") for row in rows for x in row)
        sheets.append(
            {
                "name": sheet.title,
                "n_rows_including_headers": len(rows),
                "n_columns": sheet.max_column,
                "first_two_rows": rows[:2],
                "rows": rows,
            }
        )
    book.close()
    return sheets


def flat(sheet):
    header, *body = sheet["rows"]
    assert all(isinstance(x, str) for x in header) and len(set(header)) == len(header)
    return [dict(zip(header, row, strict=True)) for row in body]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    meta, parsed = [], {}
    for name in ("project-download.xlsx", "sample-download.xlsx", "datasets-download.xlsx"):
        path = args.data_dir / name
        content = path.read_bytes()
        receipt = json.loads(path.with_suffix(".xlsx.receipt.json").read_text())
        assert hashlib.sha256(content).hexdigest() == receipt["sha256"]
        parsed[name] = read(path)
        meta.append(
            {
                "path": str(path),
                "receipt": receipt,
                "sheets": [{k: v for k, v in s.items() if k != "rows"} for s in parsed[name]],
            }
        )
    projects = flat(parsed["project-download.xlsx"][0])
    samples = flat(parsed["sample-download.xlsx"][0])
    pkey, skey = "PeptidomicsDB project ID", "PeptidomicsDB sample ID"
    pindex = {x[pkey]: x for x in projects}
    assert len(pindex) == len(projects)
    assert len({x[skey] for x in samples}) == len(samples)
    assert all(x[pkey] in pindex for x in samples)
    groups = defaultdict(list)
    for item in samples:
        groups[item[pkey]].append(item)
    candidates, nonmhc, discrepancies = [], [], []
    for project in projects:
        project_id = project[pkey]
        hits = []
        for field in ("Title", "Description", "Tissue"):
            matches = TERMS.findall(str(project.get(field, "")))
            if matches:
                hits.append({"scope": "project", "field": field, "terms": sorted(set(matches))})
        for sample in groups[project_id]:
            for field in ("Tissue", "Condition"):
                matches = TERMS.findall(str(sample.get(field, "")))
                if matches:
                    hits.append(
                        {"scope": sample[skey], "field": field, "terms": sorted(set(matches))}
                    )
        item = {
            "project": project,
            "n_sample_rows": len(groups[project_id]),
            "sample_categories": dict(
                Counter(x["Peptidomics category"] for x in groups[project_id])
            ),
            "sample_species": sorted({str(x["Species"]) for x in groups[project_id]}),
            "sample_tissues": sorted({str(x["Tissue"]) for x in groups[project_id]}),
            "metadata_term_hits": hits,
            "eligibility": "primary_source_review_required",
        }
        if hits:
            candidates.append(item)
        category = str(project.get("Peptidomics category", ""))
        species = str(project.get("Species", ""))
        if not re.search(r"MHC|immunopeptid", category, re.IGNORECASE) and re.search(
            r"Homo sapiens|Mus musculus", species, re.IGNORECASE
        ):
            nonmhc.append(item)
        declared = project["No. of biological samples"]
        if declared != len(groups[project_id]):
            discrepancies.append(
                {
                    "project_id": project_id,
                    "declared_biological_samples": declared,
                    "n_sample_rows": len(groups[project_id]),
                }
            )
    result = {
        "stage": "metadata_only_no_training_admission",
        "n_projects": len(projects),
        "n_sample_rows": len(samples),
        "n_unique_raw_file_names": len({x["Raw file"] for x in samples}),
        "project_categories": dict(Counter(x["Peptidomics category"] for x in projects)),
        "sample_categories": dict(Counter(x["Peptidomics category"] for x in samples)),
        "n_immune_term_candidate_projects": len(candidates),
        "n_human_mouse_non_mhc_label_projects": len(nonmhc),
        "project_sample_count_discrepancies": discrepancies,
        "search_regex": TERMS.pattern,
        "assets": meta,
        "elapsed_seconds": time.perf_counter() - start,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "limitations": [
            "Metadata labels are source assertions, not independently verified assay identities",
            "Immune-cell term matches can occur in background text; they do not prove sample eligibility",
            "Enzyme=None can be a search setting and does not prove native peptide extraction",
            "Sample rows/raw-file names do not establish independent biological specimens",
        ],
    }
    for name, value in [
        ("inventory.json", result),
        ("immune-candidates.json", candidates),
        ("human-mouse-non-mhc-labels.json", nonmhc),
    ]:
        (args.out / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "assets"}, indent=2))
    for item in candidates:
        p = item["project"]
        print(
            json.dumps(
                {
                    k: p.get(k)
                    for k in [
                        pkey,
                        "Original project accession",
                        "Title",
                        "Peptidomics category",
                        "Tissue",
                    ]
                }
            )
        )


if __name__ == "__main__":
    main()
