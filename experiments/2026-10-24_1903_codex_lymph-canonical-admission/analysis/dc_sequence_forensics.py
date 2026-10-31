"""Test source-annotation hypotheses; never mutate the original peptide strings."""

import gzip
import json
import re
import resource
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from Bio.SeqUtils import molecular_weight
from pyteomics.mass import fast_mass

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT.parents[1] / "data/lymph_source_qualification/2026-10-24_1903_canonical"
DELTA = re.compile(r"\(([+-][0-9]+(?:\.[0-9]+)?)\)")


def masses(raw):
    sequence = DELTA.sub("", raw)
    assert re.fullmatch("[ACDEFGHIKLMNPQRSTVWY]+", sequence)
    first = fast_mass(sequence)
    second = molecular_weight(sequence, seq_type="protein", monoisotopic=True)
    assert abs(first - second) <= 0.002, (sequence, first, second)
    return sequence, first + sum(float(delta) for delta in DELTA.findall(raw))


def main():
    started = time.perf_counter()
    output = ROOT / "results/dc-sequence-forensics.json"
    assert not output.exists()
    shutil.copy2(__file__, ROOT / "reproduce/source/dc_sequence_forensics.py")
    typed = PACKAGE / "nanaware_lymph/typed"
    records = {
        row["source_record_id"]: row
        for row in pq.read_table(typed / "source_records.parquet").to_pylist()
    }
    observations = pq.read_table(typed / "observations.parquet").to_pylist()
    identified = {
        row["peptide_sequence"]
        for row in pq.read_table(typed / "identifications.parquet").to_pylist()
        if row["source_file"].startswith("MSV000094761-")
    }
    counts = defaultdict(Counter)
    details = []
    for row in observations:
        block = row["block_id"]
        if not block.startswith("dc_"):
            continue
        cells = json.loads(records[row["source_record_id"]]["source_row_json"])["cells"]
        source_mass = float(cells["E" if block == "dc_dia_database" else "AO"])
        source_length = int(cells["F" if block == "dc_dia_database" else "AP"])
        raw = row["peptide_raw"]
        prefix = re.match(
            r"^[ACDEFGHIKLMNPQRSTVWY](?:\([+-][0-9]+(?:\.[0-9]+)?\))*", raw
        )
        assert prefix is not None
        full, full_mass = masses(raw)
        shortened, shortened_mass = masses(raw[prefix.end() :])
        item = {
            "observation_id": row["observation_id"],
            "source_record_id": row["source_record_id"],
            "block": block,
            "source_peptide_raw": raw,
            "source_mass": source_mass,
            "source_length": source_length,
            "literal_sequence": full,
            "minus_first_sequence_hypothesis": shortened,
            "literal_mass_residual_Da": source_mass - full_mass,
            "minus_first_mass_residual_Da": source_mass - shortened_mass,
            "literal_found_in_deposit": full in identified,
            "minus_first_found_in_deposit": shortened in identified,
            "source_original_unchanged": True,
        }
        counter = counts[block]
        counter["n_source_rows"] += 1
        counter["literal_mass_match"] += abs(item["literal_mass_residual_Da"]) <= 0.1
        counter["minus_first_mass_match"] += (
            abs(item["minus_first_mass_residual_Da"]) <= 0.1
        )
        counter["literal_length_match"] += len(full) == source_length
        counter["minus_first_length_match"] += len(shortened) == source_length
        counter["literal_found_in_deposit"] += item["literal_found_in_deposit"]
        counter["minus_first_found_in_deposit"] += item["minus_first_found_in_deposit"]
        details.append(item)
    with gzip.GzipFile(
        filename=str(ROOT / "results/dc-sequence-forensics-rows.json.gz"),
        mode="wb",
        mtime=0,
    ) as handle:
        handle.write(json.dumps(details, separators=(",", ":")).encode())
    result = {
        "status": "completed_annotation_hypothesis_check",
        "blocks": {block: dict(count) for block, count in counts.items()},
        "mass_tolerance_Da": 0.1,
        "independent_calculator_tolerance_Da": 0.002,
        "source_correction_or_training_admission": False,
        "model_training_or_predictions": False,
        "seconds": time.perf_counter() - started,
        "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3,
    }
    assert result["seconds"] <= 600 and result["peak_rss_gib"] <= 3
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
