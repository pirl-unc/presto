"""Inventory original deposited layers; this does not create training labels."""

import csv
import hashlib
import io
import json
import shutil
import time
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path("/private/tmp/presto-digestion-assets")


def sha(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def dump(name, value):
    (ROOT / "results" / name).write_text(json.dumps(value, indent=2) + "\n")


def main():
    started = time.monotonic()
    archive = CACHE / "aSPIRE_outputs_zip.zip"
    assert archive.stat().st_size == 963763954
    prior = json.loads((ROOT / "results/outer-zip-inventory.json").read_text())
    prior = {x["name"]: x for x in prior["members"]}
    output = CACHE / "original-layers"
    output.mkdir(exist_ok=True)
    inventories, summaries = [], []
    with zipfile.ZipFile(archive) as outer:
        assert set(outer.namelist()) == set(prior)
        for info in outer.infolist():
            assert info.file_size == prior[info.filename]["size"]
            assert info.CRC == prior[info.filename]["crc32"]
            assert info.header_offset == prior[info.filename]["header_offset"]
            if info.is_dir():
                continue
            destination = output / Path(info.filename).name
            # ZipExtFile validates the original CRC while copying.
            with outer.open(info) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
            layer_dir = output / destination.stem
            layer_dir.mkdir(exist_ok=True)
            members = []
            with zipfile.ZipFile(destination) as inner:
                assert inner.testzip() is None
                for entry in inner.infolist():
                    members.append(
                        {
                            "name": entry.filename,
                            "bytes": entry.file_size,
                            "compressed_bytes": entry.compress_size,
                            "crc32": entry.CRC,
                        }
                    )
                    if entry.is_dir():
                        continue
                    name = Path(entry.filename).name
                    if "/plots/" not in entry.filename and (
                        name.endswith((".RData", "finalKinetics.csv"))
                        or name == "log.txt"
                    ):
                        with (
                            inner.open(entry) as source,
                            (layer_dir / name).open("wb") as target,
                        ):
                            shutil.copyfileobj(source, target)
                    if name.endswith("finalKinetics.csv"):
                        types, schedules, reps = Counter(), Counter(), Counter()
                        peptides, paired, sequences = set(), Counter(), set()
                        n_rows = n_values = zeros = negatives = missing = 0
                        with inner.open(entry) as source:
                            reader = csv.DictReader(io.TextIOWrapper(source))
                            columns = reader.fieldnames
                            for row in reader:
                                n_rows += 1
                                times = row["digestTimes"].split(";")
                                values = row["intensities"].split(";")
                                assert len(times) == len(values)
                                n_values += len(values)
                                zeros += sum(
                                    x != "NA" and float(x) == 0 for x in values
                                )
                                negatives += sum(
                                    x != "NA" and float(x) < 0 for x in values
                                )
                                missing += values.count("NA")
                                types[row["productType"]] += 1
                                schedules[row["digestTimes"]] += 1
                                reps[row["biological_replicate"]] += 1
                                peptides.add(row["pepSeq"])
                                paired[row["pepSeq"]] += 1
                                sequences.add(row["substrateSeq"])
                        summaries.append(
                            {
                                "archive": destination.name,
                                "member": entry.filename,
                                "columns": columns,
                                "rows": n_rows,
                                "unique_peptides": len(peptides),
                                "values": n_values,
                                "zeros": zeros,
                                "negatives": negatives,
                                "missing": missing,
                                "product_types": dict(types),
                                "schedules": dict(schedules),
                                "biological_replicates": dict(reps),
                                "rows_per_peptide": dict(Counter(paired.values())),
                                "substrate_sequence_count": len(sequences),
                            }
                        )
                    elif name == "MS1_HPR.csv":
                        with inner.open(entry) as source:
                            reader = csv.reader(io.TextIOWrapper(source))
                            members[-1]["columns"] = next(reader)
                            members[-1]["rows"] = sum(1 for _ in reader)
            inventories.append(
                {
                    "archive": destination.name,
                    "bytes": destination.stat().st_size,
                    "sha256": sha(destination),
                    "all_member_crcs_verified": True,
                    "members": members,
                }
            )
            print(destination.name, "verified", len(members), flush=True)
            dump("archive-inventory.json", inventories)
            dump("final-table-schema.json", summaries)
    receipt = {
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": sha(archive),
        "original_tail_directory_matches": True,
        "nested_archives": len(inventories),
        "all_outer_and_inner_crcs_verified": True,
        "final_tables": len(summaries),
        "final_rows": sum(x["rows"] for x in summaries),
        "final_vector_values": sum(x["values"] for x in summaries),
        "runtime_seconds": time.monotonic() - started,
        "scope": "Source schema and coverage only; no fit or prediction metrics",
    }
    dump("archive-audit.json", receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
