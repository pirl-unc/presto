"""Artificial input fixture checks whether current Hitlist preserves PTM fields."""

import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile

SOURCE = Path("/private/tmp/presto-mhc2-20261026/hitlist-current")
sys.path.insert(0, str(SOURCE))

from hitlist.supplement import scan_supplementary  # noqa: E402


def main():
    family = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="hitlist-chemistry-fixture-") as scratch:
        path = Path(scratch) / "artificial.csv"
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=[
                    "peptide",
                    "mhc_class",
                    "mhc_restriction",
                    "has_ptm",
                    "peptide_modifications",
                ],
            )
            writer.writeheader()
            for peptide, modified, text in [
                ("ACDEFGHIKLMN", False, ""),
                ("NPQRSTVWYACD", True, "artificial modification annotation"),
            ]:
                writer.writerow(
                    dict(
                        peptide=peptide,
                        mhc_class="II",
                        mhc_restriction="HLA-DPA1*01:03/DPB1*04:01",
                        has_ptm=modified,
                        peptide_modifications=text,
                    )
                )
        frame = scan_supplementary(
            classify_source=False,
            allow_download=False,
            directory=scratch,
            entries=[
                {
                    "pmid": 37301199,
                    "file": path.name,
                    "defaults": {"species": "Homo sapiens", "source_organism": "Homo sapiens"},
                }
            ],
        )
        result = {
            "fixture_kind": "artificial software fixture; not experimental evidence",
            "source_commit": "13b91424295e01286db46e8e9818fcc2fe816834",
            "source_module": str(SOURCE / "hitlist/supplement.py"),
            "source_sha256": hashlib.sha256(
                (SOURCE / "hitlist/supplement.py").read_bytes()
            ).hexdigest(),
            "input_rows": 2,
            "output_rows": len(frame),
            "input_has_ptm": [False, True],
            "output_has_ptm_column": "has_ptm" in frame,
            "output_peptide_modifications_column": "peptide_modifications" in frame,
        }
        (family / "results/supplement-chemistry-fixture.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
